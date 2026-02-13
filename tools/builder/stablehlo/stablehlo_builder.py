# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import inspect
import math
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Callable, Dict, Any, Sequence
import torch
from enum import Enum, auto
import re
from collections import OrderedDict
import math

from ttmlir.ir import *
from ttmlir.dialects import stablehlo, sdy, mpmd, func

from builder.base.builder import *
from builder.base.builder_utils import *

from golden import get_golden_function, apply_sharding, apply_unsharding



class StableHLOBuilder(Builder):

    # ----- Methods -----

    def __init__(
        self,
        ctx: Context,
        location: Location,
        mesh_name: Union[List[str], str] = "mesh",
        mesh_dict: Union[
            List[OrderedDict[str, int]], OrderedDict[str, int]
        ] = OrderedDict([("x", 1), ("y", 1)]),
        disable_golden_check: bool = False,
    ):
            super().__init__(ctx, location, mesh_name, mesh_dict, disable_golden_check)

        # ----- Class helper methods -----

    @classmethod
    def build_opname_to_opview_map(cls):
        for name, obj in inspect.getmembers(stablehlo, inspect.isclass):
            if issubclass(obj, OpView) and obj is not OpView:
                op_name = getattr(obj, "OPERATION_NAME", None)

                if op_name is not None:
                    cls.opname_to_opview_map[op_name] = obj

        # ----- Private Methods ----

    def _create_mesh_attr_from_ordered_dict(
        self,
        mesh_dict: OrderedDict[str, int],
    ) -> sdy.MeshAttr:
        axes = [
            self.mesh_axis_attr(name=axis_name, size=size)
            for axis_name, size in mesh_dict.items()
        ]
        return self.mesh_attr(axes)

    def _get_mesh_attr(self, mesh_name: str) -> sdy.MeshAttr:
        if mesh_name not in self._meshes:
            raise ValueError(
                f"Mesh '{mesh_name}' not found. Available meshes: {list(self._meshes.keys())}"
            )

        mesh_dict = self._meshes[mesh_name]
        axes = [
            self.mesh_axis_attr(name=axis_name, size=size)
            for axis_name, size in mesh_dict.items()
        ]
        return self.mesh_attr(axes)

    def _get_mesh(self, mesh_name: str = "mesh") -> sdy.Mesh:
        return self.mesh(mesh_name, self._get_mesh_attr(mesh_name))

    def _get_output_shape_and_type(
        self,
        organize_golden_args: Callable,
        inputs: List[Operand],
        op_stablehlo_function: Callable,
        golden_kwargs: dict = {},
    ):
        op_golden_function = get_golden_function(op_stablehlo_function, **golden_kwargs)
        if op_golden_function is None:
            return None

        # If the op has no input, just call golden function with kwargs (e.g., zeros).
        if len(inputs) == 0:
            golden_output = op_golden_function(**golden_kwargs)
        else:
            golden_output = op_golden_function(
                *(organize_golden_args(inputs)), **golden_kwargs
            )

        return golden_output.shape, golden_output.dtype

    def _op_proxy(
        self,
        op_stablehlo_function: Callable,
        inputs: List[Operand],
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
        organize_stablehlo_args: Optional[Callable] = None,
        organize_golden_args: Optional[Callable] = None,
        output_shape: Optional[Shape] = None,
        output_type: Optional[Type] = None,
        output_create_fn: Optional[Callable] = None,
        golden_kwargs: dict = {},
        stablehlo_kwargs: dict = {},
        loc: Optional[Union[str, Location]] = None,
        skip_golden: bool = False,
    ) -> Any:
        if not golden_kwargs:
            golden_kwargs = stablehlo_kwargs

        if organize_golden_args is None:
            organize_golden_args = self._organize_eltwise_golden

        with self._ctx, self._loc:
            id = self._get_next_global_id()
            loc = (
                self._get_loc_from_str(loc)
                if loc is not None
                else self._get_loc_of_extra_file_callee(id=id)
            )

            # Most StableHLO ops have MLIR type inference, so output is not needed.
            # Only create output if user explicitly provides output_shape, output_type, or output_create_fn
            # (e.g., for ops like broadcast_in_dim that don't have type inference)
            output = None
            if (
                output_shape is not None
                or output_type is not None
                or output_create_fn is not None
            ):
                # User explicitly requested output creation
                # Try to get shape/type from golden function if not fully provided
                output_shape_and_type = self._get_output_shape_and_type(
                    organize_golden_args, inputs, op_stablehlo_function, golden_kwargs
                )

                if not output_shape_and_type:
                    # No golden function - user must provide both shape and type
                    assert (
                        output_shape is not None
                    ), "Output shape must be provided if there is no golden function for this op"
                    assert (
                        output_type is not None
                    ), "Output type must be provided if there is no golden function for this op"
                else:
                    (
                        calculated_output_shape,
                        calculated_output_type,
                    ) = output_shape_and_type
                    # Use provided values if available, otherwise use calculated
                    output_shape = (
                        calculated_output_shape
                        if output_shape is None
                        else output_shape
                    )
                    output_type = (
                        self._get_type_from_torch_dtype(calculated_output_type)
                        if output_type is None
                        else output_type
                    )

                # Create output tensor
                if output_create_fn is not None:
                    output = output_create_fn(output_shape, output_type)
                else:
                    output = self._create_ranked_tensor_type(output_shape, output_type)

            # Custom argument organization and create the stabelhlo op
            if organize_stablehlo_args is not None:
                stablehlo_args = organize_stablehlo_args(
                    inputs, output, stablehlo_kwargs
                )
                op = op_stablehlo_function(*stablehlo_args, loc=loc, **stablehlo_kwargs)
            else:
                # Default: elementwise binary operations
                op = op_stablehlo_function(*inputs, loc=loc, **stablehlo_kwargs)

            if unit_attrs is not None:
                for attr_name in unit_attrs:
                    op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

            if sharding_attr is not None:
                op.operation.attributes["sdy.sharding"] = sharding_attr

            if not skip_golden and not self._disable_golden_check:
                op_golden_function = get_golden_function(
                    op_stablehlo_function, **golden_kwargs
                )
                if op_golden_function is not None:
                    golden_output = op_golden_function(
                        *(organize_golden_args(inputs)), **golden_kwargs
                    )
                    self._set_golden_tensor(op.result, golden_output)

            return op.result

    def _eltwise_proxy(
        self,
        op_stablehlo_function: Callable,
        inputs: List[Operand],
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        return self._op_proxy(op_stablehlo_function, inputs, unit_attrs, sharding_attr)

    def create_tensor_encoding(
        self, shape: Shape, element_type: Union[torch.dtype, TypeInfo]
    ) -> ttnn.ir.TTNNLayoutAttr:
        return None

    def _get_location(self) -> Location:
        stack = inspect.stack()
        caller_frame = stack[2]
        filename = caller_frame.filename
        lineno = caller_frame.lineno
        return Location.name(f"{filename}:{lineno}")

    # ----- Public StableHLO Op Generators ----

    ############### stablehlo.ReduceScatterOp ###############

    @tag(stablehlo.ReduceScatterOp)
    def reduce_scatter(
        self,
        input: Operand,
        scatter_dimensions: int,
        replica_groups: List[List[int]],
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.reduce_scatter)

        scatter_dimensions_attr = IntegerAttr.get(
            IntegerType.get_signless(64), scatter_dimensions
        )
        replica_groups_attr = DenseElementsAttr.get(np.array(replica_groups))
        input0 = self._get_golden_tensor(input)
        op_golden_function = get_golden_function(stablehlo_op)
        golden_output = op_golden_function(
            input0, scatter_dimensions_attr, replica_groups_attr
        )
        result = self._create_ranked_tensor_type(
            golden_output.shape, self.get_type(input)
        )

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            result,
            input,
            scatter_dimensions_attr,
            replica_groups_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.ReduceScatterOp)
    def reduce_scatter_parser(
        self,
        old_op: stablehlo.ReduceScatterOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(
            StableHLOBuilder.reduce_scatter_parser
        )

        input = global_dict[old_op.operand]
        scatter_dimensions_attr = old_op.scatter_dimensions
        replica_groups_attr = old_op.replica_groups

        new_op = stablehlo_op(
            old_op.result.type,
            input,
            scatter_dimensions_attr,
            replica_groups_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, scatter_dimensions_attr, replica_groups_attr
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### stablehlo.CollectivePermuteOp ###############

    @tag(stablehlo.CollectivePermuteOp)
    def collective_permute(
        self,
        input: Operand,
        source_target_pairs: List[Tuple[int, int]],
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.collective_permute)

        source_target_pairs_attr = DenseElementsAttr.get(np.array(source_target_pairs))

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            input,
            source_target_pairs_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, source_target_pairs_attr)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.CollectivePermuteOp)
    def collective_permute_parser(
        self,
        old_op: stablehlo.CollectivePermuteOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(
            StableHLOBuilder.collective_permute_parser
        )

        input = global_dict[old_op.operand]
        source_target_pairs_attr = old_op.source_target_pairs

        new_op = stablehlo_op(
            old_op.result.type,
            input,
            source_target_pairs_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, source_target_pairs_attr)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### stablehlo.CollectiveBroadcastOp ###############

    @tag(stablehlo.CollectiveBroadcastOp)
    def collective_broadcast(
        self,
        input: Operand,
        replica_groups: List[List[int]],
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(
            StableHLOBuilder.collective_broadcast
        )

        replica_groups_attr = DenseElementsAttr.get(np.array(replica_groups))

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            input,
            replica_groups_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, replica_groups_attr)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.CollectiveBroadcastOp)
    def collective_broadcast_parser(
        self,
        old_op: stablehlo.CollectiveBroadcastOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(
            StableHLOBuilder.collective_broadcast_parser
        )

        input = global_dict[old_op.operand]
        replica_groups_attr = old_op.replica_groups

        new_op = stablehlo_op(
            old_op.result.type,
            input,
            replica_groups_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, replica_groups_attr)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### stablehlo.AllToAllOp ###############

    @tag(stablehlo.AllToAllOp)
    def all_to_all(
        self,
        input: Operand,
        split_dim: int,
        concat_dim: int,
        split_count: int,
        replica_groups: List[List[int]],
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.all_to_all)

        split_dim_attr = IntegerAttr.get(IntegerType.get_signless(64), split_dim)
        concat_dim_attr = IntegerAttr.get(IntegerType.get_signless(64), concat_dim)
        split_count_attr = IntegerAttr.get(IntegerType.get_signless(64), split_count)
        replica_groups_attr = DenseElementsAttr.get(np.array(replica_groups))

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            input,
            split_dim_attr,
            concat_dim_attr,
            split_count_attr,
            replica_groups_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0,
                split_dim_attr,
                concat_dim_attr,
                split_count_attr,
                replica_groups_attr,
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.AllToAllOp)
    def all_to_all_parser(
        self,
        old_op: stablehlo.AllToAllOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.all_to_all_parser)

        input = global_dict[old_op.operand]
        split_dim_attr = old_op.split_dim
        concat_dim_attr = old_op.concat_dim
        split_count_attr = old_op.split_count
        replica_groups_attr = old_op.replica_groups

        new_op = stablehlo_op(
            input,
            split_dim_attr,
            concat_dim_attr,
            split_count_attr,
            replica_groups_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0,
                split_dim_attr,
                concat_dim_attr,
                split_count_attr,
                replica_groups_attr,
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### stablehlo.AllReduceOp ###############

    @tag(stablehlo.AllReduceOp)
    def all_reduce(
        input: Operand,
        replica_groups: Optional[List[List[int]]],
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.all_reduce)

        replica_groups_attr = DenseElementsAttr.get(np.array(replica_groups))
        input0 = self._get_golden_tensor(input)
        op_golden_function = get_golden_function(stablehlo_op)
        golden_output = op_golden_function(input0, replica_groups_attr)
        result = self._create_ranked_tensor_type(
            golden_output.shape, self.get_type(input)
        )

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            result,
            input,
            replica_groups_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.AllReduceOp)
    def all_reduce_parser(
        self,
        old_op: stablehlo.AllReduceOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.all_reduce_parser)

        input = global_dict[old_op.operand]
        replica_groups_attr = old_op.replica_groups

        new_op = stablehlo_op(
            old_op.result.type,
            input,
            replica_groups_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, replica_groups_attr)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### stablehlo.AllGatherOp ###############

    def infer_all_gather_output_shape(
        self,
        input: Operand,
        all_gather_dim: int,
        replica_groups: Optional[List[List[int]]],
    ) -> List[int]:
        input_type = input.type
        input_shape = input_type.shape

        if replica_groups is None:
            return input_shape

        group_size = len(replica_groups[0])
        output_shape = list(input_shape)
        output_shape[all_gather_dim] = input_shape[all_gather_dim] * group_size
        return output_shape

    @tag(stablehlo.AllGatherOp)
    def all_gather(
        self,
        input: Operand,
        all_gather_dim: int,
        replica_groups: Optional[List[List[int]]],
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.all_gather)

        all_gather_dim_attr = IntegerAttr.get(
            IntegerType.get_signless(64), all_gather_dim
        )
        replica_groups_attr = DenseElementsAttr.get(np.array(replica_groups))
        output_shape = self.infer_all_gather_output_shape(
            input, all_gather_dim, replica_groups
        )
        result = self._create_ranked_tensor_type(output_shape, self.get_type(input))

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            [result],
            [input],
            all_gather_dim_attr,
            replica_groups_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, all_gather_dim_attr, replica_groups_attr
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.AllGatherOp)
    def all_gather_parser(
        self,
        old_op: stablehlo.AllGatherOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.all_gather_parser)

        input = global_dict[old_op.operand]
        all_gather_dim_attr = old_op.all_gather_dim
        replica_groups_attr = old_op.replica_groups

        new_op = stablehlo_op(
            [old_op.result.type],
            [input],
            all_gather_dim_attr,
            replica_groups_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, all_gather_dim_attr, replica_groups_attr
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### stablehlo.DynamicUpdateSliceOp ###############

    @tag(stablehlo.DynamicUpdateSliceOp)
    def dynamic_update_slice(
        self,
        input: Operand,
        update: Operand,
        start_indices: List[Operand],
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(
            StableHLOBuilder.dynamic_update_slice
        )

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            input,
            update,
            start_indices,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input_tensor = self._get_golden_tensor(input)
            update_tensor = self._get_golden_tensor(update)
            start_indices_tensors = [
                self._get_golden_tensor(idx) for idx in start_indices
            ]
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input_tensor,
                update_tensor,
                start_indices_tensors,
                op.result.type.element_type,
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.DynamicUpdateSliceOp)
    def dynamic_update_slice_parser(
        self,
        old_op: stablehlo.DynamicUpdateSliceOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(
            StableHLOBuilder.dynamic_update_slice_parser
        )

        input = global_dict[old_op.operand]
        update = global_dict[old_op.update]
        start_indices = [global_dict[idx] for idx in old_op.start_indices]

        new_op = stablehlo_op(
            input,
            update,
            start_indices,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input_tensor = self._get_golden_tensor(input)
            update_tensor = self._get_golden_tensor(update)
            start_indices_tensors = [
                self._get_golden_tensor(idx) for idx in start_indices
            ]
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input_tensor,
                update_tensor,
                start_indices_tensors,
                new_op_result.type.element_type,
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.DynamicUpdateSliceOp)
    def dynamic_update_slice_split(
        self,
        old_op: stablehlo.DynamicUpdateSliceOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(
            StableHLOBuilder.dynamic_update_slice_split
        )

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            dynamic_update_slice_module = Module.create()
            dynamic_update_slice_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.operand.type,
                old_op.update.type,
            ] + [idx.type for idx in old_op.start_indices]

            with InsertionPoint(dynamic_update_slice_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="dynamic_update_slice_module")
                def decorated_func(*inputs):
                    input = inputs[0]
                    update = inputs[1]
                    start_indices = inputs[2 : 2 + len(old_op.start_indices)]

                    new_op = stablehlo_op(
                        input, update, start_indices, loc=old_op.location
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input_tensor = self._get_golden_tensor(old_op.operand)
                        update_tensor = self._get_golden_tensor(old_op.update)
                        start_indices_tensors = [
                            self._get_golden_tensor(idx) for idx in old_op.start_indices
                        ]
                        golden_output = op_golden_function(
                            input_tensor,
                            update_tensor,
                            start_indices_tensors,
                            new_op_result.type.element_type,
                        )
                        dynamic_update_slice_builder._set_golden_tensor(
                            new_op_result, golden_output
                        )
                        dynamic_update_slice_builder._set_golden_tensor(
                            input, input_tensor
                        )
                        dynamic_update_slice_builder._set_golden_tensor(
                            update, update_tensor
                        )
                        ordered_inputs.extend([input, update] + list(start_indices))
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                dynamic_update_slice_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return dynamic_update_slice_module, dynamic_update_slice_builder

        ############### stablehlo.AddOp ###############

    @tag(stablehlo.AddOp)
    def add(
        self,
        in0: Operand,
        in1: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.add)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            in1,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            input1 = self._get_golden_tensor(in1)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, op.result.type.element_type,
                comparison_direction="EQ"
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.AddOp)
    def add_parser(
        self,
        old_op: stablehlo.AddOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.add_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]

        new_op = stablehlo_op(
            lhs,
            rhs,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, new_op_result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.AddOp)
    def add_split(
        self,
        old_op: stablehlo.AddOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.add_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            add_module = Module.create()
            add_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(add_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="add_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]

                    new_op = stablehlo_op(lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, new_op_result.type.element_type
                        )
                        add_builder._set_golden_tensor(new_op_result, golden_output)
                        add_builder._set_golden_tensor(lhs, input0)
                        add_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                add_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return add_module, add_builder

        ################ stablehlo.AndOp ###############

    @tag(stablehlo.AndOp)
    def logical_and(
        self,
        in0: Operand,
        in1: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.logical_and)

        op = stablehlo_op(
            in0,
            in1,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            input1 = self._get_golden_tensor(in1)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, op_result.type.element_type
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.AndOp)
    def logical_and_parser(
        self,
        old_op: stablehlo.AndOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.logical_and_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]

        new_op = stablehlo_op(
            lhs,
            rhs,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, new_op_result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.AndOp)
    def logical_and_split(
        self,
        old_op: stablehlo.AndOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.logical_and_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            and_module = Module.create()
            and_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(and_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="and_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]

                    new_op = stablehlo_op(lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, new_op_result.type.element_type
                        )
                        and_builder._set_golden_tensor(new_op_result, golden_output)
                        and_builder._set_golden_tensor(lhs, input0)
                        and_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                and_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return and_module, and_builder

        ################ stablehlo.AbsOp ###############

    @tag(stablehlo.AbsOp)
    def abs(
        self,
        in0: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.abs)

        op = stablehlo_op(
            in0,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, op_result.type.element_type)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.AbsOp)
    def abs_parser(
        self,
        old_op: stablehlo.AbsOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.abs_parser)
        operand = global_dict[old_op.operand]

        new_op = stablehlo_op(
            operand,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, new_op_result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.AbsOp)
    def abs_split(
        self,
        old_op: stablehlo.AbsOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.abs_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            abs_module = Module.create()
            abs_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.operand.type,
            ]

            with InsertionPoint(abs_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="abs_module")
                def decorated_func(*inputs):
                    operand = inputs[0]

                    new_op = stablehlo_op(operand, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.operand)
                        golden_output = op_golden_function(
                            input0, new_op_result.type.element_type
                        )
                        abs_builder._set_golden_tensor(new_op_result, golden_output)
                        abs_builder._set_golden_tensor(operand, input0)
                        ordered_inputs.append(operand)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                abs_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return abs_module, abs_builder

        ################ stablehlo.GetDimensionSizeOp ###############

    @tag(stablehlo.GetDimensionSizeOp)
    def get_dimension_size(
        self,
        in0: Operand,
        dimension: int = 0,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.get_dimension_size)

        dimension_attr = IntegerAttr.get(
            IntegerType.get_signless(64, self._ctx), dimension
        )

        op = stablehlo_op(
            in0,
            dimension=dimension_attr,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, dimension_attr, op_result.type.element_type
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.GetDimensionSizeOp)
    def get_dimension_size_parser(
        self,
        old_op: stablehlo.GetDimensionSizeOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(
            StableHLOBuilder.get_dimension_size_parser
        )
        operand = global_dict[old_op.operand]

        new_op = stablehlo_op(
            operand,
            dimension=old_op.dimension,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, old_op.dimension, new_op_result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.GetDimensionSizeOp)
    def get_dimension_size_split(
        self,
        old_op: stablehlo.GetDimensionSizeOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(
            StableHLOBuilder.get_dimension_size_split
        )

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            get_dimension_size_module = Module.create()
            get_dimension_size_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.operand.type,
            ]

            with InsertionPoint(get_dimension_size_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="get_dimension_size_module")
                def decorated_func(*inputs):
                    operand = inputs[0]

                    new_op = stablehlo_op(
                        operand, dimension=old_op.dimension, loc=old_op.location
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.operand)
                        golden_output = op_golden_function(
                            input0, old_op.dimension, new_op_result.type.element_type
                        )
                        get_dimension_size_builder._set_golden_tensor(
                            new_op_result, golden_output
                        )
                        get_dimension_size_builder._set_golden_tensor(operand, input0)
                        ordered_inputs.append(operand)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                get_dimension_size_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return get_dimension_size_module, get_dimension_size_builder

        ################ stablehlo.CeilOp ###############

    @tag(stablehlo.CeilOp)
    def ceil(
        self,
        in0: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.ceil)

        op = stablehlo_op(
            in0,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, op_result.type.element_type)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.CeilOp)
    def ceil_parser(
        self,
        old_op: stablehlo.CeilOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.ceil_parser)
        operand = global_dict[old_op.operand]

        new_op = stablehlo_op(
            operand,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, new_op_result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.CeilOp)
    def ceil_split(
        self,
        old_op: stablehlo.CeilOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.ceil_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            ceil_module = Module.create()
            ceil_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.operand.type,
            ]

            with InsertionPoint(ceil_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="ceil_module")
                def decorated_func(*inputs):
                    operand = inputs[0]

                    new_op = stablehlo_op(operand, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.operand)
                        golden_output = op_golden_function(
                            input0, new_op_result.type.element_type
                        )
                        ceil_builder._set_golden_tensor(new_op_result, golden_output)
                        ceil_builder._set_golden_tensor(operand, input0)
                        ordered_inputs.append(operand)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                ceil_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return ceil_module, ceil_builder

        ################ stablehlo.DivOp ###############

    @tag(stablehlo.DivOp)
    def divide(
        self,
        in0: Operand,
        in1: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.divide)

        op = stablehlo_op(
            in0,
            in1,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            input1 = self._get_golden_tensor(in1)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, op_result.type.element_type
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.DivOp)
    def divide_parser(
        self,
        old_op: stablehlo.DivOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.divide_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]

        new_op = stablehlo_op(
            lhs,
            rhs,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, new_op_result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.DivOp)
    def divide_split(
        self,
        old_op: stablehlo.DivOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.divide_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            divide_module = Module.create()
            divide_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(divide_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="divide_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]

                    new_op = stablehlo_op(lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, new_op_result.type.element_type
                        )
                        divide_builder._set_golden_tensor(new_op_result, golden_output)
                        divide_builder._set_golden_tensor(lhs, input0)
                        divide_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                divide_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return divide_module, divide_builder

        ################ stablehlo.CosineOp ###############

    @tag(stablehlo.CosineOp)
    def cosine(
        self,
        in0: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.cosine)

        op = stablehlo_op(
            in0,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, op_result.type.element_type)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.CosineOp)
    def cosine_parser(
        self,
        old_op: stablehlo.CosineOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.cosine_parser)
        operand = global_dict[old_op.operand]

        new_op = stablehlo_op(
            operand,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, new_op_result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.CosineOp)
    def cosine_split(
        self,
        old_op: stablehlo.CosineOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.cosine_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            cosine_module = Module.create()
            cosine_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.operand.type,
            ]

            with InsertionPoint(cosine_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="cosine_module")
                def decorated_func(*inputs):
                    operand = inputs[0]

                    new_op = stablehlo_op(operand, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.operand)
                        golden_output = op_golden_function(
                            input0, new_op_result.type.element_type
                        )
                        cosine_builder._set_golden_tensor(new_op_result, golden_output)
                        cosine_builder._set_golden_tensor(operand, input0)
                        ordered_inputs.append(operand)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                cosine_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return cosine_module, cosine_builder

        ################ stablehlo.ExpOp ###############

    @tag(stablehlo.ExpOp)
    def exp(
        self,
        in0: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.exp)

        op = stablehlo_op(
            in0,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, op_result.type.element_type)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.ExpOp)
    def exp_parser(
        self,
        old_op: stablehlo.ExpOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.exp_parser)
        operand = global_dict[old_op.operand]

        new_op = stablehlo_op(
            operand,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, new_op_result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.ExpOp)
    def exp_split(
        self,
        old_op: stablehlo.ExpOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.exp_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            exp_module = Module.create()
            exp_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.operand.type,
            ]

            with InsertionPoint(exp_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="exp_module")
                def decorated_func(*inputs):
                    operand = inputs[0]

                    new_op = stablehlo_op(operand, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.operand)
                        golden_output = op_golden_function(
                            input0, new_op_result.type.element_type
                        )
                        exp_builder._set_golden_tensor(new_op_result, golden_output)
                        exp_builder._set_golden_tensor(operand, input0)
                        ordered_inputs.append(operand)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                exp_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return exp_module, exp_builder

        ################ stablehlo.FloorOp ###############

    @tag(stablehlo.FloorOp)
    def floor(
        self,
        in0: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.floor)

        op = stablehlo_op(
            in0,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, op_result.type.element_type)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.FloorOp)
    def floor_parser(
        self,
        old_op: stablehlo.FloorOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.floor_parser)
        operand = global_dict[old_op.operand]

        new_op = stablehlo_op(
            operand,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, new_op_result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.FloorOp)
    def floor_split(
        self,
        old_op: stablehlo.FloorOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.floor_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            floor_module = Module.create()
            floor_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.operand.type,
            ]

            with InsertionPoint(floor_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="floor_module")
                def decorated_func(*inputs):
                    operand = inputs[0]

                    new_op = stablehlo_op(operand, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.operand)
                        golden_output = op_golden_function(
                            input0, new_op_result.type.element_type
                        )
                        floor_builder._set_golden_tensor(new_op_result, golden_output)
                        floor_builder._set_golden_tensor(operand, input0)
                        ordered_inputs.append(operand)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                floor_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return floor_module, floor_builder

    def broadcast_in_dim(
        self,
        in0: Operand,
        broadcast_dimensions: List[int],
        output_shape: List[int],
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.broadcast_in_dim``.
        *Tensor broadcast operation.*
        Broadcasts a tensor to a new shape by replicating its values along specified dimensions.
        The broadcast_dimensions parameter specifies how dimensions of the input map to
        dimensions of the output.
        .. code-block:: mlir
            // Broadcast a 1D tensor to 2D
            %result = stablehlo.broadcast_in_dim(%input) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<2x3xf32>
            // Input tensor:
            // [1.0, 2.0, 3.0]
            // Output tensor:
            // [[1.0, 2.0, 3.0],
            //  [1.0, 2.0, 3.0]]
        Parameters
        ----------
        in0 : Operand
            Input tensor to broadcast
        broadcast_dimensions : *List[int]*
            List of dimension mappings from input to output
        output_shape : *List[int]*
            Target shape for the broadcasted tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes
        sharding_attr : *Optional[sdy.TensorShardingPerValueAttr]*, optional
            Optional sharding attribute for distributed execution
        Returns
        -------
        (*OpView*)
            The broadcasted tensor
        """
        output_type = self._get_type(in0).element_type
        return self._op_proxy(
            stablehlo.BroadcastInDimOp,
            [in0],
            organize_golden_args=self._organize_eltwise_golden,
            organize_stablehlo_args=lambda inputs, output, _: (output, inputs[0]),
            output_shape=output_shape,
            output_type=output_type,
            golden_kwargs={"size": output_shape},
            stablehlo_kwargs={
                "broadcast_dimensions": broadcast_dimensions,
            },
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    ################ stablehlo.LogOp ###############

    @tag(stablehlo.LogOp)
    def log(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.log)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(stablehlo_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.LogOp)
    def log_parser(
        self,
        old_op: stablehlo.LogOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.log_parser)
        in0 = global_dict[old_op.operand]

        new_op = stablehlo_op(
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.LogOp)
    def log_split(
        self,
        old_op: stablehlo.LogOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.log_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            log_module = Module.create()
            log_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(log_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="log_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    new_op = stablehlo_op(in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        log_builder._set_golden_tensor(new_op_result, golden_output)
                        log_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                log_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return log_module, log_builder

        ############### stablehlo.NegOp ###############

    @tag(stablehlo.NegOp)
    def neg(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.neg)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(stablehlo_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.NegOp)
    def neg_parser(
        self,
        old_op: stablehlo.NegOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.neg_parser)
        in0 = global_dict[old_op.operand]

        new_op = stablehlo_op(
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.NegOp)
    def neg_split(
        self,
        old_op: stablehlo.NegOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.neg_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            neg_module = Module.create()
            neg_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(neg_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="neg_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    new_op = stablehlo_op(in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        neg_builder._set_golden_tensor(new_op_result, golden_output)
                        neg_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                neg_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return neg_module, neg_builder

        ############### stablehlo.RsqrtOp ###############

    @tag(stablehlo.RsqrtOp)
    def rsqrt(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.rsqrt)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(stablehlo_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.RsqrtOp)
    def rsqrt_parser(
        self,
        old_op: stablehlo.RsqrtOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.rsqrt_parser)
        in0 = global_dict[old_op.operand]

        new_op = stablehlo_op(
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.RsqrtOp)
    def rsqrt_split(
        self,
        old_op: stablehlo.RsqrtOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.rsqrt_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            rsqrt_module = Module.create()
            rsqrt_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(rsqrt_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="rsqrt_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    new_op = stablehlo_op(in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        rsqrt_builder._set_golden_tensor(new_op_result, golden_output)
                        rsqrt_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                rsqrt_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return rsqrt_module, rsqrt_builder

        ############### stablehlo.SineOp ###############

    @tag(stablehlo.SineOp)
    def sine(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.sine)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, mlir_output_type)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.SineOp)
    def sine_parser(
        self,
        old_op: stablehlo.SineOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.sine_parser)
        in0 = global_dict[old_op.operand]

        new_op = stablehlo_op(
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.SineOp)
    def sine_split(
        self,
        old_op: stablehlo.SineOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.sine_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            sine_module = Module.create()
            sine_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(sine_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="sine_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    new_op = stablehlo_op(in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        sine_builder._set_golden_tensor(new_op_result, golden_output)
                        sine_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                sine_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return sine_module, sine_builder

        ############### stablehlo.SqrtOp ###############

    @tag(stablehlo.SqrtOp)
    def sqrt(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.sqrt)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(stablehlo_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.SqrtOp)
    def sqrt_parser(
        self,
        old_op: stablehlo.SqrtOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.sqrt_parser)
        in0 = global_dict[old_op.operand]

        new_op = stablehlo_op(
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.SqrtOp)
    def sqrt_split(
        self,
        old_op: stablehlo.SqrtOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.sqrt_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            sqrt_module = Module.create()
            sqrt_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(sqrt_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="sqrt_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    new_op = stablehlo_op(in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        sqrt_builder._set_golden_tensor(new_op_result, golden_output)
                        sqrt_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

        return sqrt_module, sqrt_builder

        ############### stablehlo.TanOp ###############

    @tag(stablehlo.TanOp)
    def tan(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.tan)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(stablehlo_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.TanOp)
    def tan_parser(
        self,
        old_op: stablehlo.TanOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.tan_parser)
        in0 = global_dict[old_op.operand]

        new_op = stablehlo_op(
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.TanOp)
    def tan_split(
        self,
        old_op: stablehlo.TanOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.tan_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            tan_module = Module.create()
            tan_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(tan_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="tan_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    new_op = stablehlo_op(in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        tan_builder._set_golden_tensor(new_op_result, golden_output)
                        tan_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                tan_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return tan_module, tan_builder

        ############### stablehlo.TanhOp ###############

    @tag(stablehlo.TanhOp)
    def tanh(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.tanh)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(stablehlo_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.TanhOp)
    def tanh_parser(
        self,
        old_op: stablehlo.TanhOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.tanh_parser)
        in0 = global_dict[old_op.operand]

        new_op = stablehlo_op(
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.TanhOp)
    def tanh_split(
        self,
        old_op: stablehlo.TanhOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.tanh_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            tanh_module = Module.create()
            tanh_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(tanh_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="tanh_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    new_op = stablehlo_op(in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        tanh_builder._set_golden_tensor(new_op_result, golden_output)
                        tanh_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                tanh_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return tanh_module, tanh_builder

        ############### stablehlo.Log1pOp ###############

    @tag(stablehlo.Log1pOp)
    def log1p(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.log1p)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(stablehlo_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.Log1pOp)
    def log1p_parser(
        self,
        old_op: stablehlo.Log1pOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.log1p_parser)
        in0 = global_dict[old_op.operand]

        new_op = stablehlo_op(
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.Log1pOp)
    def log1p_split(
        self,
        old_op: stablehlo.Log1pOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.log1p_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            log1p_module = Module.create()
            log1p_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(log1p_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="log1p_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    new_op = stablehlo_op(in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        log1p_builder._set_golden_tensor(new_op_result, golden_output)
                        log1p_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                log1p_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return log1p_module, log1p_builder

        ############### stablehlo.LogisticOp ###############

    @tag(stablehlo.LogisticOp)
    def logistic(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.logistic)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(stablehlo_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.LogisticOp)
    def logistic_parser(
        self,
        old_op: stablehlo.LogisticOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.logistic_parser)
        in0 = global_dict[old_op.operand]

        new_op = stablehlo_op(
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.LogisticOp)
    def logistic_split(
        self,
        old_op: stablehlo.LogisticOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.logistic_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            logistic_module = Module.create()
            logistic_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(logistic_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="logistic_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    new_op = stablehlo_op(in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        logistic_builder._set_golden_tensor(
                            new_op_result, golden_output
                        )
                        logistic_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                logistic_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return logistic_module, logistic_builder

        ############### stablehlo.SliceOp ###############

    @tag(stablehlo.SliceOp)
    def slice(
        self,
        in0: Operand,
        start_indices: List[int],
        limit_indices: List[int],
        strides: List[int],
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.slice)

        start_indices_attr = DenseI64ArrayAttr.get(start_indices)
        limit_indices_attr = DenseI64ArrayAttr.get(limit_indices)
        strides_attr = DenseI64ArrayAttr.get(strides)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            start_indices_attr,
            limit_indices_attr,
            strides_attr,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0,
                start_indices_attr,
                limit_indices_attr,
                strides_attr,
                mlir_output_type,
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.SliceOp)
    def slice_parser(
        self,
        old_op: stablehlo.SliceOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.slice_parser)
        in0 = global_dict[old_op.operand]
        start_indices_attr = old_op.start_indices
        limit_indices_attr = old_op.limit_indices
        strides_attr = old_op.strides

        new_op = stablehlo_op(
            in0,
            start_indices_attr,
            limit_indices_attr,
            strides_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0,
                start_indices_attr,
                limit_indices_attr,
                strides_attr,
                old_op.result.type.element_type,
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.SliceOp)
    def slice_split(
        self,
        old_op: stablehlo.SliceOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.slice_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            slice_module = Module.create()
            slice_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(slice_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="slice_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    start_indices_attr = old_op.start_indices
                    limit_indices_attr = old_op.limit_indices
                    strides_attr = old_op.strides

                    new_op = stablehlo_op(
                        in0,
                        start_indices_attr,
                        limit_indices_attr,
                        strides_attr,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0,
                            start_indices_attr,
                            limit_indices_attr,
                            strides_attr,
                            old_op.result.type.element_type,
                        )
                        slice_builder._set_golden_tensor(new_op_result, golden_output)
                        slice_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                slice_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return slice_module, slice_builder

        ############### stablehlo.DynamicSliceOp ###############

    @tag(stablehlo.DynamicSliceOp)
    def dynamic_slice(
        self,
        operand: Operand,
        start_indices: List[Operand],
        slice_sizes: List[int],
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        # If all start indices are Python ints, use static slice
        if all(isinstance(i, int) for i in start_indices):
            limit_indices = [s + sz for s, sz in zip(start_indices, slice_sizes)]
            strides = [1] * len(start_indices)
            return self.slice(
                operand,
                start_indices,
                limit_indices,
                strides,
                loc=loc,
                unit_attrs=unit_attrs,
                sharding_attr=sharding_attr,
            )

        # Otherwise, ensure every start index is a Value (wrap ints as constants)
        start_indices_vals = [
            (
                self.constant(torch.tensor(i, dtype=torch.int64)).result
                if isinstance(i, int)
                else i
            )
            for i in start_indices
        ]

        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.dynamic_slice)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            operand,
            start_indices_vals,
            slice_sizes=slice_sizes,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            operand_tensor = self._get_golden_tensor(operand)
            start_indices_tensors = [
                self._get_golden_tensor(idx) for idx in start_indices_vals
            ]
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                operand_tensor,
                start_indices_tensors,
                slice_sizes,
                op.result.type.element_type,
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.DynamicSliceOp)
    def dynamic_slice_parser(
        self,
        old_op: stablehlo.DynamicSliceOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(
            StableHLOBuilder.dynamic_slice_parser
        )

        operand = global_dict[old_op.operand]
        start_indices = [global_dict[idx] for idx in old_op.start_indices]
        slice_sizes = list(old_op.slice_sizes)

        new_op = stablehlo_op(
            operand,
            start_indices,
            slice_sizes=slice_sizes,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            operand_tensor = self._get_golden_tensor(operand)
            start_indices_tensors = [
                self._get_golden_tensor(idx) for idx in start_indices
            ]
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                operand_tensor,
                start_indices_tensors,
                slice_sizes,
                new_op_result.type.element_type,
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {old_op.result: new_op_result}
        return new_op, op_map_dictionary

    @split(stablehlo.DynamicSliceOp)
    def dynamic_slice_split(
        self,
        old_op: stablehlo.DynamicSliceOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.dynamic_slice_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            dynamic_slice_module = Module.create()
            dynamic_slice_builder = StableHLOBuilder(old_context, old_loc)

            op_input_types = [old_op.operand.type] + [
                idx.type for idx in old_op.start_indices
            ]
            slice_sizes = list(old_op.slice_sizes)

            with InsertionPoint(dynamic_slice_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="dynamic_slice_module")
                def decorated_func(*inputs):
                    operand = inputs[0]
                    start_indices = inputs[1:]

                    new_op = stablehlo_op(
                        operand,
                        start_indices,
                        slice_sizes=slice_sizes,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        operand_tensor = self._get_golden_tensor(old_op.operand)
                        start_indices_tensors = [
                            self._get_golden_tensor(idx) for idx in old_op.start_indices
                        ]
                        golden_output = op_golden_function(
                            operand_tensor,
                            start_indices_tensors,
                            slice_sizes,
                            new_op_result.type.element_type,
                        )
                        dynamic_slice_builder._set_golden_tensor(
                            new_op_result, golden_output
                        )
                        dynamic_slice_builder._set_golden_tensor(
                            operand, operand_tensor
                        )

                        ordered_inputs.extend([operand] + list(start_indices))
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                dynamic_slice_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return dynamic_slice_module, dynamic_slice_builder

        ############### stablehlo.TransposeOp ###############

    @tag(stablehlo.TransposeOp)
    def transpose(
        self,
        in0: Operand,
        permutation: List[int],
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.transpose)
        permutation_attr = DenseI64ArrayAttr.get(permutation)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            permutation_attr,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, permutation_attr, op_result.type.element_type
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.TransposeOp)
    def transpose_parser(
        self,
        old_op: stablehlo.TransposeOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.transpose_parser)
        in0 = global_dict[old_op.operand]
        permutation_attr = old_op.permutation

        new_op = stablehlo_op(
            in0,
            permutation_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, permutation_attr, old_op.result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.TransposeOp)
    def transpose_split(
        self,
        old_op: stablehlo.TransposeOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.transpose_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            transpose_module = Module.create()
            transpose_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(transpose_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="transpose_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    permutation_attr = old_op.permutation

                    new_op = stablehlo_op(
                        in0,
                        permutation_attr,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, permutation_attr, old_op.result.type.element_type
                        )
                        transpose_builder._set_golden_tensor(
                            new_op_result, golden_output
                        )
                        transpose_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                transpose_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return transpose_module, transpose_builder

        ############### stablehlo.PadOp ###############

    @tag(stablehlo.PadOp)
    def pad(
        self,
        in0: Operand,
        padding_value: Operand,
        padding: List[int],
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.pad)
        element_type = in0.type.element_type

        rank = len(in0.type.shape)
        edge_low = [padding[2 * i] for i in range(rank)]
        edge_high = [padding[2 * i + 1] for i in range(rank)]
        # Currently adding PadOp to support interior padding with [0,0..0] only
        interior = [0] * rank

        edge_low_attr = DenseI64ArrayAttr.get(edge_low)
        edge_high_attr = DenseI64ArrayAttr.get(edge_high)
        interior_attr = DenseI64ArrayAttr.get(interior)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            padding_value,
            edge_low_attr,
            edge_high_attr,
            interior_attr,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            padding_value_golden = self._get_golden_tensor(padding_value)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0,
                padding_value_golden,
                edge_low_attr,
                edge_high_attr,
                interior_attr,
                op_result.type.element_type,
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.PadOp)
    def pad_parser(
        self,
        old_op: stablehlo.PadOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.pad_parser)
        in0 = global_dict[old_op.operand]
        padding_value = global_dict[old_op.padding_value]
        edge_low_attr = old_op.edge_padding_low
        edge_high_attr = old_op.edge_padding_high
        interior_attr = old_op.interior_padding

        new_op = stablehlo_op(
            in0,
            padding_value,
            edge_low_attr,
            edge_high_attr,
            interior_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            padding_value_golden = self._get_golden_tensor(padding_value)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0,
                padding_value_golden,
                edge_low_attr,
                edge_high_attr,
                interior_attr,
                old_op.result.type.element_type,
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.PadOp)
    def pad_split(
        self,
        old_op: stablehlo.PadOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.pad_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            pad_module = Module.create()
            pad_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type, old_op.padding_value.type]

            with InsertionPoint(pad_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="pad_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    padding_value = inputs[1]
                    edge_low_attr = old_op.edge_padding_low
                    edge_high_attr = old_op.edge_padding_high
                    interior_attr = old_op.interior_padding

                    new_op = stablehlo_op(
                        in0,
                        padding_value,
                        edge_low_attr,
                        edge_high_attr,
                        interior_attr,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        padding_value_golden = self._get_golden_tensor(
                            old_op.padding_value
                        )
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0,
                            padding_value_golden,
                            edge_low_attr,
                            edge_high_attr,
                            interior_attr,
                            old_op.result.type.element_type,
                        )
                        pad_builder._set_golden_tensor(new_op_result, golden_output)
                        pad_builder._set_golden_tensor(in0, input0)
                        pad_builder._set_golden_tensor(
                            padding_value, padding_value_golden
                        )
                        ordered_inputs.append(in0)
                        ordered_inputs.append(padding_value)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                pad_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return pad_module, pad_builder

        ############### stablehlo.ReshapeOp ###############

    @tag(stablehlo.ReshapeOp)
    def reshape(
        self,
        in0: Operand,
        shape: List[int],
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.reshape)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        result = self._create_ranked_tensor_type(shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, result.shape, op_result.type.element_type
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.ReshapeOp)
    def reshape_parser(
        self,
        old_op: stablehlo.ReshapeOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.reshape_parser)
        in0 = global_dict[old_op.operand]
        shape_attr = old_op.result.type.shape
        result = old_op.result.type

        new_op = stablehlo_op(old_op.result.type, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, shape_attr, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.ReshapeOp)
    def reshape_split(
        self,
        old_op: stablehlo.ReshapeOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.reshape_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            reshape_module = Module.create()
            reshape_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(reshape_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="reshape_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    shape_attr = old_op.result.type.shape
                    result = old_op.result.type

                    new_op = stablehlo_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, shape_attr, result.element_type
                        )
                        reshape_builder._set_golden_tensor(new_op_result, golden_output)
                        reshape_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                reshape_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return reshape_module, reshape_builder

        ############### stablehlo.MaxOp ###############

    @tag(stablehlo.MaxOp)
    def max(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.max)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            in1,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            input1 = self._get_golden_tensor(in1)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, input1, mlir_output_type)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.MaxOp)
    def max_parser(
        self,
        old_op: stablehlo.MaxOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.max_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]

        new_op = stablehlo_op(
            lhs,
            rhs,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, new_op_result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.MaxOp)
    def max_split(
        self,
        old_op: stablehlo.MaxOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.max_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            max_module = Module.create()
            max_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(max_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="max_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]

                    new_op = stablehlo_op(lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, new_op_result.type.element_type
                        )
                        max_builder._set_golden_tensor(new_op_result, golden_output)
                        max_builder._set_golden_tensor(lhs, input0)
                        max_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                max_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return max_module, max_builder

        ############### stablehlo.MinOp ###############

    @tag(stablehlo.MinOp)
    def min(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.min)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            in1,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            input1 = self._get_golden_tensor(in1)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, input1, mlir_output_type)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.MinOp)
    def min_parser(
        self,
        old_op: stablehlo.MinOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.min_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]

        new_op = stablehlo_op(
            lhs,
            rhs,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, new_op_result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.MinOp)
    def min_split(
        self,
        old_op: stablehlo.MinOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.min_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            min_module = Module.create()
            min_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(min_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="min_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]

                    new_op = stablehlo_op(lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, new_op_result.type.element_type
                        )
                        min_builder._set_golden_tensor(new_op_result, golden_output)
                        min_builder._set_golden_tensor(lhs, input0)
                        min_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                min_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return min_module, min_builder

        ############### stablehlo.MulOp ###############

    @tag(stablehlo.MulOp)
    def mul(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.mul)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            in1,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            input1 = self._get_golden_tensor(in1)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, input1, mlir_output_type)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.MulOp)
    def mul_parser(
        self,
        old_op: stablehlo.MulOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.mul_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]

        new_op = stablehlo_op(
            lhs,
            rhs,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, new_op_result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.MulOp)
    def mul_split(
        self,
        old_op: stablehlo.MulOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.mul_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            mul_module = Module.create()
            mul_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(mul_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="mul_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]

                    new_op = stablehlo_op(lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, new_op_result.type.element_type
                        )
                        mul_builder._set_golden_tensor(new_op_result, golden_output)
                        mul_builder._set_golden_tensor(lhs, input0)
                        mul_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                mul_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return mul_module, mul_builder

        ############### stablehlo.SubtractOp ###############

    @tag(stablehlo.SubtractOp)
    def subtract(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.subtract)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            in1,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            input1 = self._get_golden_tensor(in1)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, input1, mlir_output_type)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.SubtractOp)
    def subtract_parser(
        self,
        old_op: stablehlo.SubtractOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.subtract_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]

        new_op = stablehlo_op(
            lhs,
            rhs,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, new_op_result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.SubtractOp)
    def subtract_split(
        self,
        old_op: stablehlo.SubtractOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.subtract_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            sub_module = Module.create()
            sub_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(sub_module.body):

                ordered_inputs = []
                ordered_outputs = []







    @tag(stablehlo.RemOp)
    def remainder(
        self,
        in0: Operand,
        in1: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.remainder)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            in1,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            input1 = self._get_golden_tensor(in1)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, op.result.type.element_type,
                comparison_direction="NE"
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

        
    @parse(stablehlo.RemOp)
    def remainder_parser(
        self,
        old_op: stablehlo.RemOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.remainder_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]

        new_op = stablehlo_op(
            lhs,
            rhs,
            loc=old_op.location,
            unit_attrs=old_op.unit_attrs,
            sharding_attr=old_op.sharding_attr,
        )
        return new_op, {old_op.result: new_op.result}

    @split(stablehlo.RemOp)
    def remainder_split(
        self,
        old_op: stablehlo.RemOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.remainder_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            remainder_module = Module.create()
            remainder_builder = StableHLOBuilder(old_context, old_loc)
            lhs = remainder_builder.placeholder(old_op.lhs.type, "lhs")
            rhs = remainder_builder.placeholder(old_op.rhs.type, "rhs")
            result = remainder_builder.remainder(lhs, rhs)
            remainder_builder.return_op([result])
        
        return remainder_module, remainder_builder

    ############### stablehlo.CompareOp - Equal ###############

    @tag(stablehlo.CompareOp)
    def equal(
        self,
        in0: Operand,
        in1: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.equal)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            in1,
            comparison_direction="EQ",
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            input1 = self._get_golden_tensor(in1)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, op.result.type.element_type,
                comparison_direction="GE"
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

        
    @parse(stablehlo.CompareOp)
    def equal_parser(
        self,
        old_op: stablehlo.CompareOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.equal_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]

        new_op = stablehlo_op(
            lhs,
            rhs,
            loc=old_op.location,
            unit_attrs=old_op.unit_attrs,
            sharding_attr=old_op.sharding_attr,
        )
        return new_op, {old_op.result: new_op.result}

    @split(stablehlo.CompareOp)
    def equal_split(
        self,
        old_op: stablehlo.CompareOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.equal_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            equal_module = Module.create()
            equal_builder = StableHLOBuilder(old_context, old_loc)
            lhs = equal_builder.placeholder(old_op.lhs.type, "lhs")
            rhs = equal_builder.placeholder(old_op.rhs.type, "rhs")
            result = equal_builder.equal(lhs, rhs)
            equal_builder.return_op([result])
        
        return equal_module, equal_builder

    ############### stablehlo.CompareOp - Not Equal ###############

    @tag(stablehlo.CompareOp)
    def not_equal(
        self,
        in0: Operand,
        in1: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.not_equal)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            in1,
            comparison_direction="NE",
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            input1 = self._get_golden_tensor(in1)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, op.result.type.element_type,
                comparison_direction="GT"
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

        
    @parse(stablehlo.CompareOp)
    def not_equal_parser(
        self,
        old_op: stablehlo.CompareOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.not_equal_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]

        new_op = stablehlo_op(
            lhs,
            rhs,
            loc=old_op.location,
            unit_attrs=old_op.unit_attrs,
            sharding_attr=old_op.sharding_attr,
        )
        return new_op, {old_op.result: new_op.result}

    @split(stablehlo.CompareOp)
    def not_equal_split(
        self,
        old_op: stablehlo.CompareOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.not_equal_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            not_equal_module = Module.create()
            not_equal_builder = StableHLOBuilder(old_context, old_loc)
            lhs = not_equal_builder.placeholder(old_op.lhs.type, "lhs")
            rhs = not_equal_builder.placeholder(old_op.rhs.type, "rhs")
            result = not_equal_builder.not_equal(lhs, rhs)
            not_equal_builder.return_op([result])
        
        return not_equal_module, not_equal_builder

    ############### stablehlo.CompareOp - Greater Equal ###############

    @tag(stablehlo.CompareOp)
    def greater_equal(
        self,
        in0: Operand,
        in1: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.greater_equal)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            in1,
            comparison_direction="GE",
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            input1 = self._get_golden_tensor(in1)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, op.result.type.element_type,
                comparison_direction="LE"
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

        
    @parse(stablehlo.CompareOp)
    def greater_equal_parser(
        self,
        old_op: stablehlo.CompareOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.greater_equal_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]

        new_op = stablehlo_op(
            lhs,
            rhs,
            loc=old_op.location,
            unit_attrs=old_op.unit_attrs,
            sharding_attr=old_op.sharding_attr,
        )
        return new_op, {old_op.result: new_op.result}

    @split(stablehlo.CompareOp)
    def greater_equal_split(
        self,
        old_op: stablehlo.CompareOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.greater_equal_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            greater_equal_module = Module.create()
            greater_equal_builder = StableHLOBuilder(old_context, old_loc)
            lhs = greater_equal_builder.placeholder(old_op.lhs.type, "lhs")
            rhs = greater_equal_builder.placeholder(old_op.rhs.type, "rhs")
            result = greater_equal_builder.greater_equal(lhs, rhs)
            greater_equal_builder.return_op([result])
        
        return greater_equal_module, greater_equal_builder

    ############### stablehlo.CompareOp - Greater Than ###############

    @tag(stablehlo.CompareOp)
    def greater_than(
        self,
        in0: Operand,
        in1: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.greater_than)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            in1,
            comparison_direction="GT",
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            input1 = self._get_golden_tensor(in1)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, op.result.type.element_type,
                comparison_direction="LT"
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

        
    @parse(stablehlo.CompareOp)
    def greater_than_parser(
        self,
        old_op: stablehlo.CompareOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.greater_than_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]

        new_op = stablehlo_op(
            lhs,
            rhs,
            loc=old_op.location,
            unit_attrs=old_op.unit_attrs,
            sharding_attr=old_op.sharding_attr,
        )
        return new_op, {old_op.result: new_op.result}

    @split(stablehlo.CompareOp)
    def greater_than_split(
        self,
        old_op: stablehlo.CompareOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.greater_than_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            greater_than_module = Module.create()
            greater_than_builder = StableHLOBuilder(old_context, old_loc)
            lhs = greater_than_builder.placeholder(old_op.lhs.type, "lhs")
            rhs = greater_than_builder.placeholder(old_op.rhs.type, "rhs")
            result = greater_than_builder.greater_than(lhs, rhs)
            greater_than_builder.return_op([result])
        
        return greater_than_module, greater_than_builder

    ############### stablehlo.CompareOp - Less Equal ###############

    @tag(stablehlo.CompareOp)
    def less_equal(
        self,
        in0: Operand,
        in1: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.less_equal)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            in1,
            comparison_direction="LE",
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            input1 = self._get_golden_tensor(in1)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, op.result.type.element_type
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

        
    @parse(stablehlo.CompareOp)
    def less_equal_parser(
        self,
        old_op: stablehlo.CompareOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.less_equal_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]

        new_op = stablehlo_op(
            lhs,
            rhs,
            loc=old_op.location,
            unit_attrs=old_op.unit_attrs,
            sharding_attr=old_op.sharding_attr,
        )
        return new_op, {old_op.result: new_op.result}

    @split(stablehlo.CompareOp)
    def less_equal_split(
        self,
        old_op: stablehlo.CompareOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.less_equal_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            less_equal_module = Module.create()
            less_equal_builder = StableHLOBuilder(old_context, old_loc)
            lhs = less_equal_builder.placeholder(old_op.lhs.type, "lhs")
            rhs = less_equal_builder.placeholder(old_op.rhs.type, "rhs")
            result = less_equal_builder.less_equal(lhs, rhs)
            less_equal_builder.return_op([result])
        
        return less_equal_module, less_equal_builder

    ############### stablehlo.CompareOp - Less Than ###############

    @tag(stablehlo.CompareOp)
    def less_than(
        self,
        in0: Operand,
        in1: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.less_than)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            in1,
            comparison_direction="LT",
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            input1 = self._get_golden_tensor(in1)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, op.result.type.element_type
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

        
    @parse(stablehlo.CompareOp)
    def less_than_parser(
        self,
        old_op: stablehlo.CompareOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.less_than_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]

        new_op = stablehlo_op(
            lhs,
            rhs,
            loc=old_op.location,
            unit_attrs=old_op.unit_attrs,
            sharding_attr=old_op.sharding_attr,
        )
        return new_op, {old_op.result: new_op.result}

    @split(stablehlo.CompareOp)
    def less_than_split(
        self,
        old_op: stablehlo.CompareOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.less_than_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            less_than_module = Module.create()
            less_than_builder = StableHLOBuilder(old_context, old_loc)
            lhs = less_than_builder.placeholder(old_op.lhs.type, "lhs")
            rhs = less_than_builder.placeholder(old_op.rhs.type, "rhs")
            result = less_than_builder.less_than(lhs, rhs)
            less_than_builder.return_op([result])
        
        return less_than_module, less_than_builder
