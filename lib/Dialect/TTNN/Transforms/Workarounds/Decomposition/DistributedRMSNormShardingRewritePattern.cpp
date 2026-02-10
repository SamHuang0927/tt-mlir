// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/DistributedRMSNormShardingRewritePattern.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// This rewrite pattern forces width-sharded L1 layout on the input tensor of
// DistributedRMSNormOp. The ttnn::fused_rms_minimal runtime function requires
// the input to be width-sharded, and the program config is derived from the
// input's shard spec at runtime.
//
// The conversion is done in two steps because the tilize operation (ROW_MAJOR
// -> TILE) does not support WIDTH_SHARDED input:
//   1. ToLayoutOp: convert to TILE layout with L1 interleaved memory
//   2. ToMemoryConfigOp: reshard from interleaved to WIDTH_SHARDED
LogicalResult DistributedRMSNormShardingRewritePattern::matchAndRewrite(
    ttnn::DistributedRMSNormOp op, PatternRewriter &rewriter) const {

  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(op.getInput().getType());

  auto inputElementType = inputType.getElementType();
  if (ttcore::TileType inputTileType =
          mlir::dyn_cast_or_null<ttcore::TileType>(inputElementType)) {
    inputElementType = inputTileType.getElementType();
  }

  // Retrieve the physical grid shape for the device.
  auto physicalGrid =
      ttcore::getCurrentScopeSystemDesc(op).getChipDescs()[0].getGrid();

  // For width sharding, the virtual grid is [1, numCores]. Cap the number of
  // cores so that each shard gets at least one full tile in width (tile_width
  // = 32). Without this, the runtime would promote sub-tile shard widths to
  // tile_width, causing a mismatch with the compiler's serialized layout.
  constexpr int64_t tileWidth = 32;
  int64_t totalCores = physicalGrid[0] * physicalGrid[1];
  int64_t lastDim = inputType.getShape().back();
  int64_t lastDimInTiles = (lastDim + tileWidth - 1) / tileWidth;
  int64_t numCores = std::min(totalCores, lastDimInTiles);
  SmallVector<int64_t> virtualGridSize = {1, numCores};

  // Create an affine map for width-sharded layout.
  auto affineMap = mlir::tt::ttnn::optimizer_utils::
      createSingleDeviceVirtualToPhysicalAffineMap(
          rewriter.getContext(), ttnn::TensorMemoryLayout::WidthSharded,
          physicalGrid);
  auto widthShardedGrid = mlir::tt::ttcore::GridAttr::get(
      rewriter.getContext(), virtualGridSize, affineMap);
  auto widthShardedMemLayout = mlir::tt::ttnn::TensorMemoryLayoutAttr::get(
      rewriter.getContext(), ttnn::TensorMemoryLayout::WidthSharded);

  // Create the desired width-sharded L1 layout attribute.
  auto tileType = ttcore::TileType::get(inputElementType);
  ttnn::TTNNLayoutAttr desiredInputLayout = ttnn::TTNNLayoutAttr::get(
      rewriter.getContext(), inputType.getShape(), tileType,
      ttnn::BufferType::L1, widthShardedGrid, widthShardedMemLayout);

  ttnn::TTNNLayoutAttr currentInputLayout =
      mlir::dyn_cast_or_null<ttnn::TTNNLayoutAttr>(
          op.getInput().getType().getEncoding());
  // If the input already has the desired layout, no transformation is needed.
  if (currentInputLayout == desiredInputLayout) {
    return failure();
  }

  // Step 1: Convert to TILE layout with L1 interleaved memory.
  // This is needed because the tilize operation (ROW_MAJOR -> TILE) does not
  // support WIDTH_SHARDED input.
  auto interleavedMemLayout = mlir::tt::ttnn::TensorMemoryLayoutAttr::get(
      rewriter.getContext(), ttnn::TensorMemoryLayout::Interleaved);
  auto defaultGrid =
      mlir::tt::ttcore::GridAttr::get(rewriter.getContext(), {1, 1});
  ttnn::TTNNLayoutAttr interleavedLayout = ttnn::TTNNLayoutAttr::get(
      rewriter.getContext(), inputType.getShape(), tileType,
      ttnn::BufferType::L1, defaultGrid, interleavedMemLayout);
  ttnn::MemoryConfigAttr interleavedMemConfig =
      ttnn::MemoryConfigAttr::get(interleavedLayout, defaultGrid);
  RankedTensorType interleavedType =
      inputType.cloneWithEncoding(interleavedLayout);
  auto toTileOp = rewriter.create<ttnn::ToLayoutOp>(
      ttmlir::utils::appendLocationSuffix(op.getLoc(), "_to_tile"),
      interleavedType, op.getInput(), tt::ttnn::Layout::Tile,
      ttcore::DataTypeAttr::get(rewriter.getContext(),
                                ttcore::elementTypeToDataType(inputElementType)),
      interleavedMemConfig);

  // Step 2: Reshard from interleaved to width-sharded using ToMemoryConfigOp.
  ttnn::MemoryConfigAttr widthShardedMemConfig =
      ttnn::MemoryConfigAttr::get(desiredInputLayout, widthShardedGrid);
  RankedTensorType widthShardedType =
      inputType.cloneWithEncoding(desiredInputLayout);
  auto toMemConfigOp = rewriter.create<ttnn::ToMemoryConfigOp>(
      ttmlir::utils::appendLocationSuffix(op.getLoc(), "_to_width_sharded"),
      widthShardedType, toTileOp.getResult(), widthShardedMemConfig);

  // Force weight to ROW_MAJOR layout on device (L1 interleaved).
  // fused_rms_minimal requires weight (gamma) in ROW_MAJOR layout with
  // padded_shape[-1] == TILE_WIDTH (32). Reshape from (N,) to (N/32, 32)
  // before converting to ROW_MAJOR.
  mlir::Value weightValue = op.getWeight();
  if (weightValue) {
    auto weightType = mlir::cast<RankedTensorType>(weightValue.getType());
    auto weightElementType = weightType.getElementType();
    if (ttcore::TileType weightTileType =
            mlir::dyn_cast_or_null<ttcore::TileType>(weightElementType)) {
      weightElementType = weightTileType.getElementType();
    }

    // Reshape weight from (N,) to (N/32, 32) so padded_shape[-1] == 32.
    int64_t weightSize = weightType.getNumElements();
    SmallVector<int64_t> reshapedShape = {weightSize / tileWidth, tileWidth};
    auto reshapedWeightType =
        ttnn::utils::RankedTensorTypeFactory::create(weightType,
                                                     reshapedShape);
    SmallVector<int32_t> reshapedShapeI32(reshapedShape.begin(),
                                          reshapedShape.end());
    auto reshapeOp = rewriter.create<ttnn::ReshapeOp>(
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_weight_reshape"),
        reshapedWeightType, weightValue,
        rewriter.getI32ArrayAttr(reshapedShapeI32), ttnn::MemoryConfigAttr());

    // Convert reshaped weight to ROW_MAJOR layout on L1.
    ttnn::TTNNLayoutAttr weightLayout = ttnn::TTNNLayoutAttr::get(
        rewriter.getContext(), reshapedShape, weightElementType,
        ttnn::BufferType::L1, defaultGrid, interleavedMemLayout);
    ttnn::MemoryConfigAttr weightMemConfig =
        ttnn::MemoryConfigAttr::get(weightLayout, defaultGrid);
    RankedTensorType rowMajorWeightType =
        reshapedWeightType.cloneWithEncoding(weightLayout);
    auto weightToLayoutOp = rewriter.create<ttnn::ToLayoutOp>(
        ttmlir::utils::appendLocationSuffix(op.getLoc(),
                                            "_weight_to_row_major"),
        rowMajorWeightType, reshapeOp.getResult(), tt::ttnn::Layout::RowMajor,
        ttcore::DataTypeAttr::get(
            rewriter.getContext(),
            ttcore::elementTypeToDataType(weightElementType)),
        weightMemConfig);
    weightValue = weightToLayoutOp.getResult();
  }

  // Replace the original DistributedRMSNormOp with one that takes the
  // properly sharded input tensor and ROW_MAJOR weight.
  auto newOp = rewriter.create<ttnn::DistributedRMSNormOp>(
      op.getLoc(), op.getResult().getType(), toMemConfigOp.getResult(),
      weightValue, op.getClusterAxisAttr(),
      op.getEpsilonAttr(), op.getSubDeviceIdAttr(), op.getMemoryConfigAttr(),
      op.getNumLinksAttr(), op.getTopologyAttr(), op.getComputeConfigAttr());

  rewriter.replaceOp(op, newOp.getResult());
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
