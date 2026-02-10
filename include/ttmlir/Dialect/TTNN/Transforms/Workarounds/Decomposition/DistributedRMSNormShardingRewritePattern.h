// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_DISTRIBUTEDRMSNORMSHARDDINGREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_DISTRIBUTEDRMSNORMSHARDDINGREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Workaround which forces width-sharded L1 layout on the input tensor of
// DistributedRMSNormOp. The ttnn::fused_rms_minimal runtime function requires
// the input to be width-sharded. This pattern inserts a ToLayoutOp before the
// op to enforce this layout if not already present.
class DistributedRMSNormShardingRewritePattern
    : public OpRewritePattern<ttnn::DistributedRMSNormOp> {
public:
  using OpRewritePattern<ttnn::DistributedRMSNormOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::DistributedRMSNormOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_DISTRIBUTEDRMSNORMSHARDDINGREWRITEPATTERN_H
