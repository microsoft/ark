// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_MATMUL_HPP_
#define ARK_OPS_MATMUL_HPP_

#include "model/model_op.hpp"
#include "ops_copy.hpp"

namespace ark {

class ModelOpMatmul : public ModelOp {
   public:
    ModelOpMatmul() = default;
    ModelOpMatmul(ModelTensorRef input, ModelTensorRef other,
                  ModelTensorRef output, bool trans_input, bool trans_other);

    std::string impl_name(const Json &config) const override;

    std::vector<ModelOpArg> impl_args(const Json &config) const override;

    Json default_config(const ArchRef arch = ARCH_ANY) const override;
};

/// Matmul with GELU activation fused into the CUTLASS epilogue.
/// Output = gelu(input @ other). Same interface as ModelOpMatmul.
class ModelOpMatmulGelu : public ModelOpMatmul {
   public:
    ModelOpMatmulGelu() = default;
    ModelOpMatmulGelu(ModelTensorRef input, ModelTensorRef other,
                      ModelTensorRef output, bool trans_input,
                      bool trans_other);

    std::string impl_name(const Json &config) const override;
};

/// Matmul with residual addition: output = input @ other + residual.
/// The residual tensor is added via CUTLASS beta=1 epilogue.
class ModelOpMatmulAdd : public ModelOp {
   public:
    ModelOpMatmulAdd() = default;
    ModelOpMatmulAdd(ModelTensorRef input, ModelTensorRef other,
                     ModelTensorRef residual, ModelTensorRef output,
                     bool trans_input, bool trans_other);

    std::string impl_name(const Json &config) const override;

    std::vector<ModelOpArg> impl_args(const Json &config) const override;

    Json default_config(const ArchRef arch = ARCH_ANY) const override;
};

/// Matmul with scale applied on register accumulators before epilogue.
/// Output = (input @ other) * scale. The scale is fused into the CUTLASS
/// accumulator registers, eliminating a separate global memory round-trip.
class ModelOpMatmulScale : public ModelOpMatmul {
   public:
    ModelOpMatmulScale() = default;
    ModelOpMatmulScale(ModelTensorRef input, ModelTensorRef other,
                       ModelTensorRef output, bool trans_input,
                       bool trans_other, float scale);

    std::string impl_name(const Json &config) const override;
};

/// MMA op: matmul that produces a REGISTER tensor.
/// Currently behaves identically to Matmul but tags the output with
/// TensorLocation::REGISTER. When codegen detects a REGISTER output
/// followed by elementwise ops in the same sync=False block, it can
/// fuse them at the register level.
class ModelOpMma : public ModelOpMatmul {
   public:
    ModelOpMma() = default;
    ModelOpMma(ModelTensorRef input, ModelTensorRef other,
               ModelTensorRef output, bool trans_input, bool trans_other);

    std::string impl_name(const Json &config) const override;
};

/// Store op: write a register tensor to global memory.
/// This marks the end of a register-level fusion chain.
/// When codegen detects mma → elementwise → store within a sync=False block,
/// it generates a single gemm_with_functor kernel.
/// Currently implemented as a no-op (the output IS the input buffer).
class ModelOpStore : public ModelOpCopy {
   public:
    ModelOpStore() = default;
    ModelOpStore(ModelTensorRef input, ModelTensorRef output);
    // Override to use "copy" kernel (not "store" which clashes with load_store.h)
    std::string impl_name(const Json &config) const override;
};

}  // namespace ark

#endif  // ARK_OPS_MATMUL_HPP_
