// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// GEMM with scale: D = alpha * (A @ B) where alpha = scale.
// Uses CUTLASS LinearCombination epilogue with custom alpha.
// The scale is applied on accumulator register fragments during the epilogue,
// eliminating a separate global memory round-trip.

#ifndef ARK_KERNELS_GEMM_SCALE_H_
#define ARK_KERNELS_GEMM_SCALE_H_

#include "gemm_cutlass.h"

namespace ark {

/// CUDA GeMM with scale: D = scale * A*B.
/// Uses the standard CUTLASS epilogue with alpha=scale, beta=0.
/// The scale is applied on register fragments in the epilogue thread function,
/// before writing to global memory — NO extra global memory read/write.
template <typename DataTypeA, int LeadingDimA, bool IsColumnA,
          typename DataTypeB, int LeadingDimB, bool IsColumnB,
          typename DataTypeC, int LeadingDimC, int ProblemSizeM,
          int ProblemSizeN, int ProblemSizeK, int TileSizeM, int TileSizeN,
          typename UnitOp, uint32_t ScaleBits>
DEVICE void gemm_cuda_scale(DataTypeC *C, DataTypeA *A, DataTypeB *B,
                            int uop_idx, int smem_per_warp) {
#if (ARK_TARGET_CUDA_ARCH == 70)
    using ArchTag = cutlass::arch::Sm70;
#elif (ARK_TARGET_CUDA_ARCH == 80)
    using ArchTag = cutlass::arch::Sm80;
#elif (ARK_TARGET_CUDA_ARCH == 90)
    using ArchTag = cutlass::arch::Sm80;
#else
    static_assert(false, "Unsupported CUDA arch.");
#endif

    using LayoutA = typename cutlass::platform::conditional<
        IsColumnA, cutlass::layout::ColumnMajor,
        cutlass::layout::RowMajor>::type;
    using LayoutB = typename cutlass::platform::conditional<
        IsColumnB, cutlass::layout::ColumnMajor,
        cutlass::layout::RowMajor>::type;
    using LayoutC = cutlass::layout::RowMajor;

    static constexpr int TileSizeK = std::is_same_v<DataTypeC, float> ? 32 : 64;
    using GemmKernel = typename ark::GemmConfiguration<
        UnitOp, cutlass::arch::OpClassTensorOp, ArchTag, DataTypeA, LayoutA,
        DataTypeB, LayoutB, DataTypeC, LayoutC,
        cutlass::gemm::GemmShape<TileSizeM, TileSizeN,
                                 TileSizeK>>::Gemm::GemmKernel;
    using OutputOp = typename GemmKernel::OutputOp;

    IsEq<GemmKernel::kThreadCount, UnitOp::NumThreads>();
    IsEq<sizeof(typename GemmKernel::SharedStorage), UnitOp::SmemBytes>();

    LayoutA layout_a(LeadingDimA);
    LayoutB layout_b(LeadingDimB);
    LayoutC layout_c(LeadingDimC);
    cutlass::TensorRef<DataTypeA, LayoutA> ref_a(A, layout_a);
    cutlass::TensorRef<DataTypeB, LayoutB> ref_b(B, layout_b);
    cutlass::TensorRef<DataTypeC, LayoutC> ref_c(C, layout_c);

    cutlass::gemm::GemmCoord problem_size(ProblemSizeM, ProblemSizeN,
                                          ProblemSizeK);
    cutlass::gemm::GemmCoord threadblock_shape(TileSizeM, TileSizeN, TileSizeK);

    ark::GemmThreadblockSwizzle<UnitOp> swizzle;
    cutlass::gemm::GemmCoord tiled_shape(swizzle.get_tiled_shape());

    // Decode scale from bits
    union {
        uint32_t u;
        float f;
    } conv;
    conv.u = ScaleBits;

    // Create OutputOp params with alpha=scale, beta=0
    typename OutputOp::Params output_op_params(
        static_cast<typename OutputOp::ElementCompute>(conv.f),
        static_cast<typename OutputOp::ElementCompute>(0));

    typename GemmKernel::Params params(problem_size, tiled_shape, ref_a, ref_b,
                                       ref_c, ref_c, output_op_params);

    params.swizzle_log_tile = uop_idx;

    typename GemmKernel::SharedStorage *ps =
        UnitOp::template shared_memory<typename GemmKernel::SharedStorage>(
            smem_per_warp);

    UnitOp::sync_threads();

    GemmKernel gemm_kernel{};
    gemm_kernel(params, *ps);
}

}  // namespace ark

#endif  // ARK_KERNELS_GEMM_SCALE_H_
