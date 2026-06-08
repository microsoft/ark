// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Decomposed GEMM approach: Insert a user-defined functor between MMA
// accumulation and the epilogue store. The functor operates on accumulator
// registers before they're written to global memory, eliminating the
// global→global data path for fused elementwise ops.
//
// Data flow:
//   global(A,B) → shared → MMA → accum(registers) → Functor → Epilogue →
//   global(C)
//
// vs current matmul + separate elementwise:
//   global(A,B) → shared → MMA → Epilogue → global(C)
//   global(C) → elementwise → global(C)   ← EXTRA global read+write

#ifndef ARK_KERNELS_GEMM_FUSED_H_
#define ARK_KERNELS_GEMM_FUSED_H_

#include "gemm_cutlass.h"

namespace ark {

// ============================================================================
// Functors that operate on CUTLASS accumulator fragments (in registers)
// ============================================================================

struct FunctorIdentity {
    template <typename FragmentC>
    DEVICE static void apply(FragmentC &) {}
};

struct FunctorScale {
    float scale;
    template <typename FragmentC>
    DEVICE void apply(FragmentC &accum) const {
        using Element = typename FragmentC::Element;
        for (int i = 0; i < FragmentC::kElements; i++) {
            float val = static_cast<float>(accum[i]);
            accum[i] = static_cast<Element>(val * scale);
        }
    }
};

struct FunctorGelu {
    template <typename FragmentC>
    DEVICE static void apply(FragmentC &accum) {
        using Element = typename FragmentC::Element;
        for (int i = 0; i < FragmentC::kElements; i++) {
            float x = static_cast<float>(accum[i]);
            accum[i] = static_cast<Element>(
                x * 0.5f * (1.0f + erff(x * 0.7071067811865475f)));
        }
    }
};

struct FunctorRelu {
    template <typename FragmentC>
    DEVICE static void apply(FragmentC &accum) {
        using Element = typename FragmentC::Element;
        for (int i = 0; i < FragmentC::kElements; i++) {
            float val = static_cast<float>(accum[i]);
            accum[i] = static_cast<Element>(val > 0.f ? val : 0.f);
        }
    }
};

// Scale + Exp (for attention: exp(score * scale))
struct FunctorScaleExp {
    float scale;
    template <typename FragmentC>
    DEVICE void apply(FragmentC &accum) const {
        using Element = typename FragmentC::Element;
        for (int i = 0; i < FragmentC::kElements; i++) {
            float val = static_cast<float>(accum[i]);
            accum[i] = static_cast<Element>(expf(val * scale));
        }
    }
};

// ============================================================================
// gemm_with_functor: CUTLASS Mma + Functor on accumulators + Epilogue
//
// IMPORTANT: DataTypeA/B/C must be CUTLASS types (cutlass::half_t, etc.),
// NOT ARK types (ark::fp16). Use the gemm_cutlass_fused() wrapper below
// for ARK type conversion.
// ============================================================================

template <typename DataTypeA, int LeadingDimA, bool IsColumnA,
          typename DataTypeB, int LeadingDimB, bool IsColumnB,
          typename DataTypeC, int LeadingDimC, int ProblemSizeM,
          int ProblemSizeN, int ProblemSizeK, int TileSizeM, int TileSizeN,
          typename UnitOp, typename Functor>
DEVICE void gemm_with_functor(DataTypeC *C, DataTypeA *A, DataTypeB *B,
                              Functor functor, int uop_idx, int smem_per_warp) {
#if (ARK_TARGET_CUDA_ARCH == 60)
    using ArchTag = cutlass::arch::Sm60;
#elif (ARK_TARGET_CUDA_ARCH == 70)
    using ArchTag = cutlass::arch::Sm70;
#elif (ARK_TARGET_CUDA_ARCH == 80)
    using ArchTag = cutlass::arch::Sm80;
#elif (ARK_TARGET_CUDA_ARCH == 90)
    using ArchTag = cutlass::arch::Sm80;  // SM80 CUTLASS 2.x path for compat
#else
    static_assert(false, "Unsupported CUDA arch for gemm_with_functor");
#endif

    using LayoutA = typename cutlass::platform::conditional<
        IsColumnA, cutlass::layout::ColumnMajor,
        cutlass::layout::RowMajor>::type;
    using LayoutB = typename cutlass::platform::conditional<
        IsColumnB, cutlass::layout::ColumnMajor,
        cutlass::layout::RowMajor>::type;
    using LayoutC = cutlass::layout::RowMajor;

    static constexpr int TileSizeK = std::is_same_v<DataTypeC, float> ? 32 : 64;
    // NOTE: GemmConfiguration uses ElementC as accumulator for fp16 (half_t),
    // and float for bf16/fp32. Functors that need fp32 precision throughout
    // (e.g., exp, erff) should use a dedicated GemmConfiguration with
    // ElementAccumulator = float. FunctorScale's float cast is sufficient
    // for simple multiply, but FunctorScaleExp may lose precision with fp16
    // accumulators.
    using GemmKernel = typename ark::GemmConfiguration<
        UnitOp, cutlass::arch::OpClassTensorOp, ArchTag, DataTypeA, LayoutA,
        DataTypeB, LayoutB, DataTypeC, LayoutC,
        cutlass::gemm::GemmShape<TileSizeM, TileSizeN,
                                 TileSizeK>>::Gemm::GemmKernel;
    using Mma = typename GemmKernel::Mma;
    using Epilogue = typename GemmKernel::Epilogue;
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

    ark::GemmThreadblockSwizzle<UnitOp> swizzle;
    cutlass::gemm::GemmCoord tiled_shape(swizzle.get_tiled_shape());

    typename GemmKernel::Params params(problem_size, tiled_shape, ref_a, ref_b,
                                       ref_c, ref_c);
    params.swizzle_log_tile = uop_idx;

    typename GemmKernel::SharedStorage *ps =
        UnitOp::template shared_memory<typename GemmKernel::SharedStorage>(
            smem_per_warp);

    UnitOp::sync_threads();

    // --- Phase 1: Mma mainloop (same as standard CUTLASS) ---

    cutlass::gemm::GemmCoord threadblock_tile_offset =
        swizzle.get_tile_offset(uop_idx);

    if (tiled_shape.m() <= threadblock_tile_offset.m() ||
        tiled_shape.n() <= threadblock_tile_offset.n()) {
        return;
    }

    cutlass::MatrixCoord tb_offset_A{
        threadblock_tile_offset.m() * Mma::Shape::kM, 0};
    cutlass::MatrixCoord tb_offset_B{
        0, threadblock_tile_offset.n() * Mma::Shape::kN};

    int gemm_k_iterations =
        (ProblemSizeK + Mma::Shape::kK - 1) / Mma::Shape::kK;
    int thread_idx = threadIdx.x % GemmKernel::kThreadCount;

    typename Mma::IteratorA iterator_A(params.params_A, params.ref_A.data(),
                                       {ProblemSizeM, ProblemSizeK}, thread_idx,
                                       tb_offset_A);

    typename Mma::IteratorB iterator_B(params.params_B, params.ref_B.data(),
                                       {ProblemSizeK, ProblemSizeN}, thread_idx,
                                       tb_offset_B);

    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0) %
                   GemmKernel::WarpCount::kCount;
    int lane_idx = threadIdx.x % 32;

    Mma mma(ps->main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;
    accumulators.clear();
    mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

    // --- Phase 2: Apply functor on accumulator registers ---
    functor.apply(accumulators);

    // --- Phase 3: Epilogue (write to global memory) ---
    // Uses default OutputOp (alpha=1, beta=0) — just stores the
    // (functor-modified) accumulators to global memory.
    OutputOp output_op(params.output_op);

    threadblock_tile_offset = swizzle.get_tile_offset(params.swizzle_log_tile);

    typename Epilogue::OutputTileIterator iterator_C(
        params.params_C, params.ref_C.data(), params.problem_size.mn(),
        thread_idx,
        {threadblock_tile_offset.m() * Mma::Shape::kM,
         threadblock_tile_offset.n() * Mma::Shape::kN});

    typename Epilogue::OutputTileIterator iterator_D(
        params.params_D, params.ref_D.data(), params.problem_size.mn(),
        thread_idx,
        {threadblock_tile_offset.m() * Mma::Shape::kM,
         threadblock_tile_offset.n() * Mma::Shape::kN});

    Epilogue epilogue(ps->epilogue, thread_idx, warp_idx, lane_idx);
    epilogue(output_op, iterator_D, accumulators, iterator_C);
}

// ============================================================================
// gemm_cutlass_fused: ARK type conversion wrapper for gemm_with_functor.
// Converts ark::fp16/bf16/fp32 → cutlass::half_t/bfloat16_t/float, then
// calls gemm_with_functor.
// ============================================================================

template <typename DataTypeA, int LeadingDimA, bool IsColumnA,
          typename DataTypeB, int LeadingDimB, bool IsColumnB,
          typename DataTypeC, int LeadingDimC, int ProblemSizeM,
          int ProblemSizeN, int ProblemSizeK, int TileSizeM, int TileSizeN,
          typename UnitOp, typename Functor>
DEVICE void gemm_cutlass_fused(DataTypeC *C, DataTypeA *A, DataTypeB *B,
                               Functor functor, int uop_idx,
                               int smem_per_warp) {
    using CutDataTypeA = typename cutlass::platform::conditional<
        std::is_same<DataTypeA, fp16>::value, cutlass::half_t,
        typename cutlass::platform::conditional<
            std::is_same<DataTypeA, bf16>::value, cutlass::bfloat16_t,
            DataTypeA>::type>::type;

    using CutDataTypeB = typename cutlass::platform::conditional<
        std::is_same<DataTypeB, fp16>::value, cutlass::half_t,
        typename cutlass::platform::conditional<
            std::is_same<DataTypeB, bf16>::value, cutlass::bfloat16_t,
            DataTypeB>::type>::type;

    using CutDataTypeC = typename cutlass::platform::conditional<
        std::is_same<DataTypeC, fp16>::value, cutlass::half_t,
        typename cutlass::platform::conditional<
            std::is_same<DataTypeC, bf16>::value, cutlass::bfloat16_t,
            DataTypeC>::type>::type;

    CutDataTypeC *pC = reinterpret_cast<CutDataTypeC *>(C);
    CutDataTypeA *pA = reinterpret_cast<CutDataTypeA *>(A);
    CutDataTypeB *pB = reinterpret_cast<CutDataTypeB *>(B);

#if (ARK_TARGET_CUDA_ARCH == 60 || ARK_TARGET_CUDA_ARCH == 70 || \
     ARK_TARGET_CUDA_ARCH == 80 || ARK_TARGET_CUDA_ARCH == 90)
    gemm_with_functor<CutDataTypeA, LeadingDimA, IsColumnA, CutDataTypeB,
                      LeadingDimB, IsColumnB, CutDataTypeC, LeadingDimC,
                      ProblemSizeM, ProblemSizeN, ProblemSizeK, TileSizeM,
                      TileSizeN, UnitOp, Functor>(pC, pA, pB, functor, uop_idx,
                                                  smem_per_warp);
#else
    static_assert(false, "Unsupported CUDA arch.");
#endif
}

}  // namespace ark

#endif  // ARK_KERNELS_GEMM_FUSED_H_
