#ifndef ARK_KERNELS_GEMM_H_
#define ARK_KERNELS_GEMM_H_

#include "cutlass/coord.h"
#include "cutlass/cutlass.h"

#include "cutlass/arch/wmma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/threadblock/epilogue.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm.h"
#include "cutlass/gemm/kernel/gemm_pipelined.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/epilogue/threadblock/predicated_tile_static_iterator.h"

#if defined(CUTLASS_ARCH_WMMA_ENABLED)
#include "cutlass/epilogue/threadblock/default_epilogue_wmma_tensor_op.h"
#endif // defined(CUTLASS_ARCH_WMMA_ENABLED)

#include "smem.h"
#include "utils.h"

namespace ark {

////////////////////////////////////////////////////////////////////////////////

template <typename ElementC, typename OperatorClass>
struct GemmEpilogueOutputElementCount
{
    static const int value = 128 / cutlass::sizeof_bits<ElementC>::value;
};
template <typename ElementC>
struct GemmEpilogueOutputElementCount<ElementC, cutlass::arch::OpClassSimt>
{
    static const int value = 1;
};

////////////////////////////////////////////////////////////////////////////////

// For configuration detials, refer to the following files:
//  - cutlass/gemm/device/default_gemm_configuration.h
//  - cutlass/epilogue/threadblock/default_epilogue_tensor_op.h
//  - cutlass/gemm/kernel/default_gemm.h
template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Leading dimension of C matrix operand
    int LeadingDimC,
    /// Shape of the unit multiplication
    typename Shape,
    ///
    int ThreadsNum>
struct DefaultGemm;

////////////////////////////////////////////////////////////////////////////////
/// Partial specialization for Ampere Architecture
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Leading dimension of C matrix operand
    int LeadingDimC,
    /// Shape of the unit multiplication
    typename Shape,
    ///
    int ThreadsNum>
struct DefaultGemm<ElementA, LayoutA, ElementB, LayoutB, ElementC,
                   cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp,
                   cutlass::arch::Sm80, EpilogueOutputOp, LeadingDimC, Shape,
                   ThreadsNum>
{
    static const int WarpsNum = ThreadsNum / 32;
    static const int NumM = Shape::kM > Shape::kN ? WarpsNum / 2 : 2;
    static const int NumN = WarpsNum / NumM;
    using ThreadblockShape = Shape;
    using WarpShape =
        cutlass::gemm::GemmShape<Shape::kM / NumM, Shape::kN / NumN, Shape::kK>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using Operator = typename cutlass::platform::conditional<
        (cutlass::platform::is_same<ElementA, int8_t>::value ||
         cutlass::platform::is_same<ElementA, cutlass::int4b_t>::value ||
         cutlass::platform::is_same<ElementA, uint8_t>::value ||
         cutlass::platform::is_same<ElementA, cutlass::uint4b_t>::value),
        cutlass::arch::OpMultiplyAddSaturate,
        cutlass::arch::OpMultiplyAdd>::type;

    /// Define the threadblock-scoped matrix multiply-accumulate
    using Mma = typename cutlass::gemm::threadblock::DefaultMma<
        ElementA, LayoutA, 128 / cutlass::sizeof_bits<ElementA>::value,
        ElementB, LayoutB, 128 / cutlass::sizeof_bits<ElementA>::value,
        ElementC, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, ThreadblockShape, WarpShape, InstructionShape, 3,
        Operator>::ThreadblockMma;

    static const int kWarpsNum = ThreadblockShape::kCount / WarpShape::kCount;
    static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

    /// Define the epilogue
    using DefaultEpilogue =
        cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
            ThreadblockShape, typename Mma::Operator, kPartitionsK,
            EpilogueOutputOp, EpilogueOutputOp::kCount>;

    using OutputTileIterator =
        cutlass::epilogue::threadblock::PredicatedTileStaticIterator<
            typename DefaultEpilogue::OutputTileThreadMap,
            typename DefaultEpilogue::ElementOutput, LeadingDimC>;

    using Epilogue = cutlass::epilogue::threadblock::Epilogue<
        typename DefaultEpilogue::Shape,
        typename DefaultEpilogue::WarpMmaTensorOp,
        DefaultEpilogue::kPartitionsK, OutputTileIterator,
        typename DefaultEpilogue::AccumulatorFragmentIterator,
        typename DefaultEpilogue::WarpTileIterator,
        typename DefaultEpilogue::SharedLoadIterator,
        typename DefaultEpilogue::OutputOp, typename DefaultEpilogue::Padding>;
};

////////////////////////////////////////////////////////////////////////////////
/// Partial specialization for Turing Architecture
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Leading dimension of C matrix operand
    int LeadingDimC,
    /// Shape of the unit multiplication
    typename Shape,
    ///
    int ThreadsNum>
struct DefaultGemm<ElementA, LayoutA, ElementB, LayoutB, ElementC,
                   cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp,
                   cutlass::arch::Sm75, EpilogueOutputOp, LeadingDimC, Shape,
                   ThreadsNum>
{
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
    using Operator = typename cutlass::platform::conditional<
        (cutlass::platform::is_same<ElementA, int8_t>::value ||
         cutlass::platform::is_same<ElementA, cutlass::int4b_t>::value ||
         cutlass::platform::is_same<ElementA, uint8_t>::value ||
         cutlass::platform::is_same<ElementA, cutlass::uint4b_t>::value),
        cutlass::arch::OpMultiplyAddSaturate,
        cutlass::arch::OpMultiplyAdd>::type;

    /// Define the threadblock-scoped matrix multiply-accumulate
    using Mma = typename cutlass::gemm::threadblock::DefaultMma<
        ElementA, LayoutA, 128 / cutlass::sizeof_bits<ElementA>::value,
        ElementB, LayoutB, 128 / cutlass::sizeof_bits<ElementA>::value,
        ElementC, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm75, ThreadblockShape, WarpShape, InstructionShape, 2,
        Operator>::ThreadblockMma;

    static const int kWarpsNum = ThreadblockShape::kCount / WarpShape::kCount;
    static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

    /// Define the epilogue
    using Epilogue = cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
        ThreadblockShape, typename Mma::Operator, kPartitionsK,
        EpilogueOutputOp, EpilogueOutputOp::kCount>;
};

////////////////////////////////////////////////////////////////////////////////
/// Partial specialization for Volta architecture
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Leading dimension of C matrix operand
    int LeadingDimC,
    /// Shape of the unit multiplication
    typename Shape,
    ///
    int ThreadsNum>
struct DefaultGemm<ElementA, LayoutA, ElementB, LayoutB, ElementC,
                   cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp,
                   cutlass::arch::Sm70, EpilogueOutputOp, LeadingDimC, Shape,
                   ThreadsNum>
{
    static const int WarpsNum = ThreadsNum / 32;
    static const int NumM = Shape::kM > Shape::kN ? WarpsNum / 2 : 2;
    static const int NumN = WarpsNum / NumM;
    using ThreadblockShape = Shape;
    using WarpShape =
        cutlass::gemm::GemmShape<Shape::kM / NumM, Shape::kN / NumN, Shape::kK>;
    using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;

    /// Define the threadblock-scoped matrix multiply-accumulate
    using Mma = typename cutlass::gemm::threadblock::DefaultMma<
        ElementA, LayoutA, 128 / cutlass::sizeof_bits<ElementA>::value,
        ElementB, LayoutB, 128 / cutlass::sizeof_bits<ElementB>::value,
        ElementC, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm70, ThreadblockShape, WarpShape, InstructionShape, 2,
        cutlass::arch::OpMultiplyAdd>::ThreadblockMma;

    static const int kWarpsNum = ThreadblockShape::kCount / WarpShape::kCount;
    static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

    /// Define the epilogue
    using DefaultEpilogue =
        cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
            ThreadblockShape, typename Mma::Operator, kPartitionsK,
            EpilogueOutputOp, EpilogueOutputOp::kCount>;

    using OutputTileIterator =
        cutlass::epilogue::threadblock::PredicatedTileStaticIterator<
            typename DefaultEpilogue::OutputTileThreadMap,
            typename DefaultEpilogue::ElementOutput, LeadingDimC>;

    using Epilogue = cutlass::epilogue::threadblock::Epilogue<
        typename DefaultEpilogue::Shape,
        typename DefaultEpilogue::WarpMmaTensorOp,
        DefaultEpilogue::kPartitionsK, OutputTileIterator,
        typename DefaultEpilogue::AccumulatorFragmentIterator,
        typename DefaultEpilogue::WarpTileIterator,
        typename DefaultEpilogue::SharedLoadIterator,
        typename DefaultEpilogue::OutputOp, typename DefaultEpilogue::Padding>;
};

////////////////////////////////////////////////////////////////////////////////
/// Partial specialization for SIMT
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Leading dimension of C matrix operand
    int LeadingDimC,
    /// Shape of the unit multiplication
    typename Shape,
    ///
    int ThreadsNum>
struct DefaultGemm<ElementA, LayoutA, ElementB, LayoutB, ElementC,
                   cutlass::layout::RowMajor, cutlass::arch::OpClassSimt,
                   ArchTag, EpilogueOutputOp, LeadingDimC, Shape, ThreadsNum>
{
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
    using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using Operator = cutlass::arch::OpMultiplyAdd;

    /// Define the threadblock-scoped matrix multiply-accumulate
    using Mma = typename cutlass::gemm::threadblock::DefaultMma<
        ElementA, LayoutA, 1, ElementB, LayoutB, 1, ElementC,
        cutlass::layout::RowMajor, cutlass::arch::OpClassSimt,
        ArchTag, // cutlass::arch::Sm50 ?
        ThreadblockShape, WarpShape, InstructionShape, 2,
        Operator>::ThreadblockMma;

    static const int kWarpsNum = ThreadblockShape::kCount / WarpShape::kCount;
    static int const kEpilogueElementsPerAccess = EpilogueOutputOp::kCount;
    static_assert(kEpilogueElementsPerAccess == 1,
                  "simt epilogue must operate on scalars");

    /// Define the epilogue
    using Epilogue = cutlass::epilogue::threadblock::DefaultEpilogueSimt<
        ThreadblockShape, typename Mma::Operator, EpilogueOutputOp,
        kEpilogueElementsPerAccess>;
};

////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_WMMA_ENABLED)
////////////////////////////////////////////////////////////////////////////////
/// Partial specialization for Wmma Gemm Kernel
template <
    ///< Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Leading dimension of C matrix operand
    int LeadingDimC,
    /// Shape of the unit multiplication
    typename Shape,
    ///
    int ThreadsNum>
struct DefaultGemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                   cutlass::arch::OpClassWmmaTensorOp, ArchTag,
                   EpilogueOutputOp, LeadingDimC, Shape, ThreadsNum>
{
    using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>; // ??
    using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;        // ??
    using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>; // ??

    /// Define the threadblock-scoped matrix multiply-accumulate
    using Mma = typename cutlass::gemm::threadblock::DefaultMma<
        ElementA, LayoutA, 128 / cutlass::sizeof_bits<ElementA>::value,
        ElementB, LayoutB, 128 / cutlass::sizeof_bits<ElementB>::value,
        ElementC, LayoutC, cutlass::arch::OpClassWmmaTensorOp, ArchTag,
        ThreadblockShape, WarpShape, InstructionShape, 1,
        cutlass::arch::OpMultiplyAdd>::ThreadblockMma;

    static const int kWarpsNum = ThreadblockShape::kCount / WarpShape::kCount;
    static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

    /// Define the epilogue
    using DefaultEpilogue =
        cutlass::epilogue::threadblock::DefaultEpilogueWmmaTensorOp<
            ThreadblockShape, typename Mma::Operator, kPartitionsK,
            EpilogueOutputOp, EpilogueOutputOp::kCount>;

    using OutputTileIterator =
        cutlass::epilogue::threadblock::PredicatedTileStaticIterator<
            typename DefaultEpilogue::OutputTileThreadMap,
            typename DefaultEpilogue::ElementOutput, LeadingDimC>;

    using Epilogue = cutlass::epilogue::threadblock::Epilogue<
        typename DefaultEpilogue::Shape,
        typename DefaultEpilogue::WarpMmaTensorOp,
        DefaultEpilogue::kPartitionsK, OutputTileIterator,
        typename DefaultEpilogue::AccumulatorFragmentIterator,
        typename DefaultEpilogue::WarpTileIterator,
        typename DefaultEpilogue::SharedLoadIterator,
        typename DefaultEpilogue::OutputOp, typename DefaultEpilogue::Padding>;
};
////////////////////////////////////////////////////////////////////////////////
#endif // defined(CUTLASS_ARCH_WMMA_ENABLED)

////////////////////////////////////////////////////////////////////////////////

template <
    // Shape of the unit multiplication
    typename Shape,
    //
    int ThreadsNum,
    //
    typename ProblemSize,
    //
    typename LeadingDims,
    // Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    // Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    // Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    // Operator class tag
    typename OperatorClass,
    // Tag indicating architecture to tune for
    typename ArchTag,
    // Whether to use ReLU or not
    bool IsRelu = false>
struct GemmKernelBase
{
    using GemmEpilogueOutputOp = typename cutlass::platform::conditional<
        IsRelu,
        cutlass::epilogue::thread::LinearCombinationRelu<
            ElementC,
            GemmEpilogueOutputElementCount<ElementC, OperatorClass>::value,
            ElementC, ElementC>,
        cutlass::epilogue::thread::LinearCombination<
            ElementC,
            GemmEpilogueOutputElementCount<ElementC, OperatorClass>::value,
            ElementC, ElementC>>::type;
    using Gemm = DefaultGemm<
        ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, OperatorClass,
        ArchTag, GemmEpilogueOutputOp, LeadingDims::D1,
        cutlass::gemm::GemmShape<Shape::X, Shape::Y, Shape::Z>, ThreadsNum>;
    // Define the threadblock-scoped matrix multiply-accumulate
    using Mma = typename Gemm::Mma;
    // Define the epilogue
    using Epilogue = typename Gemm::Epilogue;
    using OutputOp = typename Epilogue::OutputOp;
    //
    typedef typename Gemm::ThreadblockShape ThreadblockShape;
    using GridTiledShape =
        Vec<math::div_up<ProblemSize::Y, ThreadblockShape::kN>::value,
            math::div_up<ProblemSize::X, ThreadblockShape::kM>::value, 1>;

    // Shared memory storage structure
    union SharedStorage {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };

    // Compute initial location in logical coordinates
    static int const GemmKSize =
        math::div_up<math::div_up<ProblemSize::Z, Mma::Shape::kK>::value,
                     GridTiledShape::Z>::value *
        Mma::Shape::kK;
    // Problem size is a function of threadblock index in the K dimension
    static int const ProblemSizeK =
        ProblemSize::Z < GemmKSize ? ProblemSize::Z : GemmKSize;
    // Thread mask
    static int const ThreadMask = Gemm::kWarpsNum * 32 - 1;

    static DEVICE void run(ElementA *pA, ElementB *pB, ElementC *pC,
                           ElementC *pD, ElementC alpha, ElementC beta,
                           SharedStorage &shared_storage, int tx, int ty)
    {
        // Early exit if CTA is out of range
        if (GridTiledShape::X <= tx || GridTiledShape::Y <= ty)
            return;
        cutlass::MatrixCoord tb_offset_A{ty * Mma::Shape::kM, 0};
        cutlass::MatrixCoord tb_offset_B{0, tx * Mma::Shape::kN};

        // Compute threadblock-scoped matrix multiply-add
        int gemm_k_iterations =
            (ProblemSizeK - tb_offset_A.column() + Mma::Shape::kK - 1) /
            Mma::Shape::kK;

        // Compute position within threadblock
        int thread_idx = threadIdx.x & ThreadMask;

        // Construct iterators to A and B operands
        typename Mma::IteratorA iterator_A(
            Mma::IteratorA::Params(LeadingDims::D0), pA,
            {ProblemSize::X, ProblemSizeK}, thread_idx, tb_offset_A);

        typename Mma::IteratorB iterator_B(
            Mma::IteratorB::Params(LeadingDims::D3), pB,
            {ProblemSizeK, ProblemSize::Y}, thread_idx, tb_offset_B);

        // Broadcast the warp_id computed by lane 0 to ensure dependent code
        // is compiled as warp-uniform.
        // int warp_idx = __shfl_sync(0x1f, thread_idx >> 5, 0);
        int warp_idx = thread_idx >> 5;
        int lane_idx = thread_idx & 31;

        //
        // Main loop
        //

        // Construct thread-scoped matrix multiply
        Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

        typename Mma::FragmentC accumulators;
        accumulators.clear();
        // Compute threadblock-scoped matrix multiply-add
        mma(gemm_k_iterations, accumulators, iterator_A, iterator_B,
            accumulators);

        //
        // Epilogue
        //

        OutputOp output_op(OutputOp::Params(alpha, beta));

        //
        // Masked tile iterators constructed from members
        //

        // assume identity swizzle
        cutlass::MatrixCoord threadblock_offset(ty * Mma::Shape::kM,
                                                tx * Mma::Shape::kN);

        // Tile iterator loading from source tensor.
        typename Epilogue::OutputTileIterator iterator_C(
            pC, cutlass::make_Coord(ProblemSize::X, ProblemSize::Y), thread_idx,
            threadblock_offset);

        // Tile iterator writing to destination tensor.
        typename Epilogue::OutputTileIterator iterator_D(
            pD, cutlass::make_Coord(ProblemSize::X, ProblemSize::Y), thread_idx,
            threadblock_offset);

        Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx,
                          lane_idx);

        // Execute the epilogue operator to update the destination tensor.
        epilogue(output_op, iterator_D, accumulators, iterator_C);
    }
};

// Half-precision GEMM. Row-major.
// TODO: this kernel returns the error code 716 'misaligned address'
// when TA is false and the kernel is compiled with "-G" option
// (which turns off all optimizations on device code).
template <int M, int N, int K, bool TA, bool TB, int BcastType, bool IsRelu,
          int ThreadsNum, int SmemBytes, int TDimM, int TDimN, int TDimK>
DEVICE void gemm(ark::half *C, ark::half *A, ark::half *B, ark::half alpha,
                 ark::half beta, int tx, int ty, int tz)
{
    // BcastType = 0: both A and B are batched with the same size.
    // BcastType = 1: only A is batched.
    // BcastType = 2: only B is batched.
    static_assert(BcastType == 0 || BcastType == 1 || BcastType == 2,
                  "invalid broadcast type.");
    static_assert(M % TDimM == 0, "");
    static_assert(N % TDimN == 0, "");
    static_assert(K % TDimK == 0, "");
    using LayoutA = typename cutlass::platform::conditional<
        TA, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type;
    using LayoutB = typename cutlass::platform::conditional<
        TB, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type;
    constexpr int LdA = TA ? M : K;
    constexpr int LdB = TB ? K : N;
    using GemmKernel = GemmKernelBase<
        Vec<TDimM, TDimN, TDimK>, ThreadsNum, Vec<M, N, K>, Vec<LdA, N, N, LdB>,
        cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, cutlass::half_t,
        cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp,
#if (ARK_TARGET_CUDA_ARCH == 60)
        cutlass::arch::Sm60,
#elif (ARK_TARGET_CUDA_ARCH == 70)
        cutlass::arch::Sm70,
#elif (ARK_TARGET_CUDA_ARCH == 75)
        cutlass::arch::Sm75,
#elif (ARK_TARGET_CUDA_ARCH == 80)
        cutlass::arch::Sm80,
#else
        cutlass::arch::Sm60,
#endif
        IsRelu>;
    static_assert(GemmKernel::ThreadMask == ThreadsNum - 1,
                  "traits mismatch with the actual implementation.");
    static_assert(sizeof(typename GemmKernel::SharedStorage) <= SmemBytes,
                  "traits mismatch with the actual implementation.");
    using Smem = SharedMemory<typename GemmKernel::SharedStorage, ThreadsNum>;

    constexpr int SizeA = K * M;
    constexpr int SizeB = K * N;
    constexpr int SizeC = N * M;
    cutlass::half_t *pA, *pB;
    cutlass::half_t *pC = &C[tz * SizeC];
    if (BcastType == 0) {
        pA = &A[tz * SizeA];
        pB = &B[tz * SizeB];
    } else if (BcastType == 1) {
        pA = &A[tz * SizeA];
        pB = B;
    } else if (BcastType == 2) {
        pA = A;
        pB = &B[tz * SizeB];
    }
    typename GemmKernel::SharedStorage *ps = Smem();
    GemmKernel::run(pA, pB, pC, pC, alpha, beta, *ps, tx, ty);
}

} // namespace ark

#endif // ARK_KERNELS_GEMM_H_