// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_GEMM_H_
#define ARK_KERNELS_GEMM_H_

#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm.h"

#include "common.h"

namespace ark {

template <typename OperatorClass, typename ArchTag, typename ElementA,
          typename ElementB, typename ElementC, typename ElementAccumulator,
          typename Shape>
struct GemmConfiguration;

template <typename ArchTag, typename ElementA, typename ElementB,
          typename ElementC, typename ElementAccumulator>
struct GemmConfiguration<cutlass::arch::OpClassSimt, ArchTag, ElementA,
                         ElementB, ElementC, ElementAccumulator,
                         cutlass::gemm::GemmShape<128, 128, 8>>
{
    static int const kAlignmentA = 1;
    static int const kAlignmentB = 1;

    using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    static int const kStages = 2;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementC, 1, ElementAccumulator, ElementAccumulator>;

    using Operator = cutlass::arch::OpMultiplyAdd;
};

template <typename ArchTag, typename ElementC>
struct GemmConfiguration<cutlass::arch::OpClassSimt, ArchTag, int8_t, int8_t,
                         ElementC, int32_t,
                         cutlass::gemm::GemmShape<128, 128, 32>>
{
    static int const kAlignmentA = 4;
    static int const kAlignmentB = 4;

    using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 4>;
    static int const kStages = 2;

    using EpilogueOutputOp =
        cutlass::epilogue::thread::LinearCombinationClamp<ElementC, 1, int32_t,
                                                          float>;

    using Operator = cutlass::arch::OpMultiplyAdd;
};

template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementAccumulator>
struct GemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70,
                         ElementA, ElementB, ElementC, ElementAccumulator,
                         cutlass::gemm::GemmShape<64, 64, 32>>
{
    static int const kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static int const kAlignmentB = 128 / cutlass::sizeof_bits<ElementA>::value;

    using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
    static int const kStages = 3;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementAccumulator>;

    using Operator = typename cutlass::platform::conditional<
        (cutlass::platform::is_same<ElementA, int8_t>::value ||
         cutlass::platform::is_same<ElementA, cutlass::int4b_t>::value ||
         cutlass::platform::is_same<ElementA, uint8_t>::value ||
         cutlass::platform::is_same<ElementA, cutlass::uint4b_t>::value),
        cutlass::arch::OpMultiplyAddSaturate,
        cutlass::arch::OpMultiplyAdd>::type;
};

template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementAccumulator>
struct GemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70,
                         ElementA, ElementB, ElementC, ElementAccumulator,
                         cutlass::gemm::GemmShape<128, 64, 32>>
{
    static int const kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static int const kAlignmentB = 128 / cutlass::sizeof_bits<ElementA>::value;

    using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
    static int const kStages = 3;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementAccumulator>;

    using Operator = typename cutlass::platform::conditional<
        (cutlass::platform::is_same<ElementA, int8_t>::value ||
         cutlass::platform::is_same<ElementA, cutlass::int4b_t>::value ||
         cutlass::platform::is_same<ElementA, uint8_t>::value ||
         cutlass::platform::is_same<ElementA, cutlass::uint4b_t>::value),
        cutlass::arch::OpMultiplyAddSaturate,
        cutlass::arch::OpMultiplyAdd>::type;
};

template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementAccumulator>
struct GemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70,
                         ElementA, ElementB, ElementC, ElementAccumulator,
                         cutlass::gemm::GemmShape<64, 128, 32>>
{
    static int const kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static int const kAlignmentB = 128 / cutlass::sizeof_bits<ElementA>::value;

    using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
    static int const kStages = 3;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementAccumulator>;

    using Operator = typename cutlass::platform::conditional<
        (cutlass::platform::is_same<ElementA, int8_t>::value ||
         cutlass::platform::is_same<ElementA, cutlass::int4b_t>::value ||
         cutlass::platform::is_same<ElementA, uint8_t>::value ||
         cutlass::platform::is_same<ElementA, cutlass::uint4b_t>::value),
        cutlass::arch::OpMultiplyAddSaturate,
        cutlass::arch::OpMultiplyAdd>::type;
};

template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementAccumulator>
struct GemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70,
                         ElementA, ElementB, ElementC, ElementAccumulator,
                         cutlass::gemm::GemmShape<128, 128, 32>>
{
    static int const kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static int const kAlignmentB = 128 / cutlass::sizeof_bits<ElementA>::value;

    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
    static int const kStages = 3;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementAccumulator>;

    using Operator = typename cutlass::platform::conditional<
        (cutlass::platform::is_same<ElementA, int8_t>::value ||
         cutlass::platform::is_same<ElementA, cutlass::int4b_t>::value ||
         cutlass::platform::is_same<ElementA, uint8_t>::value ||
         cutlass::platform::is_same<ElementA, cutlass::uint4b_t>::value),
        cutlass::arch::OpMultiplyAddSaturate,
        cutlass::arch::OpMultiplyAdd>::type;
};

template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementAccumulator>
struct GemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                         ElementA, ElementB, ElementC, ElementAccumulator,
                         cutlass::gemm::GemmShape<64, 64, 32>>
{
    static int const kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static int const kAlignmentB = 128 / cutlass::sizeof_bits<ElementA>::value;

    using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
    static int const kStages = 3;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementAccumulator>;

    using Operator = typename cutlass::platform::conditional<
        (cutlass::platform::is_same<ElementA, int8_t>::value ||
         cutlass::platform::is_same<ElementA, cutlass::int4b_t>::value ||
         cutlass::platform::is_same<ElementA, uint8_t>::value ||
         cutlass::platform::is_same<ElementA, cutlass::uint4b_t>::value),
        cutlass::arch::OpMultiplyAddSaturate,
        cutlass::arch::OpMultiplyAdd>::type;
};

template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementAccumulator>
struct GemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                         ElementA, ElementB, ElementC, ElementAccumulator,
                         cutlass::gemm::GemmShape<64, 64, 64>>
{
    static int const kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static int const kAlignmentB = 128 / cutlass::sizeof_bits<ElementA>::value;

    using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    static int const kStages = 3;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementAccumulator>;

    using Operator = typename cutlass::platform::conditional<
        (cutlass::platform::is_same<ElementA, int8_t>::value ||
         cutlass::platform::is_same<ElementA, cutlass::int4b_t>::value ||
         cutlass::platform::is_same<ElementA, uint8_t>::value ||
         cutlass::platform::is_same<ElementA, cutlass::uint4b_t>::value),
        cutlass::arch::OpMultiplyAddSaturate,
        cutlass::arch::OpMultiplyAdd>::type;
};

template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementAccumulator>
struct GemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                         ElementA, ElementB, ElementC, ElementAccumulator,
                         cutlass::gemm::GemmShape<128, 128, 32>>
{
    static int const kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static int const kAlignmentB = 128 / cutlass::sizeof_bits<ElementA>::value;

    using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
    static int const kStages = 3;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementAccumulator>;

    using Operator = typename cutlass::platform::conditional<
        (cutlass::platform::is_same<ElementA, int8_t>::value ||
         cutlass::platform::is_same<ElementA, cutlass::int4b_t>::value ||
         cutlass::platform::is_same<ElementA, uint8_t>::value ||
         cutlass::platform::is_same<ElementA, cutlass::uint4b_t>::value),
        cutlass::arch::OpMultiplyAddSaturate,
        cutlass::arch::OpMultiplyAdd>::type;
};

template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementAccumulator>
struct GemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                         ElementA, ElementB, ElementC, ElementAccumulator,
                         cutlass::gemm::GemmShape<128, 128, 64>>
{
    static int const kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static int const kAlignmentB = 128 / cutlass::sizeof_bits<ElementA>::value;

    using WarpShape = cutlass::gemm::GemmShape<64, 32, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    static int const kStages = 3;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementAccumulator>;

    using Operator = typename cutlass::platform::conditional<
        (cutlass::platform::is_same<ElementA, int8_t>::value ||
         cutlass::platform::is_same<ElementA, cutlass::int4b_t>::value ||
         cutlass::platform::is_same<ElementA, uint8_t>::value ||
         cutlass::platform::is_same<ElementA, cutlass::uint4b_t>::value),
        cutlass::arch::OpMultiplyAddSaturate,
        cutlass::arch::OpMultiplyAdd>::type;
};

template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementAccumulator>
struct GemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                         ElementA, ElementB, ElementC, ElementAccumulator,
                         cutlass::gemm::GemmShape<128, 256, 32>>
{
    static int const kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static int const kAlignmentB = 128 / cutlass::sizeof_bits<ElementA>::value;

    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
    static int const kStages = 3;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementAccumulator>;

    using Operator = typename cutlass::platform::conditional<
        (cutlass::platform::is_same<ElementA, int8_t>::value ||
         cutlass::platform::is_same<ElementA, cutlass::int4b_t>::value ||
         cutlass::platform::is_same<ElementA, uint8_t>::value ||
         cutlass::platform::is_same<ElementA, cutlass::uint4b_t>::value),
        cutlass::arch::OpMultiplyAddSaturate,
        cutlass::arch::OpMultiplyAdd>::type;
};

template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementAccumulator>
struct GemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                         ElementA, ElementB, ElementC, ElementAccumulator,
                         cutlass::gemm::GemmShape<128, 256, 64>>
{
    static int const kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static int const kAlignmentB = 128 / cutlass::sizeof_bits<ElementA>::value;

    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    static int const kStages = 3;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementAccumulator>;

    using Operator = typename cutlass::platform::conditional<
        (cutlass::platform::is_same<ElementA, int8_t>::value ||
         cutlass::platform::is_same<ElementA, cutlass::int4b_t>::value ||
         cutlass::platform::is_same<ElementA, uint8_t>::value ||
         cutlass::platform::is_same<ElementA, cutlass::uint4b_t>::value),
        cutlass::arch::OpMultiplyAddSaturate,
        cutlass::arch::OpMultiplyAdd>::type;
};

template <typename ElementC, typename ElementAccumulator>
struct GemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                         double, double, ElementC, ElementAccumulator,
                         cutlass::gemm::GemmShape<128, 256, 64>>
{
    static int const kAlignmentA = 1;
    static int const kAlignmentB = 1;

    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    static int const kStages = 3;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementAccumulator>;

    using Operator = cutlass::arch::OpMultiplyAdd;
};

template <typename ElementC>
struct GemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                         int8_t, int8_t, ElementC, int32_t,
                         cutlass::gemm::GemmShape<128, 256, 64>>
{
    static int const kAlignmentA = 128 / cutlass::sizeof_bits<int8_t>::value;
    static int const kAlignmentB = 128 / cutlass::sizeof_bits<int8_t>::value;

    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
    static int const kStages = 3;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationClamp<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, int32_t, float>;

    using Operator = cutlass::arch::OpMultiplyAddSaturate;
};

template <typename ElementC>
struct GemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                         int8_t, uint8_t, ElementC, int32_t,
                         cutlass::gemm::GemmShape<128, 256, 64>>
{
    static int const kAlignmentA = 128 / cutlass::sizeof_bits<int8_t>::value;
    static int const kAlignmentB = 128 / cutlass::sizeof_bits<uint8_t>::value;

    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
    static int const kStages = 3;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationClamp<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, int32_t, float>;

    using Operator = cutlass::arch::OpMultiplyAddSaturate;
};

template <typename ElementC>
struct GemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                         uint8_t, int8_t, ElementC, int32_t,
                         cutlass::gemm::GemmShape<128, 256, 64>>
{
    static int const kAlignmentA = 128 / cutlass::sizeof_bits<uint8_t>::value;
    static int const kAlignmentB = 128 / cutlass::sizeof_bits<int8_t>::value;

    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
    static int const kStages = 3;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationClamp<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, int32_t, float>;

    using Operator = cutlass::arch::OpMultiplyAddSaturate;
};

template <typename ElementC>
struct GemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                         uint8_t, uint8_t, ElementC, int32_t,
                         cutlass::gemm::GemmShape<128, 256, 64>>
{
    static int const kAlignmentA = 128 / cutlass::sizeof_bits<uint8_t>::value;
    static int const kAlignmentB = 128 / cutlass::sizeof_bits<uint8_t>::value;

    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
    static int const kStages = 3;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationClamp<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, int32_t, float>;

    using Operator = cutlass::arch::OpMultiplyAddSaturate;
};

template <typename ElementC>
struct GemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                         cutlass::int4b_t, cutlass::int4b_t, ElementC, int32_t,
                         cutlass::gemm::GemmShape<128, 256, 128>>
{
    static int const kAlignmentA =
        128 / cutlass::sizeof_bits<cutlass::int4b_t>::value;
    static int const kAlignmentB =
        128 / cutlass::sizeof_bits<cutlass::int4b_t>::value;

    using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
    static int const kStages = 3;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationClamp<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, int32_t, float>;

    using Operator = cutlass::arch::OpMultiplyAddSaturate;
};

template <typename ElementC>
struct GemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                         cutlass::int4b_t, cutlass::uint4b_t, ElementC, int32_t,
                         cutlass::gemm::GemmShape<128, 256, 128>>
{
    static int const kAlignmentA =
        128 / cutlass::sizeof_bits<cutlass::int4b_t>::value;
    static int const kAlignmentB =
        128 / cutlass::sizeof_bits<cutlass::uint4b_t>::value;

    using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
    static int const kStages = 3;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationClamp<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, int32_t, float>;

    using Operator = cutlass::arch::OpMultiplyAddSaturate;
};

template <typename ElementC>
struct GemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                         cutlass::uint4b_t, cutlass::int4b_t, ElementC, int32_t,
                         cutlass::gemm::GemmShape<128, 256, 128>>
{
    static int const kAlignmentA =
        128 / cutlass::sizeof_bits<cutlass::uint4b_t>::value;
    static int const kAlignmentB =
        128 / cutlass::sizeof_bits<cutlass::int4b_t>::value;

    using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
    static int const kStages = 3;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationClamp<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, int32_t, float>;

    using Operator = cutlass::arch::OpMultiplyAddSaturate;
};

template <typename ElementC>
struct GemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                         cutlass::uint4b_t, cutlass::uint4b_t, ElementC,
                         int32_t, cutlass::gemm::GemmShape<128, 256, 128>>
{
    static int const kAlignmentA =
        128 / cutlass::sizeof_bits<cutlass::uint4b_t>::value;
    static int const kAlignmentB =
        128 / cutlass::sizeof_bits<cutlass::uint4b_t>::value;

    using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
    static int const kStages = 3;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationClamp<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, int32_t, float>;

    using Operator = cutlass::arch::OpMultiplyAddSaturate;
};

template <typename ElementC>
struct GemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                         cutlass::uint1b_t, cutlass::uint1b_t, ElementC,
                         int32_t, cutlass::gemm::GemmShape<128, 256, 512>>
{
    static int const kAlignmentA =
        128 / cutlass::sizeof_bits<cutlass::uint1b_t>::value;
    static int const kAlignmentB =
        128 / cutlass::sizeof_bits<cutlass::uint1b_t>::value;

    using WarpShape = cutlass::gemm::GemmShape<64, 64, 512>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 256>;
    static int const kStages = 3;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationClamp<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, int32_t, float>;

    using Operator = cutlass::arch::OpMultiplyAdd;
};

/// Custom ThreadblockSwizzle for ARK.
template <typename UnitOp> struct GemmThreadblockSwizzle
{
    DEVICE GemmThreadblockSwizzle()
    {
    }

    DEVICE cutlass::gemm::GemmCoord get_tiled_shape() const
    {
        return cutlass::gemm::GemmCoord(UnitOp::UnitOpDims::H,
                                        UnitOp::UnitOpDims::W, 1);
    }

    DEVICE int get_log_tile(cutlass::gemm::GemmCoord) const
    {
        return 0;
    }

    DEVICE cutlass::gemm::GemmCoord get_tile_offset(int log_tile) const
    {
        // log_tile is actually uop_idx here.
        int uh = UnitOp::uop_idx_h(log_tile);
        int uw = UnitOp::uop_idx_w(log_tile);
        return cutlass::gemm::GemmCoord{uh, uw, 0};
    }
};

/// Half-precision GEMM. Row-major.
template <typename OutDims, typename NCA, typename NCB, typename Shape,
          typename ProblemSize, typename LeadingDims, bool IsColumnA,
          bool IsColumnB, int NumThreads, int SmemBytes, typename DataTypeA,
          typename DataTypeB, typename DataTypeC, typename AccumulateType>
DEVICE void gemm(DataTypeC *C, DataTypeA *A, DataTypeB *B, int uop_idx,
                 int smem_per_warp)
{
    static_assert(NCA::D2 == 1 && NCA::D3 == 1,
                  "NCA should be two dimensional.");
    static_assert(NCB::D2 == 1 && NCB::D3 == 1,
                  "NCB should be two dimensional.");
    static_assert(Shape::D3 == 1, "Shape should be three dimensional.");
    static_assert(ProblemSize::D3 == 1,
                  "ProblemSize should be three dimensional.");

    // N dimension of C is max(N dimension of A, N dimension of B)
    constexpr int NC = (NCA::D0 > NCB::D0) ? NCA::D0 : NCB::D0;
    // C dimension of C is max(C dimension of A, C dimension of B)
    constexpr int CC = (NCA::D1 > NCB::D1) ? NCA::D1 : NCB::D1;

    using OutShape = Vec<NC, CC, ProblemSize::D0, ProblemSize::D1>;
    using UnitOutDims = Vec<1, 1, Shape::D0, Shape::D1>;
    using UnitOp =
        UnitOp<OutDims, OutShape, UnitOutDims, NumThreads, SmemBytes>;

    using LayoutA = typename cutlass::platform::conditional<
        IsColumnA, cutlass::layout::ColumnMajor,
        cutlass::layout::RowMajor>::type;
    using LayoutB = typename cutlass::platform::conditional<
        IsColumnB, cutlass::layout::ColumnMajor,
        cutlass::layout::RowMajor>::type;
    using LayoutC = cutlass::layout::RowMajor;

#if (ARK_TARGET_CUDA_ARCH == 60)
    using ArchTag = cutlass::arch::Sm60;
#elif (ARK_TARGET_CUDA_ARCH == 70)
    using ArchTag = cutlass::arch::Sm70;
#elif (ARK_TARGET_CUDA_ARCH == 80)
    using ArchTag = cutlass::arch::Sm80;
#else
    using ArchTag = cutlass::arch::Sm60;
#endif

    using ThreadblockSwizzle = ark::GemmThreadblockSwizzle<UnitOp>;

    using GemmShape = cutlass::gemm::GemmShape<Shape::D0, Shape::D1, Shape::D2>;
    using GemmConfig =
        typename ark::GemmConfiguration<cutlass::arch::OpClassTensorOp, ArchTag,
                                        DataTypeA, DataTypeB, DataTypeC,
                                        AccumulateType, GemmShape>;
    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
        DataTypeA, LayoutA, GemmConfig::kAlignmentA, DataTypeB, LayoutB,
        GemmConfig::kAlignmentB, DataTypeC, LayoutC, AccumulateType,
        cutlass::arch::OpClassTensorOp, ArchTag, GemmShape,
        typename GemmConfig::WarpShape, typename GemmConfig::InstructionShape,
        typename GemmConfig::EpilogueOutputOp, ThreadblockSwizzle,
        GemmConfig::kStages, false, typename GemmConfig::Operator>::GemmKernel;

    IsEq<GemmKernel::kThreadCount, NumThreads>();
    IsEq<sizeof(GemmKernel::SharedStorage), SmemBytes>();

    constexpr int SizeA = math::mul<ProblemSize::D0, ProblemSize::D2>::value;
    constexpr int SizeB = math::mul<ProblemSize::D1, ProblemSize::D2>::value;
    constexpr int SizeC = math::mul<ProblemSize::D0, ProblemSize::D1>::value;

    int un = UnitOp::uop_idx_n(uop_idx);
    int uc = UnitOp::uop_idx_c(uop_idx);

    // Broadcasting
    DataTypeA *pA;
    DataTypeB *pB;
    DataTypeC *pC = &C[un * math::mul<CC, SizeC>::value + uc * SizeC];
    if (NCA::D0 == 1 && NCA::D1 == 1) {
        pA = A;
    } else if (NCA::D0 == 1) {
        pA = &A[uc * SizeA];
    } else if (NCA::D1 == 1) {
        pA = &A[un * SizeA];
    } else {
        pA = &A[un * math::mul<CC, SizeA>::value + uc * SizeA];
    }
    if (NCB::D0 == 1 && NCB::D1 == 1) {
        pB = B;
    } else if (NCB::D0 == 1) {
        pB = &B[uc * SizeB];
    } else if (NCB::D1 == 1) {
        pB = &B[un * SizeB];
    } else {
        pB = &B[un * math::mul<CC, SizeB>::value + uc * SizeB];
    }

    LayoutA layout_a(LeadingDims::D0);
    LayoutB layout_b(LeadingDims::D3);
    LayoutC layout_c(LeadingDims::D1);
    cutlass::TensorRef<DataTypeA, LayoutA> ref_a(pA, layout_a);
    cutlass::TensorRef<DataTypeB, LayoutB> ref_b(pB, layout_b);
    cutlass::TensorRef<DataTypeC, LayoutC> ref_c(pC, layout_c);

    cutlass::gemm::GemmCoord problem_size(ProblemSize::D0, ProblemSize::D1,
                                          ProblemSize::D2);

    cutlass::gemm::GemmCoord threadblock_shape(Shape::D0, Shape::D1, Shape::D2);

    ThreadblockSwizzle swizzle;

    cutlass::gemm::GemmCoord tiled_shape(swizzle.get_tiled_shape());

    typename GemmKernel::Params params(problem_size, tiled_shape, ref_a, ref_b,
                                       ref_c, ref_c);

    // A hack for custom threadblock swizzle. swizzle_log_tile is useless
    // for ARK, instead we need uop_idx to determine the tile offset.
    // Since swizzle_log_tile is the input to get_tile_offset(), we can
    // use it to pass uop_idx.
    params.swizzle_log_tile = uop_idx;

    typename GemmKernel::SharedStorage *ps =
        UnitOp::template shared_memory<GemmKernel::SharedStorage>(
            smem_per_warp);

    GemmKernel gemm_kernel;

    gemm_kernel(params, *ps);
}

} // namespace ark

#endif // ARK_KERNELS_GEMM_H_
