// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_GEMM_H_
#define ARK_KERNELS_GEMM_H_

// clang-format off
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/sm70_epilogue_vectorized.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
// clang-format on

#include "common.h"

namespace ark {

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

template <typename UnitOp, typename OperatorClass, typename ArchTag,
          typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB, typename ElementC, typename LayoutC, typename Shape>
struct GemmConfiguration;

////////////////////////////////////////////////////////////////////////////////
/// SM70 FP16
////////////////////////////////////////////////////////////////////////////////

template <typename UnitOp, typename LayoutA, typename LayoutB, typename LayoutC>
struct GemmConfiguration<UnitOp, cutlass::arch::OpClassTensorOp,
                         cutlass::arch::Sm70, cutlass::half_t, LayoutA,
                         cutlass::half_t, LayoutB, cutlass::half_t, LayoutC,
                         cutlass::gemm::GemmShape<128, 256, 32>>
{
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = cutlass::half_t;

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, ElementOutput,
        LayoutC, ElementAccumulator, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 256, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<8, 8, 4>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        ark::GemmThreadblockSwizzle<UnitOp>, 2>;
};

////////////////////////////////////////////////////////////////////////////////
/// SM80 FP16
////////////////////////////////////////////////////////////////////////////////

template <typename UnitOp, typename LayoutA, typename LayoutB, typename LayoutC>
struct GemmConfiguration<UnitOp, cutlass::arch::OpClassTensorOp,
                         cutlass::arch::Sm80, cutlass::half_t, LayoutA,
                         cutlass::half_t, LayoutB, cutlass::half_t, LayoutC,
                         cutlass::gemm::GemmShape<128, 256, 64>>
{
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = cutlass::half_t;

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, ElementOutput,
        LayoutC, ElementAccumulator, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 256, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        ark::GemmThreadblockSwizzle<UnitOp>, 3>;
};

template <typename UnitOp, typename LayoutA, typename LayoutB, typename LayoutC>
struct GemmConfiguration<UnitOp, cutlass::arch::OpClassTensorOp,
                         cutlass::arch::Sm80, cutlass::half_t, LayoutA,
                         cutlass::half_t, LayoutB, cutlass::half_t, LayoutC,
                         cutlass::gemm::GemmShape<256, 128, 64>>
{
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = cutlass::half_t;

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, ElementOutput,
        LayoutC, ElementAccumulator, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<256, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        ark::GemmThreadblockSwizzle<UnitOp>, 3>;
};

template <typename UnitOp, typename LayoutA, typename LayoutB, typename LayoutC>
struct GemmConfiguration<UnitOp, cutlass::arch::OpClassTensorOp,
                         cutlass::arch::Sm80, cutlass::half_t, LayoutA,
                         cutlass::half_t, LayoutB, cutlass::half_t, LayoutC,
                         cutlass::gemm::GemmShape<128, 128, 64>>
{
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = cutlass::half_t;

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, ElementOutput,
        LayoutC, ElementAccumulator, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        ark::GemmThreadblockSwizzle<UnitOp>, 3>;
};

template <typename UnitOp, typename LayoutA, typename LayoutB, typename LayoutC>
struct GemmConfiguration<UnitOp, cutlass::arch::OpClassTensorOp,
                         cutlass::arch::Sm80, cutlass::half_t, LayoutA,
                         cutlass::half_t, LayoutB, cutlass::half_t, LayoutC,
                         cutlass::gemm::GemmShape<256, 64, 64>>
{
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = cutlass::half_t;

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, ElementOutput,
        LayoutC, ElementAccumulator, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<256, 64, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        ark::GemmThreadblockSwizzle<UnitOp>, 3>;
};

template <typename UnitOp, typename LayoutA, typename LayoutB, typename LayoutC>
struct GemmConfiguration<UnitOp, cutlass::arch::OpClassTensorOp,
                         cutlass::arch::Sm80, cutlass::half_t, LayoutA,
                         cutlass::half_t, LayoutB, cutlass::half_t, LayoutC,
                         cutlass::gemm::GemmShape<64, 256, 64>>
{
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = cutlass::half_t;

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, ElementOutput,
        LayoutC, ElementAccumulator, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 256, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        ark::GemmThreadblockSwizzle<UnitOp>, 3>;
};

template <typename UnitOp, typename LayoutA, typename LayoutB, typename LayoutC>
struct GemmConfiguration<UnitOp, cutlass::arch::OpClassTensorOp,
                         cutlass::arch::Sm80, cutlass::half_t, LayoutA,
                         cutlass::half_t, LayoutB, cutlass::half_t, LayoutC,
                         cutlass::gemm::GemmShape<64, 128, 64>>
{
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = cutlass::half_t;

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, ElementOutput,
        LayoutC, ElementAccumulator, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 128, 64>,
        cutlass::gemm::GemmShape<32, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        ark::GemmThreadblockSwizzle<UnitOp>, 4>;
};

template <typename UnitOp, typename LayoutA, typename LayoutB, typename LayoutC>
struct GemmConfiguration<UnitOp, cutlass::arch::OpClassTensorOp,
                         cutlass::arch::Sm80, cutlass::half_t, LayoutA,
                         cutlass::half_t, LayoutB, cutlass::half_t, LayoutC,
                         cutlass::gemm::GemmShape<128, 64, 64>>
{
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = cutlass::half_t;

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, ElementOutput,
        LayoutC, ElementAccumulator, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 64, 64>,
        cutlass::gemm::GemmShape<64, 32, 64>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        ark::GemmThreadblockSwizzle<UnitOp>, 4>;
};

template <typename UnitOp, typename LayoutA, typename LayoutB, typename LayoutC>
struct GemmConfiguration<UnitOp, cutlass::arch::OpClassTensorOp,
                         cutlass::arch::Sm80, cutlass::half_t, LayoutA,
                         cutlass::half_t, LayoutB, cutlass::half_t, LayoutC,
                         cutlass::gemm::GemmShape<64, 64, 64>>
{
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = cutlass::half_t;

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, ElementOutput,
        LayoutC, ElementAccumulator, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<32, 32, 64>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        ark::GemmThreadblockSwizzle<UnitOp>, 6>;
};

template <typename UnitOp, typename LayoutA, typename LayoutB, typename LayoutC>
struct GemmConfiguration<UnitOp, cutlass::arch::OpClassTensorOp,
                         cutlass::arch::Sm80, cutlass::half_t, LayoutA,
                         cutlass::half_t, LayoutB, cutlass::half_t, LayoutC,
                         cutlass::gemm::GemmShape<128, 256, 32>>
{
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = cutlass::half_t;

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, ElementOutput,
        LayoutC, ElementAccumulator, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 256, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        ark::GemmThreadblockSwizzle<UnitOp>, 3>;
};

template <typename UnitOp, typename LayoutA, typename LayoutB, typename LayoutC>
struct GemmConfiguration<UnitOp, cutlass::arch::OpClassTensorOp,
                         cutlass::arch::Sm80, cutlass::half_t, LayoutA,
                         cutlass::half_t, LayoutB, cutlass::half_t, LayoutC,
                         cutlass::gemm::GemmShape<256, 128, 32>>
{
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = cutlass::half_t;

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, ElementOutput,
        LayoutC, ElementAccumulator, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<256, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        ark::GemmThreadblockSwizzle<UnitOp>, 3>;
};

template <typename UnitOp, typename LayoutA, typename LayoutB, typename LayoutC>
struct GemmConfiguration<UnitOp, cutlass::arch::OpClassTensorOp,
                         cutlass::arch::Sm80, cutlass::half_t, LayoutA,
                         cutlass::half_t, LayoutB, cutlass::half_t, LayoutC,
                         cutlass::gemm::GemmShape<128, 128, 32>>
{
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = cutlass::half_t;

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, ElementOutput,
        LayoutC, ElementAccumulator, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        ark::GemmThreadblockSwizzle<UnitOp>, 4>;
};

template <typename UnitOp, typename LayoutA, typename LayoutB, typename LayoutC>
struct GemmConfiguration<UnitOp, cutlass::arch::OpClassTensorOp,
                         cutlass::arch::Sm80, cutlass::half_t, LayoutA,
                         cutlass::half_t, LayoutB, cutlass::half_t, LayoutC,
                         cutlass::gemm::GemmShape<256, 64, 32>>
{
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = cutlass::half_t;

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, ElementOutput,
        LayoutC, ElementAccumulator, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<256, 64, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        ark::GemmThreadblockSwizzle<UnitOp>, 4>;
};

template <typename UnitOp, typename LayoutA, typename LayoutB, typename LayoutC>
struct GemmConfiguration<UnitOp, cutlass::arch::OpClassTensorOp,
                         cutlass::arch::Sm80, cutlass::half_t, LayoutA,
                         cutlass::half_t, LayoutB, cutlass::half_t, LayoutC,
                         cutlass::gemm::GemmShape<64, 256, 32>>
{
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = cutlass::half_t;

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, ElementOutput,
        LayoutC, ElementAccumulator, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 256, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        ark::GemmThreadblockSwizzle<UnitOp>, 4>;
};

template <typename UnitOp, typename LayoutA, typename LayoutB, typename LayoutC>
struct GemmConfiguration<UnitOp, cutlass::arch::OpClassTensorOp,
                         cutlass::arch::Sm80, cutlass::half_t, LayoutA,
                         cutlass::half_t, LayoutB, cutlass::half_t, LayoutC,
                         cutlass::gemm::GemmShape<64, 128, 32>>
{
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = cutlass::half_t;

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, ElementOutput,
        LayoutC, ElementAccumulator, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 128, 32>,
        cutlass::gemm::GemmShape<32, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        ark::GemmThreadblockSwizzle<UnitOp>, 6>;
};

template <typename UnitOp, typename LayoutA, typename LayoutB, typename LayoutC>
struct GemmConfiguration<UnitOp, cutlass::arch::OpClassTensorOp,
                         cutlass::arch::Sm80, cutlass::half_t, LayoutA,
                         cutlass::half_t, LayoutB, cutlass::half_t, LayoutC,
                         cutlass::gemm::GemmShape<128, 64, 32>>
{
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = cutlass::half_t;

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, ElementOutput,
        LayoutC, ElementAccumulator, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 64, 32>,
        cutlass::gemm::GemmShape<64, 32, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        ark::GemmThreadblockSwizzle<UnitOp>, 6>;
};

template <typename UnitOp, typename LayoutA, typename LayoutB, typename LayoutC>
struct GemmConfiguration<UnitOp, cutlass::arch::OpClassTensorOp,
                         cutlass::arch::Sm80, cutlass::half_t, LayoutA,
                         cutlass::half_t, LayoutB, cutlass::half_t, LayoutC,
                         cutlass::gemm::GemmShape<64, 64, 32>>
{
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = cutlass::half_t;

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, ElementOutput,
        LayoutC, ElementAccumulator, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<32, 32, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        ark::GemmThreadblockSwizzle<UnitOp>, 10>;
};

////////////////////////////////////////////////////////////////////////////////
/// SM80 FP32
////////////////////////////////////////////////////////////////////////////////

template <typename UnitOp, typename LayoutA, typename LayoutB, typename LayoutC>
struct GemmConfiguration<UnitOp, cutlass::arch::OpClassTensorOp,
                         cutlass::arch::Sm80, float, LayoutA, float, LayoutB,
                         float, LayoutC, cutlass::gemm::GemmShape<128, 256, 32>>
{
    using ElementOutput = float;
    using ElementAccumulator = float;

    using Gemm = cutlass::gemm::device::Gemm<
        float, LayoutA, float, LayoutB, ElementOutput, LayoutC,
        ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 256, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 8>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        ark::GemmThreadblockSwizzle<UnitOp>, 3>;
};

template <typename UnitOp, typename LayoutA, typename LayoutB, typename LayoutC>
struct GemmConfiguration<UnitOp, cutlass::arch::OpClassTensorOp,
                         cutlass::arch::Sm80, float, LayoutA, float, LayoutB,
                         float, LayoutC, cutlass::gemm::GemmShape<128, 128, 32>>
{
    using ElementOutput = float;
    using ElementAccumulator = float;

    using Gemm = cutlass::gemm::device::Gemm<
        float, LayoutA, float, LayoutB, ElementOutput, LayoutC,
        ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 8>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        ark::GemmThreadblockSwizzle<UnitOp>, 3>;
};

template <typename UnitOp, typename LayoutA, typename LayoutB, typename LayoutC>
struct GemmConfiguration<UnitOp, cutlass::arch::OpClassTensorOp,
                         cutlass::arch::Sm80, float, LayoutA, float, LayoutB,
                         float, LayoutC, cutlass::gemm::GemmShape<64, 64, 32>>
{
    using ElementOutput = float;
    using ElementAccumulator = float;

    using Gemm = cutlass::gemm::device::Gemm<
        float, LayoutA, float, LayoutB, ElementOutput, LayoutC,
        ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<32, 32, 32>,
        cutlass::gemm::GemmShape<16, 8, 8>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        ark::GemmThreadblockSwizzle<UnitOp>, 3>;
};

#if 0
template <typename UnitOp>
struct GemmConfiguration<
    UnitOp, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm90,
    cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
    cutlass::layout::RowMajor, cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::gemm::GemmShape<64, 128, 64>>
{
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::ColumnMajor;

    using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, cutlass::half_t,
        LayoutA, 8, cutlass::half_t, LayoutB, 8, cutlass::half_t,
        cute::Shape<cute::_64, cute::_128, cute::_64>, cute::Shape<cute::_1, cute::_1, cute::_1>,
        cutlass::gemm::collective::StageCountAuto,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
            cute::Shape<cute::_64, cute::_128, cute::_64>, cute::Shape<cute::_1, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto, cutlass::half_t,
            cutlass::half_t, cutlass::half_t, LayoutC, 8, cutlass::half_t,
            LayoutC, 8,
            cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

    using GemmKernel =
        cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
                                             CollectiveOp, CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename UnitOp>
struct GemmConfiguration<
    UnitOp, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm90,
    cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
    cutlass::layout::RowMajor, cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::gemm::GemmShape<128, 128, 32>>
{
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::ColumnMajor;

    using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, cutlass::half_t,
        LayoutA, 8, cutlass::half_t, LayoutB, 8, cutlass::half_t,
        cute::Shape<cute::_128, cute::_128, cute::_32>, cute::Shape<cute::_1, cute::_1, cute::_1>,
        cutlass::gemm::collective::StageCountAuto,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
            cute::Shape<cute::_128, cute::_128, cute::_32>, cute::Shape<cute::_1, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto, cutlass::half_t,
            cutlass::half_t, cutlass::half_t, LayoutC, 8, cutlass::half_t,
            LayoutC, 8,
            cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

    using GemmKernel =
        cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
                                             CollectiveOp, CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename UnitOp, typename LayoutA, typename LayoutB, typename LayoutC>
struct GemmConfiguration<UnitOp, cutlass::arch::OpClassTensorOp,
                         cutlass::arch::Sm90, cutlass::half_t, LayoutA,
                         cutlass::half_t, LayoutB, cutlass::half_t, LayoutC,
                         cutlass::gemm::GemmShape<64, 64, 64>>
{
    using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, cutlass::half_t,
        LayoutA, 8, cutlass::half_t, LayoutB, 8, cutlass::half_t,
        cute::Shape<cute::_64, cute::_64, cute::_64>, cute::Shape<cute::_1, cute::_1, cute::_1>,
        cutlass::gemm::collective::StageCountAuto,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
            cute::Shape<cute::_64, cute::_64, cute::_64>, cute::Shape<cute::_1, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto, cutlass::half_t,
            cutlass::half_t, cutlass::half_t, LayoutC, 8, cutlass::half_t,
            LayoutC, 8,
            cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

    using GemmKernel =
        cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
                                             CollectiveOp, CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};
#endif

/// CUDA GeMM for arch equal to or earlier than 80.
template <typename DataTypeA, int LeadingDimA, bool IsColumnA,
          typename DataTypeB, int LeadingDimB, bool IsColumnB,
          typename DataTypeC, int LeadingDimC, int ProblemSizeM,
          int ProblemSizeN, int ProblemSizeK, int TileSizeM, int TileSizeN,
          int TileSizeK, typename UnitOp, int NumThreads, int SmemBytes>
DEVICE void gemm_cuda(DataTypeC *C, DataTypeA *A, DataTypeB *B, int uop_idx,
                      int smem_per_warp)
{
#if (ARK_TARGET_CUDA_ARCH == 60)
    using ArchTag = cutlass::arch::Sm60;
#elif (ARK_TARGET_CUDA_ARCH == 70)
    using ArchTag = cutlass::arch::Sm70;
#elif (ARK_TARGET_CUDA_ARCH == 80)
    using ArchTag = cutlass::arch::Sm80;
#elif (ARK_TARGET_CUDA_ARCH == 90)
    static_assert(false, "Use gemm_cuda_90 instead.");
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

    using GemmKernel = typename ark::GemmConfiguration<
        UnitOp, cutlass::arch::OpClassTensorOp, ArchTag, DataTypeA, LayoutA,
        DataTypeB, LayoutB, DataTypeC, LayoutC,
        cutlass::gemm::GemmShape<TileSizeM, TileSizeN,
                                 TileSizeK>>::Gemm::GemmKernel;

    IsEq<GemmKernel::kThreadCount, NumThreads>();
    IsEq<sizeof(GemmKernel::SharedStorage), SmemBytes>();

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

    GemmKernel gemm_kernel{};
    gemm_kernel(params, *ps);
}

/// CUDA GeMM for arch 90.
template <typename DataTypeA, int LeadingDimA, bool IsColumnA,
          typename DataTypeB, int LeadingDimB, bool IsColumnB,
          typename DataTypeC, int LeadingDimC, int ProblemSizeM,
          int ProblemSizeN, int ProblemSizeK, int TileSizeM, int TileSizeN,
          int TileSizeK, typename UnitOp, int NumThreads, int SmemBytes>
DEVICE void gemm_cuda_90(DataTypeC *C, DataTypeA *A, DataTypeB *B, int uop_idx,
                         int smem_per_warp)
{
#if 1
    static_assert(false, "Not implemented yet.");
#else
#if (ARK_TARGET_CUDA_ARCH == 90)
    using ArchTag = cutlass::arch::Sm90;
#elif (ARK_TARGET_CUDA_ARCH == 60 || ARK_TARGET_CUDA_ARCH == 70 ||             \
       ARK_TARGET_CUDA_ARCH == 80)
    static_assert(false, "Use gemm_cuda instead.");
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

    using GemmConfig = typename ark::GemmConfiguration<
        UnitOp, cutlass::arch::OpClassTensorOp, ArchTag, DataTypeA, LayoutA,
        DataTypeB, LayoutB, DataTypeC, LayoutC,
        cutlass::gemm::GemmShape<TileSizeM, TileSizeN, TileSizeK>>;
    using GemmKernel = typename GemmConfig::Gemm::GemmKernel;

    IsEq<GemmKernel::MaxThreadsPerBlock, NumThreads>();
    IsEq<GemmKernel::SharedStorageSize, SmemBytes>();

    // Construct params

    // typename GemmKernel::Arguments args;
    // args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    // args.problem_shape = GemmKernel::ProblemShape(ProblemSizeM, ProblemSizeN,
    // ProblemSizeK, 1);

    // args.mainloop.ptr_A = A;
    // args.mainloop.dA = cute::make_int_tuple_from<typename
    // GemmKernel::StrideA>(
    //     LeadingDimA, 1);
    // args.mainloop.ptr_B = B;
    // args.mainloop.dB = cute::make_int_tuple_from<typename
    // GemmKernel::StrideB>(
    //     LeadingDimB, 1);

    // args.epilogue.thread.alpha = 1;
    // args.epilogue.thread.beta = 0;
    // args.epilogue.thread.alpha_ptr = nullptr;
    // args.epilogue.thread.beta_ptr = nullptr;

    auto dA =
        cute::make_int_tuple_from<typename GemmKernel::StrideA>(LeadingDimA, 1);
    auto dB =
        cute::make_int_tuple_from<typename GemmKernel::StrideB>(LeadingDimB, 1);

    typename GemmKernel::Params params;

    params.mode = cutlass::gemm::GemmUniversalMode::kGemm;

    if constexpr (cutlass::gemm::kernel::detail::IF_SWAP_AB<
                      typename GemmKernel::CollectiveMainloop>::value) {
        // swap M/N
        params.problem_shape = GemmKernel::ProblemShape(
            ProblemSizeN, ProblemSizeM, ProblemSizeK, 1);
    } else {
        params.problem_shape = GemmKernel::ProblemShape(
            ProblemSizeM, ProblemSizeN, ProblemSizeK, 1);
    }

    // Below is copied from CollectiveMma::to_underlying_arguments().

    auto [M, N, K, L] = params.problem_shape;

    auto ptr_A = reinterpret_cast<
        typename GemmKernel::CollectiveMainloop::InternalElementA const *>(A);
    auto ptr_B = reinterpret_cast<
        typename GemmKernel::CollectiveMainloop::InternalElementB const *>(B);

    cute::Tensor tensor_a = cute::make_tensor(
        ptr_A, cute::make_layout(cute::make_shape(M, K, L), dA));
    cute::Tensor tensor_b = cute::make_tensor(
        ptr_B, cute::make_layout(cute::make_shape(N, K, L), dB));

    params.mainloop.tma_load_a = cute::make_tma_copy(
        typename GemmKernel::CollectiveMainloop::GmemTiledCopyA{}, tensor_a,
        typename GemmKernel::CollectiveMainloop::SmemLayoutA{}(cute::_, cute::_,
                                                               cute::Int<0>{}),
        cute::make_shape(
            cute::shape<0>(
                typename GemmKernel::CollectiveMainloop::TileShape{}),
            cute::shape<2>(
                typename GemmKernel::CollectiveMainloop::TileShape{})),
        cute::size<1>(typename GemmKernel::CollectiveMainloop::DispatchPolicy::
                          ClusterShape{}));
    params.mainloop.tma_load_b = cute::make_tma_copy(
        typename GemmKernel::CollectiveMainloop::GmemTiledCopyB{}, tensor_b,
        typename GemmKernel::CollectiveMainloop::SmemLayoutB{}(cute::_, cute::_,
                                                               cute::Int<0>{}),
        cute::make_shape(
            cute::shape<1>(
                typename GemmKernel::CollectiveMainloop::TileShape{}),
            cute::shape<2>(
                typename GemmKernel::CollectiveMainloop::TileShape{})),
        cute::size<0>(typename GemmKernel::CollectiveMainloop::DispatchPolicy::
                          ClusterShape{}));

    // params.epilogue.thread.alpha = 1;
    // params.epilogue.thread.beta = 0;
    // params.epilogue.thread.alpha_ptr = nullptr;
    // params.epilogue.thread.beta_ptr = nullptr;
    // params.epilogue.ptr_C = C;
    // params.epilogue.dC = LeadingDimC;
    // params.epilogue.ptr_D = C;
    // params.epilogue.dD = LeadingDimC;

    union GemmCuda90SharedStorage {
        typename GemmKernel::CollectiveMainloop::SharedStorage mainloop;
        typename GemmKernel::CollectiveEpilogue::SharedStorage epilogue;
    };

    GemmCuda90SharedStorage *ps =
        UnitOp::template shared_memory<GemmCuda90SharedStorage>(smem_per_warp);

    GemmKernel gemm_kernel{};
    gemm_kernel(params, (char *)ps);
#endif
}

/// Row-major GeMM.
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

    constexpr int LeadingDimA = LeadingDims::D0;
    constexpr int LeadingDimB = LeadingDims::D3;
    constexpr int LeadingDimC = LeadingDims::D1;
    constexpr int ProblemSizeM = ProblemSize::D0;
    constexpr int ProblemSizeN = ProblemSize::D1;
    constexpr int ProblemSizeK = ProblemSize::D2;
    constexpr int TileSizeM = Shape::D0;
    constexpr int TileSizeN = Shape::D1;
    constexpr int TileSizeK = Shape::D2;

    constexpr int SizeA = math::mul<ProblemSizeM, ProblemSizeK>::value;
    constexpr int SizeB = math::mul<ProblemSizeN, ProblemSizeK>::value;
    constexpr int SizeC = math::mul<ProblemSizeM, ProblemSizeN>::value;

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

#if (ARK_TARGET_CUDA_ARCH == 60 || ARK_TARGET_CUDA_ARCH == 70 ||               \
     ARK_TARGET_CUDA_ARCH == 80)
    gemm_cuda<DataTypeA, LeadingDimA, IsColumnA, DataTypeB, LeadingDimB,
              IsColumnB, DataTypeC, LeadingDimC, ProblemSizeM, ProblemSizeN,
              ProblemSizeK, TileSizeM, TileSizeN, TileSizeK, UnitOp, NumThreads,
              SmemBytes>(C, A, B, uop_idx, smem_per_warp);
#elif (ARK_TARGET_CUDA_ARCH == 90)
    gemm_cuda_90<DataTypeA, LeadingDimA, IsColumnA, DataTypeB, LeadingDimB,
                 IsColumnB, DataTypeC, LeadingDimC, ProblemSizeM, ProblemSizeN,
                 ProblemSizeK, TileSizeM, TileSizeN, TileSizeK, UnitOp,
                 NumThreads, SmemBytes>(C, A, B, uop_idx, smem_per_warp);
#else
    static_assert(false, "Unsupported CUDA arch.");
#endif
}

} // namespace ark

#endif // ARK_KERNELS_GEMM_H_
