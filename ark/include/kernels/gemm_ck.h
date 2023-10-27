// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_GEMM_CK_H_
#define ARK_KERNELS_GEMM_CK_H_

#include <cassert>
#include <type_traits>

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl.hpp"
#include "common.h"

/// Common aliases for CK GeMM configurations.

using F16 = ck::half_t;
using F32 = float;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using LoopScheduler = ck::LoopScheduler;
using PipelineVersion = ck::PipelineVersion;
static constexpr auto GemmDefault =
    ck::tensor_operation::device::GemmSpecialization::Default;

namespace ark {

template <typename DataTypeA, typename DataTypeB, typename DataTypeC,
          typename AccumulateType, typename LayoutA, typename LayoutB,
          int NumThreads, int TileSizeM, int TileSizeN,
          LoopScheduler LoopSched = LoopScheduler::Default,
          PipelineVersion PipelineVer = PipelineVersion::v1,
          int CShuffleNumStage = 1>
struct CkGemm;

////////////////////////////////////////////////////////////////////////////////

template <typename LayoutA, typename LayoutB, int NumThreads, int TileSizeM,
          int TileSizeN, LoopScheduler LoopSched, PipelineVersion PipelineVer,
          int CShuffleNumStage>
struct CkGemm<fp32, fp32, fp32, fp32, LayoutA, LayoutB, NumThreads, TileSizeM,
              TileSizeN, LoopSched, PipelineVer, CShuffleNumStage> {
    static constexpr bool IsColA = std::is_same<LayoutA, Col>::value;
    static constexpr bool IsColB = std::is_same<LayoutB, Col>::value;
    static constexpr auto MNXdlPerWave =
        TileSizeM * TileSizeN / 16 / NumThreads;
    static constexpr auto LogMNXdlPerWave = math::log2_up<MNXdlPerWave>::value;
    static constexpr auto MXdlPerWave =
        (TileSizeM < TileSizeN) ? 1 << (LogMNXdlPerWave / 2)
                                : 1 << (LogMNXdlPerWave - LogMNXdlPerWave / 2);
    static constexpr auto NXdlPerWave = MNXdlPerWave / MXdlPerWave;
    static constexpr bool Is_128x128x64 =
        NumThreads == 128 && TileSizeM == 128 && TileSizeN == 64;
    static constexpr bool Is_128x64x128 =
        NumThreads == 128 && TileSizeM == 64 && TileSizeN == 128;
    static constexpr bool Is_128x128x128 =
        NumThreads == 128 && TileSizeM == 128 && TileSizeN == 128;

    using ImplXdl = ck::tensor_operation::device::DeviceGemmXdl<
        F32, F32, F32, F32, LayoutA, LayoutB, Row, PassThrough, PassThrough,
        PassThrough, GemmDefault, NumThreads, TileSizeM, TileSizeN, 4, 4, 32,
        32, MXdlPerWave, NXdlPerWave, S<4, NumThreads / 4, 1>,
        typename std::conditional<IsColA, S<0, 2, 1>, S<1, 0, 2>>::type,
        typename std::conditional<IsColA, S<0, 2, 1>, S<1, 0, 2>>::type,
        (IsColA ? 1 : 2), ((!IsColA || Is_128x128x64) ? 4 : MXdlPerWave), 4,
        true, S<4, NumThreads / 4, 1>,
        typename std::conditional<IsColB, S<1, 0, 2>, S<0, 2, 1>>::type,
        typename std::conditional<IsColB, S<1, 0, 2>, S<0, 2, 1>>::type,
        (IsColB ? 2 : 1),
        ((IsColB || Is_128x64x128 || Is_128x128x128) ? 4 : NXdlPerWave), 4,
        true, 7, 1>;
};

template <typename LayoutA, typename LayoutB, int NumThreads, int TileSizeM,
          int TileSizeN, LoopScheduler LoopSched, PipelineVersion PipelineVer,
          int CShuffleNumStage>
struct CkGemm<fp16, fp16, fp16, fp32, LayoutA, LayoutB, NumThreads, TileSizeM,
              TileSizeN, LoopSched, PipelineVer, CShuffleNumStage> {
    static constexpr auto Is_16 = (TileSizeM == 16 || TileSizeN == 16);
    static constexpr auto MPerXdl = Is_16 ? 16 : 32;
    static constexpr auto NPerXdl = MPerXdl;
    static constexpr bool IsColA = std::is_same<LayoutA, Col>::value;
    static constexpr bool IsColB = std::is_same<LayoutB, Col>::value;
    static constexpr auto MNXdlPerWave =
        Is_16 ? (TileSizeM * TileSizeN / 4 / NumThreads)
              : (TileSizeM * TileSizeN / 16 / NumThreads);
    static constexpr auto LogMNXdlPerWave = math::log2_up<MNXdlPerWave>::value;
    static constexpr auto MXdlPerWave =
        (TileSizeM == 16) ? 1
                          : (TileSizeM < TileSizeN)
                                ? 1 << (LogMNXdlPerWave / 2)
                                : 1 << (LogMNXdlPerWave - LogMNXdlPerWave / 2);
    static constexpr auto NXdlPerWave = MNXdlPerWave / MXdlPerWave;

    static constexpr bool Is_128x128x64 =
        NumThreads == 128 && TileSizeM == 128 && TileSizeN == 64;

    static constexpr bool Is_128x32x256 =
        NumThreads == 128 && TileSizeM == 32 && TileSizeN == 256;

    static constexpr bool Is_128x32x128 =
        NumThreads == 128 && TileSizeM == 32 && TileSizeN == 128;
    static constexpr bool Is_128x64x128 =
        NumThreads == 128 && TileSizeM == 64 && TileSizeN == 128;
    static constexpr bool Is_128x128x128 =
        NumThreads == 128 && TileSizeM == 128 && TileSizeN == 128;

    static constexpr bool Is_128x32x64 =
        NumThreads == 128 && TileSizeM == 32 && TileSizeN == 64;
    static constexpr bool Is_64x32x32 =
        NumThreads == 64 && TileSizeM == 32 && TileSizeN == 32;

    using ImplXdl = ck::tensor_operation::device::DeviceGemmXdl<
        F16, F16, F16, F32, LayoutA, LayoutB, Row, PassThrough, PassThrough,
        PassThrough, GemmDefault, NumThreads, TileSizeM, TileSizeN, 4, 8,
        MPerXdl, NPerXdl, MXdlPerWave, NXdlPerWave,
        S<4, Is_16 ? 16 : (NumThreads / 4), 1>,
        typename std::conditional<IsColA, S<0, 2, 1>, S<1, 0, 2>>::type,
        typename std::conditional<IsColA, S<0, 2, 1>, S<1, 0, 2>>::type,
        (IsColA ? 1 : 2), (!IsColA ? 8 : Is_128x128x64 ? 4 : MXdlPerWave), 8,
        true, S<4, NumThreads / 4, 1>,
        typename std::conditional<IsColB, S<1, 0, 2>, S<0, 2, 1>>::type,
        typename std::conditional<IsColB, S<1, 0, 2>, S<0, 2, 1>>::type,
        (IsColB ? 2 : 1),
        (IsColB ? 8
                : Is_128x32x256
                      ? 8
                      : (Is_128x32x128 || Is_128x64x128 || Is_128x128x128)
                            ? 4
                            : (Is_128x32x64 || Is_64x32x32) ? 2 : NXdlPerWave),
        8, true, 7, 1, 1, LoopSched, PipelineVer>;
};

////////////////////////////////////////////////////////////////////////////////

/// Row-major GeMM.
template <typename DataTypeA, int LeadingDimA, bool IsColumnA,
          typename DataTypeB, int LeadingDimB, bool IsColumnB,
          typename DataTypeC, int LeadingDimC, int ProblemSizeM,
          int ProblemSizeN, int ProblemSizeK, int TileSizeM, int TileSizeN,
          int TileSizeK, typename UnitOp>
DEVICE void gemm_ck(DataTypeC *C, DataTypeA *A, DataTypeB *B, int uop_idx,
                    int smem_per_warp) {
    static_assert(LeadingDimA >= 0, "");
    static_assert(LeadingDimB >= 0, "");
    static_assert(LeadingDimC >= 0, "");
    static_assert(ProblemSizeM >= 0, "");
    static_assert(ProblemSizeN >= 0, "");
    static_assert(ProblemSizeK >= 0, "");
    static_assert(TileSizeM >= 0, "");
    static_assert(TileSizeN >= 0, "");
    static_assert(TileSizeK >= 0, "");

    using AccumulateType = fp32;
    using LayoutA = typename std::conditional<IsColumnA, Col, Row>::type;
    using LayoutB = typename std::conditional<IsColumnB, Col, Row>::type;
    using CkGemm =
        CkGemm<DataTypeA, DataTypeB, DataTypeC, AccumulateType, LayoutA,
               LayoutB, UnitOp::NumThreads, TileSizeM, TileSizeN>;

    using GridwiseGemm = typename CkGemm::ImplXdl::GridwiseGemm;

    using CkDataTypeA = typename std::conditional<
        std::is_same<DataTypeA, fp16>::value, ck::half_t,
        typename std::conditional<std::is_same<DataTypeA, bf16>::value,
                                  ck::bhalf_t, DataTypeA>::type>::type;

    using CkDataTypeB = typename std::conditional<
        std::is_same<DataTypeB, fp16>::value, ck::half_t,
        typename std::conditional<std::is_same<DataTypeB, bf16>::value,
                                  ck::bhalf_t, DataTypeB>::type>::type;

    using CkDataTypeC = typename std::conditional<
        std::is_same<DataTypeC, fp16>::value, ck::half_t,
        typename std::conditional<std::is_same<DataTypeC, bf16>::value,
                                  ck::bhalf_t, DataTypeC>::type>::type;

    CkDataTypeC *pC = reinterpret_cast<CkDataTypeC *>(C);
    CkDataTypeA *pA = reinterpret_cast<CkDataTypeA *>(A);
    CkDataTypeB *pB = reinterpret_cast<CkDataTypeB *>(B);

    static constexpr auto I0 = ck::Number<0>{};
    static constexpr auto I1 = ck::Number<1>{};
    static constexpr auto I2 = ck::Number<2>{};

    const ck::index_t K0 = ProblemSizeK / GridwiseGemm::K1;

    const auto a_element_op = PassThrough{};
    const auto b_element_op = PassThrough{};
    const auto c_element_op = PassThrough{};

    auto arg = CkGemm::ImplXdl::MakeArgument(
        pA, pB, pC, ProblemSizeM, ProblemSizeN, ProblemSizeK, LeadingDimA,
        LeadingDimB, LeadingDimC, a_element_op, b_element_op, c_element_op);

    if (UnitOp::thread_id() == 0) {
        assert(GridwiseGemm::CheckValidity(
            arg.a_grid_desc_k0_m_k1_, arg.b_grid_desc_k0_n_k1_,
            arg.c_grid_desc_m_n_, arg.block_2_ctile_map_));
    }
    UnitOp::sync_threads();

    const auto K = arg.a_grid_desc_k0_m_k1_.GetLength(I0) *
                   arg.a_grid_desc_k0_m_k1_.GetLength(I2);

    char *p_shared = UnitOp::template shared_memory<char>(smem_per_warp);

    constexpr int SmemBytes = GridwiseGemm::GetSharedMemoryNumberOfByte();
    IsEq<SmemBytes, UnitOp::SmemBytes>();

    if (GridwiseGemm::CalculateHasMainKBlockLoop(K)) {
        GridwiseGemm::template Run<true>(
            pA, pB, pC, p_shared, arg.a_grid_desc_k0_m_k1_,
            arg.b_grid_desc_k0_n_k1_, arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
            a_element_op, b_element_op, c_element_op, arg.block_2_ctile_map_,
            uop_idx);
    } else {
        GridwiseGemm::template Run<false>(
            pA, pB, pC, p_shared, arg.a_grid_desc_k0_m_k1_,
            arg.b_grid_desc_k0_n_k1_, arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
            a_element_op, b_element_op, c_element_op, arg.block_2_ctile_map_,
            uop_idx);
    }
}

}  // namespace ark

#endif  // ARK_KERNELS_GEMM_CK_H_
