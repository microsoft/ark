// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_GEMM_CK_H_
#define ARK_KERNELS_GEMM_CK_H_

#include <cassert>
#include <type_traits>

// TODO: temporal until CK officially supports gfx941/942
#if defined(__gfx941__) || defined(__gfx942__)
#define __gfx940__
#endif

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp"
#include "common/checker.h"
#include "common/unit_op.h"

/// Common aliases for CK GeMM configurations.

using F16 = ck::half_t;
using BF16 = ck::bhalf_t;
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

#define DEBUG_CK 0

namespace ark {

template <typename DataTypeA, typename DataTypeB, typename DataTypeC,
          typename AccumulateType, typename LayoutA, typename LayoutB,
          int NumThreads, int TileSizeM, int TileSizeN,
          LoopScheduler LoopSched = LoopScheduler::Default,
          PipelineVersion PipelineVer = PipelineVersion::v1,
          int CShuffleNumStage = 1>
struct CkGemmConfig;

template <int BlockSize, int MPerBlock, int NPerBlock, int KPerBlock, int AK1,
          int BK1, int MPerXDL, int NPerXDL, int MXdlPerWave, int NXdlPerWave,
          int ABlockTransferSrcScalarPerVector,
          int BBlockTransferSrcScalarPerVector,
          int CShuffleMXdlPerWavePerShuffle, int CShuffleNXdlPerWavePerShuffle>
struct PrintDeviceGemmXdlCShuffle;

////////////////////////////////////////////////////////////////////////////////

template <typename LayoutA, typename LayoutB, int NumThreads, int TileSizeM,
          int TileSizeN, LoopScheduler LoopSched, PipelineVersion PipelineVer,
          int CShuffleNumStage>
struct CkGemmConfig<fp32, fp32, fp32, fp32, LayoutA, LayoutB, NumThreads,
                    TileSizeM, TileSizeN, LoopSched, PipelineVer,
                    CShuffleNumStage> {
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
struct CkGemmConfig<fp16, fp16, fp16, fp32, LayoutA, LayoutB, NumThreads,
                    TileSizeM, TileSizeN, LoopSched, PipelineVer,
                    CShuffleNumStage> {
    static constexpr auto Is_16 = (TileSizeM == 16 || TileSizeN == 16);
    static constexpr auto MPerXdl = Is_16 ? 16 : 32;
    static constexpr auto NPerXdl = MPerXdl;
    static constexpr bool IsColA = std::is_same<LayoutA, Col>::value;
    static constexpr bool IsColB = std::is_same<LayoutB, Col>::value;
    static constexpr auto AK1 = 8;  // or (!IsColA) ? 8 : 2;
    static constexpr auto BK1 = 8;  // or IsColB ? 8 : 2;
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

    static constexpr bool Is_256x256x128 =
        NumThreads == 256 && TileSizeM == 256 && TileSizeN == 128;
    static constexpr bool Is_256x128x256 =
        NumThreads == 256 && TileSizeM == 128 && TileSizeN == 256;
    static constexpr bool Is_256x64x128 =
        NumThreads == 256 && TileSizeM == 64 && TileSizeN == 128;
    static constexpr bool Is_256x128x64 =
        NumThreads == 256 && TileSizeM == 128 && TileSizeN == 64;

    static constexpr bool Is_128x128x64 =
        NumThreads == 128 && TileSizeM == 128 && TileSizeN == 64;
    static constexpr bool Is_128x128x32 =
        NumThreads == 128 && TileSizeM == 128 && TileSizeN == 32;

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

    static constexpr auto A_Lengths_K0 =
        (!IsColA || AK1 == 8 || Is_256x256x128 || Is_128x128x128 ||
         Is_128x128x64)
            ? 4
            : Is_256x64x128 ? 16 : 8;

    static constexpr auto B_Lengths_K0 =
        (IsColB || BK1 == 8 || Is_256x128x256 || Is_128x128x128 ||
         Is_128x64x128)
            ? 4
            : Is_256x128x64 ? 16 : 8;

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

    using ImplXdlCShuffle =
        ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
            LayoutA, LayoutB, Row, F16, F16, F16, F32, F16, PassThrough,
            PassThrough, PassThrough, GemmDefault, 1, NumThreads, TileSizeM,
            TileSizeN, 32, AK1, BK1, 32, 32, MXdlPerWave, NXdlPerWave,
            S<A_Lengths_K0, (NumThreads / A_Lengths_K0), 1>,
            typename std::conditional<IsColA, S<0, 2, 1>, S<1, 0, 2>>::type,
            typename std::conditional<IsColA, S<0, 2, 1>, S<1, 0, 2>>::type,
            (IsColA ? 1 : 2),
            (!IsColA ? 8 : (AK1 == 2 || Is_128x128x64) ? 4 : MXdlPerWave), AK1,
            (AK1 == 8), S<B_Lengths_K0, (NumThreads / B_Lengths_K0), 1>,
            typename std::conditional<IsColB, S<1, 0, 2>, S<0, 2, 1>>::type,
            typename std::conditional<IsColB, S<1, 0, 2>, S<0, 2, 1>>::type,
            (IsColB ? 2 : 1),
            (IsColB ? 8
                    : (BK1 == 2 || Is_256x128x256 || Is_128x128x128 ||
                       Is_128x64x128)
                          ? 4
                          : NXdlPerWave),
            BK1, (BK1 == 8), 1, 1,
            S<1,
              (Is_128x128x128 || Is_128x64x128 || NumThreads == 64) ? 16 : 32,
              1,
              ((Is_128x128x64 || Is_128x128x32 || NumThreads == 64) ? 4 : 8)>,
            8>;

#if (DEBUG_CK != 0)
    PrintDeviceGemmXdlCShuffle<
        NumThreads, TileSizeM, TileSizeN, 32, AK1, BK1, 32, 32, MXdlPerWave,
        NXdlPerWave,
        (!IsColA ? 8 : (AK1 == 2 || Is_128x128x64) ? 4 : MXdlPerWave),
        (IsColB
             ? 8
             : (BK1 == 2 || Is_256x128x256 || Is_128x128x128 || Is_128x64x128)
                   ? 4
                   : NXdlPerWave),
        1, 1>
        p;
#endif  // (DEBUG_CK != 0)
};

template <typename LayoutA, typename LayoutB, int NumThreads, int TileSizeM,
          int TileSizeN, LoopScheduler LoopSched, PipelineVersion PipelineVer,
          int CShuffleNumStage>
struct CkGemmConfig<bf16, bf16, bf16, fp32, LayoutA, LayoutB, NumThreads,
                    TileSizeM, TileSizeN, LoopSched, PipelineVer,
                    CShuffleNumStage> {
    static constexpr auto Is_16 = (TileSizeM == 16 || TileSizeN == 16);
    static constexpr bool IsColA = std::is_same_v<LayoutA, Col>;
    static constexpr bool IsColB = std::is_same_v<LayoutB, Col>;
    static constexpr auto AK1 = (!IsColA) ? 8 : 2;  // or 8
    static constexpr auto BK1 = IsColB ? 8 : 2;     // or 8
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

    static constexpr bool Is_256x256x128 =
        NumThreads == 256 && TileSizeM == 256 && TileSizeN == 128;
    static constexpr bool Is_128x128x128 =
        NumThreads == 128 && TileSizeM == 128 && TileSizeN == 128;
    static constexpr bool Is_128x128x64 =
        NumThreads == 128 && TileSizeM == 128 && TileSizeN == 64;
    static constexpr bool Is_128x128x32 =
        NumThreads == 128 && TileSizeM == 128 && TileSizeN == 32;

    static constexpr bool Is_256x64x128 =
        NumThreads == 256 && TileSizeM == 64 && TileSizeN == 128;

    static constexpr auto A_Lengths_K0 =
        (!IsColA || AK1 == 8 || Is_256x256x128 || Is_128x128x128 ||
         Is_128x128x64)
            ? 4
            : Is_256x64x128 ? 16 : 8;

    static constexpr bool Is_256x128x256 =
        NumThreads == 256 && TileSizeM == 128 && TileSizeN == 256;
    static constexpr bool Is_128x64x128 =
        NumThreads == 128 && TileSizeM == 64 && TileSizeN == 128;

    static constexpr bool Is_256x128x64 =
        NumThreads == 256 && TileSizeM == 128 && TileSizeN == 64;

    static constexpr auto B_Lengths_K0 =
        (IsColB || BK1 == 8 || Is_256x128x256 || Is_128x128x128 ||
         Is_128x64x128)
            ? 4
            : Is_256x128x64 ? 16 : 8;

    using ImplXdlCShuffle =
        ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
            LayoutA, LayoutB, Row, BF16, BF16, BF16, F32, BF16, PassThrough,
            PassThrough, PassThrough, GemmDefault, 1, NumThreads, TileSizeM,
            TileSizeN, 32, AK1, BK1, 32, 32, MXdlPerWave, NXdlPerWave,
            S<A_Lengths_K0, (NumThreads / A_Lengths_K0), 1>,
            typename std::conditional<IsColA, S<0, 2, 1>, S<1, 0, 2>>::type,
            typename std::conditional<IsColA, S<0, 2, 1>, S<1, 0, 2>>::type,
            (IsColA ? 1 : 2),
            (!IsColA ? 8 : (AK1 == 2 || Is_128x128x64) ? 4 : MXdlPerWave), AK1,
            (AK1 == 8), S<B_Lengths_K0, (NumThreads / B_Lengths_K0), 1>,
            typename std::conditional<IsColB, S<1, 0, 2>, S<0, 2, 1>>::type,
            typename std::conditional<IsColB, S<1, 0, 2>, S<0, 2, 1>>::type,
            (IsColB ? 2 : 1),
            (IsColB ? 8
                    : (BK1 == 2 || Is_256x128x256 || Is_128x128x128 ||
                       Is_128x64x128)
                          ? 4
                          : NXdlPerWave),
            BK1, (BK1 == 8), 1, 1,
            S<1,
              (Is_128x128x128 || Is_128x64x128 || NumThreads == 64) ? 16 : 32,
              1,
              ((Is_128x128x64 || Is_128x128x32 || NumThreads == 64) ? 4 : 8)>,
            8>;
};

////////////////////////////////////////////////////////////////////////////////

template <typename CkGemmConfig, typename = void>
struct HasXdl : std::false_type {};

template <typename CkGemmConfig>
struct HasXdl<CkGemmConfig, std::void_t<typename CkGemmConfig::ImplXdl>>
    : std::true_type {};

template <typename CkGemmConfig, typename = void>
struct HasXdlCShuffle : std::false_type {};

template <typename CkGemmConfig>
struct HasXdlCShuffle<CkGemmConfig,
                      std::void_t<typename CkGemmConfig::ImplXdlCShuffle>>
    : std::true_type {};

template <typename DataTypeA, int LeadingDimA, bool IsColumnA,
          typename DataTypeB, int LeadingDimB, bool IsColumnB,
          typename DataTypeC, int LeadingDimC, int ProblemSizeM,
          int ProblemSizeN, int ProblemSizeK, int TileSizeM, int TileSizeN,
          int TileSizeK, typename UnitOp>
struct CkGemm {
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
    using CkGemmConfig =
        CkGemmConfig<DataTypeA, DataTypeB, DataTypeC, AccumulateType, LayoutA,
                     LayoutB, UnitOp::NumThreads, TileSizeM, TileSizeN>;

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

    DEVICE void RunXdl(DataTypeC *C, DataTypeA *A, DataTypeB *B, int uop_idx,
                       int smem_per_warp) const {
        using Impl = typename CkGemmConfig::ImplXdl;
        using GridwiseGemm = typename Impl::GridwiseGemm;

        CkDataTypeC *pC = reinterpret_cast<CkDataTypeC *>(C);
        CkDataTypeA *pA = reinterpret_cast<CkDataTypeA *>(A);
        CkDataTypeB *pB = reinterpret_cast<CkDataTypeB *>(B);

        const auto a_element_op = PassThrough{};
        const auto b_element_op = PassThrough{};
        const auto c_element_op = PassThrough{};

        auto arg = Impl::MakeArgument(
            pA, pB, pC, ProblemSizeM, ProblemSizeN, ProblemSizeK, LeadingDimA,
            LeadingDimB, LeadingDimC, a_element_op, b_element_op, c_element_op);

        if (UnitOp::thread_id() == 0) {
            assert(GridwiseGemm::CheckValidity(arg));
        }
        UnitOp::sync_threads();

        char *p_shared = UnitOp::template shared_memory<char>(smem_per_warp);

        constexpr int SmemBytes = GridwiseGemm::GetSharedMemoryNumberOfByte();
        IsEq<GridwiseGemm::ThisThreadBlock::GetNumOfThread(),
             UnitOp::NumThreads>();
        IsEq<SmemBytes, UnitOp::SmemBytes>();

        const auto a_grid_desc_k0_m_k1 =
            amd_wave_read_first_lane(GridwiseGemm::MakeAGridDescriptor_K0_M_K1(
                arg.M, arg.MPadded, arg.K, arg.K0, arg.StrideA));
        const auto b_grid_desc_k0_n_k1 =
            amd_wave_read_first_lane(GridwiseGemm::MakeBGridDescriptor_K0_N_K1(
                arg.K, arg.N, arg.NPadded, arg.K0, arg.StrideB));
        const auto c_grid_desc_m_n =
            amd_wave_read_first_lane(GridwiseGemm::MakeCGridDescriptor_M_N(
                arg.M, arg.MPadded, arg.N, arg.NPadded, arg.StrideC));

        if (GridwiseGemm::CalculateHasMainKBlockLoop(ProblemSizeK)) {
            GridwiseGemm::template Run<true>(
                pA, pB, pC, p_shared, a_grid_desc_k0_m_k1, b_grid_desc_k0_n_k1,
                c_grid_desc_m_n, uop_idx);
        } else {
            GridwiseGemm::template Run<false>(
                pA, pB, pC, p_shared, a_grid_desc_k0_m_k1, b_grid_desc_k0_n_k1,
                c_grid_desc_m_n, uop_idx);
        }
    }

    DEVICE void RunXdlCShuffle(DataTypeC *C, DataTypeA *A, DataTypeB *B,
                               int uop_idx, int smem_per_warp) const {
        using GridwiseGemm =
            typename CkGemmConfig::ImplXdlCShuffle::GridwiseGemm;

        CkDataTypeC *pC = reinterpret_cast<CkDataTypeC *>(C);
        CkDataTypeA *pA = reinterpret_cast<CkDataTypeA *>(A);
        CkDataTypeB *pB = reinterpret_cast<CkDataTypeB *>(B);

        typename GridwiseGemm::Problem problem(ProblemSizeM, ProblemSizeN,
                                               ProblemSizeK, LeadingDimA,
                                               LeadingDimB, LeadingDimC);

        if (UnitOp::thread_id() == 0) {
            assert(GridwiseGemm::CheckValidity(problem));
        }
        UnitOp::sync_threads();

        char *p_shared = UnitOp::template shared_memory<char>(smem_per_warp);

        constexpr int SmemBytes = GridwiseGemm::GetSharedMemoryNumberOfByte();
        IsEq<GridwiseGemm::ThisThreadBlock::GetNumOfThread(),
             UnitOp::NumThreads>();
        IsEq<SmemBytes, UnitOp::SmemBytes>();

        if (GridwiseGemm::CalculateHasMainKBlockLoop(ProblemSizeK)) {
            GridwiseGemm::template Run<true>(pA, pB, pC, p_shared, problem,
                                             uop_idx);
        } else {
            GridwiseGemm::template Run<false>(pA, pB, pC, p_shared, problem,
                                              uop_idx);
        }
    }

    DEVICE void Run(DataTypeC *C, DataTypeA *A, DataTypeB *B, int uop_idx,
                    int smem_per_warp) const {
        if constexpr (HasXdlCShuffle<CkGemmConfig>::value) {
            RunXdlCShuffle(C, A, B, uop_idx, smem_per_warp);
        } else if constexpr (HasXdl<CkGemmConfig>::value) {
            RunXdl(C, A, B, uop_idx, smem_per_warp);
        } else {
            assert(false);
        }
    }
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
    using CkGemm =
        CkGemm<DataTypeA, LeadingDimA, IsColumnA, DataTypeB, LeadingDimB,
               IsColumnB, DataTypeC, LeadingDimC, ProblemSizeM, ProblemSizeN,
               ProblemSizeK, TileSizeM, TileSizeN, TileSizeK, UnitOp>;
    CkGemm gemm;
    gemm.Run(C, A, B, uop_idx, smem_per_warp);
}

}  // namespace ark

#endif  // ARK_KERNELS_GEMM_CK_H_
