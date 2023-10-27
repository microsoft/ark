// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_GEMM_CK_H_
#define ARK_KERNELS_GEMM_CK_H_

#include <cassert>
#include <type_traits>

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl.hpp"
#include "type_intrinsics.h"
#include "unit_op.h"

/// Common aliases for CK GeMM configurations.

using F16 = ck::half_t;
using F32 = float;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
static constexpr auto GemmDefault =
    ck::tensor_operation::device::GemmSpecialization::Default;

namespace ark {

template <typename DataTypeA, typename DataTypeB, typename DataTypeC,
          typename AccumulateType, bool IsColumnA, bool IsColumnB,
          int NumThreads, int TileSizeM, int TileSizeN>
struct CkGemm;

template <>
struct CkGemm<fp32, fp32, fp32, fp32, false, false, 256, 128, 256> {
    using Impl = ck::tensor_operation::device::DeviceGemmXdl<
        F32, F32, F32, F32, Row, Row, Row, PassThrough, PassThrough,
        PassThrough, GemmDefault, 256, 128, 256, 4, 4, 32, 32, 2, 4,
        S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 4, 4, true, S<4, 64, 1>,
        S<0, 2, 1>, S<0, 2, 1>, 1, 4, 4, true, 7, 1>;
};

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

    using AccumulateType =
        typename std::conditional<std::is_same<DataTypeC, bf16>::value, float,
                                  DataTypeC>::type;

    using CkGemm =
        CkGemm<DataTypeA, DataTypeB, DataTypeC, AccumulateType, IsColumnA,
               IsColumnB, UnitOp::NumThreads, TileSizeM, TileSizeN>;

    using GridwiseGemm = typename CkGemm::Impl::GridwiseGemm;

    static constexpr auto I0 = ck::Number<0>{};
    static constexpr auto I1 = ck::Number<1>{};
    static constexpr auto I2 = ck::Number<2>{};

    const ck::index_t K0 = ProblemSizeK / GridwiseGemm::K1;

    const auto a_element_op = PassThrough{};
    const auto b_element_op = PassThrough{};
    const auto c_element_op = PassThrough{};

    auto arg = CkGemm::Impl::MakeArgument(
        A, B, C, ProblemSizeM, ProblemSizeN, ProblemSizeK, LeadingDimA,
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

    if (GridwiseGemm::CalculateHasMainKBlockLoop(K)) {
        GridwiseGemm::template Run<true>(
            A, B, C, p_shared, arg.a_grid_desc_k0_m_k1_,
            arg.b_grid_desc_k0_n_k1_, arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
            a_element_op, b_element_op, c_element_op, arg.block_2_ctile_map_,
            uop_idx);
    } else {
        GridwiseGemm::template Run<false>(
            A, B, C, p_shared, arg.a_grid_desc_k0_m_k1_,
            arg.b_grid_desc_k0_n_k1_, arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
            a_element_op, b_element_op, c_element_op, arg.block_2_ctile_map_,
            uop_idx);
    }
}

}  // namespace ark

#endif  // ARK_KERNELS_GEMM_CK_H_
