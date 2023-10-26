// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_GEMM_CK_H_
#define ARK_KERNELS_GEMM_CK_H_

#include <type_traits>

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl.hpp"
#include "unit_op.h"

namespace ark {

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

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    constexpr ck::index_t K1 = 4;

    using DeviceGemmXdl = ck::tensor_operation::device::DeviceGemmXdl<
        DataTypeA, DataTypeB, DataTypeC, AccumulateType,
        ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::RowMajor,
        ck::tensor_layout::gemm::RowMajor, PassThrough, PassThrough,
        PassThrough, ck::tensor_operation::device::GemmSpecialization::Default,
        UnitOp::NumThreads, 256, 128, 4, K1, 32, 32, 4, 2,
        ck::Sequence<4, 64, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>, 2,
        4, 4, true, ck::Sequence<4, 64, 1>, ck::Sequence<0, 2, 1>,
        ck::Sequence<0, 2, 1>, 1, 2, 4, true, 7, 1>;
    using GridwiseGemm = typename DeviceGemmXdl::GridwiseGemm;

    char *p_shared = UnitOp::template shared_memory<char>(smem_per_warp);

    static constexpr auto I0 = ck::Number<0>{};
    static constexpr auto I1 = ck::Number<1>{};
    static constexpr auto I2 = ck::Number<2>{};

    static constexpr auto K1Number = ck::Number<K1>{};

    const ck::index_t K0 = ProblemSizeK / K1;

    const auto a_grid_desc_m_k_func = [&]() {
        if constexpr (IsColumnA) {
            return ck::make_naive_tensor_descriptor(
                ck::make_tuple(ProblemSizeM, ProblemSizeK),
                ck::make_tuple(I1, LeadingDimA));
        } else {
            return ck::make_naive_tensor_descriptor(
                ck::make_tuple(ProblemSizeM, ProblemSizeK),
                ck::make_tuple(LeadingDimA, I1));
        }
    }();

    const auto b_grid_desc_k_n_func = [&]() {
        if constexpr (IsColumnB) {
            return ck::make_naive_tensor_descriptor(
                ck::make_tuple(ProblemSizeK, ProblemSizeN),
                ck::make_tuple(I1, LeadingDimB));
        } else {
            return ck::make_naive_tensor_descriptor(
                ck::make_tuple(ProblemSizeK, ProblemSizeN),
                ck::make_tuple(LeadingDimB, I1));
        }
    }();

    const auto c_grid_desc_m_n_func = [&]() {
        return ck::make_naive_tensor_descriptor(
            ck::make_tuple(ProblemSizeM, ProblemSizeN),
            ck::make_tuple(LeadingDimC, I1));
    }();

    auto a_grid_desc_k0_m_k1 = ck::transform_tensor_descriptor(
        a_grid_desc_m_k_func,
        ck::make_tuple(ck::make_unmerge_transform(ck::make_tuple(K0, K1Number)),
                       ck::make_pass_through_transform(ProblemSizeM)),
        ck::make_tuple(ck::Sequence<1>{}, ck::Sequence<0>{}),
        ck::make_tuple(ck::Sequence<0, 2>{}, ck::Sequence<1>{}));

    auto b_grid_desc_k0_n_k1 = ck::transform_tensor_descriptor(
        b_grid_desc_k_n_func,
        ck::make_tuple(ck::make_unmerge_transform(make_tuple(K0, K1Number)),
                       ck::make_pass_through_transform(ProblemSizeN)),
        ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}),
        ck::make_tuple(ck::Sequence<0, 2>{}, ck::Sequence<1>{}));

    auto c_grid_desc_m_n = ck::transform_tensor_descriptor(
        c_grid_desc_m_n_func,
        ck::make_tuple(ck::make_pass_through_transform(ProblemSizeM),
                       ck::make_pass_through_transform(ProblemSizeN)),
        ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}),
        ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}));

    const DimType K =
        a_grid_desc_k0_m_k1.GetLength(I0) * a_grid_desc_k0_m_k1.GetLength(I2);

    // static constexpr bool HasMainKBlockLoop =
    //     GridwiseGemm::CalculateHasMainKBlockLoop(K);

    auto block_2_ctile_map =
        GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n, 1, 1);

    // static_assert(
    //     GridwiseGemm::CheckValidity(a_grid_desc_k0_m_k1, b_grid_desc_k0_n_k1,
    //                                 c_grid_desc_m_n, block_2_ctile_map),
    //     "");

    auto c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
        GridwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(
            c_grid_desc_m_n);

    const auto a_element_op = PassThrough{};
    const auto b_element_op = PassThrough{};
    const auto c_element_op = PassThrough{};

    if (GridwiseGemm::CalculateHasMainKBlockLoop(K)) {
        GridwiseGemm::template Run<true>(
            A, B, C, p_shared, a_grid_desc_k0_m_k1, b_grid_desc_k0_n_k1,
            c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2, a_element_op, b_element_op,
            c_element_op, block_2_ctile_map);
    } else {
        GridwiseGemm::template Run<false>(
            A, B, C, p_shared, a_grid_desc_k0_m_k1, b_grid_desc_k0_n_k1,
            c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2, a_element_op, b_element_op,
            c_element_op, block_2_ctile_map);
    }
}

}  // namespace ark

#endif  // ARK_KERNELS_GEMM_CK_H_
