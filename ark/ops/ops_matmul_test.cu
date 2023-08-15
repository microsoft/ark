// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cublas_v2.h>
#include <type_traits>
#include <cassert>
#include "ark_utils.h"
#include "ops_test_common.h"

cublasHandle_t globalCublasHandle = nullptr;

cublasHandle_t get_cublas_handle()
{
    if (globalCublasHandle == nullptr) {
        cublasStatus_t status = cublasCreate(&globalCublasHandle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cublas handle");
        }
    }
    return globalCublasHandle;
}

void cublas_matmul_float_nn(int m, int n, int k, const float *a, int lda,
                            const float *b, int ldb, float *c, int ldc)
{
    auto cublasH = get_cublas_handle();
    float alpha = 1;
    float beta = 0;
    cublasStatus_t status =
        cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a, lda,
                    b, ldb, &beta, c, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to call cublasSgemm");
    }
}

void cublas_matmul_float_nt(int m, int n, int k, const float *a, int lda,
                            const float *b, int ldb, float *c, int ldc)
{
    auto cublasH = get_cublas_handle();
    float alpha = 1;
    float beta = 0;
    cublasStatus_t status =
        cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, a, lda,
                    b, ldb, &beta, c, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to call cublasSgemm");
    }
}

void cublas_matmul_float_tn(int m, int n, int k, const float *a, int lda,
                            const float *b, int ldb, float *c, int ldc)
{
    auto cublasH = get_cublas_handle();
    float alpha = 1;
    float beta = 0;
    cublasStatus_t status =
        cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, a, lda,
                    b, ldb, &beta, c, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to call cublasSgemm");
    }
}

void cublas_matmul_float_tt(int m, int n, int k, const float *a, int lda,
                            const float *b, int ldb, float *c, int ldc)
{
    auto cublasH = get_cublas_handle();
    float alpha = 1;
    float beta = 0;
    cublasStatus_t status =
        cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, a, lda,
                    b, ldb, &beta, c, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to call cublasSgemm");
    }
}

void cublas_matmul_half_nn(int m, int n, int k, const half *a, int lda,
                           const half *b, int ldb, half *c, int ldc)
{
    auto cublasH = get_cublas_handle();
    half alpha = half(ark::half_t(1));
    half beta = half(ark::half_t(0));
    cublasStatus_t status =
        cublasHgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a, lda,
                    b, ldb, &beta, c, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to call cublasHgemm");
    }
}

void cublas_matmul_half_nt(int m, int n, int k, const half *a, int lda,
                           const half *b, int ldb, half *c, int ldc)
{
    auto cublasH = get_cublas_handle();
    half alpha = half(ark::half_t(1));
    half beta = half(ark::half_t(0));
    cublasStatus_t status =
        cublasHgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, a, lda,
                    b, ldb, &beta, c, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to call cublasHgemm");
    }
}

void cublas_matmul_half_tn(int m, int n, int k, const half *a, int lda,
                           const half *b, int ldb, half *c, int ldc)
{
    auto cublasH = get_cublas_handle();
    half alpha = half(ark::half_t(1));
    half beta = half(ark::half_t(0));
    cublasStatus_t status =
        cublasHgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, a, lda,
                    b, ldb, &beta, c, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to call cublasHgemm");
    }
}

void cublas_matmul_half_tt(int m, int n, int k, const half *a, int lda,
                           const half *b, int ldb, half *c, int ldc)
{
    auto cublasH = get_cublas_handle();
    half alpha = half(ark::half_t(1));
    half beta = half(ark::half_t(0));
    cublasStatus_t status =
        cublasHgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, a, lda,
                    b, ldb, &beta, c, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to call cublasHgemm");
    }
}

template <typename T>
void baseline_matmul_nn(std::vector<void *> &outputs,
                        const std::vector<ark::Dims> &output_shapes,
                        const std::vector<void *> &inputs,
                        const std::vector<ark::Dims> &input_shapes)
{
    // baseline inputs & outputs have no padding
    int m = output_shapes[0].dims4()[2];
    int n = output_shapes[0].dims4()[3];
    int k = input_shapes[0].dims4()[3];
    int lda = k;
    int ldb = n;
    int ldc = n;

    // matmul using cublas
    if constexpr (std::is_same_v<T, float>) {
        cublas_matmul_float_nn(m, n, k, reinterpret_cast<float *>(inputs[0]),
                               lda, reinterpret_cast<float *>(inputs[1]), ldb,
                               reinterpret_cast<float *>(outputs[0]), ldc);
    } else if constexpr (std::is_same_v<T, half>) {
        cublas_matmul_half_nn(m, n, k, reinterpret_cast<half *>(inputs[0]), lda,
                              reinterpret_cast<half *>(inputs[1]), ldb,
                              reinterpret_cast<half *>(outputs[0]), ldc);
    } else {
        throw std::runtime_error("Unsupported data type");
    }
}

ark::unittest::State test_matmul_gran0()
{
    ark::Model m;
    ark::Tensor *a = m.tensor(ark::Dims(1, 64), ark::FP32);
    ark::Tensor *b = m.tensor(ark::Dims(64, 64), ark::FP32);
    ark::Tensor *c = m.matmul(a, b);

    auto result = ark::op_test("matmul_gran0", m, {a, b}, {c}, baseline_matmul_nn<float>, true);
    ark::op_test_log(result);
    // test_matmul_internal(/*m=*/64, /*n=*/64, /*k=*/32, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/0);
    // test_matmul_internal(/*m=*/128, /*n=*/64, /*k=*/32, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/0);
    // test_matmul_internal(/*m=*/64, /*n=*/128, /*k=*/32, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/0);
    // test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/32, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/0);

    // test_matmul_internal(/*m=*/64, /*n=*/64, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/0);
    // test_matmul_internal(/*m=*/128, /*n=*/64, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/0);
    // test_matmul_internal(/*m=*/64, /*n=*/128, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/0);
    // test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/0);
    // test_matmul_internal(/*m=*/256, /*n=*/128, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/0);

    // test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/256, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/1, /*gran_lev=*/0);

    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/1, /*gran_lev=*/0);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_gran1()
{
    // test_matmul_internal(/*m=*/64, /*n=*/64, /*k=*/32, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/1);
    // test_matmul_internal(/*m=*/128, /*n=*/64, /*k=*/32, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/1);
    // test_matmul_internal(/*m=*/64, /*n=*/128, /*k=*/32, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/1);
    // test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/32, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/1);

    // test_matmul_internal(/*m=*/64, /*n=*/64, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/1);
    // test_matmul_internal(/*m=*/128, /*n=*/64, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/1);
    // test_matmul_internal(/*m=*/64, /*n=*/128, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/1);
    // test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/1);
    // test_matmul_internal(/*m=*/256, /*n=*/128, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/1);

    // test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/256, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/1, /*gran_lev=*/1);

    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/1, /*gran_lev=*/1);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_gran2()
{
    // test_matmul_internal(/*m=*/64, /*n=*/64, /*k=*/32, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/2);
    // test_matmul_internal(/*m=*/128, /*n=*/64, /*k=*/32, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/2);
    // test_matmul_internal(/*m=*/64, /*n=*/128, /*k=*/32, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/2);
    // test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/32, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/2);

    // test_matmul_internal(/*m=*/64, /*n=*/64, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/2);
    // test_matmul_internal(/*m=*/128, /*n=*/64, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/2);
    // test_matmul_internal(/*m=*/64, /*n=*/128, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/2);
    // test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/2);
    // test_matmul_internal(/*m=*/256, /*n=*/128, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/2);

    // test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/256, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/1,
    //                      /*gran_lev=*/2);

    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/1, /*gran_lev=*/2);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_gran3()
{
    // test_matmul_internal(/*m=*/64, /*n=*/64, /*k=*/32, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/3);
    // test_matmul_internal(/*m=*/128, /*n=*/64, /*k=*/32, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/3);
    // test_matmul_internal(/*m=*/64, /*n=*/128, /*k=*/32, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/3);
    // test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/32, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/3);

    // test_matmul_internal(/*m=*/64, /*n=*/64, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/3);
    // test_matmul_internal(/*m=*/128, /*n=*/64, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/3);
    // test_matmul_internal(/*m=*/64, /*n=*/128, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/3);
    // test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/3);
    // test_matmul_internal(/*m=*/256, /*n=*/128, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/3);

    // test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/256, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/1, /*gran_lev=*/3);

    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/1, /*gran_lev=*/3);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_split()
{
    // test_matmul_internal(/*m=*/64, /*n=*/64, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/2, /*gran_lev=*/2);
    // test_matmul_internal(/*m=*/128, /*n=*/64, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/2, /*gran_lev=*/2);
    // test_matmul_internal(/*m=*/64, /*n=*/128, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/2, /*gran_lev=*/2);
    // test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/2, /*gran_lev=*/2);
    // test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/256, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/2, /*gran_lev=*/2);
    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/2, /*gran_lev=*/2);

    // test_matmul_internal(/*m=*/64, /*n=*/64, /*k=*/128, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/4, /*gran_lev=*/2);
    // test_matmul_internal(/*m=*/128, /*n=*/64, /*k=*/128, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/4, /*gran_lev=*/2);
    // test_matmul_internal(/*m=*/64, /*n=*/128, /*k=*/128, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/4, /*gran_lev=*/2);
    // test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/128, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/4, /*gran_lev=*/2);
    // test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/256, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/4, /*gran_lev=*/2);
    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/4, /*gran_lev=*/2);

    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/4, /*gran_lev=*/0);
    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/4, /*gran_lev=*/1);
    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/4, /*gran_lev=*/2);

    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/3, /*gran_lev=*/0);
    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/3, /*gran_lev=*/1);
    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/3, /*gran_lev=*/2);
    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/3, /*gran_lev=*/2);

    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/5, /*gran_lev=*/0);
    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/5, /*gran_lev=*/1);
    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/5, /*gran_lev=*/2);
    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/5, /*gran_lev=*/2);

    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/6, /*gran_lev=*/0);
    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/6, /*gran_lev=*/1);
    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/6, /*gran_lev=*/2);
    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/6, /*gran_lev=*/2);

    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/7, /*gran_lev=*/0);
    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/7, /*gran_lev=*/1);
    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/7, /*gran_lev=*/2);
    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/7, /*gran_lev=*/2);

    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/8, /*gran_lev=*/0);
    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/8, /*gran_lev=*/1);
    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/8, /*gran_lev=*/2);
    // test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
    //                      /*bs_b=*/1, /*split_k=*/8, /*gran_lev=*/2);

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_perf()
{
    // test_matmul_internal(/*m=*/64, /*n=*/64, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/-1, /*iter=*/1000);
    // test_matmul_internal(/*m=*/64, /*n=*/128, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/-1, /*iter=*/1000);
    // test_matmul_internal(/*m=*/128, /*n=*/64, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/-1, /*iter=*/1000);
    // test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/-1, /*iter=*/1000);
    // test_matmul_internal(/*m=*/256, /*n=*/128, /*k=*/64, /*bs_a=*/1,
    // /*bs_b=*/1,
    //                      /*split_k=*/1, /*gran_lev=*/-1, /*iter=*/1000);
    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_matmul_gran0);
    UNITTEST(test_matmul_gran1);
    UNITTEST(test_matmul_gran2);
    // UNITTEST(test_matmul_gran3);
    UNITTEST(test_matmul_split);
    UNITTEST(test_matmul_perf);

    cublasDestroy(get_cublas_handle());
    return ark::unittest::SUCCESS;
}
