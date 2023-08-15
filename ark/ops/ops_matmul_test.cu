// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark_utils.h"
#include "ops_test_common.h"
#include <cassert>
#include <cublas_v2.h>
#include <type_traits>

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
        cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, ldb,
                    a, lda, &beta, c, ldc);
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
        cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, b, ldb,
                    a, lda, &beta, c, ldc);
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
        cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha, b, ldb,
                    a, lda, &beta, c, ldc);
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
        cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, &alpha, b, ldb,
                    a, lda, &beta, c, ldc);
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
        cublasHgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, ldb,
                    a, lda, &beta, c, ldc);
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
        cublasHgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, b, ldb,
                    a, lda, &beta, c, ldc);
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
        cublasHgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha, b, ldb,
                    a, lda, &beta, c, ldc);
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
        cublasHgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, &alpha, b, ldb,
                    a, lda, &beta, c, ldc);
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

    auto memA = ark::to_gpu(inputs[0], input_shapes[0].size() * sizeof(T));
    auto memB = ark::to_gpu(inputs[1], input_shapes[1].size() * sizeof(T));
    auto memC = ark::to_gpu(outputs[0], output_shapes[0].size() * sizeof(T));

    T *devA = static_cast<T *>(memA.get());
    T *devB = static_cast<T *>(memB.get());
    T *devC = static_cast<T *>(memC.get());

    // matmul using cublas
    if constexpr (std::is_same_v<T, float>) {
        cublas_matmul_float_nn(m, n, k, devA, lda, devB, ldb, devC, ldc);
    } else if constexpr (std::is_same_v<T, half>) {
        cublas_matmul_half_nn(m, n, k, devA, lda, devB, ldb, devC, ldc);
    } else {
        throw std::runtime_error("Unsupported data type");
    }
    ark::sync_gpu();

    // copy back to host
    ark::from_gpu(memC, outputs[0]);
}

template <typename T>
void baseline_matmul_nt(std::vector<void *> &outputs,
                        const std::vector<ark::Dims> &output_shapes,
                        const std::vector<void *> &inputs,
                        const std::vector<ark::Dims> &input_shapes)
{
    // baseline inputs & outputs have no padding
    int m = output_shapes[0].dims4()[2];
    int n = output_shapes[0].dims4()[3];
    int k = input_shapes[0].dims4()[3];
    int lda = k;
    int ldb = k;
    int ldc = n;

    auto memA = ark::to_gpu(inputs[0], input_shapes[0].size() * sizeof(T));
    auto memB = ark::to_gpu(inputs[1], input_shapes[1].size() * sizeof(T));
    auto memC = ark::to_gpu(outputs[0], output_shapes[0].size() * sizeof(T));

    T *devA = static_cast<T *>(memA.get());
    T *devB = static_cast<T *>(memB.get());
    T *devC = static_cast<T *>(memC.get());

    // matmul using cublas
    if constexpr (std::is_same_v<T, float>) {
        cublas_matmul_float_nt(m, n, k, devA, lda, devB, ldb, devC, ldc);
    } else if constexpr (std::is_same_v<T, half>) {
        cublas_matmul_half_nt(m, n, k, devA, lda, devB, ldb, devC, ldc);
    } else {
        throw std::runtime_error("Unsupported data type");
    }
    ark::sync_gpu();

    // copy back to host
    ark::from_gpu(memC, outputs[0]);
}

template <typename T>
void baseline_matmul_tn(std::vector<void *> &outputs,
                        const std::vector<ark::Dims> &output_shapes,
                        const std::vector<void *> &inputs,
                        const std::vector<ark::Dims> &input_shapes)
{
    // baseline inputs & outputs have no padding
    int m = output_shapes[0].dims4()[2];
    int n = output_shapes[0].dims4()[3];
    int k = input_shapes[0].dims4()[2];
    int lda = m;
    int ldb = n;
    int ldc = n;

    auto memA = ark::to_gpu(inputs[0], input_shapes[0].size() * sizeof(T));
    auto memB = ark::to_gpu(inputs[1], input_shapes[1].size() * sizeof(T));
    auto memC = ark::to_gpu(outputs[0], output_shapes[0].size() * sizeof(T));

    T *devA = static_cast<T *>(memA.get());
    T *devB = static_cast<T *>(memB.get());
    T *devC = static_cast<T *>(memC.get());

    // matmul using cublas
    if constexpr (std::is_same_v<T, float>) {
        cublas_matmul_float_tn(m, n, k, devA, lda, devB, ldb, devC, ldc);
    } else if constexpr (std::is_same_v<T, half>) {
        cublas_matmul_half_tn(m, n, k, devA, lda, devB, ldb, devC, ldc);
    } else {
        throw std::runtime_error("Unsupported data type");
    }
    ark::sync_gpu();

    // copy back to host
    ark::from_gpu(memC, outputs[0]);
}

template <typename T>
void baseline_matmul_tt(std::vector<void *> &outputs,
                        const std::vector<ark::Dims> &output_shapes,
                        const std::vector<void *> &inputs,
                        const std::vector<ark::Dims> &input_shapes)
{
    // baseline inputs & outputs have no padding
    int m = output_shapes[0].dims4()[2];
    int n = output_shapes[0].dims4()[3];
    int k = input_shapes[0].dims4()[2];
    int lda = m;
    int ldb = k;
    int ldc = n;

    auto memA = ark::to_gpu(inputs[0], input_shapes[0].size() * sizeof(T));
    auto memB = ark::to_gpu(inputs[1], input_shapes[1].size() * sizeof(T));
    auto memC = ark::to_gpu(outputs[0], output_shapes[0].size() * sizeof(T));

    T *devA = static_cast<T *>(memA.get());
    T *devB = static_cast<T *>(memB.get());
    T *devC = static_cast<T *>(memC.get());

    // matmul using cublas
    if constexpr (std::is_same_v<T, float>) {
        cublas_matmul_float_tt(m, n, k, devA, lda, devB, ldb, devC, ldc);
    } else if constexpr (std::is_same_v<T, half>) {
        cublas_matmul_half_tt(m, n, k, devA, lda, devB, ldb, devC, ldc);
    } else {
        throw std::runtime_error("Unsupported data type");
    }
    ark::sync_gpu();

    // copy back to host
    ark::from_gpu(memC, outputs[0]);
}

ark::unittest::State test_matmul_gran0()
{
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(128, 64), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(64, 256), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, false, "matmul", 0);

        auto result = ark::op_test("matmul_gran0", m, {a, b}, {c},
                                   baseline_matmul_nn<half>);
        ark::op_test_log(result);
    }
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(4096, 8192), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(8192, 16384), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, false, "matmul", 0);

        auto result = ark::op_test("matmul_gran0", m, {a, b}, {c},
                                   baseline_matmul_nn<half>);
        ark::op_test_log(result);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_gran1()
{
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(128, 64), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(64, 256), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, false, "matmul", 1);

        auto result = ark::op_test("matmul_gran1", m, {a, b}, {c},
                                   baseline_matmul_nn<half>);
        ark::op_test_log(result);
    }
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(4096, 8192), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(8192, 16384), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, false, "matmul", 1);

        auto result = ark::op_test("matmul_gran1", m, {a, b}, {c},
                                   baseline_matmul_nn<half>);
        ark::op_test_log(result);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_gran2()
{
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(128, 64), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(64, 256), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, false, "matmul", 2);

        auto result = ark::op_test("matmul_gran2", m, {a, b}, {c},
                                   baseline_matmul_nn<half>);
        ark::op_test_log(result);
    }
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(4096, 8192), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(8192, 16384), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, false, "matmul", 2);

        auto result = ark::op_test("matmul_gran2", m, {a, b}, {c},
                                   baseline_matmul_nn<half>);
        ark::op_test_log(result);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_split()
{
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(4096, 8192), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(8192, 16384), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 7, false, false, "matmul", 0);

        auto result = ark::op_test("matmul_split", m, {a, b}, {c},
                                   baseline_matmul_nn<half>);
        ark::op_test_log(result);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_fp32()
{
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(128, 64), ark::FP32);
        ark::Tensor *b = m.tensor(ark::Dims(64, 256), ark::FP32);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, false, "matmul", 0);

        auto result = ark::op_test("matmul_fp32", m, {a, b}, {c},
                                   baseline_matmul_nn<float>);
        ark::op_test_log(result);
    }
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(4096, 8192), ark::FP32);
        ark::Tensor *b = m.tensor(ark::Dims(8192, 16384), ark::FP32);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, false, "matmul", 0);

        auto result = ark::op_test("matmul_fp32", m, {a, b}, {c},
                                   baseline_matmul_nn<float>);
        ark::op_test_log(result);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_nt()
{
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(128, 64), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(256, 64), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, true, "matmul", 0);

        auto result =
            ark::op_test("matmul_nt", m, {a, b}, {c}, baseline_matmul_nt<half>);
        ark::op_test_log(result);
    }
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(4096, 8192), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(16384, 8192), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, true, "matmul", 0);

        auto result =
            ark::op_test("matmul_nt", m, {a, b}, {c}, baseline_matmul_nt<half>);
        ark::op_test_log(result);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_tn()
{
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(64, 128), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(64, 256), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, true, false, "matmul", 0);

        auto result =
            ark::op_test("matmul_tn", m, {a, b}, {c}, baseline_matmul_tn<half>, "ones", true);
        ark::op_test_log(result);
    }
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(8192, 4096), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(8192, 16384), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, true, false, "matmul", 0);

        auto result =
            ark::op_test("matmul_tn", m, {a, b}, {c}, baseline_matmul_tn<half>);
        ark::op_test_log(result);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_tt()
{
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(64, 128), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(256, 64), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, true, true, "matmul", 0);

        auto result =
            ark::op_test("matmul_tt", m, {a, b}, {c}, baseline_matmul_tt<half>);
        ark::op_test_log(result);
    }
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(8192, 4096), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(16384, 8192), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, true, true, "matmul", 0);

        auto result =
            ark::op_test("matmul_tt", m, {a, b}, {c}, baseline_matmul_tt<half>);
        ark::op_test_log(result);
    }
    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_matmul_gran0);
    UNITTEST(test_matmul_gran1);
    UNITTEST(test_matmul_gran2);
    UNITTEST(test_matmul_split);
    UNITTEST(test_matmul_fp32);
    UNITTEST(test_matmul_nt);
    UNITTEST(test_matmul_tn);
    UNITTEST(test_matmul_tt);

    cublasDestroy(get_cublas_handle());
    return ark::unittest::SUCCESS;
}
