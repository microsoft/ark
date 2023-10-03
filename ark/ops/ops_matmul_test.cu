// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cublas_v2.h>

#include <cassert>
#include <type_traits>

#include "ark_utils.h"
#include "ops_test_common.h"

cublasHandle_t globalCublasHandle = nullptr;

cublasHandle_t get_cublas_handle() {
    if (globalCublasHandle == nullptr) {
        cublasStatus_t status = cublasCreate(&globalCublasHandle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cublas handle");
        }
    }
    return globalCublasHandle;
}

template <int CubOpTypeA, int CubOpTypeB, int CudaDataType, int CubComputeType>
void cublas_matmul(int m, int n, int k, void *alpha, const void *a, int lda,
                   const void *b, int ldb, void *beta, void *c, int ldc,
                   int batch_size = 1) {
    auto cublasH = get_cublas_handle();
    cublasStatus_t status;
    cublasOperation_t optypeA = (cublasOperation_t)CubOpTypeA;
    cublasOperation_t optypeB = (cublasOperation_t)CubOpTypeB;
    cudaDataType dtype = (cudaDataType)CudaDataType;
    cublasComputeType_t ctype = (cublasComputeType_t)CubComputeType;
    if (batch_size == 1) {
        status = cublasGemmEx(cublasH, optypeB, optypeA, n, m, k, alpha, b,
                              dtype, ldb, a, dtype, lda, beta, c, dtype, ldc,
                              ctype, CUBLAS_GEMM_DEFAULT);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to call cublasGemmEx");
        }
    } else {
        status = cublasGemmStridedBatchedEx(
            cublasH, optypeB, optypeA, n, m, k, alpha, b, dtype, ldb, n * k, a,
            dtype, lda, k * m, beta, c, dtype, ldc, n * m, batch_size, ctype,
            CUBLAS_GEMM_DEFAULT);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(
                "Failed to call cublasGemmStridedBatchedEx");
        }
    }
}

template <typename T>
void baseline_matmul_nn(std::vector<void *> &outputs,
                        const std::vector<ark::Dims> &output_shapes,
                        const std::vector<void *> &inputs,
                        const std::vector<ark::Dims> &input_shapes) {
    auto out_shape_dims4 = output_shapes[0].dims4();

    // baseline inputs & outputs have no padding
    int m = out_shape_dims4[2];
    int n = out_shape_dims4[3];
    int k = input_shapes[0].dims4()[3];
    int lda = k;
    int ldb = n;
    int ldc = n;

    int batch_size = out_shape_dims4[0] * out_shape_dims4[1];

    auto memA = ark::to_gpu(inputs[0], input_shapes[0].size() * sizeof(T));
    auto memB = ark::to_gpu(inputs[1], input_shapes[1].size() * sizeof(T));
    auto memC = ark::to_gpu(outputs[0], output_shapes[0].size() * sizeof(T));

    T *devA = static_cast<T *>(memA.get());
    T *devB = static_cast<T *>(memB.get());
    T *devC = static_cast<T *>(memC.get());

    // matmul using cublas
    if constexpr (std::is_same_v<T, float>) {
        float alpha = 1;
        float beta = 0;
        cublas_matmul<CUBLAS_OP_N, CUBLAS_OP_N, CUDA_R_32F,
                      CUBLAS_COMPUTE_32F_FAST_16F>(m, n, k, &alpha, devA, lda,
                                                   devB, ldb, &beta, devC, ldc,
                                                   batch_size);
    } else if constexpr (std::is_same_v<T, half>) {
        half alpha = 1;
        half beta = 0;
        cublas_matmul<CUBLAS_OP_N, CUBLAS_OP_N, CUDA_R_16F, CUBLAS_COMPUTE_16F>(
            m, n, k, &alpha, devA, lda, devB, ldb, &beta, devC, ldc,
            batch_size);
    } else if constexpr (std::is_same_v<T, ark::bfloat16_t>) {
        float alpha = 1;
        float beta = 0;
        cublas_matmul<CUBLAS_OP_N, CUBLAS_OP_N, CUDA_R_16BF,
                      CUBLAS_COMPUTE_32F>(m, n, k, &alpha, devA, lda, devB, ldb,
                                          &beta, devC, ldc, batch_size);
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
                        const std::vector<ark::Dims> &input_shapes) {
    auto out_shape_dims4 = output_shapes[0].dims4();

    // baseline inputs & outputs have no padding
    int m = out_shape_dims4[2];
    int n = out_shape_dims4[3];
    int k = input_shapes[0].dims4()[3];
    int lda = k;
    int ldb = k;
    int ldc = n;

    int batch_size = out_shape_dims4[0] * out_shape_dims4[1];

    auto memA = ark::to_gpu(inputs[0], input_shapes[0].size() * sizeof(T));
    auto memB = ark::to_gpu(inputs[1], input_shapes[1].size() * sizeof(T));
    auto memC = ark::to_gpu(outputs[0], output_shapes[0].size() * sizeof(T));

    T *devA = static_cast<T *>(memA.get());
    T *devB = static_cast<T *>(memB.get());
    T *devC = static_cast<T *>(memC.get());

    // matmul using cublas
    if constexpr (std::is_same_v<T, float>) {
        float alpha = 1;
        float beta = 0;
        cublas_matmul<CUBLAS_OP_N, CUBLAS_OP_T, CUDA_R_32F,
                      CUBLAS_COMPUTE_32F_FAST_16F>(m, n, k, &alpha, devA, lda,
                                                   devB, ldb, &beta, devC, ldc,
                                                   batch_size);
    } else if constexpr (std::is_same_v<T, half>) {
        half alpha = 1;
        half beta = 0;
        cublas_matmul<CUBLAS_OP_N, CUBLAS_OP_T, CUDA_R_16F, CUBLAS_COMPUTE_16F>(
            m, n, k, &alpha, devA, lda, devB, ldb, &beta, devC, ldc,
            batch_size);
    } else if constexpr (std::is_same_v<T, ark::bfloat16_t>) {
        float alpha = 1;
        float beta = 0;
        cublas_matmul<CUBLAS_OP_N, CUBLAS_OP_T, CUDA_R_16BF,
                      CUBLAS_COMPUTE_32F>(m, n, k, &alpha, devA, lda, devB, ldb,
                                          &beta, devC, ldc, batch_size);
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
                        const std::vector<ark::Dims> &input_shapes) {
    auto out_shape_dims4 = output_shapes[0].dims4();

    // baseline inputs & outputs have no padding
    int m = out_shape_dims4[2];
    int n = out_shape_dims4[3];
    int k = input_shapes[0].dims4()[2];
    int lda = m;
    int ldb = n;
    int ldc = n;

    int batch_size = out_shape_dims4[0] * out_shape_dims4[1];

    auto memA = ark::to_gpu(inputs[0], input_shapes[0].size() * sizeof(T));
    auto memB = ark::to_gpu(inputs[1], input_shapes[1].size() * sizeof(T));
    auto memC = ark::to_gpu(outputs[0], output_shapes[0].size() * sizeof(T));

    T *devA = static_cast<T *>(memA.get());
    T *devB = static_cast<T *>(memB.get());
    T *devC = static_cast<T *>(memC.get());

    // matmul using cublas
    if constexpr (std::is_same_v<T, float>) {
        float alpha = 1;
        float beta = 0;
        cublas_matmul<CUBLAS_OP_T, CUBLAS_OP_N, CUDA_R_32F,
                      CUBLAS_COMPUTE_32F_FAST_16F>(m, n, k, &alpha, devA, lda,
                                                   devB, ldb, &beta, devC, ldc,
                                                   batch_size);
    } else if constexpr (std::is_same_v<T, half>) {
        half alpha = 1;
        half beta = 0;
        cublas_matmul<CUBLAS_OP_T, CUBLAS_OP_N, CUDA_R_16F, CUBLAS_COMPUTE_16F>(
            m, n, k, &alpha, devA, lda, devB, ldb, &beta, devC, ldc,
            batch_size);
    } else if constexpr (std::is_same_v<T, ark::bfloat16_t>) {
        float alpha = 1;
        float beta = 0;
        cublas_matmul<CUBLAS_OP_T, CUBLAS_OP_N, CUDA_R_16BF,
                      CUBLAS_COMPUTE_32F>(m, n, k, &alpha, devA, lda, devB, ldb,
                                          &beta, devC, ldc, batch_size);
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
                        const std::vector<ark::Dims> &input_shapes) {
    auto out_shape_dims4 = output_shapes[0].dims4();

    // baseline inputs & outputs have no padding
    int m = out_shape_dims4[2];
    int n = out_shape_dims4[3];
    int k = input_shapes[0].dims4()[2];
    int lda = m;
    int ldb = k;
    int ldc = n;

    int batch_size = out_shape_dims4[0] * out_shape_dims4[1];

    auto memA = ark::to_gpu(inputs[0], input_shapes[0].size() * sizeof(T));
    auto memB = ark::to_gpu(inputs[1], input_shapes[1].size() * sizeof(T));
    auto memC = ark::to_gpu(outputs[0], output_shapes[0].size() * sizeof(T));

    T *devA = static_cast<T *>(memA.get());
    T *devB = static_cast<T *>(memB.get());
    T *devC = static_cast<T *>(memC.get());

    // matmul using cublas
    if constexpr (std::is_same_v<T, float>) {
        float alpha = 1;
        float beta = 0;
        cublas_matmul<CUBLAS_OP_T, CUBLAS_OP_T, CUDA_R_32F,
                      CUBLAS_COMPUTE_32F_FAST_16F>(m, n, k, &alpha, devA, lda,
                                                   devB, ldb, &beta, devC, ldc,
                                                   batch_size);
    } else if constexpr (std::is_same_v<T, half>) {
        half alpha = 1;
        half beta = 0;
        cublas_matmul<CUBLAS_OP_T, CUBLAS_OP_T, CUDA_R_16F, CUBLAS_COMPUTE_16F>(
            m, n, k, &alpha, devA, lda, devB, ldb, &beta, devC, ldc,
            batch_size);
    } else if constexpr (std::is_same_v<T, ark::bfloat16_t>) {
        float alpha = 1;
        float beta = 0;
        cublas_matmul<CUBLAS_OP_T, CUBLAS_OP_T, CUDA_R_16BF,
                      CUBLAS_COMPUTE_32F>(m, n, k, &alpha, devA, lda, devB, ldb,
                                          &beta, devC, ldc, batch_size);
    } else {
        throw std::runtime_error("Unsupported data type");
    }
    ark::sync_gpu();

    // copy back to host
    ark::from_gpu(memC, outputs[0]);
}

ark::unittest::State test_matmul_fp16_gran0() {
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(128, 64), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(64, 256), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, false, "matmul", 0);

        auto result = ark::op_test("matmul_fp16_gran0", m, {a, b}, {c},
                                   baseline_matmul_nn<half>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(4096, 8192), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(8192, 16384), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, false, "matmul", 0);

        auto result = ark::op_test("matmul_fp16_gran0", m, {a, b}, {c},
                                   baseline_matmul_nn<half>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_fp16_gran1() {
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(128, 64), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(64, 256), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, false, "matmul", 1);

        auto result = ark::op_test("matmul_fp16_gran1", m, {a, b}, {c},
                                   baseline_matmul_nn<half>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(4096, 8192), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(8192, 16384), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, false, "matmul", 1);

        auto result = ark::op_test("matmul_fp16_gran1", m, {a, b}, {c},
                                   baseline_matmul_nn<half>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_fp16_gran2() {
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(128, 64), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(64, 256), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, false, "matmul", 2);

        auto result = ark::op_test("matmul_fp16_gran2", m, {a, b}, {c},
                                   baseline_matmul_nn<half>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(4096, 8192), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(8192, 16384), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, false, "matmul", 2);

        auto result = ark::op_test("matmul_fp16_gran2", m, {a, b}, {c},
                                   baseline_matmul_nn<half>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_split() {
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(4096, 8192), ark::FP32);
        ark::Tensor *b = m.tensor(ark::Dims(8192, 16384), ark::FP32);
        ark::Tensor *c = m.matmul(a, b, nullptr, 7, false, false, "matmul", 0);

        auto result = ark::op_test("matmul_split", m, {a, b}, {c},
                                   baseline_matmul_nn<float>);
        UNITTEST_LOG(result);
        UNITTEST_TRUE(result.max_diff[0] < 1e-4f);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_fp32() {
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(128, 64), ark::FP32);
        ark::Tensor *b = m.tensor(ark::Dims(64, 256), ark::FP32);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, false, "matmul", 0);

        auto result = ark::op_test("matmul_fp32", m, {a, b}, {c},
                                   baseline_matmul_nn<float>);
        UNITTEST_LOG(result);
        UNITTEST_TRUE(result.max_diff[0] < 1e-4f);
    }
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(4096, 8192), ark::FP32);
        ark::Tensor *b = m.tensor(ark::Dims(8192, 16384), ark::FP32);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, false, "matmul", 0);

        auto result = ark::op_test("matmul_fp32", m, {a, b}, {c},
                                   baseline_matmul_nn<float>);
        UNITTEST_LOG(result);
        UNITTEST_TRUE(result.max_diff[0] < 1e-4f);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_bf16() {
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(128, 64), ark::BF16);
        ark::Tensor *b = m.tensor(ark::Dims(64, 256), ark::BF16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, false, "matmul", 0);

        auto result = ark::op_test("matmul_bf16", m, {a, b}, {c},
                                   baseline_matmul_nn<ark::bfloat16_t>);
        UNITTEST_LOG(result);
        UNITTEST_TRUE(result.max_diff[0] < 1e-4f);
    }
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(4096, 8192), ark::BF16);
        ark::Tensor *b = m.tensor(ark::Dims(8192, 16384), ark::BF16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, false, "matmul", 0);

        auto result = ark::op_test("matmul_bf16", m, {a, b}, {c},
                                   baseline_matmul_nn<ark::bfloat16_t>);
        UNITTEST_LOG(result);
        UNITTEST_TRUE(result.max_diff[0] < 1e-4f);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_fp16_nt() {
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(128, 64), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(256, 64), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, true, "matmul", 0);

        auto result =
            ark::op_test("matmul_nt", m, {a, b}, {c}, baseline_matmul_nt<half>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(4096, 8192), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(16384, 8192), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, true, "matmul", 0);

        auto result =
            ark::op_test("matmul_nt", m, {a, b}, {c}, baseline_matmul_nt<half>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_fp16_tn() {
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(64, 128), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(64, 256), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, true, false, "matmul", 0);

        auto ones_a = ark::utils::ones<ark::half_t>(a->shape.size());
        auto ones_b = ark::utils::ones<ark::half_t>(b->shape.size());
        auto result =
            ark::op_test("matmul_tn", m, {a, b}, {c}, baseline_matmul_tn<half>,
                         {ones_a.get(), ones_b.get()}, true);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(8192, 4096), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(8192, 16384), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, true, false, "matmul", 0);

        auto result =
            ark::op_test("matmul_tn", m, {a, b}, {c}, baseline_matmul_tn<half>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_fp16_tt() {
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(64, 128), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(256, 64), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, true, true, "matmul", 0);

        auto result =
            ark::op_test("matmul_tt", m, {a, b}, {c}, baseline_matmul_tt<half>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(8192, 4096), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(16384, 8192), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, true, true, "matmul", 0);

        auto result =
            ark::op_test("matmul_tt", m, {a, b}, {c}, baseline_matmul_tt<half>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_batched() {
    ark::Model m;
    ark::Tensor *a = m.tensor(ark::Dims(3, 7, 64, 128), ark::FP16);
    ark::Tensor *b = m.tensor(ark::Dims(3, 7, 128, 256), ark::FP16);
    ark::Tensor *c = m.matmul(a, b);

    auto result = ark::op_test("matmul_batched", m, {a, b}, {c},
                               baseline_matmul_nn<half>);
    UNITTEST_LOG(result);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_batched_padded() {
    ark::Model m;
    ark::Tensor *a = m.tensor(ark::Dims(3, 7, 2, 9), ark::FP16);
    ark::Tensor *b = m.tensor(ark::Dims(3, 7, 9, 2), ark::FP16);
    ark::Tensor *c = m.matmul(a, b);

    auto result = ark::op_test("matmul_batched_padded", m, {a, b}, {c},
                               baseline_matmul_nn<half>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-3f);
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_matmul_fp16_gran0);
    UNITTEST(test_matmul_fp16_gran1);
    UNITTEST(test_matmul_fp16_gran2);
    UNITTEST(test_matmul_split);
    UNITTEST(test_matmul_fp32);
    UNITTEST(test_matmul_bf16);
    UNITTEST(test_matmul_fp16_nt);
    UNITTEST(test_matmul_fp16_tn);
    UNITTEST(test_matmul_fp16_tt);
    UNITTEST(test_matmul_batched);
    UNITTEST(test_matmul_batched_padded);

    cublasDestroy(get_cublas_handle());
    return ark::unittest::SUCCESS;
}
