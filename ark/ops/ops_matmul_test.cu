// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>
#include <type_traits>

#include "ops_test_common.h"

#define ARK_GPU_DEFINE_FUNC_ALIAS(alias, func)    \
    template <typename... Args>                   \
    inline auto alias(Args &&... args) {          \
        return func(std::forward<Args>(args)...); \
    }

#if defined(ARK_CUDA)

#include <cublas_v2.h>

typedef cublasHandle_t blasHandle;
typedef cublasStatus_t blasStatus;
typedef cublasOperation_t blasOperation;
typedef cudaDataType blasDataType;
typedef cublasComputeType_t blasComputeType;
constexpr auto blasSuccess = CUBLAS_STATUS_SUCCESS;
constexpr auto BLAS_OP_N = CUBLAS_OP_N;
constexpr auto BLAS_OP_T = CUBLAS_OP_T;
constexpr auto BLAS_R_32F = CUDA_R_32F;
constexpr auto BLAS_R_16F = CUDA_R_16F;
constexpr auto BLAS_R_16BF = CUDA_R_16BF;
constexpr auto BLAS_COMPUTE_32F = CUBLAS_COMPUTE_32F;
constexpr auto BLAS_COMPUTE_32F_FAST_TF32 = CUBLAS_COMPUTE_32F_FAST_TF32;
constexpr auto BLAS_COMPUTE_16F = CUBLAS_COMPUTE_16F;

ARK_GPU_DEFINE_FUNC_ALIAS(blasCreate, cublasCreate);
ARK_GPU_DEFINE_FUNC_ALIAS(blasDestroy, cublasDestroy);

inline auto blasGemmEx(blasHandle handle, blasOperation transA,
                       blasOperation transB, int m, int n, int k,
                       const void *alpha, const void *A, blasDataType Atype,
                       int lda, const void *B, blasDataType Btype, int ldb,
                       const void *beta, void *C, blasDataType Ctype, int ldc,
                       blasComputeType computeType) {
    return cublasGemmEx(handle, transA, transB, m, n, k, alpha, A, Atype, lda,
                        B, Btype, ldb, beta, C, Ctype, ldc, computeType,
                        CUBLAS_GEMM_DEFAULT);
}

inline auto blasGemmStridedBatchedEx(
    blasHandle handle, blasOperation transA, blasOperation transB, int m, int n,
    int k, const void *alpha, const void *A, blasDataType Atype, int lda,
    int strideA, const void *B, blasDataType Btype, int ldb, int strideB,
    const void *beta, void *C, blasDataType Ctype, int ldc, int strideC,
    int batchCount, blasComputeType computeType) {
    return cublasGemmStridedBatchedEx(
        handle, transA, transB, m, n, k, alpha, A, Atype, lda, strideA, B,
        Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount,
        computeType, CUBLAS_GEMM_DEFAULT);
}

#elif defined(ARK_ROCM)

#include <rocblas/rocblas.h>

typedef rocblas_handle blasHandle;
typedef rocblas_status blasStatus;
typedef rocblas_operation blasOperation;
typedef rocblas_datatype blasDataType;
typedef rocblas_datatype blasComputeType;
constexpr auto blasSuccess = rocblas_status_success;
constexpr auto BLAS_OP_N = rocblas_operation_none;
constexpr auto BLAS_OP_T = rocblas_operation_transpose;
constexpr auto BLAS_R_32F = rocblas_datatype_f32_r;
constexpr auto BLAS_R_16F = rocblas_datatype_f16_r;
constexpr auto BLAS_R_16BF = rocblas_datatype_bf16_r;
constexpr auto BLAS_COMPUTE_32F = rocblas_datatype_f32_r;
constexpr auto BLAS_COMPUTE_32F_FAST_TF32 = rocblas_datatype_f32_r;
constexpr auto BLAS_COMPUTE_16F = rocblas_datatype_f16_r;

ARK_GPU_DEFINE_FUNC_ALIAS(blasCreate, rocblas_create_handle);
ARK_GPU_DEFINE_FUNC_ALIAS(blasDestroy, rocblas_destroy_handle);

inline auto blasGemmEx(blasHandle handle, blasOperation transA,
                       blasOperation transB, int m, int n, int k,
                       const void *alpha, const void *A, blasDataType Atype,
                       int lda, const void *B, blasDataType Btype, int ldb,
                       const void *beta, void *C, blasDataType Ctype, int ldc,
                       blasComputeType computeType) {
    return rocblas_gemm_ex(handle, transA, transB, m, n, k, alpha, A, Atype,
                           lda, B, Btype, ldb, beta, C, Ctype, ldc, nullptr,
                           Ctype, ldc, computeType, rocblas_gemm_algo_standard,
                           0, 0);
}

inline auto blasGemmStridedBatchedEx(
    blasHandle handle, blasOperation transA, blasOperation transB, int m, int n,
    int k, const void *alpha, const void *A, blasDataType Atype, int lda,
    int strideA, const void *B, blasDataType Btype, int ldb, int strideB,
    const void *beta, void *C, blasDataType Ctype, int ldc, int strideC,
    int batchCount, blasComputeType computeType) {
    return rocblas_gemm_strided_batched_ex(
        handle, transA, transB, m, n, k, alpha, A, Atype, lda, strideA, B,
        Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, nullptr, Ctype, ldc,
        strideC, batchCount, computeType, rocblas_gemm_algo_standard, 0, 0);
}

#endif

class BlasHandle {
   public:
    BlasHandle() {
        if (blasCreate(&handle_) != blasSuccess) {
            throw std::runtime_error("Failed to create blas handle");
        }
    }

    ~BlasHandle() {
        if (blasDestroy(handle_) != blasSuccess) {
            // do nothing.
        }
    }

    blasHandle get() const { return handle_; }

   private:
    blasHandle handle_;
};

static BlasHandle globalBlasHandle;

template <int BlasOpTypeA, int BlasOpTypeB, int BlasDataType,
          int BlasComputeType>
void blas_matmul(int m, int n, int k, void *alpha, const void *a, int lda,
                 const void *b, int ldb, void *beta, void *c, int ldc,
                 int batch_size = 1) {
    auto blasH = globalBlasHandle.get();
    blasStatus status;
    blasOperation optypeA = (blasOperation)BlasOpTypeA;
    blasOperation optypeB = (blasOperation)BlasOpTypeB;
    blasDataType dtype = (blasDataType)BlasDataType;
    blasComputeType ctype = (blasComputeType)BlasComputeType;
    if (batch_size == 1) {
        status = blasGemmEx(blasH, optypeB, optypeA, n, m, k, alpha, b, dtype,
                            ldb, a, dtype, lda, beta, c, dtype, ldc, ctype);
        if (status != blasSuccess) {
            throw std::runtime_error("Failed to call blasGemmEx");
        }
    } else {
        status = blasGemmStridedBatchedEx(
            blasH, optypeB, optypeA, n, m, k, alpha, b, dtype, ldb, n * k, a,
            dtype, lda, k * m, beta, c, dtype, ldc, n * m, batch_size, ctype);
        if (status != blasSuccess) {
            throw std::runtime_error("Failed to call blasGemmStridedBatchedEx");
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
        blas_matmul<BLAS_OP_N, BLAS_OP_N, BLAS_R_32F,
                    BLAS_COMPUTE_32F_FAST_TF32>(m, n, k, &alpha, devA, lda,
                                                devB, ldb, &beta, devC, ldc,
                                                batch_size);
    } else if constexpr (std::is_same_v<T, ark::half_t>) {
        ark::half_t alpha = 1;
        ark::half_t beta = 0;
        blas_matmul<BLAS_OP_N, BLAS_OP_N, BLAS_R_16F, BLAS_COMPUTE_16F>(
            m, n, k, &alpha, devA, lda, devB, ldb, &beta, devC, ldc,
            batch_size);
    } else if constexpr (std::is_same_v<T, ark::bfloat16_t>) {
        float alpha = 1;
        float beta = 0;
        blas_matmul<BLAS_OP_N, BLAS_OP_N, BLAS_R_16BF, BLAS_COMPUTE_32F>(
            m, n, k, &alpha, devA, lda, devB, ldb, &beta, devC, ldc,
            batch_size);
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
        blas_matmul<BLAS_OP_N, BLAS_OP_T, BLAS_R_32F,
                    BLAS_COMPUTE_32F_FAST_TF32>(m, n, k, &alpha, devA, lda,
                                                devB, ldb, &beta, devC, ldc,
                                                batch_size);
    } else if constexpr (std::is_same_v<T, ark::half_t>) {
        ark::half_t alpha = 1;
        ark::half_t beta = 0;
        blas_matmul<BLAS_OP_N, BLAS_OP_T, BLAS_R_16F, BLAS_COMPUTE_16F>(
            m, n, k, &alpha, devA, lda, devB, ldb, &beta, devC, ldc,
            batch_size);
    } else if constexpr (std::is_same_v<T, ark::bfloat16_t>) {
        float alpha = 1;
        float beta = 0;
        blas_matmul<BLAS_OP_N, BLAS_OP_T, BLAS_R_16BF, BLAS_COMPUTE_32F>(
            m, n, k, &alpha, devA, lda, devB, ldb, &beta, devC, ldc,
            batch_size);
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
        blas_matmul<BLAS_OP_T, BLAS_OP_N, BLAS_R_32F,
                    BLAS_COMPUTE_32F_FAST_TF32>(m, n, k, &alpha, devA, lda,
                                                devB, ldb, &beta, devC, ldc,
                                                batch_size);
    } else if constexpr (std::is_same_v<T, ark::half_t>) {
        ark::half_t alpha = 1;
        ark::half_t beta = 0;
        blas_matmul<BLAS_OP_T, BLAS_OP_N, BLAS_R_16F, BLAS_COMPUTE_16F>(
            m, n, k, &alpha, devA, lda, devB, ldb, &beta, devC, ldc,
            batch_size);
    } else if constexpr (std::is_same_v<T, ark::bfloat16_t>) {
        float alpha = 1;
        float beta = 0;
        blas_matmul<BLAS_OP_T, BLAS_OP_N, BLAS_R_16BF, BLAS_COMPUTE_32F>(
            m, n, k, &alpha, devA, lda, devB, ldb, &beta, devC, ldc,
            batch_size);
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
        blas_matmul<BLAS_OP_T, BLAS_OP_T, BLAS_R_32F,
                    BLAS_COMPUTE_32F_FAST_TF32>(m, n, k, &alpha, devA, lda,
                                                devB, ldb, &beta, devC, ldc,
                                                batch_size);
    } else if constexpr (std::is_same_v<T, ark::half_t>) {
        ark::half_t alpha = 1;
        ark::half_t beta = 0;
        blas_matmul<BLAS_OP_T, BLAS_OP_T, BLAS_R_16F, BLAS_COMPUTE_16F>(
            m, n, k, &alpha, devA, lda, devB, ldb, &beta, devC, ldc,
            batch_size);
    } else if constexpr (std::is_same_v<T, ark::bfloat16_t>) {
        float alpha = 1;
        float beta = 0;
        blas_matmul<BLAS_OP_T, BLAS_OP_T, BLAS_R_16BF, BLAS_COMPUTE_32F>(
            m, n, k, &alpha, devA, lda, devB, ldb, &beta, devC, ldc,
            batch_size);
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
                                   baseline_matmul_nn<ark::half_t>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(4096, 8192), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(8192, 16384), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, false, "matmul", 0);

        auto result = ark::op_test("matmul_fp16_gran0", m, {a, b}, {c},
                                   baseline_matmul_nn<ark::half_t>);
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
                                   baseline_matmul_nn<ark::half_t>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(4096, 8192), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(8192, 16384), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, false, "matmul", 1);

        auto result = ark::op_test("matmul_fp16_gran1", m, {a, b}, {c},
                                   baseline_matmul_nn<ark::half_t>);
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
                                   baseline_matmul_nn<ark::half_t>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(4096, 8192), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(8192, 16384), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, false, "matmul", 2);

        auto result = ark::op_test("matmul_fp16_gran2", m, {a, b}, {c},
                                   baseline_matmul_nn<ark::half_t>);
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
        UNITTEST_TRUE(result.max_err_rate[0] < 1e-4f);
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
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(4096, 8192), ark::FP32);
        ark::Tensor *b = m.tensor(ark::Dims(8192, 16384), ark::FP32);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, false, "matmul", 0);

        auto result = ark::op_test("matmul_fp32", m, {a, b}, {c},
                                   baseline_matmul_nn<float>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
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
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(4096, 8192), ark::BF16);
        ark::Tensor *b = m.tensor(ark::Dims(8192, 16384), ark::BF16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, false, "matmul", 0);

        auto result = ark::op_test("matmul_bf16", m, {a, b}, {c},
                                   baseline_matmul_nn<ark::bfloat16_t>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_fp16_nt() {
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(128, 64), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(256, 64), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, true, "matmul", 0);

        auto result = ark::op_test("matmul_nt", m, {a, b}, {c},
                                   baseline_matmul_nt<ark::half_t>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(4096, 8192), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(16384, 8192), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, false, true, "matmul", 0);

        auto result = ark::op_test("matmul_nt", m, {a, b}, {c},
                                   baseline_matmul_nt<ark::half_t>);
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

        std::vector<ark::half_t> ones_a(a->shape.size(), ark::half_t(1.0f));
        std::vector<ark::half_t> ones_b(b->shape.size(), ark::half_t(1.0f));
        auto result = ark::op_test("matmul_tn", m, {a, b}, {c},
                                   baseline_matmul_tn<ark::half_t>,
                                   {ones_a.data(), ones_b.data()}, true);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(8192, 4096), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(8192, 16384), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, true, false, "matmul", 0);

        auto result = ark::op_test("matmul_tn", m, {a, b}, {c},
                                   baseline_matmul_tn<ark::half_t>);
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

        auto result = ark::op_test("matmul_tt", m, {a, b}, {c},
                                   baseline_matmul_tt<ark::half_t>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    {
        ark::Model m;
        ark::Tensor *a = m.tensor(ark::Dims(8192, 4096), ark::FP16);
        ark::Tensor *b = m.tensor(ark::Dims(16384, 8192), ark::FP16);
        ark::Tensor *c = m.matmul(a, b, nullptr, 1, true, true, "matmul", 0);

        auto result = ark::op_test("matmul_tt", m, {a, b}, {c},
                                   baseline_matmul_tt<ark::half_t>);
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
                               baseline_matmul_nn<ark::half_t>);
    UNITTEST_LOG(result);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_batched_padded() {
    ark::Model m;
    ark::Tensor *a = m.tensor(ark::Dims(3, 7, 2, 9), ark::FP16);
    ark::Tensor *b = m.tensor(ark::Dims(3, 7, 9, 2), ark::FP16);
    ark::Tensor *c = m.matmul(a, b);

    auto result = ark::op_test("matmul_batched_padded", m, {a, b}, {c},
                               baseline_matmul_nn<ark::half_t>);
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

    return ark::unittest::SUCCESS;
}
