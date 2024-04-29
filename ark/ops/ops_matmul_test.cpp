// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>

#include "ark/model.hpp"
#include "gpu/gpu.h"
#include "logging.h"
#include "model/model_node.hpp"
#include "model/model_op.hpp"
#include "model/model_tensor.hpp"
#include "ops_test_common.hpp"
#include "unittest/unittest_utils.h"

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
[[maybe_unused]] constexpr auto BLAS_COMPUTE_32F_FAST_TF32 =
    rocblas_datatype_f32_r;
[[maybe_unused]] constexpr auto BLAS_COMPUTE_16F = rocblas_datatype_f16_r;

inline auto blasGemmEx(blasHandle handle, blasOperation transA,
                       blasOperation transB, int m, int n, int k,
                       const void *alpha, const void *A, blasDataType Atype,
                       int lda, const void *B, blasDataType Btype, int ldb,
                       const void *beta, void *C, blasDataType Ctype, int ldc,
                       blasComputeType computeType) {
    return rocblas_gemm_ex(handle, transA, transB, m, n, k, alpha, A, Atype,
                           lda, B, Btype, ldb, beta, C, Ctype, ldc, C, Ctype,
                           ldc, computeType, rocblas_gemm_algo_standard, 0, 0);
}

inline auto blasGemmStridedBatchedEx(
    blasHandle handle, blasOperation transA, blasOperation transB, int m, int n,
    int k, const void *alpha, const void *A, blasDataType Atype, int lda,
    int strideA, const void *B, blasDataType Btype, int ldb, int strideB,
    const void *beta, void *C, blasDataType Ctype, int ldc, int strideC,
    int batchCount, blasComputeType computeType) {
    return rocblas_gemm_strided_batched_ex(
        handle, transA, transB, m, n, k, alpha, A, Atype, lda, strideA, B,
        Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, C, Ctype, ldc,
        strideC, batchCount, computeType, rocblas_gemm_algo_standard, 0, 0);
}

#endif

ARK_GPU_DEFINE_FUNC_ALIAS(blasCreate, cublasCreate, rocblas_create_handle);
ARK_GPU_DEFINE_FUNC_ALIAS(blasDestroy, cublasDestroy, rocblas_destroy_handle);

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

template <int BlasOpTypeA, int BlasOpTypeB, typename DataType>
void blas_matmul(int m, int n, int k, const DataType *a, int lda,
                 const DataType *b, int ldb, DataType *c, int ldc,
                 int batch_size = 1) {
    static_assert(std::is_same_v<DataType, float> ||
                      std::is_same_v<DataType, ark::half_t> ||
                      std::is_same_v<DataType, ark::bfloat16_t>,
                  "Unsupported data type");

    auto blasH = globalBlasHandle.get();
    blasStatus status;
    blasOperation optypeA = (blasOperation)BlasOpTypeA;
    blasOperation optypeB = (blasOperation)BlasOpTypeB;

#if defined(ARK_CUDA)
    using CompType =
        typename std::conditional_t<std::is_same_v<DataType, ark::half_t>,
                                    ark::half_t, float>;
    blasComputeType ctype =
        std::is_same_v<DataType, float>
            ? BLAS_COMPUTE_32F_FAST_TF32
            : (std::is_same_v<DataType, ark::half_t> ? BLAS_COMPUTE_16F
                                                     : BLAS_COMPUTE_32F);
#elif defined(ARK_ROCM)
    // CK uses only fp32 compute type for fp16/bf16
    using CompType = float;
    blasComputeType ctype = BLAS_COMPUTE_32F;
#endif
    CompType alpha = 1;
    CompType beta = 0;

    blasDataType dtype =
        std::is_same_v<DataType, float>
            ? BLAS_R_32F
            : (std::is_same_v<DataType, ark::half_t> ? BLAS_R_16F
                                                     : BLAS_R_16BF);
    if (batch_size == 1) {
        status = blasGemmEx(blasH, optypeB, optypeA, n, m, k, &alpha, b, dtype,
                            ldb, a, dtype, lda, &beta, c, dtype, ldc, ctype);
        if (status != blasSuccess) {
            throw std::runtime_error("Failed to call blasGemmEx");
        }
    } else {
        status = blasGemmStridedBatchedEx(
            blasH, optypeB, optypeA, n, m, k, &alpha, b, dtype, ldb, n * k, a,
            dtype, lda, k * m, &beta, c, dtype, ldc, n * m, batch_size, ctype);
        if (status != blasSuccess) {
            throw std::runtime_error("Failed to call blasGemmStridedBatchedEx");
        }
    }
}

template <typename T>
void baseline_matmul_nn(std::vector<void *> &outputs,
                        const std::vector<ark::Dims> &output_shapes,
                        const std::vector<void *> &inputs,
                        const std::vector<ark::Dims> &input_shapes, int) {
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

    blas_matmul<BLAS_OP_N, BLAS_OP_N>(m, n, k, devA, lda, devB, ldb, devC, ldc,
                                      batch_size);
    ark::sync_gpu();

    // copy back to host
    ark::from_gpu(memC, outputs[0]);
}

template <typename T>
void baseline_matmul_nt(std::vector<void *> &outputs,
                        const std::vector<ark::Dims> &output_shapes,
                        const std::vector<void *> &inputs,
                        const std::vector<ark::Dims> &input_shapes, int) {
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

    blas_matmul<BLAS_OP_N, BLAS_OP_T>(m, n, k, devA, lda, devB, ldb, devC, ldc,
                                      batch_size);
    ark::sync_gpu();

    // copy back to host
    ark::from_gpu(memC, outputs[0]);
}

template <typename T>
void baseline_matmul_tn(std::vector<void *> &outputs,
                        const std::vector<ark::Dims> &output_shapes,
                        const std::vector<void *> &inputs,
                        const std::vector<ark::Dims> &input_shapes, int) {
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

    blas_matmul<BLAS_OP_T, BLAS_OP_N>(m, n, k, devA, lda, devB, ldb, devC, ldc,
                                      batch_size);
    ark::sync_gpu();

    // copy back to host
    ark::from_gpu(memC, outputs[0]);
}

template <typename T>
void baseline_matmul_tt(std::vector<void *> &outputs,
                        const std::vector<ark::Dims> &output_shapes,
                        const std::vector<void *> &inputs,
                        const std::vector<ark::Dims> &input_shapes, int) {
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

    blas_matmul<BLAS_OP_T, BLAS_OP_T>(m, n, k, devA, lda, devB, ldb, devC, ldc,
                                      batch_size);
    ark::sync_gpu();

    // copy back to host
    ark::from_gpu(memC, outputs[0]);
}

template <typename T>
float max_diff(float max_abs, int reduction_length) {
    constexpr int NumFracBits =
        (std::is_same_v<T, float> ? 23
                                  : (std::is_same_v<T, ark::half_t> ? 10 : 7));
    // If the reduction length is too large, the error will be dominated by
    // the rounding error of the reduction itself.
    assert(reduction_length <= (1 << (NumFracBits + 1)));
    float max_diff =
        reduction_length * 2 * max_abs * 1.0f / (1 << (NumFracBits + 1));
    // *2 because the baseline is also a computed value.
    return max_diff * 2;
}

ark::unittest::State test_matmul_model() {
    // Hidden dimension of the dense layer.
    unsigned int units = 1024;
    // Input dimension of the dense layer.
    unsigned int in_dim = 1024;
    // Extra dimension of the input. CHANNEL=1 for 2D inputs.
    unsigned int channel = 128;
    // Batch size of the input.
    unsigned int batch_size = 1;

    ark::Model m;
    ark::ModelTensorRef input =
        m.tensor({batch_size, channel, in_dim}, ark::FP16);
    ark::ModelTensorRef weight = m.tensor({in_dim, units}, ark::FP16);
    m.matmul(input, weight);

    UNITTEST_TRUE(m.verify());
    auto compressed = m.compress();
    UNITTEST_TRUE(compressed.verify());

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_fp16() {
    {
        ark::Model m;
        ark::ModelTensorRef a = m.tensor(ark::Dims(128, 64), ark::FP16);
        ark::ModelTensorRef b = m.tensor(ark::Dims(64, 256), ark::FP16);
        ark::ModelTensorRef c = m.matmul(a, b);

        auto result = ark::op_test("matmul_fp16", m, {a, b}, {c},
                                   baseline_matmul_nn<ark::half_t>);
        UNITTEST_LOG(result);
        UNITTEST_TRUE(result.max_diff[0] < max_diff<ark::half_t>(0.1f, 64));
    }
    {
        ark::Model m;
        ark::ModelTensorRef a = m.tensor(ark::Dims(4096, 2048), ark::FP16);
        ark::ModelTensorRef b = m.tensor(ark::Dims(2048, 16384), ark::FP16);
        ark::ModelTensorRef c = m.matmul(a, b);

        std::vector<ark::half_t> p_ones_a(a->shape().size(), ark::half_t(0.1f));
        std::vector<ark::half_t> p_ones_b(b->shape().size(), ark::half_t(0.1f));

        auto result = ark::op_test("matmul_fp16", m, {a, b}, {c},
                                   baseline_matmul_nn<ark::half_t>,
                                   {p_ones_a.data(), p_ones_b.data()});
        UNITTEST_LOG(result);
        UNITTEST_TRUE(result.max_diff[0] < max_diff<ark::half_t>(0.1f, 2048));
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_fp32() {
    {
        ark::Model m;
        ark::ModelTensorRef a = m.tensor(ark::Dims(128, 64), ark::FP32);
        ark::ModelTensorRef b = m.tensor(ark::Dims(64, 256), ark::FP32);
        ark::ModelTensorRef c = m.matmul(a, b);

        auto result = ark::op_test("matmul_fp32", m, {a, b}, {c},
                                   baseline_matmul_nn<float>);
        UNITTEST_LOG(result);
        UNITTEST_TRUE(result.max_diff[0] < max_diff<float>(0.1f, 64));
    }
    {
        ark::Model m;
        ark::ModelTensorRef a = m.tensor(ark::Dims(4096, 8192), ark::FP32);
        ark::ModelTensorRef b = m.tensor(ark::Dims(8192, 16384), ark::FP32);
        ark::ModelTensorRef c = m.matmul(a, b);

        std::vector<float> p_ones_a(a->shape().size(), float(0.1f));
        std::vector<float> p_ones_b(b->shape().size(), float(0.1f));

        auto result = ark::op_test("matmul_fp32", m, {a, b}, {c},
                                   baseline_matmul_nn<float>,
                                   {p_ones_a.data(), p_ones_b.data()});
        UNITTEST_LOG(result);
        // TODO: #199
#if defined(ARK_CUDA)
        UNITTEST_TRUE(result.max_diff[0] < max_diff<float>(0.1f, 8192));
#endif  // defined(ARK_CUDA)
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_bf16() {
    {
        ark::Model m;
        ark::ModelTensorRef a = m.tensor(ark::Dims(128, 64), ark::BF16);
        ark::ModelTensorRef b = m.tensor(ark::Dims(64, 256), ark::BF16);
        ark::ModelTensorRef c = m.matmul(a, b);

        auto result = ark::op_test("matmul_bf16", m, {a, b}, {c},
                                   baseline_matmul_nn<ark::bfloat16_t>);
        UNITTEST_LOG(result);
        UNITTEST_TRUE(result.max_diff[0] < max_diff<ark::bfloat16_t>(0.1f, 64));
    }
    {
        ark::Model m;
        ark::ModelTensorRef a = m.tensor(ark::Dims(4096, 256), ark::BF16);
        ark::ModelTensorRef b = m.tensor(ark::Dims(256, 16384), ark::BF16);
        ark::ModelTensorRef c = m.matmul(a, b);

        auto result = ark::op_test("matmul_bf16", m, {a, b}, {c},
                                   baseline_matmul_nn<ark::bfloat16_t>);
        UNITTEST_LOG(result);
        UNITTEST_TRUE(result.max_diff[0] <
                      max_diff<ark::bfloat16_t>(0.1f, 256));
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_fp16_nt() {
    {
        ark::Model m;
        ark::ModelTensorRef a = m.tensor(ark::Dims(128, 64), ark::FP16);
        ark::ModelTensorRef b = m.tensor(ark::Dims(256, 64), ark::FP16);
        ark::ModelTensorRef c = m.matmul(a, b, nullptr, false, true);

        auto result = ark::op_test("matmul_fp16_nt", m, {a, b}, {c},
                                   baseline_matmul_nt<ark::half_t>);
        UNITTEST_LOG(result);
        UNITTEST_TRUE(result.max_diff[0] < max_diff<ark::half_t>(0.1f, 64));
    }
    {
        ark::Model m;
        ark::ModelTensorRef a = m.tensor(ark::Dims(4096, 2048), ark::FP16);
        ark::ModelTensorRef b = m.tensor(ark::Dims(16384, 2048), ark::FP16);
        ark::ModelTensorRef c = m.matmul(a, b, nullptr, false, true);

        auto result = ark::op_test("matmul_fp16_nt", m, {a, b}, {c},
                                   baseline_matmul_nt<ark::half_t>);
        UNITTEST_LOG(result);
        UNITTEST_TRUE(result.max_diff[0] < max_diff<ark::half_t>(0.1f, 2048));
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_fp16_tn() {
    {
        ark::Model m;
        ark::ModelTensorRef a = m.tensor(ark::Dims(64, 128), ark::FP16);
        ark::ModelTensorRef b = m.tensor(ark::Dims(64, 256), ark::FP16);
        ark::ModelTensorRef c = m.matmul(a, b, nullptr, true, false);

        auto result = ark::op_test("matmul_fp16_tn", m, {a, b}, {c},
                                   baseline_matmul_tn<ark::half_t>);
        UNITTEST_LOG(result);
        UNITTEST_TRUE(result.max_diff[0] < max_diff<ark::half_t>(0.1f, 64));
    }
    {
        ark::Model m;
        ark::ModelTensorRef a = m.tensor(ark::Dims(2048, 4096), ark::FP16);
        ark::ModelTensorRef b = m.tensor(ark::Dims(2048, 16384), ark::FP16);
        ark::ModelTensorRef c = m.matmul(a, b, nullptr, true, false);

        auto result = ark::op_test("matmul_fp16_tn", m, {a, b}, {c},
                                   baseline_matmul_tn<ark::half_t>);
        UNITTEST_LOG(result);
        UNITTEST_TRUE(result.max_diff[0] < max_diff<ark::half_t>(0.1f, 2048));
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_fp16_tt() {
    {
        ark::Model m;
        ark::ModelTensorRef a = m.tensor(ark::Dims(64, 128), ark::FP16);
        ark::ModelTensorRef b = m.tensor(ark::Dims(256, 64), ark::FP16);
        ark::ModelTensorRef c = m.matmul(a, b, nullptr, true, true);

        auto result = ark::op_test("matmul_fp16_tt", m, {a, b}, {c},
                                   baseline_matmul_tt<ark::half_t>);
        UNITTEST_LOG(result);
        UNITTEST_TRUE(result.max_diff[0] < max_diff<ark::half_t>(0.1f, 64));
    }
    {
        ark::Model m;
        ark::ModelTensorRef a = m.tensor(ark::Dims(2048, 4096), ark::FP16);
        ark::ModelTensorRef b = m.tensor(ark::Dims(16384, 2048), ark::FP16);
        ark::ModelTensorRef c = m.matmul(a, b, nullptr, true, true);

        auto result = ark::op_test("matmul_fp16_tt", m, {a, b}, {c},
                                   baseline_matmul_tt<ark::half_t>);
        UNITTEST_LOG(result);
        UNITTEST_TRUE(result.max_diff[0] < max_diff<ark::half_t>(0.1f, 2048));
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_fp16_batched() {
    ark::Model m;
    ark::ModelTensorRef a = m.tensor(ark::Dims(3, 7, 64, 128), ark::FP16);
    ark::ModelTensorRef b = m.tensor(ark::Dims(3, 7, 128, 256), ark::FP16);
    ark::ModelTensorRef c = m.matmul(a, b);

    auto result = ark::op_test("matmul_fp16_batched", m, {a, b}, {c},
                               baseline_matmul_nn<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < max_diff<ark::half_t>(0.1f, 128));
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_fp16_batched_padded() {
    ark::Model m;
    ark::ModelTensorRef a = m.tensor({3, 7, 2, 9}, ark::FP16, {3, 7, 64, 64});
    ark::ModelTensorRef b = m.tensor({3, 7, 9, 2}, ark::FP16, {3, 7, 64, 64});
    ark::ModelTensorRef c = m.matmul(a, b);

    auto result = ark::op_test("matmul_fp16_batched_padded", m, {a, b}, {c},
                               baseline_matmul_nn<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < max_diff<ark::half_t>(0.1f, 9));
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_fp16_offset() {
    ark::Model m;
    ark::ModelTensorRef a =
        m.tensor({1, 128, 64}, ark::FP16, {1, 128, 256}, {0, 0, 64});
    ark::ModelTensorRef b =
        m.tensor({1, 64, 128}, ark::FP16, {1, 128, 256}, {0, 64, 0});
    ark::ModelTensorRef c =
        m.tensor({1, 128, 128}, ark::FP16, {2, 256, 256}, {1, 64, 128});
    m.matmul(a, b, c);

    auto result = ark::op_test("matmul_fp16_offset", m, {a, b}, {c},
                               baseline_matmul_nn<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < max_diff<ark::half_t>(0.1f, 64));
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_fp16_perf() {
    {
        ark::Model m;
        ark::ModelTensorRef a = m.tensor(ark::Dims(256, 8192), ark::FP16);
        ark::ModelTensorRef b = m.tensor(ark::Dims(8192, 128), ark::FP16);
        ark::ModelTensorRef c = m.matmul(a, b);

        auto result = ark::op_test("matmul_fp16", m, {a, b}, {c},
                                   baseline_matmul_nn<ark::half_t>);
        UNITTEST_LOG(result);
    }
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_matmul_model);
    UNITTEST(test_matmul_fp16);
    UNITTEST(test_matmul_fp32);
    UNITTEST(test_matmul_bf16);
    // UNITTEST(test_matmul_fp16_nt);
    // UNITTEST(test_matmul_fp16_tn);
    // UNITTEST(test_matmul_fp16_tt);
    UNITTEST(test_matmul_fp16_batched);
    UNITTEST(test_matmul_fp16_batched_padded);
    UNITTEST(test_matmul_fp16_offset);
    UNITTEST(test_matmul_fp16_perf);
    return ark::unittest::SUCCESS;
}
