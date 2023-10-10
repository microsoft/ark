// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_test_common.h"

#include <cuda_runtime.h>

#include <cstring>

#include "env.h"
#include "gpu/gpu_kernel.h"
#include "logging.h"
#include "random.h"
#include "env.h"
#include "unittest/unittest_utils.h"

namespace ark {

std::ostream &operator<<(std::ostream &os, const OpsTestResult &result) {
    os << "op test: " << result.test_name << " #warp/sm "
       << result.num_warps_per_sm << ", msec/iter " << result.msec_per_iter;
    os << std::setprecision(4);
    for (size_t i = 0; i < result.mse.size(); i++) {
        float err_pcnt = result.max_err_rate[i] * 100;
        os << ", mse " << result.mse[i] << ", max_diff " << result.max_diff[i]
           << ", max_err_rate " << err_pcnt << "%";
    }
    return os;
}

/// Calculate the error rate between two values.
/// @tparam T Type of the values
/// @param a First value
/// @param b Second value
/// @return The error rate
template <typename T>
float error_rate(T a, T b) {
    T diff = abs(a - b);
    T max = std::max(abs(a), abs(b));
    if (max == 0) {
        return 0;
    }
    return (float)diff / (float)max;
}

/// Calculate the error rate between two @ref half_t values.
/// @param a First value
/// @param b Second value
/// @return The error rate
float error_rate(half_t a, half_t b) { return error_rate<half_t>(a, b); }

/// Calculate the error rate between two floats.
/// @param a First value
/// @param b Second value
/// @return The error rate
float error_rate(float a, float b) { return error_rate<float>(a, b); }

/// Return mean squared error and max error rate between two tensors.
/// @tparam T data type of the tensors.
/// @param ground_truth ground truth data array.
/// @param res input data array to compare with the ground truth.
/// @param shape shape of the tensor.
/// @param print whether to print wrong values.
/// @return a pair of mean squared error and max error rate.
template <typename T>
TensorCompareResult tensor_compare(T *ground_truth, T *res, Dims shape,
                                   bool print = false) {
    DimType nelem = shape.size();
    int ndims = shape.ndims();
    float l2_loss = 0;
    float max_err = 0;
    float max_diff = 0;
    for (DimType i = 0; i < nelem; ++i) {
        float diff = (float)(ground_truth[i] - res[i]);
        if (std::abs(diff) > max_diff) {
            max_diff = std::abs(diff);
        }
        l2_loss += diff * diff;

        float err = error_rate(ground_truth[i], res[i]);
        if (err > 0.) {
            if (print) {
                Dims idx(std::vector<DimType>(ndims, 0));
                for (int j = 0; j < ndims; ++j) {
                    DimType vol = 1;
                    for (int k = j + 1; k < ndims; ++k) {
                        vol *= shape[k];
                    }
                    idx[j] = (i / vol) % shape[j];
                }
                std::cout << idx << " expected " << ground_truth[i]
                          << ", actually " << res[i] << " (err: " << err << ")"
                          << std::endl;
            }
            if (err > max_err) {
                max_err = err;
            }
        }
    }
    TensorCompareResult result;
    result.mse = l2_loss / nelem;
    result.max_diff = max_diff;
    result.max_error_rate = max_err;
    return result;
}

/// Return mean squared error and max error rate between two @ref half_t
/// tensors.
/// @param ground_truth ground truth data array.
/// @param res input data array to compare with the ground truth.
/// @param shape shape of the tensor.
/// @param print whether to print wrong values.
/// @return a pair of mean squared error and max error rate.
TensorCompareResult tensor_compare(half_t *ground_truth, half_t *res,
                                   Dims shape, bool print) {
    return tensor_compare<half_t>(ground_truth, res, shape, print);
}

/// Return mean squared error and max error rate between two float tensors.
/// @param ground_truth ground truth data array.
/// @param res input data array to compare with the ground truth.
/// @param shape shape of the tensor.
/// @param print whether to print wrong values.
/// @return a pair of mean squared error and max error rate.
TensorCompareResult tensor_compare(float *ground_truth, float *res, Dims shape,
                                   bool print) {
    return tensor_compare<float>(ground_truth, res, shape, print);
}

/// Return mean squared error and max error rate between two int tensors.
/// @param ground_truth ground truth data array.
/// @param res input data array to compare with the ground truth.
/// @param shape shape of the tensor.
/// @param print whether to print wrong values.
/// @return a pair of mean squared error and max error rate.
TensorCompareResult tensor_compare(int *ground_truth, int *res, Dims shape,
                                   bool print) {
    return tensor_compare<int>(ground_truth, res, shape, print);
}

/// Return mean squared error and max error rate between two byte tensors.
/// @param ground_truth ground truth data array.
/// @param res input data array to compare with the ground truth.
/// @param shape shape of the tensor.
/// @param print whether to print wrong values.
/// @return a pair of mean squared error and max error rate.
TensorCompareResult tensor_compare(uint8_t *ground_truth, uint8_t *res,
                                   Dims shape, bool print) {
    return tensor_compare<uint8_t>(ground_truth, res, shape, print);
}

#define CUDA_CHECK(status)                                              \
    do {                                                                \
        cudaError_t error = status;                                     \
        if (error != cudaSuccess) {                                     \
            std::ostringstream oss;                                     \
            oss << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__;                            \
            throw std::runtime_error(oss.str());                        \
        }                                                               \
    } while (0);

OpsTestResult op_test(const std::string &test_name_prefix, Model &model,
                      const std::vector<Tensor *> &inputs,
                      const std::vector<Tensor *> &outputs,
                      OpsTestBaseline baseline,
                      const std::vector<void *> &inputs_data,
                      bool print_on_error, int rank, int world_size,
                      int num_warps_per_sm) {
    Executor exe{rank, world_size, model, "op_test_" + rand_anum(4),
                 num_warps_per_sm};
    exe.compile();

    std::vector<void *> inputs_data_storages;
    std::vector<void *> inputs_data_refs;

    if (inputs_data.empty()) {
        // Set random data.
        for (auto t : inputs) {
            void *buf = ::malloc(t->shape_bytes());
            UNITTEST_NE(buf, (void *)nullptr);

            if (t->type == FP32) {
                ::memcpy(buf, utils::rand_floats(t->shape.size(), 0.1).get(),
                         t->shape_bytes());
            } else if (t->type == FP16) {
                ::memcpy(buf, utils::rand_halfs(t->shape.size(), 0.1).get(),
                         t->shape_bytes());
            } else if (t->type == BF16) {
                ::memcpy(
                    buf,
                    utils::rand_array<bfloat16_t>(t->shape.size(), 0.1).get(),
                    t->shape_bytes());
            } else if (t->type == INT32) {
                ::memcpy(buf,
                         utils::rand_array<int>(t->shape.size(), 10000).get(),
                         t->shape_bytes());
            } else if (t->type == BYTE) {
                ::memcpy(buf, utils::rand_bytes(t->shape.size()).get(),
                         t->shape_bytes());
            } else {
                LOG(ERROR, "Unsupported data type: ", t->type);
            }
            t->write(buf);
            inputs_data_storages.push_back(buf);
            inputs_data_refs.push_back(buf);
        }
    } else {
        // Copy input data
        UNITTEST_EQ(inputs_data.size(), inputs.size());
        for (size_t i = 0; i < inputs.size(); i++) {
            inputs[i]->write(inputs_data[i]);
            inputs_data_refs.emplace_back(inputs_data[i]);
        }
    }

    exe.launch();

    // Correctness test.
    exe.run(10000);
    exe.wait();
    exe.stop();

    // Copy results of the loop kernel routine into CPU memory.
    std::vector<void *> res;
    for (auto t : outputs) {
        void *buf = ::malloc(t->shape_bytes());
        UNITTEST_NE(buf, (void *)nullptr);
        t->read(buf);
        res.push_back(buf);
    }

    std::vector<void *> gt;
    for (auto t : outputs) {
        void *buf = ::malloc(t->shape_bytes());
        UNITTEST_NE(buf, (void *)nullptr);
        gt.push_back(buf);
    }

    std::vector<ark::Dims> output_shapes;
    for (auto t : outputs) {
        output_shapes.push_back(t->shape);
    }
    std::vector<ark::Dims> input_shapes;
    for (auto t : inputs) {
        input_shapes.push_back(t->shape);
    }

    // Calculate ground truth.
    baseline(gt, output_shapes, inputs_data_refs, input_shapes, rank);

    std::stringstream test_name;
    test_name << test_name_prefix;
    for (size_t i = 0; i < inputs.size(); i++) {
        test_name << ";in" << i << "=" << inputs[i]->shape;
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        test_name << ";out" << i << "=" << outputs[i]->shape;
    }
    test_name << ";";

    OpsTestResult result;
    result.test_name = test_name.str();
    result.num_warps_per_sm = num_warps_per_sm;

    // Compare results with the ground truth.
    for (size_t i = 0; i < outputs.size(); i++) {
        TensorCompareResult comp;
        if (outputs[i]->type == FP32) {
            comp = tensor_compare(static_cast<float *>(gt[i]),
                                  static_cast<float *>(res[i]),
                                  outputs[i]->shape.dims4(), print_on_error);
        } else if (outputs[i]->type == FP16) {
            comp = tensor_compare(static_cast<ark::half_t *>(gt[i]),
                                  static_cast<ark::half_t *>(res[i]),
                                  outputs[i]->shape.dims4(), print_on_error);
        } else if (outputs[i]->type == BF16) {
            comp = tensor_compare(static_cast<ark::bfloat16_t *>(gt[i]),
                                  static_cast<ark::bfloat16_t *>(res[i]),
                                  outputs[i]->shape.dims4(), print_on_error);
        } else if (outputs[i]->type == INT32) {
            comp = tensor_compare(static_cast<int *>(gt[i]),
                                  static_cast<int *>(res[i]),
                                  outputs[i]->shape.dims4(), print_on_error);
        } else if (outputs[i]->type == BYTE) {
            comp = tensor_compare(static_cast<uint8_t *>(gt[i]),
                                  static_cast<uint8_t *>(res[i]),
                                  outputs[i]->shape.dims4(), print_on_error);
        } else {
            LOG(ERROR, "Unsupported data type: ", outputs[i]->type);
        }
        result.mse.push_back(comp.mse);
        result.max_diff.push_back(comp.max_diff);
        result.max_err_rate.push_back(comp.max_error_rate);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Throughput test.
    if (world_size > 1) {
        // For multi-GPU, we need to make sure that all GPUs run the same
        // number of iterations. Rather than doing allgather, we just
        // use a magic number here.
        int iter = 100;
        exe.launch();
        exe.run(iter);
        float msec = exe.stop();
        result.msec_per_iter = msec / iter;
    } else {
        // Rough measure.
        int warmup_iter = 3;
        float target_msec = 2000;
        exe.launch();
        exe.run(warmup_iter);
        float warmup_msec = exe.stop();

        if (warmup_msec > target_msec) {
            // Warm-up was long enough.
            result.msec_per_iter = warmup_msec / warmup_iter;
        } else {
            int iter = int(target_msec / warmup_msec);
            exe.launch();
            exe.run(iter);
            float msec = exe.stop();
            result.msec_per_iter = msec / iter;
        }
    }

    // Free resources
    for (auto ptr : inputs_data_storages) {
        ::free(ptr);
    }
    for (auto ptr : res) {
        ::free(ptr);
    }
    for (auto ptr : gt) {
        ::free(ptr);
    }

    return result;
}

OpsTestResult op_test_8(const std::string &test_name_prefix, Model &model,
                        const std::vector<Tensor *> &inputs,
                        const std::vector<Tensor *> &outputs,
                        OpsTestBaseline baseline,
                        const std::vector<void *> &inputs_data,
                        bool print_on_error, int rank, int world_size) {
    return op_test(test_name_prefix, model, inputs, outputs, baseline,
                   inputs_data, print_on_error, rank, world_size, 8);
}

OpsTestResult op_test_16(const std::string &test_name_prefix, Model &model,
                         const std::vector<Tensor *> &inputs,
                         const std::vector<Tensor *> &outputs,
                         OpsTestBaseline baseline,
                         const std::vector<void *> &inputs_data,
                         bool print_on_error, int rank, int world_size) {
    return op_test(test_name_prefix, model, inputs, outputs, baseline,
                   inputs_data, print_on_error, rank, world_size, 16);
}

OpsTestResult op_test_32(const std::string &test_name_prefix, Model &model,
                         const std::vector<Tensor *> &inputs,
                         const std::vector<Tensor *> &outputs,
                         OpsTestBaseline baseline,
                         const std::vector<void *> &inputs_data,
                         bool print_on_error, int rank, int world_size) {
    return op_test(test_name_prefix, model, inputs, outputs, baseline,
                   inputs_data, print_on_error, rank, world_size, 32);
}

OpsTestGpuMem::OpsTestGpuMem(size_t size) : size_(size) {
    CUDA_CHECK(cudaMalloc(&this->gpu_ptr_, size));
}

OpsTestGpuMem::~OpsTestGpuMem() { cudaFree(this->gpu_ptr_); }

void *OpsTestGpuMem::get() const { return this->gpu_ptr_; }

size_t OpsTestGpuMem::size() const { return this->size_; }

OpsTestGpuMem to_gpu(const void *host_ptr, size_t size) {
    OpsTestGpuMem gpu_mem(size);
    CUDA_CHECK(
        cudaMemcpy(gpu_mem.get(), host_ptr, size, cudaMemcpyHostToDevice));
    return gpu_mem;
}

void *from_gpu(const OpsTestGpuMem &test_gpu_mem, void *host_ptr) {
    if (host_ptr == nullptr) {
        host_ptr = ::malloc(test_gpu_mem.size());
    }
    CUDA_CHECK(cudaMemcpy(host_ptr, test_gpu_mem.get(), test_gpu_mem.size(),
                          cudaMemcpyDeviceToHost));
    return host_ptr;
}

void sync_gpu() { CUDA_CHECK(cudaDeviceSynchronize()); }

}  // namespace ark
