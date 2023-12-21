// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_test_common.h"

#include <cstring>

#include "env.h"
#include "gpu/gpu_logging.h"
#include "logging.h"
#include "random.h"
#include "unittest/unittest_utils.h"

namespace ark {

std::ostream &operator<<(std::ostream &os, const OpsTestResult &result) {
    os << "op test: " << result.test_name << " #warp/sm "
       << result.num_warps_per_sm << ", msec/iter " << result.msec_per_iter;
    os << std::setprecision(4);
    for (size_t i = 0; i < result.mse.size(); i++) {
        float err_pcnt = result.max_err_rate[i] * 100;
        os << ", mse " << result.mse[i] << ", max_diff " << result.max_diff[i]
           << ", max_err_rate " << err_pcnt << "%, #diff "
           << result.num_wrong[i] << "/" << result.num_total[i];
    }
    return os;
}

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
                std::vector<float> data(t->shape.size());
                for (size_t i = 0; i < data.size(); ++i) {
                    data[i] = ark::rand<float>(-0.1, 0.1);
                }
                ::memcpy(buf, data.data(), t->shape_bytes());
            } else if (t->type == FP16) {
                std::vector<ark::half_t> data(t->shape.size());
                for (size_t i = 0; i < data.size(); ++i) {
                    data[i] = ark::rand<ark::half_t>(-0.1, 0.1);
                }
                ::memcpy(buf, data.data(), t->shape_bytes());
            } else if (t->type == BF16) {
                std::vector<ark::bfloat16_t> data(t->shape.size());
                for (size_t i = 0; i < data.size(); ++i) {
                    data[i] = ark::rand<ark::bfloat16_t>(-0.1, 0.1);
                }
                ::memcpy(buf, data.data(), t->shape_bytes());
            } else if (t->type == INT32) {
                std::vector<int> data(t->shape.size());
                for (size_t i = 0; i < data.size(); ++i) {
                    data[i] = ark::rand<int>(-10000, 10000);
                }
                ::memcpy(buf, data.data(), t->shape_bytes());
            } else if (t->type == BYTE) {
                std::vector<uint8_t> data(t->shape.size());
                for (size_t i = 0; i < data.size(); ++i) {
                    data[i] = ark::rand<uint8_t>(0, 255);
                }
                ::memcpy(buf, data.data(), t->shape_bytes());
            } else {
                ERR(UnitTestError, "Unsupported data type: ", t->type);
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
    exe.run(1);
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
            ERR(UnitTestError, "Unsupported data type: ", outputs[i]->type);
        }
        result.mse.push_back(comp.mse);
        result.max_diff.push_back(comp.max_diff);
        result.max_err_rate.push_back(comp.max_error_rate);
        result.num_wrong.push_back(comp.num_wrong);
        result.num_total.push_back(comp.num_total);
    }

    GLOG(gpuDeviceSynchronize());

    // Throughput test.
    if (world_size > 1) {
        // For multi-GPU, we need to make sure that all GPUs run the same
        // number of iterations. Rather than doing allgather, we just
        // use a magic number here.
        int iter = 1000;
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
    GLOG(gpuMemAlloc(reinterpret_cast<gpuDeviceptr *>(&this->gpu_ptr_), size));
}

OpsTestGpuMem::~OpsTestGpuMem() {
    if (gpuMemFree(reinterpret_cast<gpuDeviceptr>(this->gpu_ptr_)) !=
        gpuSuccess) {
        LOG(WARN, "gpuMemFree() failed.");
    }
}

void *OpsTestGpuMem::get() const { return this->gpu_ptr_; }

size_t OpsTestGpuMem::size() const { return this->size_; }

OpsTestGpuMem to_gpu(void *host_ptr, size_t size) {
    OpsTestGpuMem gpu_mem(size);
    GLOG(gpuMemcpyHtoD(reinterpret_cast<gpuDeviceptr>(gpu_mem.get()), host_ptr,
                       size));
    return gpu_mem;
}

void *from_gpu(const OpsTestGpuMem &test_gpu_mem, void *host_ptr) {
    if (host_ptr == nullptr) {
        host_ptr = ::malloc(test_gpu_mem.size());
    }
    GLOG(gpuMemcpyDtoH(host_ptr,
                       reinterpret_cast<gpuDeviceptr>(test_gpu_mem.get()),
                       test_gpu_mem.size()));
    return host_ptr;
}

void sync_gpu() { GLOG(gpuDeviceSynchronize()); }

}  // namespace ark
