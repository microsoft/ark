// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_test_common.hpp"

#include <cstring>

#include "ark/executor.hpp"
#include "ark/model.hpp"
#include "ark/planner.hpp"
#include "ark/random.hpp"
#include "env.h"
#include "gpu/gpu_logging.h"
#include "logging.h"
#include "model/model_data_type.hpp"
#include "model/model_tensor.hpp"
#include "unittest/unittest_utils.h"

namespace ark {

std::ostream &operator<<(std::ostream &os, const OpsTestResult &result) {
    os << "op test: " << result.test_name << " #warp/sm "
       << result.num_warps_per_sm << ", #iter " << result.iter << ", msec/iter "
       << result.msec_per_iter;
    os << std::setprecision(4);
    for (size_t i = 0; i < result.mse.size(); i++) {
        float err_pcnt = result.max_err_rate[i] * 100;
        os << ", mse " << result.mse[i] << ", max_diff " << result.max_diff[i]
           << ", max_err_rate " << err_pcnt << "%, #diff "
           << result.num_wrong[i] << "/" << result.num_total[i];
    }
    return os;
}

OpsTestResult op_test(const std::string &test_name_prefix, const Model &model,
                      const std::vector<Tensor> &inputs,
                      const std::vector<Tensor> &outputs,
                      OpsTestBaseline baseline,
                      const std::vector<void *> &inputs_data,
                      bool print_on_error, int rank, int world_size,
                      int num_warps_per_sm) {
    DefaultExecutor exe(model);
    exe.compile();

    std::vector<std::shared_ptr<std::vector<char>>> inputs_data_storages;
    std::vector<void *> inputs_data_refs;

    if (inputs_data.empty()) {
        // Set random data.
        for (auto t : inputs) {
            auto buf = std::make_shared<std::vector<char>>(
                t.shape().nelems() * t.data_type().bytes());

            if (t.data_type() == FP32) {
                float *data = reinterpret_cast<float *>(buf->data());
                for (auto i = 0; i < t.shape().nelems(); ++i) {
                    data[i] = random<float>(-0.1, 0.1);
                }
            } else if (t.data_type() == FP16) {
                half_t *data = reinterpret_cast<half_t *>(buf->data());
                for (auto i = 0; i < t.shape().nelems(); ++i) {
                    data[i] = random<ark::half_t>(-0.1, 0.1);
                }
            } else if (t.data_type() == BF16) {
                bfloat16_t *data = reinterpret_cast<bfloat16_t *>(buf->data());
                for (auto i = 0; i < t.shape().nelems(); ++i) {
                    data[i] = random<ark::bfloat16_t>(-0.1, 0.1);
                }
            } else if (t.data_type() == INT32) {
                int *data = reinterpret_cast<int *>(buf->data());
                for (auto i = 0; i < t.shape().nelems(); ++i) {
                    data[i] = random<int>(-10000, 10000);
                }
            } else if (t.data_type() == BYTE) {
                uint8_t *data = reinterpret_cast<uint8_t *>(buf->data());
                for (auto i = 0; i < t.shape().nelems(); ++i) {
                    data[i] = random<uint8_t>(0, 255);
                }
            } else {
                ERR(UnitTestError,
                    "Unsupported data type: ", t.data_type().name());
            }
            exe.tensor_write(t, *buf);
            inputs_data_storages.push_back(buf);
            inputs_data_refs.push_back(buf->data());
        }
    } else {
        // Copy input data
        UNITTEST_EQ(inputs_data.size(), inputs.size());
        for (size_t i = 0; i < inputs.size(); i++) {
            std::vector<char> buf(inputs[i].shape().nelems() *
                                  inputs[i].data_type().bytes());
            std::memcpy(buf.data(), inputs_data[i], buf.size());
            exe.tensor_write(inputs[i], buf);
            inputs_data_refs.emplace_back(inputs_data[i]);
        }
    }

    exe.launch();

    // Correctness test.
    exe.run(1);
    exe.wait();
    exe.stop();

    // Copy results of the loop kernel routine into CPU memory.
    std::vector<std::shared_ptr<std::vector<char>>> res;
    for (auto t : outputs) {
        auto buf = std::make_shared<std::vector<char>>(t.shape().nelems() *
                                                       t.data_type().bytes());
        exe.tensor_read(t, *buf);
        res.push_back(buf);
    }

    std::vector<std::shared_ptr<std::vector<char>>> gt;
    for (auto t : outputs) {
        auto buf = std::make_shared<std::vector<char>>(t.shape().nelems() *
                                                       t.data_type().bytes());
        gt.push_back(buf);
    }

    std::vector<ark::Dims> output_shapes;
    for (auto t : outputs) {
        output_shapes.push_back(t.shape());
    }
    std::vector<ark::Dims> input_shapes;
    for (auto t : inputs) {
        input_shapes.push_back(t.shape());
    }

    // Calculate ground truth.
    std::vector<void *> gt_ptrs;
    for (auto t : gt) {
        gt_ptrs.push_back(t->data());
    }
    baseline(gt_ptrs, output_shapes, inputs_data_refs, input_shapes, rank);

    std::stringstream test_name;
    test_name << test_name_prefix;
    for (size_t i = 0; i < inputs.size(); i++) {
        test_name << ";in" << i << "=" << inputs[i].shape();
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        test_name << ";out" << i << "=" << outputs[i].shape();
    }
    test_name << ";";

    OpsTestResult result;
    result.test_name = test_name.str();
    result.num_warps_per_sm = num_warps_per_sm;

    // Compare results with the ground truth.
    for (size_t i = 0; i < outputs.size(); i++) {
        TensorCompareResult comp;
        if (outputs[i].data_type() == FP32) {
            comp = tensor_compare(reinterpret_cast<float *>(gt[i]->data()),
                                  reinterpret_cast<float *>(res[i]->data()),
                                  outputs[i].shape().dims4(), print_on_error);
        } else if (outputs[i].data_type() == FP16) {
            comp =
                tensor_compare(reinterpret_cast<ark::half_t *>(gt[i]->data()),
                               reinterpret_cast<ark::half_t *>(res[i]->data()),
                               outputs[i].shape().dims4(), print_on_error);
        } else if (outputs[i].data_type() == BF16) {
            comp = tensor_compare(
                reinterpret_cast<ark::bfloat16_t *>(gt[i]->data()),
                reinterpret_cast<ark::bfloat16_t *>(res[i]->data()),
                outputs[i].shape().dims4(), print_on_error);
        } else if (outputs[i].data_type() == INT32) {
            comp = tensor_compare(reinterpret_cast<int *>(gt[i]->data()),
                                  reinterpret_cast<int *>(res[i]->data()),
                                  outputs[i].shape().dims4(), print_on_error);
        } else if (outputs[i].data_type() == BYTE) {
            comp = tensor_compare(reinterpret_cast<uint8_t *>(gt[i]->data()),
                                  reinterpret_cast<uint8_t *>(res[i]->data()),
                                  outputs[i].shape().dims4(), print_on_error);
        } else {
            ERR(UnitTestError,
                "Unsupported data type: ", outputs[i].data_type().name());
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
        result.iter = iter;
        result.msec_per_iter = msec / iter;
    } else {
        // Rough measure.
        int warmup_iter = 3;
        float target_msec = 5000;
        exe.launch();
        exe.run(warmup_iter);
        float warmup_msec = exe.stop();

        if (warmup_msec > target_msec) {
            // Warm-up was long enough.
            result.iter = warmup_iter;
            result.msec_per_iter = warmup_msec / warmup_iter;
        } else {
            int iter = int(target_msec / warmup_msec) * warmup_iter;
            exe.launch();
            exe.run(iter);
            float msec = exe.stop();
            result.iter = iter;
            result.msec_per_iter = msec / iter;
        }
    }
    return result;
}

OpsTestResult op_test_8(const std::string &test_name_prefix, const Model &model,
                        const std::vector<Tensor> &inputs,
                        const std::vector<Tensor> &outputs,
                        OpsTestBaseline baseline,
                        const std::vector<void *> &inputs_data,
                        bool print_on_error, int rank, int world_size) {
    return op_test(test_name_prefix, model, inputs, outputs, baseline,
                   inputs_data, print_on_error, rank, world_size, 8);
}

OpsTestResult op_test_16(const std::string &test_name_prefix,
                         const Model &model, const std::vector<Tensor> &inputs,
                         const std::vector<Tensor> &outputs,
                         OpsTestBaseline baseline,
                         const std::vector<void *> &inputs_data,
                         bool print_on_error, int rank, int world_size) {
    return op_test(test_name_prefix, model, inputs, outputs, baseline,
                   inputs_data, print_on_error, rank, world_size, 16);
}

OpsTestResult op_test_32(const std::string &test_name_prefix,
                         const Model &model, const std::vector<Tensor> &inputs,
                         const std::vector<Tensor> &outputs,
                         OpsTestBaseline baseline,
                         const std::vector<void *> &inputs_data,
                         bool print_on_error, int rank, int world_size) {
    return op_test(test_name_prefix, model, inputs, outputs, baseline,
                   inputs_data, print_on_error, rank, world_size, 32);
}

OpsTestGpuMem::OpsTestGpuMem(size_t size) : size_(size) {
    GLOG(gpuMalloc(&this->gpu_ptr_, size));
}

OpsTestGpuMem::~OpsTestGpuMem() {
    if (gpuFree(this->gpu_ptr_) != gpuSuccess) {
        LOG(WARN, "gpuFree() failed.");
    }
}

void *OpsTestGpuMem::get() const { return this->gpu_ptr_; }

size_t OpsTestGpuMem::size() const { return this->size_; }

OpsTestGpuMem to_gpu(void *host_ptr, size_t size) {
    OpsTestGpuMem gpu_mem(size);
    GLOG(gpuMemcpy(gpu_mem.get(), host_ptr, size, gpuMemcpyHostToDevice));
    return gpu_mem;
}

void *from_gpu(const OpsTestGpuMem &test_gpu_mem, void *host_ptr) {
    if (host_ptr == nullptr) {
        host_ptr = ::malloc(test_gpu_mem.size());
    }
    GLOG(gpuMemcpy(host_ptr, test_gpu_mem.get(), test_gpu_mem.size(),
                   gpuMemcpyDeviceToHost));
    return host_ptr;
}

void sync_gpu() { GLOG(gpuDeviceSynchronize()); }

}  // namespace ark
