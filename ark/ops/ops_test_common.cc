// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_test_common.h"
#include "gpu/gpu_kernel.h"
#include "include/ark_utils.h"
#include "logging.h"
#include "random.h"
#include "unittest/unittest_utils.h"
#include <cstring>

namespace ark {

std::ostream &operator<<(std::ostream &os, const OpsTestResult &result)
{
    os << "op test: " << result.test_name << " #warp/sm "
       << result.num_warps_per_sm << ", msec/iter " << result.msec_per_iter;
    os << std::setprecision(4) << std::fixed;
    for (size_t i = 0; i < result.mse.size(); i++) {
        float err_pcnt = result.max_err[i] * 100;
        os << ", mse " << result.mse[i] << ", max_err " << err_pcnt << "%";
    }
    return os;
}

OpsTestResult op_test(const std::string &test_name_prefix, Model &model,
                      const std::vector<Tensor *> &inputs,
                      const std::vector<Tensor *> &outputs,
                      OpsTestBaseline baseline, bool print_on_error,
                      int num_warps_per_sm)
{
    Executor exe{0, 0, 1, model, "op_test_" + rand_anum(4), num_warps_per_sm};
    exe.compile();

    // Set random data.
    std::vector<void *> input_data;
    for (auto t : inputs) {
        void *buf = ::malloc(t->shape_bytes());
        UNITTEST_NE(buf, (void *)nullptr);

        if (t->type == FP32) {
            ::memcpy(buf, utils::rand_floats(t->shape.size(), 0.01).get(),
                     t->shape_bytes());
        } else if (t->type == FP16) {
            ::memcpy(buf, utils::rand_halfs(t->shape.size(), 0.01).get(),
                     t->shape_bytes());
        } else if (t->type == INT32) {
            ::memcpy(buf, utils::rand_array<int>(t->shape.size(), 10000).get(),
                     t->shape_bytes());
        } else if (t->type == BYTE) {
            ::memcpy(buf, utils::rand_bytes(t->shape.size()).get(),
                     t->shape_bytes());
        } else {
            LOG(ERROR, "Unsupported data type: ", t->type);
        }
        t->write(buf);
        input_data.push_back(buf);
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
    baseline(gt, output_shapes, input_data, input_shapes);

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
        std::pair<float, float> p;
        if (outputs[i]->type == FP32) {
            p = utils::tensor_compare(
                static_cast<float *>(gt[i]), static_cast<float *>(res[i]),
                outputs[i]->shape.dims4(), print_on_error);
        } else if (outputs[i]->type == FP16) {
            p = utils::tensor_compare(static_cast<ark::half_t *>(gt[i]),
                                      static_cast<ark::half_t *>(res[i]),
                                      outputs[i]->shape.dims4(),
                                      print_on_error);
        } else if (outputs[i]->type == INT32) {
            p = utils::tensor_compare(
                static_cast<int *>(gt[i]), static_cast<int *>(res[i]),
                outputs[i]->shape.dims4(), print_on_error);
        } else if (outputs[i]->type == BYTE) {
            p = utils::tensor_compare(
                static_cast<uint8_t *>(gt[i]), static_cast<uint8_t *>(res[i]),
                outputs[i]->shape.dims4(), print_on_error);
        } else {
            LOG(ERROR, "Unsupported data type: ", outputs[i]->type);
        }
        result.mse.push_back(p.first);
        result.max_err.push_back(p.second);
    }

    // Throughput test.

    // Restart the executor.
    exe.launch();

    // Rough measure.
    int warmup_iter = 3;
    float target_msec = 2000;
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

    exe.stop();

    // Free resources
    for (auto ptr : input_data) {
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
                        OpsTestBaseline baseline, bool print_on_error)
{
    return op_test(test_name_prefix, model, inputs, outputs, baseline, 8,
                   print_on_error);
}

OpsTestResult op_test_16(const std::string &test_name_prefix, Model &model,
                         const std::vector<Tensor *> &inputs,
                         const std::vector<Tensor *> &outputs,
                         OpsTestBaseline baseline, bool print_on_error)
{
    return op_test(test_name_prefix, model, inputs, outputs, baseline, 16,
                   print_on_error);
}

OpsTestResult op_test_32(const std::string &test_name_prefix, Model &model,
                         const std::vector<Tensor *> &inputs,
                         const std::vector<Tensor *> &outputs,
                         OpsTestBaseline baseline, bool print_on_error)
{
    return op_test(test_name_prefix, model, inputs, outputs, baseline, 32,
                   print_on_error);
}

void op_test_log(const OpsTestResult &result)
{
    LOG(INFO, result);
}

} // namespace ark
