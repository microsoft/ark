// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "env.h"
#include "ops_test_common.h"
#include "unittest/unittest_utils.h"
#include <type_traits>

template <typename T, int NumGpus>
void baseline_all_reduce(std::vector<void *> &outputs,
                         const std::vector<ark::Dims> &output_shapes,
                         const std::vector<void *> &,
                         const std::vector<ark::Dims> &, int)
{
    // Calculate sum from 1 to NumGpus.
    T expected = 0;
    for (int i = 1; i <= NumGpus; ++i) {
        expected += i;
    }

    T *out = static_cast<T *>(outputs[0]);
    ark::DimType nelem = output_shapes[0].size();
    for (ark::DimType i = 0; i < nelem; ++i) {
        out[i] = expected;
    }
}

void test_all_reduce_4gpus_internal(size_t nelem, int iter) {
    constexpr int num_gpus = 4;
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id, nelem, iter]() {
            // Each GPU's data is equal to its GPU ID + 1.
            ark::Model m{gpu_id};
            ark::Tensor *ones = m.tensor(ark::Dims(nelem), ark::FP16);
            ark::Tensor *data = m.scale(ones, float(gpu_id + 1));
            ark::Tensor *output = m.all_reduce(data, gpu_id, num_gpus);

            auto ones_data = ark::utils::ones<ark::half_t>(ones->shape.size());
            auto result =
                ark::op_test("all_reduce", m, {ones}, {output},
                             baseline_all_reduce<ark::half_t, num_gpus>,
                             {ones_data.get()}, false, gpu_id, num_gpus);
            UNITTEST_LOG(result);
            UNITTEST_EQ(result.max_diff[0], 0.0f);
            return ark::unittest::SUCCESS;
        });
    }
    ark::unittest::wait_all_processes();
}

void test_local_all_reduce_8gpus_internel(size_t nelem, int iter)
{
    constexpr int num_gpus = 8;
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id, nelem, num_gpus, iter]() {
            // Each GPU's data is equal to its GPU ID + 1.
            ark::Model m{gpu_id};
            ark::Tensor *ones = m.tensor(ark::Dims(nelem), ark::FP16);
            ark::Tensor *data = m.scale(ones, float(gpu_id + 1));
            ark::Tensor *output = m.local_all_reduce(data, gpu_id, num_gpus, nullptr);

            auto ones_data = ark::utils::ones<ark::half_t>(ones->shape.size());
            auto result =
                ark::op_test("all_reduce", m, {ones}, {output},
                             baseline_all_reduce<ark::half_t, num_gpus>,
                             {ones_data.get()}, true, gpu_id, num_gpus, 16);
            ark::op_test_log(result);
            return ark::unittest::SUCCESS;
        });
    }
    ark::unittest::wait_all_processes();
}

ark::unittest::State test_all_reduce_4gpus()
{
    test_all_reduce_4gpus_internal(8, 1);
    test_all_reduce_4gpus_internal(8192, 1);
    if (ark::get_env().use_mscclpp) {
        test_local_all_reduce_8gpus_internel(1024 * 1024 * 12, 1);
    }
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_all_reduce_4gpus);
    return ark::unittest::SUCCESS;
}
