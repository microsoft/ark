// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_test_common.h"
#include "unittest/unittest_utils.h"

template <typename T, int NumGpus>
void baseline_all_gather(std::vector<void *> &outputs,
                         const std::vector<ark::Dims> &output_shapes,
                         const std::vector<void *> &,
                         const std::vector<ark::Dims> &)
{
    for (int i = 0; i < NumGpus; ++i) {
        T *out = static_cast<T *>(outputs[i]);
        for (ark::DimType j = 0; j < output_shapes[i].size(); ++j) {
            out[j] = i + 1;
        }
    }
}

void test_all_gather_4gpus_internal(size_t nelem, int iter)
{
    constexpr int num_gpus = 4;
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id, nelem, num_gpus, iter]() {
            // Each GPU's data is equal to its GPU ID + 1.
            ark::Model m{gpu_id};
            ark::Tensor *ones = m.tensor(ark::Dims(nelem), ark::FP16);
            ark::Tensor *data = m.scale(ones, float(gpu_id + 1));
            auto outputs = m.all_gather(data, gpu_id, num_gpus);

            auto ones_data = ark::utils::ones<ark::half_t>(ones->shape.size());
            auto result =
                ark::op_test("all_gather", m, {ones}, outputs,
                             baseline_all_gather<ark::half_t, num_gpus>,
                             {ones_data.get()}, true, gpu_id, num_gpus);
            ark::op_test_log(result);
            return ark::unittest::SUCCESS;
        });
    }
    ark::unittest::wait_all_processes();
}

ark::unittest::State test_all_gather_4gpus()
{
    test_all_gather_4gpus_internal(8, 1);
    test_all_gather_4gpus_internal(8192, 1);
    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_all_gather_4gpus);
    return ark::unittest::SUCCESS;
}
