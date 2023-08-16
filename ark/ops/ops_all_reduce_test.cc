// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_kernel.h"
#include "include/ark.h"
#include "include/ark_utils.h"
#include "logging.h"
#include "ops_test_common.h"
#include "unittest/unittest_utils.h"

using namespace std;
using namespace ark;

void test_all_reduce_internal(size_t bytes, int num_gpus, int iter)
{
    // bytes/num_gpus is the number of bytes a GPU send in one iteration, the
    // bytes must be multiple of num_gpus, the tensor shape is {1, bytes /
    // sizeof(ark::half_t), 1, 1}.
    if (bytes % num_gpus != 0) {
        LOG(INFO, "bytes must be multiple of num_gpus");
        return;
    }
    // init input data and ground truth.
    ark::srand();
    vector<unique_ptr<ark::half_t[]>> input_data(num_gpus);
    for (int i = 0; i < num_gpus; i++) {
        input_data[i] =
            ark::utils::rand_halfs(bytes / sizeof(ark::half_t), 0.01);
    }

    // calculate ground truth of all_reduce.
    ark::half_t *gt = (ark::half_t *)malloc(bytes);
    UNITTEST_NE(gt, (void *)nullptr);
    // first convert the input data to float, then sum them up, finally convert
    // the result to ark::half_t.
    for (size_t i = 0; i < bytes / sizeof(ark::half_t); i++) {
        float sum = 0;
        for (int j = 0; j < num_gpus; j++) {
            sum += (float)input_data[j].get()[i];
        }
        gt[i] = ark::half_t(sum);
    }

    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id, num_gpus, &input_data, &gt, bytes,
                                      iter]() {
            // define model.
            Model model{gpu_id};
            Tensor *data = model.tensor(
                {
                    (ark::DimType)(bytes / sizeof(ark::half_t)),
                },
                FP16);
            Tensor *allreduce_result = model.all_reduce(data, gpu_id, num_gpus);
            Executor exe{gpu_id, gpu_id, num_gpus, model, "test_all_reduce"};
            exe.compile();

            // Set data.
            allreduce_result->write(input_data[gpu_id].get());

            // launch kernel
            exe.launch();
            exe.run(iter);
            float elapsed_msec = exe.stop();

            // Copy results of the loop kernel routine into CPU memory.
            ark::half_t *res = (ark::half_t *)malloc(bytes);
            UNITTEST_NE(res, (void *)nullptr);
            allreduce_result->read(res);

            // Compare results with the ground truth.
            auto comp = tensor_compare(gt, res, allreduce_result->shape);

            free(res);
            LOG(ark::INFO, " all_reduce on gpu: ", gpu_id,
                " num_gpus: ", num_gpus, " total_bytes: ", bytes,
                " iter: ", iter, setprecision(4), " mse: ", comp.mse,
                " max_err: ", comp.max_error_rate * 100, "%",
                " elapsed_msec: ", elapsed_msec, "ms");
            return ark::unittest::SUCCESS;
        });
    }

    ark::unittest::wait_all_processes();
    free(gt);
}

ark::unittest::State test_all_reduce()
{
    test_all_reduce_internal(8, 2, 1);
    test_all_reduce_internal(16, 4, 1);
    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_all_reduce);
    return ark::unittest::SUCCESS;
}
