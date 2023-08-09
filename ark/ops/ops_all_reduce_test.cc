// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_kernel.h"
#include "include/ark.h"
#include "include/ark_utils.h"
#include "logging.h"
#include "unittest/unittest_utils.h"

using namespace std;
using namespace ark;
// used for print the all_reduce result and check the correctness
//  #define PRINT_MATRIX
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
#ifdef PRINT_MATRIX
    // print input data.
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        cout << "input data of gpu_id: " << gpu_id << endl;
        for (size_t i = 0; i < bytes / sizeof(ark::half_t) && i < 10; i++) {
            cout << (float)input_data[gpu_id].get()[i] << " ";
        }
        cout << endl;
    }
    // print ground truth.
    cout << "ground truth: " << endl;
    for (size_t i = 0; i < bytes / sizeof(ark::half_t) && i < 10; i++) {
        cout << (float)gt[i] << " ";
    }
    cout << endl;
#endif
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

            // Get the auto-scheduled buffers.
            ark::GpuBuf *buf_tns =
                static_cast<ark::GpuBuf *>(allreduce_result->buf->buf);
            UNITTEST_NE(buf_tns, (ark::GpuBuf *)nullptr);

            // Set data.
            ark::gpu_memcpy(buf_tns, input_data[gpu_id].get(), bytes);

            // launch kernel
            exe.launch();
            exe.run(iter);
            float elapsed_msec = exe.stop();

            // Copy results of the loop kernel routine into CPU memory.
            ark::half_t *res = (ark::half_t *)malloc(bytes);
            UNITTEST_NE(res, (void *)nullptr);
            ark::gpu_memcpy(res, buf_tns, bytes);

            // Compare results with the ground truth.
            auto p = ark::utils::cmp_matrix((ark::half_t *)gt,
                                            (ark::half_t *)res, 1, bytes / 2);
#ifdef PRINT_MATRIX
            // print result, to avoid too long output, only print the first 10
            // elements if(gpu_id == 0)
            {
                cout << "result on gpu_id: " << gpu_id << " ";
                for (size_t i = 0; i < bytes / sizeof(ark::half_t) && i < 10;
                     i++) {
                    cout << (float)res[i] << " ";
                }
                cout << endl;
            }
#endif
            free(res);
            LOG(ark::INFO, " all_reduce on gpu: ", gpu_id,
                " num_gpus: ", num_gpus, " total_bytes: ", bytes,
                " iter: ", iter, setprecision(4), " mse: ", p.first,
                " max_err: ", p.second * 100, "%",
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
