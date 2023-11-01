// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "env.h"
#include "gpu/gpu_kernel.h"
#include "include/ark.h"
#include "ipc/ipc_coll.h"
#include "ops_test_common.h"
#include "unittest/unittest_utils.h"

using namespace std;

template <typename T, int NumGpus>
void baseline_reduce_scatter(std::vector<void *> &outputs,
                             const std::vector<ark::Dims> &output_shapes,
                             const std::vector<void *> &,
                             const std::vector<ark::Dims> &, int rank) {
    // Calculate sum from 1 to NumGpus.
    T expected = 0;
    for (int i = 1; i <= NumGpus; ++i) {
        expected += i;
    }

    T *out = static_cast<T *>(outputs[0]);
    ark::DimType nelem = output_shapes[0].size();
    int64_t nelems_per_gpu = nelem / NumGpus;
    for (ark::DimType i = 0; i < nelem; ++i) {
        out[i] = rank + 1;
    }
    for (ark::DimType i = rank * nelems_per_gpu;
         i < (rank + 1) * nelems_per_gpu; ++i) {
        out[i] = expected;
    }
}

void test_reduce_scatter_internal(size_t nelem, int iter) {
    constexpr int num_gpus = 8;
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id, nelem, iter]() {
            //
            ark::Model model{gpu_id};
            ark::Tensor *data = model.tensor(ark::Dims(nelem), ark::FP16);
            std::vector<ark::half_t> data_buf(nelem);
            for (size_t i = 0; i < nelem; ++i) {
                data_buf[i] = ark::half_t(gpu_id + 1);
            }
            ark::Tensor *output =
                model.local_reduce_scatter_msll(data, gpu_id, num_gpus);

            auto result =
                ark::op_test("reduce_scatter", model, {data}, {output},
                             baseline_reduce_scatter<ark::half_t, num_gpus>,
                             {data_buf.data()}, false, gpu_id, num_gpus, 16);
            UNITTEST_LOG(result);
            UNITTEST_EQ(result.max_diff[0], 0.0f);
            return ark::unittest::SUCCESS;
        });
    }

    ark::unittest::wait_all_processes();
}

ark::unittest::State test_reduce_scatter() {
    test_reduce_scatter_internal(1024 * 1024 * 32, 1);
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    if (ark::get_env().use_msll) {
        UNITTEST(test_reduce_scatter);
    }
    return ark::unittest::SUCCESS;
}
