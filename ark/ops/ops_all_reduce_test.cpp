// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model/model_node.hpp"
#include "model/model_op.hpp"
#include "ops_test_common.hpp"

template <typename T, int NumGpus>
void baseline_all_reduce(std::vector<void *> &outputs,
                         const std::vector<ark::Dims> &output_shapes,
                         const std::vector<void *> &,
                         const std::vector<ark::Dims> &, int) {
    // Calculate sum from 1 to NumGpus.
    T expected = 0;
    for (int i = 1; i <= NumGpus; ++i) {
        expected += T(i);
    }

    T *out = static_cast<T *>(outputs[0]);
    for (ark::DimType i = 0; i < output_shapes[0].nelems(); ++i) {
        out[i] = expected;
    }
}

template <int NumGpus>
void test_all_reduce_internal(ark::DimType nelem) {
    for (int gpu_id = 0; gpu_id < NumGpus; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id, nelem]() {
            // Each GPU's data is equal to its GPU ID + 1.
            ark::Model m(gpu_id, NumGpus);
            ark::Tensor ones = m.tensor({nelem}, ark::FP16);
            ark::Tensor data = m.mul(ones, float(gpu_id + 1));
            ark::Tensor output = m.all_reduce(data, gpu_id, NumGpus);

            std::vector<ark::half_t> ones_vec(ones.shape().nelems(),
                                              ark::half_t(1.0f));
            auto result =
                ark::op_test("all_reduce", m, {ones}, {output},
                             baseline_all_reduce<ark::half_t, NumGpus>,
                             {ones_vec.data()}, false, gpu_id, NumGpus);
            UNITTEST_LOG(result);
            UNITTEST_EQ(result.max_diff[0], 0.0f);
            return ark::unittest::SUCCESS;
        });
    }
    ark::unittest::wait_all_processes();
}

// void test_local_all_reduce_8gpus_internel(size_t nelem, int iter) {
//     constexpr int num_gpus = 8;
//     for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
//         ark::unittest::spawn_process([gpu_id, nelem, num_gpus, iter]() {
//             // Each GPU's data is equal to its GPU ID + 1.
//             ark::Model m(gpu_id, num_gpus);
//             ark::Tensor data = m.tensor({nelem}, ark::FP16);
//             std::vector<ark::half_t> data_buf(nelem);
//             for (size_t i = 0; i < nelem; ++i) {
//                 data_buf[i] = ark::half_t(gpu_id + 1);
//             }
//             ark::Tensor *output = m.local_all_reduce(data, gpu_id, num_gpus);
//             auto result =
//                 ark::op_test("all_reduce", m, {data}, {output},
//                              baseline_all_reduce<ark::half_t, 8>,
//                              {data_buf.data()}, true, gpu_id, num_gpus, 16);
//             UNITTEST_LOG(result);
//             UNITTEST_EQ(result.max_diff[0], 0.0f);
//             return ark::unittest::SUCCESS;
//         });
//     }
//     ark::unittest::wait_all_processes();
// }

// void test_local_all_reduce_packet_8gpus_internel(size_t nelem, int iter) {
//     constexpr int num_gpus = 8;
//     for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
//         ark::unittest::spawn_process([gpu_id, nelem, num_gpus, iter]() {
//             // Each GPU's data is equal to its GPU ID + 1.
//             ark::Model m{gpu_id};
//             ark::Tensor *data = m.tensor(ark::Dims(nelem), ark::FP16);
//             std::vector<ark::half_t> data_buf(nelem);
//             for (size_t i = 0; i < nelem; ++i) {
//                 data_buf[i] = ark::half_t(gpu_id + 1);
//             }
//             ark::Tensor *output =
//                 m.local_all_reduce_packet(data, gpu_id, num_gpus);
//             auto result =
//                 ark::op_test("all_reduce_packet", m, {data}, {output},
//                              baseline_all_reduce<ark::half_t, 8>,
//                              {data_buf.data()}, false, gpu_id, num_gpus, 16);
//             UNITTEST_LOG(result);
//             UNITTEST_EQ(result.max_diff[0], 0.0f);
//             return ark::unittest::SUCCESS;
//         });
//     }
//     ark::unittest::wait_all_processes();
// }

ark::unittest::State test_all_reduce_4gpus() {
    test_all_reduce_internal<4>(64);
    test_all_reduce_internal<4>(8192);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_all_reduce_8gpus() {
    test_all_reduce_internal<8>(64);
    test_all_reduce_internal<8>(8192);
    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_all_reduce_4gpus);
    UNITTEST(test_all_reduce_8gpus);
    return 0;
}
