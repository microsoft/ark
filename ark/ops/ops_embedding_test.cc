// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>
#include <type_traits>

#include "include/ark.h"
#include "include/ark_utils.h"
#include "ops_test_common.h"
#include "unittest/unittest_utils.h"

template <typename T>
void baseline_embedding(std::vector<void *> &outputs,
                        const std::vector<ark::Dims> &output_shapes,
                        const std::vector<void *> &inputs,
                        const std::vector<ark::Dims> &input_shapes) {
    T *out = static_cast<T *>(outputs[0]);
    int *in = static_cast<int *>(inputs[0]);
    T *weight = static_cast<T *>(inputs[1]);

    ark::Dims osh = output_shapes[0].dims4();
    ark::Dims wsh = input_shapes[1].dims4();

    assert(osh[3] == wsh[3]);

    int in_idx = 0;
    for (ark::DimType n = 0; n < osh[0]; ++n) {
        for (ark::DimType c = 0; c < osh[1]; ++c) {
            for (ark::DimType h = 0; h < osh[2]; ++h) {
                int weight_idx = in[in_idx++];
                T *ptr = &weight[weight_idx * wsh[3]];
                for (ark::DimType w = 0; w < osh[3]; ++w) {
                    out[n * osh[1] * osh[2] * osh[3] + c * osh[2] * osh[3] +
                        h * osh[3] + w] = ptr[w];
                }
            }
        }
    }
};

template <typename T>
ark::unittest::State test_embedding() {
    const int num_emb = 1000;
    const int emb_dim = 8192;

    const ark::TensorType *weight_type;
    if (std::is_same<T, float>::value) {
        weight_type = &ark::FP32;
    } else {
        weight_type = &ark::FP16;
    }

    ark::Model m;
    ark::Tensor *ti = m.tensor(ark::Dims(8, 3, 64), ark::INT32);
    ark::Tensor *tw = m.tensor(ark::Dims(num_emb, emb_dim), *weight_type);
    ark::Tensor *to = m.embedding(ti, tw);

    ark::srand();

    std::vector<int> ti_data;
    for (auto i = 0; i < ti->shape.size(); ++i) {
        // Random indices in [0, num_emb)
        ti_data.push_back(ark::rand() % num_emb);
    }
    auto tw_data = ark::utils::rand_array<T>(tw->shape.size(), 1.0);
    auto result =
        ark::op_test("embedding_fp32", m, {ti, tw}, {to}, baseline_embedding<T>,
                     {ti_data.data(), tw_data.get()});
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_embedding_fp32() { return test_embedding<float>(); }

ark::unittest::State test_embedding_fp16() {
    return test_embedding<ark::half_t>();
}

ark::unittest::State test_embedding_bf16() {
    return test_embedding<ark::bfloat16_t>();
}

int main() {
    ark::init();
    UNITTEST(test_embedding_fp32);
    UNITTEST(test_embedding_fp16);
    UNITTEST(test_embedding_bf16);
    return ark::unittest::SUCCESS;
}
