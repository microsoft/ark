// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>
#include <type_traits>

#include "ark/random.hpp"
#include "ops_test_common.hpp"

template <typename T>
void baseline_embedding(std::vector<void *> &outputs,
                        const std::vector<ark::Dims> &output_shapes,
                        const std::vector<void *> &inputs,
                        const std::vector<ark::Dims> &input_shapes, int) {
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
                if (weight_idx < 0) {
                    weight_idx += wsh[2];
                }
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
    const int num_emb = 100;
    const int emb_dim = 4096;

    ark::DataType weight_type;
    if (std::is_same<T, float>::value) {
        weight_type = ark::FP32;
    } else {
        weight_type = ark::FP16;
    }

    ark::Model m;
    ark::Tensor ti = m.tensor(ark::Dims(8, 3, 64), ark::INT32);
    ark::Tensor tw = m.tensor(ark::Dims(num_emb, emb_dim), weight_type);
    ark::Tensor to = m.embedding(ti, tw);

    std::vector<int> ti_data;
    for (auto i = 0; i < ti.shape().nelems(); ++i) {
        // Random indices in [0, num_emb)
        int rand_idx = ark::rand() % num_emb;
        if (i % 9 == 0) {
            // test negative tokens (padding)
            rand_idx = -rand_idx;
        }
        ti_data.push_back(rand_idx);
    }
    std::vector<T> tw_data(tw.shape().nelems());
    for (auto i = 0; i < tw.shape().nelems(); ++i) {
        tw_data[i] = ark::random<T>(-1.0, 1.0);
    }
    std::string type_str = "";
    if (std::is_same<T, float>::value) {
        type_str = "fp32";
    } else if (std::is_same<T, ark::half_t>::value) {
        type_str = "fp16";
    } else if (std::is_same<T, ark::bfloat16_t>::value) {
        type_str = "bf16";
    }
    auto result =
        ark::op_test("embedding_" + type_str, m, {ti, tw}, {to},
                     baseline_embedding<T>, {ti_data.data(), tw_data.data()});
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

ark::unittest::State test_embedding_invalid() {
    {
        ark::Model m;
        ark::Tensor ti = m.tensor(ark::Dims(4, 8, 3, 64), ark::INT32);
        ark::Tensor tw = m.tensor(ark::Dims(100, 1024), ark::FP32);
        UNITTEST_THROW(m.embedding(ti, tw), ark::ModelError);
    }
    {
        ark::Model m;
        ark::Tensor ti = m.tensor(ark::Dims(8, 3, 64), ark::INT32);
        ark::Tensor tw = m.tensor(ark::Dims(2, 100, 1024), ark::FP32);
        UNITTEST_THROW(m.embedding(ti, tw), ark::ModelError);
    }
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_embedding_fp32);
    UNITTEST(test_embedding_fp16);
    UNITTEST(test_embedding_bf16);
    UNITTEST(test_embedding_invalid);
    return ark::unittest::SUCCESS;
}
