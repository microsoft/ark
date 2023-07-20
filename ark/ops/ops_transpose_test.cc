// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_kernel.h"
#include "include/ark.h"
#include "include/ark_utils.h"
#include "logging.h"
#include "unittest/unittest_utils.h"

template <typename T>
void test_transpose_internal(ark::TensorType type, ark::DimType n,
                             ark::DimType c, ark::DimType h, ark::DimType w,
                             ark::DimType pn, ark::DimType pc, ark::DimType ph,
                             ark::DimType pw)
{
    ark::Model model;
    ark::Tensor *tns_in = model.tensor({n, c, h, w}, type);
    ark::Tensor *tns_out = model.transpose(tns_in, {pn, pc, ph, pw});

    ark::Executor exe{/*gpu_id=*/0, /*rank=*/0, /*world_size=*/1, model,
                      "test_transpose"};
    exe.compile();

    // Set data.
    ark::srand();
    auto data_in = ark::utils::rand_array<T>(n * c * h * w, 0.01);
    T *in_ptr = data_in.get();

    exe.tensor_memcpy(tns_in, in_ptr, n * c * h * w * sizeof(T));

    exe.launch();
    exe.run(1);
    exe.stop();

    // Copy results of the loop kernel routine into CPU memory.
    T *res = (T *)malloc(n * c * h * w * sizeof(T));
    UNITTEST_NE(res, (T *)nullptr);
    exe.tensor_memcpy(res, tns_out, n * c * h * w * sizeof(T));

    // int on = tns_in->shape[pn];
    ark::DimType oc = tns_in->shape[pc];
    ark::DimType oh = tns_in->shape[ph];
    ark::DimType ow = tns_in->shape[pw];

    // Check results.
    for (ark::DimType i = 0; i < n; ++i) {
        for (ark::DimType j = 0; j < c; ++j) {
            for (ark::DimType k = 0; k < h; ++k) {
                for (ark::DimType l = 0; l < w; ++l) {
                    ark::Dims axis{i, j, k, l};
                    ark::Dims new_axis;
                    new_axis[0] = axis[pn];
                    new_axis[1] = axis[pc];
                    new_axis[2] = axis[ph];
                    new_axis[3] = axis[pw];
                    ark::DimType in_idx = i * c * h * w + j * h * w + k * w + l;
                    ark::DimType res_idx = new_axis[0] * oc * oh * ow +
                                           new_axis[1] * oh * ow +
                                           new_axis[2] * ow + new_axis[3];
                    UNITTEST_TRUE(in_idx < n * c * h * w);
                    UNITTEST_TRUE(res_idx < n * c * h * w);
                    UNITTEST_EQ(in_ptr[in_idx], res[res_idx]);
                }
            }
        }
    }
}

ark::unittest::State test_transpose_fp32()
{
    test_transpose_internal<float>(ark::FP32, 3, 2048, 96, 128, 0, 2, 1, 3);
    test_transpose_internal<ark::half_t>(ark::FP16, 1, 1, 64, 32, 0, 1, 3, 2);

    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_transpose_fp32);
    return ark::unittest::SUCCESS;
}
