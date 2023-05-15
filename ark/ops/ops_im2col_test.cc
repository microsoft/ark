// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/gpu/gpu_kernel.h"
#include "ark/include/ark.h"
#include "ark/include/ark_utils.h"
#include "ark/logging.h"
#include "ark/unittest/unittest_utils.h"

using namespace std;

//
void test_im2col_internal(ark::DimType n, ark::DimType h, ark::DimType w,
                          ark::DimType c, ark::DimType kernel_height,
                          ark::DimType kernel_width, ark::DimType stride_height,
                          ark::DimType stride_width, ark::DimType pad_height,
                          ark::DimType pad_width, ark::DimType dilation_height,
                          ark::DimType dilation_width)
{
    //
    ark::Model model;
    ark::Tensor *tns_x = model.tensor({n, c, h, w}, ark::FP16);
    ark::Tensor *tns_y = model.im2col(
        tns_x, kernel_height, kernel_width, stride_height, stride_width,
        pad_height, pad_width, dilation_height, dilation_width);
    UNITTEST_EQ(tns_y->ndims(), 3);

    //
    ark::Executor exe{0, 0, 1, model, "test_im2col"};
    exe.compile();

    // Set data.
    ark::srand();
    auto data_x =
        ark::utils::range_halfs(tns_x->shape_bytes(), 0.00001, 0.00001);
    exe.tensor_memcpy(tns_x, data_x.get(), tns_x->shape_bytes());

    // ark::utils::print_matrix(data_x.get(), h * w, c, h * w, c);

    exe.launch();
    exe.run(1);
    exe.stop();

    // Copy results of the loop kernel routine into CPU memory.
    ark::half_t *res =
        (ark::half_t *)calloc(tns_y->shape.size(), sizeof(ark::half_t));
    UNITTEST_NE(res, (ark::half_t *)nullptr);
    exe.tensor_memcpy(res, tns_y, tns_y->shape_bytes());

    // Calculate CPU results
    ark::half_t *gt = (ark::half_t *)calloc(
        tns_y->shape_bytes() / sizeof(ark::half_t), sizeof(ark::half_t));
    UNITTEST_NE(gt, (ark::half_t *)nullptr);
    ark::DimType patch_num_height =
        (h - kernel_height + 2 * pad_height) / stride_height + 1;
    ark::DimType patch_num_width =
        (w - kernel_width + 2 * pad_width) / stride_width + 1;
    ark::DimType mdim = patch_num_height * patch_num_width;
    ark::DimType inner_dim = kernel_height * kernel_width * c;
    for (ark::DimType nidx = 0; nidx < inner_dim; ++nidx) {
        for (ark::DimType midx = 0; midx < patch_num_height * patch_num_width;
             ++midx) {
            ark::DimType channel_idx = nidx / (kernel_height * kernel_width);
            ark::DimType per_channel_patch_idx = midx;
            ark::DimType per_channel_patch_pos_width =
                (per_channel_patch_idx % patch_num_width) * stride_width;
            ark::DimType per_channel_patch_pos_height =
                (per_channel_patch_idx / patch_num_width) * stride_height;
            ark::DimType per_patch_elem_idx =
                nidx % (kernel_height * kernel_width);
            ark::DimType per_patch_elem_pos_width =
                per_patch_elem_idx % kernel_width;
            ark::DimType per_patch_elem_pos_height =
                per_patch_elem_idx / kernel_width;
            ark::DimType elem_width = per_channel_patch_pos_width +
                                      per_patch_elem_pos_width - pad_width;
            ark::DimType elem_height = per_channel_patch_pos_height +
                                       per_patch_elem_pos_height - pad_height;

            if (elem_height < 0 || elem_height >= h || elem_width < 0 ||
                elem_width >= w) {
                gt[midx + nidx * mdim] = ark::half_t(0);
            } else {
                ark::DimType elem_idx =
                    elem_width + elem_height * w + channel_idx * h * w;
                gt[midx + nidx * mdim] = data_x.get()[elem_idx];
            }
        }
    }

    // Compare results with the ground truth.
    auto p = ark::utils::cmp_matrix((ark::half_t *)gt, (ark::half_t *)res, mdim,
                                    inner_dim, n, mdim, inner_dim);
    float max_err = p.second;
    stringstream ss;
    ss << "im2col:n=" << n << ",c=" << c << ",h=" << h << ",w=" << w
       << ",kh=" << kernel_height << ",kw=" << kernel_width
       << ",sh=" << stride_height << ",sw=" << stride_width
       << ",ph=" << pad_height << ",pw=" << pad_width
       << ",dh=" << dilation_height << ",dw=" << dilation_width
       << setprecision(4) << " mse " << p.first << " max_err " << max_err * 100
       << "%";
    LOG(ark::INFO, ss.str());

    free(res);
    free(gt);

    UNITTEST_EQ(max_err, 0.0);
}

ark::unittest::State test_im2col()
{
    test_im2col_internal(1, 2, 2, 2, 2, 2, 1, 1, 0, 0, 1, 1);
    test_im2col_internal(1, 4, 4, 3, 2, 2, 1, 1, 0, 0, 1, 1);
    test_im2col_internal(1, 4, 4, 15, 2, 2, 1, 1, 0, 0, 1, 1);
    test_im2col_internal(1, 4, 4, 16, 2, 2, 1, 1, 0, 0, 1, 1);
    test_im2col_internal(1, 4, 4, 17, 2, 2, 1, 1, 0, 0, 1, 1);
    test_im2col_internal(1, 4, 4, 64, 2, 2, 1, 1, 0, 0, 1, 1);

    test_im2col_internal(1, 7, 7, 3, 2, 2, 1, 1, 0, 0, 1, 1);
    test_im2col_internal(1, 8, 8, 3, 2, 2, 1, 1, 0, 0, 1, 1);
    test_im2col_internal(1, 9, 9, 3, 2, 2, 1, 1, 0, 0, 1, 1);
    test_im2col_internal(1, 64, 64, 3, 2, 2, 1, 1, 0, 0, 1, 1);

    test_im2col_internal(1, 4, 4, 3, 3, 3, 1, 1, 0, 0, 1, 1);
    test_im2col_internal(1, 8, 8, 3, 3, 3, 1, 1, 0, 0, 1, 1);
    test_im2col_internal(1, 64, 64, 3, 7, 7, 1, 1, 0, 0, 1, 1);

    test_im2col_internal(1, 4, 4, 3, 2, 2, 1, 1, 1, 1, 1, 1);

    test_im2col_internal(1, 8, 8, 3, 3, 3, 1, 1, 0, 0, 1, 1);

    test_im2col_internal(1, 256, 256, 3, 3, 3, 1, 1, 0, 0, 1, 1);
    test_im2col_internal(1, 97, 97, 13, 5, 5, 1, 1, 0, 0, 1, 1);

    test_im2col_internal(1, 256, 256, 3, 3, 3, 1, 1, 1, 1, 1, 1);
    test_im2col_internal(1, 97, 97, 13, 5, 5, 1, 1, 2, 2, 1, 1);

    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_im2col);
    return ark::unittest::SUCCESS;
}
