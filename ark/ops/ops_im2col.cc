// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"
#include "tensor.h"

using namespace std;

namespace ark {

extern const OpConfigMap Im2colConfigMap;

Im2colOp::Im2colOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                   int kernel_height, int kernel_width, int stride_height,
                   int stride_width, int pad_height, int pad_width,
                   int dilation_height, int dilation_width, const string &name)
    : Op{OP_IM2COL,
         prec_type,
         {input},
         {output},
         {{kernel_height, kernel_width, stride_height, stride_width, pad_height,
           pad_width, dilation_height, dilation_width}},
         name,
         &Im2colConfigMap,
         -1,
         true}
{
}

std::string Im2colOp::function_name(const OpConfig &cfg) const
{
    Tensor *input = this->in_deps[0];
    Tensor *output = this->out_deps[0];

    int ndims = output->shape.ndims();
    const OpTile &tile_out = cfg.out_deps_tiles[0];
    CHECK(output->ldims[ndims - 1] % tile_out.y == 0);
    if (ndims > 1) {
        CHECK(output->ldims[ndims - 2] % tile_out.x == 0);
    } else {
        CHECK(tile_out.x == 1);
    }

    int kernel_height;
    int kernel_width;
    int stride_height;
    int stride_width;
    int pad_height;
    int pad_width;
    int dilation_height;
    int dilation_width;
    this->args.get(&kernel_height, 0);
    this->args.get(&kernel_width, 1);
    this->args.get(&stride_height, 2);
    this->args.get(&stride_width, 3);
    this->args.get(&pad_height, 4);
    this->args.get(&pad_width, 5);
    this->args.get(&dilation_height, 6);
    this->args.get(&dilation_width, 7);

    return Op::function_name("ark::im2col",
                             {{
                                 input->shape.dims4(), // InShape
                                 input->ldims.dims4(), // InDims
                                 kernel_height,        // KernelHeight
                                 kernel_width,         // KernelWidth
                                 stride_height,        // StrideHeight
                                 stride_width,         // StrideWidth
                                 pad_height,           // PadHeight
                                 pad_width,            // PadWidth
                                 dilation_height,      // DilationHeight
                                 dilation_width,       // DilationWidth
                                 cfg.num_warps * 32,   // TN
                                 cfg.smem_bytes,       // SB
                                 tile_out.y,           // TDM
                                 tile_out.x,           // TDN
                                 0,                    // TDK
                             }});
}

Tensor *Model::im2col(Tensor *input, int kernel_height, int kernel_width,
                      int stride_height, int stride_width, int pad_height,
                      int pad_width, int dilation_height, int dilation_width,
                      Tensor *output, const string &name)
{
    assert(input != nullptr);
    DimType n, c, h, w;
    int input_ndims = input->ndims();
    if (input_ndims == 2) {
        n = 1;
        c = 1;
        h = input->shape[0];
        w = input->shape[1];
    } else if (input_ndims == 3) {
        n = 1;
        c = input->shape[0];
        h = input->shape[1];
        w = input->shape[2];
    } else if (input_ndims == 4) {
        n = input->shape[0];
        c = input->shape[1];
        h = input->shape[2];
        w = input->shape[3];
    } else {
        LOGERR("invalid # of input dimensions. Expected 2, 3, or 4, but given ",
               input_ndims);
    }
    OpPrecType pt;
    if (input->type == FP16) {
        pt = OP_PREC_FP16;
    } else if (input->type == FP32) {
        pt = OP_PREC_FP32;
    } else {
        LOGERR("unsupported input data type: ", type_str(input->type));
    }
    DimType out_h = (h + 2 * pad_height - kernel_height) / stride_height + 1;
    DimType out_w = (w + 2 * pad_width - kernel_width) / stride_width + 1;
    assert((out_h > 0) && (out_w > 0));
    DimType out_m = out_h * out_w;
    DimType inner_dim = c * kernel_height * kernel_width;
    Dims out_shape;
    if (input_ndims <= 3) {
        out_shape = {inner_dim, out_m};
    } else {
        out_shape = {n, inner_dim, out_m};
    }
    if (output == nullptr) {
        output = this->tensor(out_shape, input->type);
    } else {
        assert(output->shape == out_shape);
    }
    Im2colOp op{pt,           input,           output,         kernel_height,
                kernel_width, stride_height,   stride_width,   pad_height,
                pad_width,    dilation_height, dilation_width, name};
    this->impl->add_op(op);
    return output;
}

const OpConfigMap Im2colConfigMap = {
    {{OP_ARCH_CUDA_70, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 0, {{1, 1}}, {{128, 128}}, true, false},
         {4, 0, {{1, 1}}, {{64, 128}}, true, false},
         {4, 0, {{1, 1}}, {{128, 64}}, true, false},
         {4, 0, {{1, 1}}, {{64, 64}}, true, false},
     }},
    {{OP_ARCH_CUDA_80, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 0, {{1, 1}}, {{128, 128}}, true, false},
         {4, 0, {{1, 1}}, {{64, 128}}, true, false},
         {4, 0, {{1, 1}}, {{128, 64}}, true, false},
         {4, 0, {{1, 1}}, {{64, 64}}, true, false},
     }},
};

} // namespace ark
