// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>

#include "logging.h"
#include "model.h"

namespace ark {

extern const OpConfigMap Im2colConfigMap;

Im2colOp::Im2colOp(const std::string &prec_type, Tensor *input, Tensor *output,
                   int kernel_height, int kernel_width, int stride_height,
                   int stride_width, int pad_height, int pad_width,
                   int dilation_height, int dilation_width,
                   const std::string &name)
    : Op{OP_IM2COL,
         prec_type,
         {input},
         {output},
         {{kernel_height, kernel_width, stride_height, stride_width, pad_height,
           pad_width, dilation_height, dilation_width}},
         name,
         &Im2colConfigMap,
         -1,
         true} {}

std::string Im2colOp::function_name(const OpConfig &cfg) const {
    Tensor *input = this->inputs[0];
    Tensor *output = this->outputs[0];

    int ndims = output->shape.ndims();
    OpTile tile_out = cfg.output_tiles[0];
    if (tile_out.x < 0) tile_out.x = output->ldims.dims4()[2];
    if (tile_out.y < 0) tile_out.y = output->ldims.dims4()[3];
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

    Dims unit_out_dims{1, 1, tile_out.x, tile_out.y};
    return Op::function_name("ark::im2col",
                             {{
                                 input->ldims.dims4(),   // InDims
                                 input->shape.dims4(),   // InShape
                                 output->ldims.dims4(),  // OutDims
                                 output->shape.dims4(),  // OutShape
                                 unit_out_dims,          // UnitOutDims
                                 cfg.num_warps,          // NumWarps
                                 cfg.smem_bytes,         // SmemBytes
                                 kernel_height,          // KernelHeight
                                 kernel_width,           // KernelWidth
                                 stride_height,          // StrideHeight
                                 stride_width,           // StrideWidth
                                 pad_height,             // PadHeight
                                 pad_width,              // PadWidth
                                 dilation_height,        // DilationHeight
                                 dilation_width,         // DilationWidth
                             }});
}

Tensor *Model::im2col(Tensor *input, int kernel_height, int kernel_width,
                      int stride_height, int stride_width, int pad_height,
                      int pad_width, int dilation_height, int dilation_width,
                      Tensor *output, const std::string &name) {
    assert(input != nullptr);
    DimType n = 1, c = 1, h = 1, w = 1;
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
        ERR(InvalidUsageError,
            "invalid # of input dimensions. Expected 2, 3, or 4, but given ",
            input_ndims);
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
    Im2colOp op{output->type.name(), input,          output,
                kernel_height,       kernel_width,   stride_height,
                stride_width,        pad_height,     pad_width,
                dilation_height,     dilation_width, name};
    return this->impl->add_op(op)[0];
}

const OpConfigMap Im2colConfigMap = {
    {{OP_ARCH_ANY, "fp16"},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 0, {{1, 1}}, {{128, 128}}, true, false},
         {4, 0, {{1, 1}}, {{64, 128}}, true, false},
         {4, 0, {{1, 1}}, {{128, 64}}, true, false},
         {4, 0, {{1, 1}}, {{64, 64}}, true, false},
     }},
};

}  // namespace ark
