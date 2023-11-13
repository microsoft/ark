// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>

#include "logging.h"
#include "model.h"

namespace ark {

extern const OpConfigMap CastConfigMap;

CastOp::CastOp(Tensor *input, Tensor *output, const std::string &name)
    : Op{OP_CAST, "none",         {input}, {output}, {},
         name,    &CastConfigMap, -1,      true} {}

std::string CastOp::function_name(const OpConfig &cfg) const {
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

    Dims unit_out_dims{1, 1, tile_out.x, tile_out.y};
    return Op::function_name("ark::cast",
                             {{
                                 input->ldims.dims4(),   // InDims
                                 input->shape.dims4(),   // InShape
                                 output->ldims.dims4(),  // OutDims
                                 output->shape.dims4(),  // OutShape
                                 unit_out_dims,          // UnitOutDims
                                 cfg.num_warps,          // NumWarps
                             }});
}

Tensor *Model::cast(Tensor *input, const TensorType &ttype, Tensor *output,
                    const std::string &name) {
    assert(input != nullptr);
    if (output == nullptr) {
        if (input->type == ttype) {
            // Casting to the same type is considered as an identity,
            // only when the output tensor is not specified.
            return this->identity(input, {}, name);
        }
        if (input->type == BYTE) {
            // Casting BYTE to other types is considered as a reshape.
            if (input->shape_bytes() < ttype.bytes()) {
                ERR(InvalidUsageError,
                    "input tensor is too small to be casted to ", ttype);
            }
            // The last greater-than-1 dimension of the input tensor should be
            // divisible by the size of the output type.
            int last_dim = input->shape.ndims() - 1;
            for (; last_dim >= 0; --last_dim) {
                if (last_dim == 0 || input->ldims[last_dim] > 1) {
                    break;
                }
            }
            if ((input->shape[last_dim] % ttype.bytes()) != 0) {
                ERR(InvalidUsageError,
                    "the last greater-than-1 dimension of the "
                    "input tensor shape ",
                    input->shape[last_dim],
                    " is not divisible by the size of the output "
                    "tensor type (",
                    ttype.bytes(), ")");
            }
            if ((input->ldims[last_dim] % ttype.bytes()) != 0) {
                ERR(InvalidUsageError,
                    "the last greater-than-1 dimension of the "
                    "input tensor ldims ",
                    input->ldims[last_dim],
                    " is not divisible by the size of the output "
                    "tensor type (",
                    ttype.bytes(), ")");
            }
            if ((input->offs[last_dim] % ttype.bytes()) != 0) {
                ERR(InvalidUsageError,
                    "the last greater-than-1 dimension of the "
                    "input tensor offs ",
                    input->offs[last_dim],
                    " is not divisible by the size of the output "
                    "tensor type (",
                    ttype.bytes(), ")");
            }
            if (input->pads[last_dim] > 1) {
                // we can ignore pads if it is 1
                if ((input->pads[last_dim] % ttype.bytes()) != 0) {
                    ERR(InvalidUsageError,
                        "the last greater-than-1 dimension of the "
                        "input tensor pads ",
                        input->pads[last_dim],
                        " is not divisible by the size of the output "
                        "tensor type (",
                        ttype.bytes(), ")");
                }
            }

            auto out_shape = input->shape;
            auto out_ldims = input->ldims;
            auto out_offs = input->offs;
            auto out_pads = input->pads;
            out_shape[last_dim] /= ttype.bytes();
            out_ldims[last_dim] /= ttype.bytes();
            out_offs[last_dim] /= ttype.bytes();
            if (out_pads[last_dim] > 1) {
                out_pads[last_dim] /= ttype.bytes();
            }
            return this->tensor(out_shape, ttype, input->buf, out_ldims,
                                out_offs, out_pads, {input}, input->exported,
                                input->imported_rank, name + "/cast");
        }
        if (ttype == BYTE) {
            // Casting other types to BYTE is considered as a reshape.
            auto out_shape = input->shape;
            auto out_ldims = input->ldims;
            auto out_offs = input->offs;
            auto out_pads = input->pads;
            out_shape[-1] *= input->type.bytes();
            out_ldims[-1] *= input->type.bytes();
            out_offs[-1] *= input->type.bytes();
            if (out_pads[-1] > 1) {
                out_pads[-1] *= input->type.bytes();
            }
            return this->tensor(out_shape, ttype, input->buf, out_ldims,
                                out_offs, out_pads, {input}, input->exported,
                                input->imported_rank, name + "/cast");
        }
        output = this->tensor(input->shape, ttype);
    } else {
        if (output->type != ttype) {
            ERR(InvalidUsageError, "invalid output data type: ", output->type);
        }
        if (output->shape != input->shape) {
            ERR(InvalidUsageError, "invalid output shape: ", output->shape);
        }
        if (input->type == ttype) {
            ERR(InvalidUsageError, "casting to the same type: ", ttype);
        }
        if (ttype == BYTE) {
            ERR(InvalidUsageError,
                "casting to BYTE with a specified output tensor is not "
                "supported as it implies a memory copy.");
        }
    }
    CastOp op{input, output, name};
    return this->impl->add_op(op)[0];
}

const OpConfigMap CastConfigMap = {
    {{OP_ARCH_ANY, "none"},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 0, {{1, 256}}, {{1, 256}}, false, false},
         {1, 0, {{1, 128}}, {{1, 128}}, false, false},
         {1, 0, {{1, 64}}, {{1, 64}}, false, false},
     }},
};

}  // namespace ark
