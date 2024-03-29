// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>
#include <cassert>

#include "logging.h"
#include "model.h"
#include "tensor.h"

namespace ark {

ReshapeOp::ReshapeOp(const std::string &prec_type, Tensor *input,
                     Tensor *output, const std::string &name)
    : Op{OP_RESHAPE, prec_type, {input}, {output}, {},
         name,       nullptr,   -1,      true} {}

// Reshape `input` to `shape`. This interface does not support -1 as a dimension
// of `shape`, because Dims does not allow -1 as a valid dimension.
static Tensor *_reshape(Model *model, Tensor *input, const Dims &shape,
                        bool allowzero, Tensor *output, const std::string &) {
    if (input == nullptr) {
        ERR(InvalidUsageError, "input is null");
    }
    // Infer the actual shape
    std::vector<DimType> inferred_shape;
    if (shape.ndims() == 0) {
        // Convert to a scalar
        inferred_shape.emplace_back(1);
        if (input->shape.size() != 1) {
            ERR(InvalidUsageError, "number of elements mismatch: reshape from ",
                input->shape, " to ", shape);
        }
    } else {
        DimType total_size = 1;
        for (int i = 0; i < shape.ndims(); i++) {
            if (shape[i] == 0) {
                if (allowzero) {
                    inferred_shape.push_back(0);
                    total_size = 0;
                } else {
                    inferred_shape.push_back(input->shape[i]);
                    total_size *= input->shape[i];
                }
            } else {
                assert(shape[i] > 0);
                inferred_shape.push_back(shape[i]);
                total_size *= shape[i];
            }
        }
        if (input->shape.size() != total_size) {
            ERR(InvalidUsageError, "number of elements mismatch: reshape from ",
                input->shape, " to ", shape);
        }
    }
    Dims new_shape{inferred_shape};

    std::stringstream ss;
    ss << "reshape failed as the ldims of the input tensor is incompatible "
          "with the new shape. A workaround is copying the input tensor to a "
          "new tensor, so that the data becomes sequential in memory. ";
    ss << "Input shape " << input->shape << ", ldims " << input->ldims
       << ", new shape " << new_shape;
    auto incompatible_ldims_error = ss.str();

    Dims new_ldims;
    Dims new_offs;
    if (!tensor_reshape_helper(input->shape, input->ldims, input->offs,
                               new_shape, new_ldims, new_offs)) {
        ERR(ModelError, incompatible_ldims_error);
    }
    if (output != nullptr) {
        // Verfiy given `output`
        if (input->type != output->type) {
            ERR(InvalidUsageError, "invalid output data type: ", output->type);
        }
        if (input->shape.size() != output->shape.size()) {
            ERR(InvalidUsageError, "shape sizes mismatch: input ", input->shape,
                ", output ", output->shape);
        }
    } else {
        output = model->tensor(new_shape, input->type, input->buf, new_ldims,
                               new_offs);
    }
    return output;
}

//
Tensor *Model::reshape(Tensor *input, const Dims &shape, bool allowzero,
                       Tensor *output, const std::string &name) {
    output = _reshape(this, input, shape, allowzero, output, name);
    ReshapeOp op{"none", input, output, name};
    return this->impl->add_op(op)[0];
}

Tensor *Model::reshape(Tensor *input,
                       const std::initializer_list<DimType> &shape,
                       bool allowzero, Tensor *output,
                       const std::string &name) {
    std::vector<DimType> shape_vec{shape};
    return this->reshape(input, shape_vec, allowzero, output, name);
}

// Reshape `input` to `shape`. If one dimension of `shape` is -1, it will be
// inferred from the `input`. If one dimension of `shape` is 0, by default
// (`allowzero` is false), that dimension is unchanged from the corresponding
// one of `input`. If `allowzero` is true, that dimension is set to 0, which
// means that the reshaped tensor is an empty tensor, i.e., `input` should also
// be an empty tensor. If `allowzero` is true, `shape` should not include both
// 0 and -1 at the same time. If `shape` is an empty vector, `input` will be
// converted to a scalar.
Tensor *Model::reshape(Tensor *input, const std::vector<DimType> &shape,
                       bool allowzero, Tensor *output,
                       const std::string &name) {
    if (input == nullptr) {
        ERR(InvalidUsageError, "input is null");
    }
    std::vector<DimType> shape_vec{shape};
    // Infer -1 dimension if exists
    int neg_idx = -1;
    bool zero_exists = false;
    std::vector<DimType> inferred_shape;
    DimType total_size = 1;
    for (size_t i = 0; i < shape_vec.size(); i++) {
        if (shape_vec[i] == -1) {
            if (neg_idx != -1) {
                ERR(InvalidUsageError,
                    "multiple -1 in shape: ", Dims(shape_vec));
            }
            neg_idx = (int)i;
        } else if (shape_vec[i] < 0) {
            ERR(InvalidUsageError,
                "shape cannot include negative values except -1. "
                "Given: ",
                Dims(shape_vec));
        } else {
            if (shape_vec[i] == 0) {
                zero_exists = true;
            }
            total_size *= shape_vec[i];
        }
        inferred_shape.push_back(shape_vec[i]);
    }
    if (neg_idx != -1) {
        if (zero_exists) {
            ERR(InvalidUsageError,
                "shape cannot include both 0 and -1 at the same "
                "time. Given: ",
                Dims(shape_vec));
        }
        // total_size is always positive at this point.
        // Infer the -1 dimension
        if (input->shape.size() % total_size != 0) {
            ERR(InvalidUsageError, "number of elements mismatch: reshape from ",
                input->shape, " to ", Dims(shape_vec));
        }
        inferred_shape[neg_idx] = input->shape.size() / total_size;
    } else if (!zero_exists && input->shape.size() != total_size) {
        ERR(InvalidUsageError, "number of elements mismatch: reshape from ",
            input->shape, " to ", Dims(shape_vec));
    }
    output =
        _reshape(this, input, Dims{inferred_shape}, allowzero, output, name);
    ReshapeOp op{"none", input, output, name};
    return this->impl->add_op(op)[0];
}

}  // namespace ark
