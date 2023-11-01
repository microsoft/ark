// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>
#include <cassert>

#include "logging.h"
#include "model.h"

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
        LOG(ERROR, "input is null");
    }
    // Infer the actual shape
    std::vector<DimType> inferred_shape;
    if (shape.ndims() == 0) {
        // Convert to a scalar
        inferred_shape.emplace_back(1);
        if (input->shape.size() != 1) {
            LOG(ERROR, "number of elements mismatch: reshape from ",
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
            LOG(ERROR, "number of elements mismatch: reshape from ",
                input->shape, " to ", shape);
        }
    }
    Dims new_shape{inferred_shape};

    // Infer the new ldims and offs
    std::vector<DimType> reverse_ldims;
    std::vector<DimType> reverse_offs;

    std::stringstream ss;
    ss << "reshape failed as the ldims of the input tensor is incompatible "
          "with the new shape. A workaround is copying the input tensor to a "
          "new tensor, so that the data becomes sequential in memory. ";
    ss << "Input shape " << input->shape << ", ldims " << input->ldims
       << ", new shape " << new_shape;
    auto incompatible_ldims_error = ss.str();

    int orig_idx = input->shape.ndims() - 1;
    int new_idx = new_shape.ndims() - 1;
    DimType orig_shape_stack = input->shape[orig_idx];
    DimType new_shape_stack = new_shape[new_idx];
    DimType orig_ldim_stack = input->ldims[orig_idx];
    DimType div_stack = 1;
    while (orig_idx >= 0 && new_idx >= 0) {
        if (orig_shape_stack == new_shape_stack) {
            if (orig_ldim_stack % div_stack != 0) {
                LOG(ERROR, incompatible_ldims_error);
            }
            DimType new_off = input->offs[orig_idx];
            for (auto i = orig_idx + 1; i < input->ldims.ndims(); i++) {
                new_off *= input->ldims[i];
            }
            std::for_each(reverse_ldims.begin(), reverse_ldims.end(),
                          [&new_off](DimType d) { new_off /= d; });

            reverse_ldims.push_back(orig_ldim_stack / div_stack);
            reverse_offs.push_back(new_off);
            div_stack = 1;
            new_idx--;
            orig_idx--;
            if (new_idx >= 0) {
                new_shape_stack = new_shape[new_idx];
            }
            if (orig_idx >= 0) {
                orig_shape_stack = input->shape[orig_idx];
                orig_ldim_stack = input->ldims[orig_idx];
            }
        } else if (orig_shape_stack > new_shape_stack) {
            div_stack *= new_shape[new_idx];
            reverse_ldims.push_back(new_shape[new_idx]);
            reverse_offs.push_back(0);
            new_idx--;
            if (new_idx >= 0) {
                new_shape_stack *= new_shape[new_idx];
            }
        } else {
            if (input->ldims[orig_idx] != input->shape[orig_idx] ||
                input->offs[orig_idx] != 0) {
                LOG(ERROR, incompatible_ldims_error);
            }
            orig_idx--;
            if (orig_idx >= 0) {
                orig_shape_stack *= input->shape[orig_idx];
                orig_ldim_stack *= input->ldims[orig_idx];
            }
        }
    }
    while (new_idx >= 0 && new_shape[new_idx] == 1) {
        reverse_ldims.push_back(1);
        reverse_offs.push_back(0);
        new_idx--;
    }
    while (orig_idx >= 0 && input->shape[orig_idx] == 1) {
        if (input->ldims[orig_idx] != input->shape[orig_idx] ||
            input->offs[orig_idx] != 0) {
            LOG(ERROR, incompatible_ldims_error);
        }
        orig_idx--;
    }
    if (orig_idx >= 0 || new_idx >= 0) {
        LOG(ERROR, incompatible_ldims_error);
    }

    std::reverse(reverse_ldims.begin(), reverse_ldims.end());
    std::reverse(reverse_offs.begin(), reverse_offs.end());
    Dims new_ldims{reverse_ldims};
    Dims new_offs{reverse_offs};

    if (output != nullptr) {
        // Verfiy given `output`
        if (input->type != output->type) {
            LOG(ERROR, "invalid output data type: ", output->type);
        }
        if (input->shape.size() != output->shape.size()) {
            LOG(ERROR, "shape sizes mismatch: input ", input->shape,
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
        LOG(ERROR, "input is null");
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
                LOG(ERROR, "multiple -1 in shape: ", Dims(shape_vec));
            }
            neg_idx = (int)i;
        } else if (shape_vec[i] < 0) {
            LOG(ERROR,
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
            LOG(ERROR,
                "shape cannot include both 0 and -1 at the same "
                "time. Given: ",
                Dims(shape_vec));
        }
        // Infer the -1 dimension
        if (total_size <= 0) {
            LOG(ERROR, "Unexpected error");
        }
        if (input->shape.size() % total_size != 0) {
            LOG(ERROR, "number of elements mismatch: reshape from ",
                input->shape, " to ", Dims(shape_vec));
        }
        inferred_shape[neg_idx] = input->shape.size() / total_size;
    } else if (!zero_exists && input->shape.size() != total_size) {
        LOG(ERROR, "number of elements mismatch: reshape from ", input->shape,
            " to ", Dims(shape_vec));
    }
    output =
        _reshape(this, input, Dims{inferred_shape}, allowzero, output, name);
    ReshapeOp op{"none", input, output, name};
    return this->impl->add_op(op)[0];
}

}  // namespace ark
