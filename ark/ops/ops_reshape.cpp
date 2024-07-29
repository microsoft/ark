// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_reshape.hpp"

#include <algorithm>
#include <cassert>

#include "logging.h"
#include "ops_common.hpp"

namespace ark {

// Reshape `input` to `shape`. This interface does not support -1 as a dimension
// of `shape`, because Dims does not allow -1 as a valid dimension.
static void reshape_helper(ModelTensorRef input, const Dims &inferred_shape,
                           bool allowzero, Dims &new_shape, Dims &new_strides,
                           Dims &new_offs) {
    const auto &orig_shape = input->shape();
    const auto &orig_strides = input->strides();
    const auto &orig_offsets = input->offsets();
    // Calculate the new shape
    std::vector<DimType> new_shape_vec;
    if (inferred_shape.ndims() == 0) {
        // Convert to a scalar
        new_shape_vec.emplace_back(1);
        if (orig_shape.nelems() != 1) {
            ERR(InvalidUsageError, "number of elements mismatch: reshape from ",
                orig_shape, " to ", inferred_shape);
        }
    } else {
        DimType total_size = 1;
        for (int i = 0; i < inferred_shape.ndims(); i++) {
            if (inferred_shape[i] == 0) {
                if (allowzero) {
                    new_shape_vec.push_back(0);
                    total_size = 0;
                } else {
                    new_shape_vec.push_back(orig_shape[i]);
                    total_size *= orig_shape[i];
                }
            } else {
                assert(inferred_shape[i] > 0);
                new_shape_vec.push_back(inferred_shape[i]);
                total_size *= inferred_shape[i];
            }
        }
        if (orig_shape.nelems() != total_size) {
            ERR(InvalidUsageError, "number of elements mismatch: reshape from ",
                orig_shape, " to ", inferred_shape);
        }
    }
    new_shape = new_shape_vec;

    std::stringstream ss;
    ss << "reshape failed as the strides of the input tensor is incompatible "
          "with the new shape. A workaround is copying the input tensor to a "
          "new tensor, so that the data becomes sequential in memory. ";
    ss << "Input shape " << orig_shape << ", strides " << orig_strides
       << ", new shape " << new_shape;
    auto incompatible_strides_error = ss.str();

    // Infer the new strides and offs
    std::vector<DimType> reverse_strides;
    std::vector<DimType> reverse_offsets;

    int orig_idx = orig_shape.ndims() - 1;
    int new_idx = new_shape.ndims() - 1;
    DimType orig_shape_stack = orig_shape[orig_idx];
    DimType new_shape_stack = new_shape[new_idx];
    DimType orig_strides_stack = orig_strides[orig_idx];
    DimType div_stack = 1;
    while (orig_idx >= 0 && new_idx >= 0) {
        if (orig_shape_stack == new_shape_stack) {
            if (orig_strides_stack % div_stack != 0) {
                ERR(ModelError, incompatible_strides_error);
            }
            DimType new_off = orig_offsets[orig_idx];
            for (auto i = orig_idx + 1; i < orig_strides.ndims(); i++) {
                new_off *= orig_strides[i];
            }
            std::for_each(reverse_strides.begin(), reverse_strides.end(),
                          [&new_off](DimType d) { new_off /= d; });
            reverse_strides.push_back(orig_strides_stack / div_stack);
            reverse_offsets.push_back(new_off);
            div_stack = 1;
            new_idx--;
            orig_idx--;
            if (new_idx >= 0) {
                new_shape_stack = new_shape[new_idx];
            }
            if (orig_idx >= 0) {
                orig_shape_stack = orig_shape[orig_idx];
                orig_strides_stack = orig_strides[orig_idx];
            }
        } else if (orig_shape_stack > new_shape_stack) {
            div_stack *= new_shape[new_idx];
            reverse_strides.push_back(new_shape[new_idx]);
            reverse_offsets.push_back(0);
            new_idx--;
            if (new_idx >= 0) {
                new_shape_stack *= new_shape[new_idx];
            }
        } else {
            if (orig_strides[orig_idx] != orig_shape[orig_idx] ||
                orig_offsets[orig_idx] != 0) {
                ERR(ModelError, incompatible_strides_error);
            }
            orig_idx--;
            if (orig_idx >= 0) {
                orig_shape_stack *= orig_shape[orig_idx];
                orig_strides_stack *= orig_strides[orig_idx];
            }
        }
    }
    while (new_idx >= 0 && new_shape[new_idx] == 1) {
        reverse_strides.push_back(1);
        reverse_offsets.push_back(0);
        new_idx--;
    }
    while (orig_idx >= 0 && orig_shape[orig_idx] == 1) {
        if (orig_strides[orig_idx] != orig_shape[orig_idx] ||
            orig_offsets[orig_idx] != 0) {
            ERR(ModelError, incompatible_strides_error);
        }
        orig_idx--;
    }
    if (orig_idx >= 0 || new_idx >= 0) {
        ERR(ModelError, incompatible_strides_error);
    }
    std::reverse(reverse_strides.begin(), reverse_strides.end());
    std::reverse(reverse_offsets.begin(), reverse_offsets.end());
    new_strides = reverse_strides;
    new_offs = reverse_offsets;
}

ModelOpReshape::ModelOpReshape(ModelTensorRef input, const Dims &shape,
                               const Dims &strides, const Dims &offsets)
    : ModelOpTensor(input->buffer(), shape, input->data_type(), strides,
                    offsets, {}) {
    read_tensors_ = {input};
    verify();
}

// Reshape `input` to `shape`. If one dimension of `shape` is -1, it will be
// inferred from the `input`. If one dimension of `shape` is 0, by default
// (`allowzero` is false), that dimension is unchanged from the corresponding
// one of `input`. If `allowzero` is true, that dimension is set to 0, which
// means that the reshaped tensor is an empty tensor, i.e., `input` should also
// be an empty tensor. If `allowzero` is true, `shape` should not include both
// 0 and -1 at the same time. If `shape` is an empty vector, `input` will be
// converted to a scalar.
Tensor Model::reshape(Tensor input, const Dims &shape, bool allowzero,
                      const std::string &name) {
    check_null(input.ref());
    // Infer -1 dimension if exists
    std::vector<DimType> inferred_shape;
    int neg_idx = -1;
    bool zero_exists = false;
    DimType total_size = 1;
    for (auto i = 0; i < shape.ndims(); i++) {
        if (shape[i] == -1) {
            if (neg_idx != -1) {
                ERR(InvalidUsageError, "multiple -1 in shape: ", shape);
            }
            neg_idx = static_cast<int>(i);
        } else if (shape[i] < 0) {
            ERR(InvalidUsageError,
                "shape cannot include negative values except -1. Given: ",
                shape);
        } else {
            if (shape[i] == 0) {
                zero_exists = true;
            }
            total_size *= shape[i];
        }
        inferred_shape.push_back(shape[i]);
    }
    if (neg_idx != -1) {
        if (zero_exists) {
            ERR(InvalidUsageError,
                "shape cannot include both 0 and -1 at the same time. Given: ",
                shape);
        }
        // total_size is always positive at this point.
        // Infer the -1 dimension
        if (input.shape().nelems() % total_size != 0) {
            ERR(InvalidUsageError, "number of elements mismatch: reshape from ",
                input.shape(), " to ", shape);
        }
        inferred_shape[neg_idx] = input.shape().nelems() / total_size;
    } else if (!zero_exists && input.shape().nelems() != total_size) {
        ERR(InvalidUsageError, "number of elements mismatch: reshape from ",
            input.shape(), " to ", shape);
    }
    Dims new_shape;
    Dims new_strides;
    Dims new_offs;
    reshape_helper(input.ref_, Dims{inferred_shape}, allowzero, new_shape,
                   new_strides, new_offs);
    return impl_
        ->create_op<ModelOpReshape>("", name, input.ref_, new_shape,
                                    new_strides, new_offs)
        ->result_tensors()[0];
}

}  // namespace ark
