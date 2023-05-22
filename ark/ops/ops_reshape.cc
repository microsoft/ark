// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/logging.h"
#include "ark/model_io.h"

using namespace std;

namespace ark {

// Reshape `input` to `shape`. This interface does not support -1 as a dimension
// of `shape`, because Dims does not allow -1 as a valid dimension.
static Tensor *_reshape(Model *model, Tensor *input, const Dims &shape,
                        bool allowzero, Tensor *output, const string &name)
{
    if (input == nullptr) {
        LOGERR("input is null");
    }
    LOG(DEBUG, "reshape ", input->shape, " ", shape);
    // Infer the actual shape
    vector<DimType> inferred_shape;
    if (shape.ndims() == 0) {
        // Convert to a scalar
        inferred_shape.emplace_back(1);
        if (input->shape.size() != 1) {
            LOGERR("number of elements mismatch: reshape from ", input->shape,
                   " to ", shape);
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
            LOGERR("number of elements mismatch: reshape from ", input->shape,
                   " to ", shape);
        }
    }
    Dims new_shape{inferred_shape};

    if (output != nullptr) {
        // Verfiy given `output`
        if (input->type != output->type) {
            LOGERR("invalid output data type: ", type_str(output->type));
        }
        if (input->shape.size() != output->shape.size()) {
            LOGERR("shape sizes mismatch: input ", input->shape, ", output ",
                   output->shape);
        }
    }

    // TODO: check if this reshape requires any copy

    if (output == nullptr) {
        output = model->tensor(new_shape, input->type, input->buf, shape);
    }
    model->create_op(OP_RESHAPE, OP_PREC_NONE, {input}, {output}, {}, name);
    return output;
}

//
Tensor *Model::reshape(Tensor *input, const Dims &shape, bool allowzero,
                       Tensor *output, const string &name)
{
    return _reshape(this, input, shape, allowzero, output, name);
}

// Reshape `input` to `shape`. If one dimension of `shape` is -1, it will be
// inferred from the `input`. If one dimension of `shape` is 0, by default
// (`allowzero` is false), that dimension is unchanged from the corresponding
// one of `input`. If `allowzero` is true, that dimension is set to 0, which
// means that the reshaped tensor is an empty tensor, i.e., `input` should also
// be an empty tensor. If `allowzero` is true, `shape` should not include both
// 0 and -1 at the same time. If `shape` is an empty vector, `input` will be
// converted to a scalar.
Tensor *Model::reshape(Tensor *input, const initializer_list<DimType> shape,
                       bool allowzero, Tensor *output, const string &name)
{
    if (input == nullptr) {
        LOGERR("input is null");
    }
    vector<DimType> shape_vec{shape};
    // Infer -1 dimension if exists
    int neg_idx = -1;
    bool zero_exists = false;
    vector<DimType> inferred_shape;
    DimType total_size = 1;
    for (size_t i = 0; i < shape_vec.size(); i++) {
        if (shape_vec[i] == -1) {
            if (neg_idx != -1) {
                LOGERR("multiple -1 in shape: ", Dims(shape_vec));
            }
            neg_idx = (int)i;
        } else if (shape_vec[i] < 0) {
            LOGERR("shape cannot include negative values except -1. "
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
            LOGERR("shape cannot include both 0 and -1 at the same "
                   "time. Given: ",
                   Dims(shape_vec));
        }
        // Infer the -1 dimension
        if (total_size <= 0) {
            LOGERR("Unexpected error");
        }
        if (input->shape.size() % total_size != 0) {
            LOGERR("number of elements mismatch: reshape from ", input->shape,
                   " to ", Dims(shape_vec));
        }
        inferred_shape[neg_idx] = input->shape.size() / total_size;
    } else if (input->shape.size() != total_size) {
        LOGERR("number of elements mismatch: reshape from ", input->shape,
               " to ", Dims(shape_vec));
    }
    return _reshape(this, input, Dims{inferred_shape}, allowzero, output, name);
}

} // namespace ark
