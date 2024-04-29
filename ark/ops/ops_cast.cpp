// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_cast.hpp"

#include "ark/model.hpp"
#include "ops_common.hpp"

namespace ark {

ModelOpCast::ModelOpCast(ModelTensorRef input, ModelDataType data_type,
                         ModelTensorRef output)
    : ModelOpBroadcast1("Cast", input,
                        output ? output
                               : std::make_shared<ModelTensor>(
                                     data_type, std::make_shared<ModelBuffer>(),
                                     input->shape())) {
    if (output) {
        check_match_data_type(output, data_type);
    }
    verify();
}

static void byte_cast_helper(ModelTensorRef input, ModelDataType data_type,
                             Dims &new_shape, Dims &new_strides,
                             Dims &new_offsets, Dims &new_pads) {
    if (input->data_type() == BYTE) {
        if (input->shape_bytes() < data_type->bytes()) {
            ERR(InvalidUsageError, "input tensor is too small to be casted to ",
                data_type);
        }
        // The last greater-than-1 dimension of the input tensor should be
        // divisible by the size of the output type.
        int last_dim = input->shape().ndims() - 1;
        for (; last_dim >= 0; --last_dim) {
            if (last_dim == 0 || input->strides()[last_dim] > 1) {
                break;
            }
        }
        if ((input->shape()[last_dim] % data_type->bytes()) != 0) {
            ERR(InvalidUsageError,
                "the last greater-than-1 dimension of the "
                "input tensor shape ",
                input->shape()[last_dim],
                " is not divisible by the size of the output "
                "tensor type (",
                data_type->bytes(), ")");
        }
        if ((input->strides()[last_dim] % data_type->bytes()) != 0) {
            ERR(InvalidUsageError,
                "the last greater-than-1 dimension of the "
                "input tensor strides ",
                input->strides()[last_dim],
                " is not divisible by the size of the output "
                "tensor type (",
                data_type->bytes(), ")");
        }
        if ((input->offsets()[last_dim] % data_type->bytes()) != 0) {
            ERR(InvalidUsageError,
                "the last greater-than-1 dimension of the "
                "input tensor offsets ",
                input->offsets()[last_dim],
                " is not divisible by the size of the output "
                "tensor type (",
                data_type->bytes(), ")");
        }
        if (input->pads()[last_dim] > 1) {
            // we can ignore pads if it is 1
            if ((input->pads()[last_dim] % data_type->bytes()) != 0) {
                ERR(InvalidUsageError,
                    "the last greater-than-1 dimension of the "
                    "input tensor pads ",
                    input->pads()[last_dim],
                    " is not divisible by the size of the output "
                    "tensor type (",
                    data_type->bytes(), ")");
            }
        }
        new_shape = input->shape();
        new_strides = input->strides();
        new_offsets = input->offsets();
        new_pads = input->pads();
        new_shape[last_dim] /= data_type->bytes();
        new_strides[last_dim] /= data_type->bytes();
        new_offsets[last_dim] /= data_type->bytes();
        if (new_pads[last_dim] > 1) {
            new_pads[last_dim] /= data_type->bytes();
        }
    } else if (data_type == BYTE) {
        new_shape = input->shape();
        new_strides = input->strides();
        new_offsets = input->offsets();
        new_pads = input->pads();
        new_shape[-1] *= input->data_type()->bytes();
        new_strides[-1] *= input->data_type()->bytes();
        new_offsets[-1] *= input->data_type()->bytes();
        if (new_pads[-1] > 1) {
            new_pads[-1] *= input->data_type()->bytes();
        }
    } else {
        ERR(ModelError, "unexpected error");
    }
}

ModelOpByteCast::ModelOpByteCast(ModelTensorRef input, ModelDataType data_type,
                                 const Dims &shape, const Dims &strides,
                                 const Dims &offsets, const Dims &pads)
    : ModelOpTensor(input->buffer(), shape, data_type, strides, offsets, pads,
                    input->exported(), input->imported_rank()) {
    read_tensors_ = {input};
    verify();
}

ModelTensorRef Model::cast(ModelTensorRef input, ModelDataType data_type,
                           ModelTensorRef output, const std::string &name) {
    check_null(input);
    if (!output) {
        if (input->data_type() == data_type) {
            // Casting to the same type without the output tensor specified is
            // considered as an identity.
            return this->identity(input, {}, name);
        } else if (data_type == BYTE || input->data_type() == BYTE) {
            // Casting to/from BYTE without the output tensor specified is
            // handled by `ModelOpByteCast`.
            Dims new_shape, new_strides, new_offsets, new_pads;
            byte_cast_helper(input, data_type, new_shape, new_strides,
                             new_offsets, new_pads);
            return impl_
                ->create_op<ModelOpByteCast>(name, input, data_type, new_shape,
                                             new_strides, new_offsets, new_pads)
                ->result_tensors()[0];
        }
    }
    return impl_->create_op<ModelOpCast>(name, input, data_type, output)
        ->result_tensors()[0];
}

}  // namespace ark
