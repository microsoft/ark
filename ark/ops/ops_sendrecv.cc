// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>

#include "env.h"
#include "logging.h"
#include "model.h"

namespace ark {

//
Tensor *Model::send(Tensor *input, int id, int dst_rank, size_t bytes,
                    const std::string &name) {
    size_t max_bytes = input->shape_bytes();
    if (max_bytes < bytes) {
        ERR(InvalidUsageError, "invalid bytes: ", bytes, ", max: ", max_bytes);
    }
    if (bytes == 0) {
        bytes = max_bytes;
    }
    input->exported = true;
    if (!input->is_sequential()) {
        ERR(InvalidUsageError, "input tensor must be sequential");
    }
    return this->send_mscclpp(input, id, dst_rank, bytes, name);
}

//
Tensor *Model::send_done(Tensor *input, int id, int dst_rank,
                         const std::string &name) {
    return this->send_done_mscclpp(input, dst_rank, name);
}

//
Tensor *Model::recv(int id, int src_rank, size_t bytes, Tensor *output,
                    const std::string &name) {
    if (output == nullptr) {
        if (bytes == 0) {
            ERR(InvalidUsageError, "receive bytes cannot be 0");
        }
        output = this->tensor({DimType(bytes)}, BYTE);
    }
    output->exported = true;
    size_t max_bytes = output->shape_bytes();
    if (max_bytes < bytes) {
        ERR(InvalidUsageError, "invalid bytes: ", bytes, ", max: ", max_bytes);
    }
    if (bytes == 0) {
        bytes = max_bytes;
    }
    if (!output->is_sequential()) {
        ERR(InvalidUsageError, "output tensor must be sequential");
    }
    return this->recv_mscclpp(id, src_rank, bytes, output, name);
}

}  // namespace ark
