// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_placeholder.hpp"

#include "logging.hpp"
#include "model_buffer_manager.hpp"
#include "ops_common.hpp"

namespace ark {

ModelOpPlaceholder::ModelOpPlaceholder(ModelBufferRef buffer, const Dims &shape,
                                       ModelDataType data_type,
                                       const Dims &strides, const Dims &offsets,
                                       const Dims &padded_shape,
                                       void *external_data)
    : ModelOp("Placeholder", true) {
    if (!buffer) {
        buffer = std::make_shared<ModelBuffer>();
    }
    const std::vector<DimType> &shape_vec = shape.vector();
    DataType dtype = ModelDataType(data_type);

    size_t external_data_size =
        std::accumulate(shape_vec.begin(), shape_vec.end(), 1,
                        std::multiplies<int64_t>()) *
        dtype.bytes();

    ModelBufferManager::get_instance().register_buffer(
        buffer->id(), external_data, external_data_size);

    ModelTensorRef tensor = std::make_shared<ModelTensor>(
        data_type, buffer, shape, strides, offsets, padded_shape);

    result_tensors_.emplace_back(tensor);

    verify();
}

Tensor Model::placeholder(const Dims &shape, const DataType &data_type,
                          const Dims &strides, const Dims &offsets,
                          const Dims &padded_shape, int rank,
                          const std::string &name, void *external_data) {
    if (rank != -1) {
        if (rank == this->rank()) {
            rank = -1;
        } else if (rank < 0 || rank >= this->world_size()) {
            ERR(ModelError, "Invalid rank %d", rank);
        }
    }
    return impl_
        ->create_op<ModelOpPlaceholder>(
            name, std::make_shared<ModelBuffer>(rank), shape, data_type.ref(),
            strides, offsets, padded_shape, external_data)
        ->result_tensors()[0];
}
}  // namespace ark