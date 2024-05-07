// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model_offset.hpp"

#include "logging.h"
#include "model_buffer.hpp"
#include "model_data_type.hpp"
#include "model_tensor.hpp"

namespace ark {

ModelOffset::ModelOffset(ModelTensorRef tensor) {
    auto st = tensor->strides();
    auto of = tensor->offsets();
    int ndims = st.ndims();
    size_t offset = 0;
    for (int idx = ndims - 1; idx >= 0; --idx) {
        size_t inc = of[idx];
        for (int j = idx + 1; j < ndims; ++j) {
            inc *= st[j];
        }
        offset += inc * tensor->data_type()->bytes();
    }
    buffer_id_ = tensor->buffer()->id();
    value_ = offset;
}

nlohmann::ordered_json ModelOffset::serialize() const {
    nlohmann::ordered_json j;
    j["BufferId"] = buffer_id_;
    j["Value"] = value_;
    return j;
}

std::shared_ptr<ModelOffset> ModelOffset::deserialize(const json &serialized) {
    if (!serialized.contains("BufferId")) {
        ERR(ModelError, "ModelOffset deserialization failed: missing BufferId");
    } else if (!serialized.contains("Value")) {
        ERR(ModelError, "ModelOffset deserialization failed: missing Value");
    }
    return std::make_shared<ModelOffset>(serialized["BufferId"],
                                         serialized["Value"]);
}

}  // namespace ark
