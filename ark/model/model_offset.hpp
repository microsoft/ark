// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_OFFSET_HPP_
#define ARK_MODEL_OFFSET_HPP_

#include "ark/model_ref.hpp"
#include "json.hpp"

namespace ark {

class ModelOffset {
   private:
    size_t buffer_id_;
    size_t value_;

   public:
    ModelOffset(size_t buffer_id, size_t value)
        : buffer_id_(buffer_id), value_(value) {}

    ModelOffset(ModelTensorRef tensor);

    size_t buffer_id() const { return buffer_id_; }

    size_t value() const { return value_; }

    nlohmann::ordered_json serialize() const;

    static std::shared_ptr<ModelOffset> deserialize(const json &serialized);
};

}  // namespace ark

#endif  // ARK_MODEL_OFFSET_HPP_
