// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model_op_arg.hpp"

#include "logging.h"
#include "model_tensor.hpp"

namespace ark {

ModelOpArg::ModelOpArg() : NamedT("") {}

nlohmann::ordered_json ModelOpArg::serialize() const {
    const std::string &type_name = this->type_name();
    nlohmann::ordered_json j;
    j.push_back(type_name);
    if (type_name == "TENSOR") {
        j.push_back(this->value<ModelTensorRef>()->serialize());
    } else if (type_name == "OFFSET") {
        j.push_back(this->value<ModelOffset>().serialize());
    } else if (type_name == "DIMS") {
        j.push_back(this->value<Dims>().vector());
    } else if (type_name == "INT") {
        j.push_back(this->value<int>());
    } else if (type_name == "INT64") {
        j.push_back(this->value<int64_t>());
    } else if (type_name == "UINT64") {
        j.push_back(this->value<uint64_t>());
    } else if (type_name == "BOOL") {
        j.push_back(this->value<bool>());
    } else if (type_name == "FLOAT") {
        j.push_back(this->value<float>());
    } else {
        ERR(InvalidUsageError,
            "Tried to serialize an unknown type of argument: ", type_name);
    }
    return j;
}

ModelOpArg ModelOpArg::deserialize(const json &serialized) {
    try {
        const std::string &type_name = serialized[0];
        auto &value = serialized[1];
        if (type_name == "TENSOR") {
            return ModelOpArg(ModelTensor::deserialize(value));
        } else if (type_name == "OFFSET") {
            return ModelOpArg(*ModelOffset::deserialize(value));
        } else if (type_name == "DIMS") {
            return ModelOpArg(Dims(value.get<std::vector<DimType>>()));
        } else if (type_name == "INT") {
            return ModelOpArg(value.get<int>());
        } else if (type_name == "INT64") {
            return ModelOpArg(value.get<int64_t>());
        } else if (type_name == "UINT64") {
            return ModelOpArg(value.get<uint64_t>());
        } else if (type_name == "BOOL") {
            return ModelOpArg(value.get<bool>());
        } else if (type_name == "FLOAT") {
            return ModelOpArg(value.get<float>());
        }
    } catch (const std::exception &e) {
        ERR(InvalidUsageError, "Failed to deserialize `", serialized.dump(),
            "`: ", e.what());
    }
    ERR(InvalidUsageError,
        "Tried to deserialize an unknown type of argument: ", serialized[0]);
    return ModelOpArg();
}

}  // namespace ark
