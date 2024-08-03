// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model_op_arg.hpp"

#include "logging.hpp"
#include "model_tensor.hpp"

namespace ark {

ModelOpArg::ModelOpArg() : ModelNamedT("") {}

Json ModelOpArg::serialize() const {
    const std::string &type_name = this->type_name();
    Json j;
    if (type_name == "TENSOR") {
        j[type_name] = this->value<ModelTensorRef>()->serialize();
    } else if (type_name == "OFFSET") {
        j[type_name] = this->value<ModelOffset>().serialize();
    } else if (type_name == "DIMS") {
        j[type_name] = this->value<Dims>().vector();
    } else if (type_name == "INT") {
        j[type_name] = this->value<int>();
    } else if (type_name == "INT64") {
        j[type_name] = this->value<int64_t>();
    } else if (type_name == "UINT64") {
        j[type_name] = this->value<uint64_t>();
    } else if (type_name == "BOOL") {
        j[type_name] = this->value<bool>();
    } else if (type_name == "FLOAT") {
        j[type_name] = this->value<float>();
    } else {
        ERR(ModelError,
            "Tried to serialize an unknown type of argument: ", type_name);
    }
    return j;
}

ModelOpArg ModelOpArg::deserialize(const Json &serialized) {
    try {
        const std::string &type_name = serialized.begin().key();
        auto &value = serialized.begin().value();
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
        ERR(ModelError, "Failed to deserialize `", serialized.dump(),
            "`: ", e.what());
    }
    ERR(ModelError, "Tried to deserialize an unknown type of argument: ",
        serialized.dump());
    return ModelOpArg();
}

}  // namespace ark
