// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model_op.hpp"

#include <algorithm>
#include <set>

#include "logging.h"
#include "model_tensor.hpp"
#include "ops/ops_arithmetic.hpp"
#include "ops/ops_cast.hpp"
#include "ops/ops_copy.hpp"
#include "ops/ops_math.hpp"
#include "ops/ops_matmul.hpp"
#include "ops/ops_noop.hpp"
#include "ops/ops_reduce.hpp"
#include "ops/ops_refer.hpp"
#include "ops/ops_reshape.hpp"
#include "ops/ops_scale.hpp"
#include "ops/ops_sendrecv.hpp"
#include "ops/ops_tensor.hpp"
#include "ops/ops_transpose.hpp"

///
/// NOTE: how to add a new operator
///   1. Define `class ModelOp.*` in `ops/` directory and include it here.
///   2. Register the new operator in `ModeOpT::from_name()` by
///      `MODEL_OP_TYPE_REGISTER(.*)`.
///   3. Add an operator function in `class Model` in `model.hpp`.
///

namespace ark {

std::shared_ptr<ModelOpFactory> model_op_factory() {
    static auto factory = std::make_shared<ModelOpFactory>();
    return factory;
}

#define MODEL_OP_TYPE_REGISTER(_name)                       \
    instances[#_name] = std::make_shared<ModelOpT>(#_name); \
    model_op_factory()->register_op<ModelOp##_name>(#_name);

const ModelOpType ModelOpT::from_name(const std::string &type_name) {
    static std::unordered_map<std::string, ModelOpType> instances;
    if (instances.empty()) {
        MODEL_OP_TYPE_REGISTER(Add);
        MODEL_OP_TYPE_REGISTER(Cast);
        MODEL_OP_TYPE_REGISTER(Copy);
        MODEL_OP_TYPE_REGISTER(Div);
        MODEL_OP_TYPE_REGISTER(Exp);
        MODEL_OP_TYPE_REGISTER(Gelu);
        MODEL_OP_TYPE_REGISTER(Matmul);
        MODEL_OP_TYPE_REGISTER(Mul);
        MODEL_OP_TYPE_REGISTER(Noop);
        MODEL_OP_TYPE_REGISTER(Recv);
        MODEL_OP_TYPE_REGISTER(ReduceMax);
        MODEL_OP_TYPE_REGISTER(ReduceMean);
        MODEL_OP_TYPE_REGISTER(ReduceSum);
        MODEL_OP_TYPE_REGISTER(Relu);
        MODEL_OP_TYPE_REGISTER(Reshape);
        MODEL_OP_TYPE_REGISTER(Rsqrt);
        MODEL_OP_TYPE_REGISTER(Scale);
        MODEL_OP_TYPE_REGISTER(Send);
        MODEL_OP_TYPE_REGISTER(SendDone);
        MODEL_OP_TYPE_REGISTER(Sigmoid);
        MODEL_OP_TYPE_REGISTER(Sqrt);
        MODEL_OP_TYPE_REGISTER(Sub);
        MODEL_OP_TYPE_REGISTER(Tensor);
        MODEL_OP_TYPE_REGISTER(Transpose);
    }
    auto it = instances.find(type_name);
    if (it == instances.end()) {
        ERR(InvalidUsageError, "Unknown model op type: ", type_name);
    }
    return it->second;
}

std::vector<ModelTensorRef> ModelOp::input_tensors() const {
    // input_tensors = read_tensors || write_tensors
    std::set<ModelTensorRef> input_tensors;
    input_tensors.insert(read_tensors_.begin(), read_tensors_.end());
    input_tensors.insert(write_tensors_.begin(), write_tensors_.end());
    std::vector<ModelTensorRef> input_tensors_vec(input_tensors.begin(),
                                                  input_tensors.end());
    return input_tensors_vec;
}

void ModelOp::verify() const {
    std::set<ModelTensorRef> inputs;
    inputs.insert(read_tensors_.begin(), read_tensors_.end());
    inputs.insert(write_tensors_.begin(), write_tensors_.end());

    for (auto &input : inputs) {
        if (input->buffer() == nullptr) {
            ERR(InvalidUsageError, "input tensor buffer is null");
        }
    }

    std::set<ModelTensorRef> outputs;
    outputs.insert(result_tensors_.begin(), result_tensors_.end());

    for (auto &output : outputs) {
        if (output->buffer() == nullptr) {
            ERR(InvalidUsageError, "output tensor buffer is null");
        }
    }

    std::set<ModelTensorRef> intersect;
    std::set_intersection(inputs.begin(), inputs.end(), outputs.begin(),
                          outputs.end(),
                          std::inserter(intersect, intersect.begin()));
    if (!intersect.empty()) {
        ERR(InvalidUsageError, "cyclic dependency detected");
    }
}

std::string ModelOp::vec_string(const Dims &dims) {
    if (dims.is_invalid()) {
        ERR(InvalidUsageError, "invalid dims given");
    }
    int ndims = dims.ndims();
    std::stringstream ss;
    ss << "Vec<";
    if (ndims > 0) {
        ss << dims[0];
        for (int i = 1; i < ndims; ++i) {
            ss << ", " << dims[i];
        }
    }
    ss << '>';
    return ss.str();
}

std::string ModelOp::function_name_string(
    const std::string &kernel_name,
    const std::vector<std::string> &template_args) {
    std::stringstream ss;
    ss << kernel_name;
    if (!template_args.empty()) {
        ss << "<" << template_args[0];
        for (size_t i = 1; i < template_args.size(); i++) {
            ss << ", " << template_args[i];
        }
        ss << ">";
    }
    return ss.str();
}

nlohmann::ordered_json ModelOp::serialize() const {
    nlohmann::ordered_json j;
    j["Type"] = type_->type_name();
    j["Name"] = name_;
    j["IsVirtual"] = is_virtual_;
    j["ReadTensors"] = nlohmann::ordered_json::array();
    for (auto &t : read_tensors_) {
        j["ReadTensors"].push_back(t->serialize());
    }
    j["WriteTensors"] = nlohmann::ordered_json::array();
    for (auto &t : write_tensors_) {
        j["WriteTensors"].push_back(t->serialize());
    }
    j["ResultTensors"] = nlohmann::ordered_json::array();
    for (auto &t : result_tensors_) {
        j["ResultTensors"].push_back(t->serialize());
    }
    j["Args"] = nlohmann::ordered_json::object();
    for (auto &arg : args_) {
        j["Args"][arg.first] = arg.second.serialize();
    }
    return j;
}

std::shared_ptr<ModelOp> ModelOp::deserialize(
    const nlohmann::json &serialized) {
    if (!serialized.contains("Type")) {
        ERR(InvalidUsageError, "ModelOp deserialization failed: missing Type");
    } else if (!serialized.contains("Name")) {
        ERR(InvalidUsageError, "ModelOp deserialization failed: missing Name");
    } else if (!serialized.contains("IsVirtual")) {
        ERR(InvalidUsageError,
            "ModelOp deserialization failed: missing IsVirtual");
    } else if (!serialized.contains("ReadTensors")) {
        ERR(InvalidUsageError,
            "ModelOp deserialization failed: missing ReadTensors");
    } else if (!serialized.contains("WriteTensors")) {
        ERR(InvalidUsageError,
            "ModelOp deserialization failed: missing WriteTensors");
    } else if (!serialized.contains("ResultTensors")) {
        ERR(InvalidUsageError,
            "ModelOp deserialization failed: missing ResultTensors");
    } else if (!serialized.contains("Args")) {
        ERR(InvalidUsageError, "ModelOp deserialization failed: missing Args");
    }
    auto ret = model_op_factory()->construct(serialized["Type"]);
    ret->type_ = ModelOpT::from_name(serialized["Type"]);
    ret->name_ = serialized["Name"];
    ret->is_virtual_ = serialized["IsVirtual"];
    for (const auto &t : serialized["ReadTensors"]) {
        ret->read_tensors_.push_back(ModelTensor::deserialize(t));
    }
    for (const auto &t : serialized["WriteTensors"]) {
        ret->write_tensors_.push_back(ModelTensor::deserialize(t));
    }
    for (const auto &t : serialized["ResultTensors"]) {
        ret->result_tensors_.push_back(ModelTensor::deserialize(t));
    }
    for (const auto &arg : serialized["Args"].items()) {
        ret->args_[arg.key()] = ModelOpArg::deserialize(arg.value());
    }
    return ret;
}

}  // namespace ark
