// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_OP_HPP_
#define ARK_MODEL_OP_HPP_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "arch.hpp"
#include "ark/model_ref.hpp"
#include "logging.hpp"
#include "model_json.hpp"
#include "model_op_arg.hpp"

namespace ark {

class ModelGraph;

class ModelOpT;
using ModelOpType = std::shared_ptr<ModelOpT>;

class ModelOp;

class ModelOpT : public ModelNamedT {
   public:
    ModelOpT(const std::string &type_name) : ModelNamedT(type_name) {}

    ModelOpT(const ModelOpT &) = default;

    static const ModelOpType from_name(const std::string &type_name);
};

class ModelOp {
   public:
    ModelOp() = default;

    ModelOp(const std::string &type_name, bool is_virtual = false)
        : type_(ModelOpT::from_name(type_name)), is_virtual_(is_virtual) {}

    ModelOp(const ModelOp &) = default;

    virtual ~ModelOp() = default;

    virtual std::string impl_name([[maybe_unused]] const Json &config) const {
        return "";
    }

    virtual std::vector<ModelOpArg> impl_args([
        [maybe_unused]] const Json &config) const {
        return {};
    }

    virtual Json default_config(
        [[maybe_unused]] const ArchRef arch = ARCH_ANY) const {
        return {{"NumTasks", 0}, {"NumWarps", 0}, {"SramBytes", 0}};
    }

    void set_name(const std::string &name) { name_ = name; }

    ModelOpType type() const { return type_; }

    const std::string &name() const { return name_; }

    bool is_virtual() const { return is_virtual_; }

    const std::vector<ModelTensorRef> &read_tensors() const {
        return read_tensors_;
    }

    const std::vector<ModelTensorRef> &write_tensors() const {
        return write_tensors_;
    }

    const std::vector<ModelTensorRef> &result_tensors() const {
        return result_tensors_;
    }

    const std::map<std::string, ModelOpArg> &args() const { return args_; }

    std::vector<ModelTensorRef> input_tensors() const;

    void verify() const;

    Json serialize() const;

    static std::shared_ptr<ModelOp> deserialize(const Json &serialized);

   protected:
    friend class ModelGraph;

    static std::string vec_string(const Dims &dims);

    static std::string function_name_string(
        const std::string &kernel_name,
        const std::vector<std::string> &template_args = {});

    ModelOpType type_;
    std::string name_;
    bool is_virtual_;
    std::vector<ModelTensorRef> read_tensors_;
    std::vector<ModelTensorRef> write_tensors_;
    std::vector<ModelTensorRef> result_tensors_;
    std::map<std::string, ModelOpArg> args_;
};

class ModelOpFactory {
   private:
    std::unordered_map<std::string, std::function<std::shared_ptr<ModelOp>()>>
        constructors_;

   public:
    ModelOpFactory() = default;

    template <class DerivedModelOp>
    void register_op(const std::string &class_name) {
        if (constructors_.find(class_name) != constructors_.end()) {
            ERR(InvalidUsageError, "Class already registered: ", class_name);
        }
        constructors_[class_name] = []() {
            return std::shared_ptr<ModelOp>(new DerivedModelOp());
        };
    }

    std::shared_ptr<ModelOp> construct(const std::string &class_name) const {
        auto it = constructors_.find(class_name);
        if (it == constructors_.end()) {
            ERR(InvalidUsageError,
                "Tried to construct an unknown class: ", class_name);
        }
        return it->second();
    }

    bool empty() const { return constructors_.empty(); }
};

std::shared_ptr<ModelOpFactory> model_op_factory();

}  // namespace ark

#endif  // ARK_MODEL_OP_HPP_
