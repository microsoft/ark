// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_REDUCE_HPP_
#define ARK_OPS_REDUCE_HPP_

#include "model/model_op.hpp"

namespace ark {

class ModelOpReduce : public ModelOp {
   public:
    ModelOpReduce() = default;
    ModelOpReduce(const std::string &type_name, ModelTensorRef input, int axis,
                  bool keepdims, ModelTensorRef output);

    std::string impl_name(const nlohmann::json &config) const override;

    std::vector<ModelOpArg> impl_args(
        [[maybe_unused]] const nlohmann::json &config) const override;

    nlohmann::ordered_json default_config() const override;
};

class ModelOpReduceMax : public ModelOpReduce {
   public:
    ModelOpReduceMax() = default;
    ModelOpReduceMax(ModelTensorRef input, int axis, bool keepdims,
                     ModelTensorRef output)
        : ModelOpReduce("ReduceMax", input, axis, keepdims, output) {}
};

class ModelOpReduceMean : public ModelOpReduce {
   public:
    ModelOpReduceMean() = default;
    ModelOpReduceMean(ModelTensorRef input, int axis, bool keepdims,
                      ModelTensorRef output)
        : ModelOpReduce("ReduceMean", input, axis, keepdims, output) {}
};

class ModelOpReduceSum : public ModelOpReduce {
   public:
    ModelOpReduceSum() = default;
    ModelOpReduceSum(ModelTensorRef input, int axis, bool keepdims,
                     ModelTensorRef output)
        : ModelOpReduce("ReduceSum", input, axis, keepdims, output) {}
};

}  // namespace ark

#endif  // ARK_OPS_REDUCE_HPP_
