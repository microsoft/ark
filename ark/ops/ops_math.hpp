// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_MATH_HPP_
#define ARK_OPS_MATH_HPP_

#include "ark/dims.hpp"
#include "ark/model.hpp"
#include "model/model_op.hpp"

namespace ark {

class ModelOpMath : public ModelOp {
   public:
    ModelOpMath() = default;
    ModelOpMath(const std::string &type_name, ModelTensorRef input,
                ModelTensorRef output);
};

class ModelOpExp : public ModelOpMath {
   public:
    ModelOpExp() = default;
    ModelOpExp(ModelTensorRef input, ModelTensorRef output);
};

class ModelOpRelu : public ModelOpMath {
   public:
    ModelOpRelu() = default;
    ModelOpRelu(ModelTensorRef input, ModelTensorRef output);
};

}  // namespace ark

#endif  // ARK_OPS_MATH_HPP_
