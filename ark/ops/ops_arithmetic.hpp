// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_ARITHMETIC_HPP_
#define ARK_OPS_ARITHMETIC_HPP_

#include "ops_broadcast.hpp"

namespace ark {

class ModelOpAdd : public ModelOpBroadcast2 {
   public:
    ModelOpAdd() = default;
    ModelOpAdd(ModelTensorRef input, ModelTensorRef other,
               ModelTensorRef output);
};

class ModelOpMul : public ModelOpBroadcast2 {
   public:
    ModelOpMul() = default;
    ModelOpMul(ModelTensorRef input, ModelTensorRef other,
               ModelTensorRef output);
};

class ModelOpSub : public ModelOpBroadcast2 {
   public:
    ModelOpSub() = default;
    ModelOpSub(ModelTensorRef input, ModelTensorRef other,
               ModelTensorRef output);
};

class ModelOpDiv : public ModelOpBroadcast2 {
   public:
    ModelOpDiv() = default;
    ModelOpDiv(ModelTensorRef input, ModelTensorRef other,
               ModelTensorRef output);
};

}  // namespace ark

#endif  // ARK_OPS_ARITHMETIC_HPP_
