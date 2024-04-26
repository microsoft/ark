// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_REF_HPP
#define ARK_MODEL_REF_HPP

#include <memory>

namespace ark {

class ModelOp;
using ModelOpRef = std::shared_ptr<ModelOp>;

class ModelBuffer;
using ModelBufferRef = std::shared_ptr<ModelBuffer>;

class ModelTensor;
using ModelTensorRef = std::shared_ptr<ModelTensor>;

class ModelNode;
using ModelNodeRef = std::shared_ptr<ModelNode>;

}  // namespace ark

#endif  // ARK_MODEL_REF_HPP
