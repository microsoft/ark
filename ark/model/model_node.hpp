// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_NODE_HPP_
#define ARK_MODEL_NODE_HPP_

#include <memory>
#include <vector>

#include "ark/model_ref.hpp"
#include "model_json.hpp"
#include "unique_list.hpp"

namespace ark {

/// A node of @ref Model.
class ModelNode {
   public:
    ModelNode() = default;

    /// @ref Op that this @ref ModelNode represents.
    ModelOpRef op;

    /// The list of @ref ModelNode that depends on this @ref ModelNode.
    UniqueList<ModelNodeRef> consumers;

    /// The list of @ref ModelNode that this @ref ModelNode depends on.
    UniqueList<ModelNodeRef> producers;

    /// Graph context of this node.
    Json context;
};

}  // namespace ark

#endif  // ARK_MODEL_NODE_HPP_
