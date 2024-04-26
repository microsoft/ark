// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_GRAPH_HPP
#define ARK_MODEL_GRAPH_HPP

#include <memory>
#include <string>
#include <vector>

#include "model_ref.hpp"

namespace ark {

class ModelGraph {
   public:
    ModelGraph();

    ModelGraph(const ModelGraph &other);

    ~ModelGraph();

    ModelGraph &operator=(const ModelGraph &other);

    /// Break a @ref ModelNode into two @ref ModelNode.
    ///
    /// The original node will have the first @p op_idx ops, and the new node
    /// will have the rest.
    ///
    /// @param node The @ref ModelNode to break.
    /// @param op_idx The index of the first op in the new @ref ModelNode.
    /// @return The new @ref ModelNode.
    ModelNodeRef break_node(ModelNodeRef node, size_t op_idx);

    void compress_nodes();

    bool verify() const;

    std::string serialize(int indent = -1) const;

    /// Get the list of @ref ModelNode in the graph.
    std::vector<ModelNodeRef> nodes() const;

   protected:
    friend class Model;

    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ark

#endif  // ARK_MODEL_GRAPH_HPP
