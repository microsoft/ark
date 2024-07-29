// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_GRAPH_HPP
#define ARK_MODEL_GRAPH_HPP

#include <ark/model_ref.hpp>
#include <memory>
#include <string>
#include <vector>

namespace ark {

class ModelGraph {
   public:
    ModelGraph(int rank, int world_size);

    ModelGraph(const ModelGraph &other);

    ~ModelGraph();

    ModelGraph &operator=(const ModelGraph &other);

    int rank() const;

    int world_size() const;

    void compress_nodes(bool merge_nodes = false);

    bool compressed() const;

    bool verify() const;

    std::string serialize(bool pretty = true) const;

    /// Get the list of @ref ModelNode in the graph.
    std::vector<ModelNodeRef> nodes() const;

   protected:
    friend class Model;
    friend class PlanManager;
    friend class ContextManager;

    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ark

#endif  // ARK_MODEL_GRAPH_HPP
