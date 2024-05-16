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
    ModelGraph();

    ModelGraph(const ModelGraph &other);

    ~ModelGraph();

    ModelGraph &operator=(const ModelGraph &other);

    void compress_nodes();

    bool verify() const;

    std::string serialize(bool pretty = true) const;

    /// Get the list of @ref ModelNode in the graph.
    std::vector<ModelNodeRef> nodes() const;

   protected:
    friend class Model;

    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ark

#endif  // ARK_MODEL_GRAPH_HPP
