// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/model_graph.hpp"

#include "logging.h"
#include "model/model_graph_impl.hpp"
#include "model/model_node.hpp"

namespace ark {

ModelGraph::ModelGraph() : impl_(std::make_unique<ModelGraph::Impl>()) {}

ModelGraph::ModelGraph(const ModelGraph &other)
    : impl_(std::make_unique<ModelGraph::Impl>(*other.impl_)) {}

ModelGraph::~ModelGraph() = default;

ModelGraph &ModelGraph::operator=(const ModelGraph &other) {
    *impl_ = *other.impl_;
    return *this;
}

ModelNodeRef ModelGraph::break_node(ModelNodeRef node, size_t op_idx) {
    return impl_->break_node(node, op_idx);
}

/// Get the list of @ref ModelNode in the graph.
std::vector<ModelNodeRef> ModelGraph::nodes() const { return impl_->nodes(); }

std::string ModelGraph::serialize(int indent) const {
    return impl_->serialize(indent);
}

void ModelGraph::compress_nodes() { impl_->compress_nodes(); }

bool ModelGraph::verify() const { return impl_->verify(); }

}  // namespace ark