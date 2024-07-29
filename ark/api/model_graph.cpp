// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/model_graph.hpp"

#include "logging.h"
#include "model/model_graph_impl.hpp"
#include "model/model_node.hpp"

namespace ark {

ModelGraph::ModelGraph(int rank, int world_size)
    : impl_(std::make_unique<ModelGraph::Impl>(rank, world_size)) {}

ModelGraph::ModelGraph(const ModelGraph &other)
    : impl_(std::make_unique<ModelGraph::Impl>(*other.impl_)) {}

ModelGraph::~ModelGraph() = default;

ModelGraph &ModelGraph::operator=(const ModelGraph &other) {
    *impl_ = *other.impl_;
    return *this;
}

/// Get the list of @ref ModelNode in the graph.
std::vector<ModelNodeRef> ModelGraph::nodes() const { return impl_->nodes(); }

std::string ModelGraph::serialize(bool pretty) const {
    return impl_->serialize(pretty);
}

int ModelGraph::rank() const { return impl_->rank(); }

int ModelGraph::world_size() const { return impl_->world_size(); }

void ModelGraph::compress_nodes(bool merge_nodes) {
    impl_->compress_nodes(merge_nodes);
}

bool ModelGraph::compressed() const { return impl_->compressed(); }

bool ModelGraph::verify() const { return impl_->verify(); }

}  // namespace ark
