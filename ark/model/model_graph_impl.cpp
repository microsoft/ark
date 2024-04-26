// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model_graph_impl.hpp"

#include "logging.h"
#include "model_node.hpp"
#include "model_tensor.hpp"

#define DEBUG_MODEL_GRAPH 0
#define MODEL_GRAPH_DEBUG(...)       \
    do {                             \
        if (DEBUG_MODEL_GRAPH) {     \
            LOG(DEBUG, __VA_ARGS__); \
        }                            \
    } while (0);

namespace ark {

ModelGraph::Impl::Impl(const ModelGraph::Impl &other) { *this = other; }

ModelGraph::Impl &ModelGraph::Impl::operator=(const ModelGraph::Impl &other) {
    std::map<ModelNodeRef, ModelNodeRef> node_map;
    nodes_.clear();
    for (const auto &node : other.nodes_) {
        ModelNodeRef new_node = std::make_shared<ModelNode>();
        new_node->ops = node->ops;
        node_map.emplace(node, new_node);
        nodes_.push_back(new_node);
    }
    for (const auto &node : other.nodes_) {
        auto it = node_map.find(node);
        if (it == node_map.end()) {
            ERR(ModelError, "unexpected error");
        }
        ModelNodeRef new_node = it->second;
        for (auto &producer : node->producers) {
            auto it2 = node_map.find(producer);
            if (it2 == node_map.end()) {
                ERR(ModelError, "unexpected error");
            }
            new_node->producers.push_back(it2->second);
        }
        for (auto &consumer : node->consumers) {
            auto it2 = node_map.find(consumer);
            if (it2 == node_map.end()) {
                ERR(ModelError, "unexpected error");
            }
            new_node->consumers.push_back(it2->second);
        }
    }
    op_to_node_.clear();
    for (const auto &p : other.op_to_node_) {
        auto it = node_map.find(p.second);
        if (it == node_map.end()) {
            ERR(ModelError, "unexpected error");
        }
        op_to_node_[p.first] = it->second;
    }
    tensor_to_producer_op_ = other.tensor_to_producer_op_;
    return *this;
}

ModelNodeRef ModelGraph::Impl::break_node(ModelNodeRef node, size_t op_idx) {
    if (op_idx == 0) {
        return node;
    }
    if (op_idx >= node->ops.size()) {
        ERR(ModelError, "unexpected error: op_idx out of range");
    }
    ModelNodeRef new_node = std::make_shared<ModelNode>();
    nodes_.push_back(new_node);
    new_node->ops.insert(new_node->ops.end(), node->ops.begin() + op_idx,
                         node->ops.end());
    for (auto &op : new_node->ops) {
        op_to_node_[op] = new_node;
    }
    new_node->consumers = node->consumers;
    new_node->producers.push_back(node);
    for (auto &consumer : node->consumers) {
        consumer->producers.erase(node);
        consumer->producers.push_back(new_node);
    }
    node->ops.erase(node->ops.begin() + op_idx, node->ops.end());
    node->consumers.clear();
    node->consumers.push_back(new_node);
    return new_node;
}

void ModelGraph::Impl::compress_nodes() {
    this->recursive_remove_virtual_nodes();
    this->recursive_merge_nodes();
}

bool ModelGraph::Impl::verify() const {
    for (auto &node : nodes_) {
        if (node->ops.size() == 0) {
            LOG(DEBUG, "node has no ops");
            return false;
        }
        for (auto &op : node->ops) {
            if (op_to_node_.find(op) == op_to_node_.end()) {
                LOG(DEBUG, "op has not been added to the graph");
                return false;
            }
            if (op_to_node_.at(op) != node) {
                LOG(DEBUG, "op is not in the correct node");
                return false;
            }
            op->verify();
            for (auto &tns : op->result_tensors()) {
                if (tensor_to_producer_op_.find(tns) ==
                    tensor_to_producer_op_.end()) {
                    LOG(DEBUG, "result tensor has not been produced by any op");
                    return false;
                }
                if (tensor_to_producer_op_.at(tns) != op) {
                    LOG(DEBUG, "result tensor has been produced by another op");
                    return false;
                }
            }
            for (auto &tns : op->input_tensors()) {
                if (tensor_to_producer_op_.find(tns) ==
                    tensor_to_producer_op_.end()) {
                    LOG(DEBUG, "input tensor has not been produced by any op");
                    return false;
                }
            }
        }
        for (auto &producer : node->producers) {
            if (producer->consumers.find(node) == producer->consumers.end()) {
                LOG(DEBUG, "producer does not have this node as consumer");
                return false;
            }
        }
        for (auto &consumer : node->consumers) {
            if (consumer->producers.find(node) == consumer->producers.end()) {
                LOG(DEBUG, "consumer does not have this node as producer");
                return false;
            }
        }
    }
    return true;
}

ModelNodeRef ModelGraph::Impl::add_op(ModelOpRef op) {
    for (auto &tns : op->input_tensors()) {
        if (tensor_to_producer_op_.find(tns) == tensor_to_producer_op_.end()) {
            // This tensor has not been produced by any op - assume it is a
            // Tensor op.
            ModelOpRef tensor_op = std::make_shared<ModelOp>("Tensor", true);
            tensor_op->result_tensors_ = {tns};
            this->add_op(tensor_op);
        }
    }
    for (auto &tns : op->result_tensors()) {
        if (tensor_to_producer_op_.find(tns) != tensor_to_producer_op_.end()) {
            ERR(ModelError, "Tensor has already been produced by an op. ",
                tns->serialize().dump(), "; ",
                tensor_to_producer_op_.at(tns)->serialize().dump());
        }
        tensor_to_producer_op_.emplace(tns, op);
    }

    ModelNodeRef node = std::make_shared<ModelNode>();
    node->ops.push_back(op);
    op_to_node_[op] = node;

    for (auto &tns : op->input_tensors()) {
        auto it = tensor_to_producer_op_.find(tns);
        if (it == tensor_to_producer_op_.end()) {
            ERR(ModelError, "Tensor has not been produced by any op. ",
                tns->serialize().dump(), " ", tns.get());
        }
        auto it2 = op_to_node_.find(it->second);
        if (it2 == op_to_node_.end()) {
            ERR(ModelError, "Op has not been added to the graph");
        }
        auto producer = it2->second;
        node->producers.push_back(producer);
        producer->consumers.push_back(node);
    }

    nodes_.push_back(node);
    return node;
}

void ModelGraph::Impl::remove_node(ModelNodeRef node) {
    auto it = nodes_.find(node);
    if (it == nodes_.end()) {
        ERR(ModelError, "attempted to remove a node that is not in the graph");
    }
    // Remove node from consumers and producers.
    for (auto &consumer : node->consumers) {
        consumer->producers.erase(node);
    }
    for (auto &producer : node->producers) {
        producer->consumers.erase(node);
    }
    // Connect consumers and producers.
    for (auto &consumer : node->consumers) {
        for (auto &producer : node->producers) {
            consumer->producers.push_back(producer);
            producer->consumers.push_back(consumer);
        }
    }
    for (auto &op : node->ops) {
        auto it = op_to_node_.find(op);
        if (it == op_to_node_.end()) {
            ERR(ModelError, "unexpected error");
        }
        if (it->second == node) {
            op_to_node_.erase(it);
        }
    }
    nodes_.erase(it);
}

bool ModelGraph::Impl::depends_on(ModelNodeRef node1,
                                  ModelNodeRef node2) const {
    if (node1 == node2) {
        return false;
    }
    std::set<ModelNodeRef> seen_nodes;
    std::vector<ModelNodeRef> boundary_nodes;
    boundary_nodes.emplace_back(node1);
    while (boundary_nodes.size() > 0) {
        std::vector<ModelNodeRef> new_boundary_nodes;
        for (auto &boundary_node : boundary_nodes) {
            if (boundary_node == node2) {
                return true;
            }
            for (auto &producer : boundary_node->producers) {
                if (seen_nodes.find(producer) != seen_nodes.end()) {
                    continue;
                }
                new_boundary_nodes.emplace_back(producer);
            }
        }
        boundary_nodes = new_boundary_nodes;
    }
    return false;
}

void ModelGraph::Impl::recursive_remove_virtual_nodes() {
    std::vector<ModelNodeRef> leaf_nodes;
    for (auto &node : nodes_) {
        if (node->consumers.empty()) {
            leaf_nodes.emplace_back(node);
        }
    }
    UniqueList<ModelNodeRef> seen_nodes;
    this->recursive_remove_virtual_nodes(seen_nodes, leaf_nodes);
}

void ModelGraph::Impl::recursive_remove_virtual_nodes(
    UniqueList<ModelNodeRef> &seen_nodes,
    const std::vector<ModelNodeRef> &boundary_nodes) {
    if (boundary_nodes.size() == 0) {
        return;
    }
    MODEL_GRAPH_DEBUG("remove virtual nodes");
    std::vector<ModelNodeRef> new_boundary_nodes;
    for (auto &boundary_node : boundary_nodes) {
        if (boundary_node->ops.size() == 0) {
            ERR(ModelError, "unexpected error: empty node");
        } else if (boundary_node->ops.size() > 1) {
            ERR(ModelError, "unexpected error: multiple ops in node");
        }
        MODEL_GRAPH_DEBUG("  boundary node");
        MODEL_GRAPH_DEBUG("    node: ", to_json(boundary_node).dump());
        for (auto &producer : boundary_node->producers) {
            // Exception: if any consumer of the producer (rather than the
            // current boundary_node) is unseen, we should not add the producer
            // to the next boundary.
            bool should_add = true;
            for (auto &consumer : producer->consumers) {
                if (consumer == boundary_node) {
                    continue;
                }
                if (!seen_nodes.contains(consumer)) {
                    should_add = false;
                    break;
                }
            }
            if (!should_add) {
                continue;
            }
            if (seen_nodes.contains(producer)) {
                ERR(ModelError,
                    "circular dependency detected: ", to_json(producer).dump());
            }
            MODEL_GRAPH_DEBUG("      added to next boundary: ",
                              to_json(producer).dump());
            new_boundary_nodes.emplace_back(producer);
        }
        if (boundary_node->ops[0]->is_virtual()) {
            MODEL_GRAPH_DEBUG("    remove node: ",
                              to_json(boundary_node).dump());
            // Remove this node from the graph.
            this->remove_node(boundary_node);
            MODEL_GRAPH_DEBUG("      nodes.size() ", nodes_.size());
        } else {
            seen_nodes.push_back(boundary_node);
        }
    }
    this->recursive_remove_virtual_nodes(seen_nodes, new_boundary_nodes);
}

void ModelGraph::Impl::recursive_merge_nodes() {
    std::vector<ModelNodeRef> leaf_nodes;
    for (auto &node : nodes_) {
        if (node->consumers.empty()) {
            leaf_nodes.emplace_back(node);
        }
    }
    UniqueList<ModelNodeRef> seen_nodes;
    this->recursive_merge_nodes(seen_nodes, leaf_nodes);
}

void ModelGraph::Impl::recursive_merge_nodes(
    UniqueList<ModelNodeRef> &seen_nodes,
    const std::vector<ModelNodeRef> &boundary_nodes) {
    if (boundary_nodes.size() == 0) {
        return;
    }
    MODEL_GRAPH_DEBUG("merge ops");
    std::vector<ModelNodeRef> new_boundary_nodes;
    for (auto &boundary_node : boundary_nodes) {
        MODEL_GRAPH_DEBUG("  boundary node");
        MODEL_GRAPH_DEBUG("    node: ", to_json(boundary_node).dump());
        if (boundary_node->producers.size() == 0) {
            // This node is a root.
            seen_nodes.push_back(boundary_node);
            MODEL_GRAPH_DEBUG("    root");
            continue;
        }
        // Add all producers of this node to the next boundary.
        for (auto &producer : boundary_node->producers) {
            // Exception: if any consumer of the producer (rather than the
            // current boundary_node) is unseen, we should not add the producer
            // to the next boundary.
            bool should_add = true;
            for (auto &consumer : producer->consumers) {
                if (consumer == boundary_node) {
                    continue;
                }
                if (!seen_nodes.contains(consumer)) {
                    should_add = false;
                    break;
                }
            }
            if (!should_add) {
                continue;
            }
            if (seen_nodes.contains(producer)) {
                ERR(ModelError,
                    "unexpected error: circular dependency detected");
            }
            new_boundary_nodes.emplace_back(producer);
        }
        ModelNodeRef merge_candidate;
        if (boundary_node->producers.size() > 1) {
            // This node has multiple producers. We can merge only if one
            // producer depends on all other producers.
            for (auto &producer : boundary_node->producers) {
                bool depends_on_all = true;
                for (auto &other_producer : boundary_node->producers) {
                    if (other_producer == producer) {
                        continue;
                    }
                    if (!this->depends_on(producer, other_producer)) {
                        depends_on_all = false;
                        break;
                    }
                }
                if (depends_on_all) {
                    merge_candidate = producer;
                    break;
                }
            }
            if (!merge_candidate) {
                // At least one producer does not depend on others.
                // Cannot merge.
                seen_nodes.push_back(boundary_node);
                MODEL_GRAPH_DEBUG("    multiple producers");
                continue;
            }
        } else {
            // This node has only one producer.
            merge_candidate = *(boundary_node->producers.begin());
        }
        if (merge_candidate->consumers.size() == 0) {
            ERR(ModelError, "unexpected error: graph is incomplete");
        }
        if (merge_candidate->consumers.size() > 1) {
            // The candidate has multiple consumers. We can merge only if all
            // other consumers depend on the current boundary_node.
            bool depends_on_one = true;
            for (auto &consumer : merge_candidate->consumers) {
                if (consumer == boundary_node) {
                    continue;
                }
                if (!this->depends_on(consumer, boundary_node)) {
                    depends_on_one = false;
                    break;
                }
            }
            if (!depends_on_one) {
                // At least one consumer does not depend on the boundary_node.
                // Cannot merge.
                seen_nodes.push_back(boundary_node);
                MODEL_GRAPH_DEBUG("    multiple consumers");
                continue;
            }
        }
        // We can merge the two nodes.
        // Merge `boundary_node` into `merge_candidate`.
        MODEL_GRAPH_DEBUG("  merge: ", to_json(merge_candidate).dump(), " -> ",
                          to_json(boundary_node).dump());
        auto &ops = boundary_node->ops;
        merge_candidate->ops.insert(merge_candidate->ops.end(), ops.begin(),
                                    ops.end());
        for (auto &op : ops) {
            op_to_node_[op] = merge_candidate;
        }
        for (auto &consumer : boundary_node->consumers) {
            consumer->producers.erase(boundary_node);
            consumer->producers.push_back(merge_candidate);
            merge_candidate->consumers.push_back(consumer);
        }
        for (auto &producer : boundary_node->producers) {
            if (producer == merge_candidate) {
                continue;
            }
            producer->consumers.erase(boundary_node);
            producer->consumers.push_back(merge_candidate);
            merge_candidate->producers.push_back(producer);
        }
        merge_candidate->consumers.erase(boundary_node);

        // Remove `boundary_node` from `nodes_`.
        auto it = nodes_.find(boundary_node);
        if (it == nodes_.end()) {
            ERR(ModelError, "unexpected error");
        }
        nodes_.erase(it);

        // Since producer is already in the next boundary and boundary_node is
        // merged into producer, we don't need to add anything to
        // seen_nodes here.
    }
    this->recursive_merge_nodes(seen_nodes, new_boundary_nodes);
}

nlohmann::ordered_json ModelGraph::Impl::to_json(
    const ModelNodeRef &node) const {
    nlohmann::ordered_json j;
    j["Id"] = nodes_.index(node);
    j["ProducerNodeIds"] = nlohmann::json::array();
    for (auto producer : node->producers) {
        j["ProducerNodeIds"].emplace_back(nodes_.index(producer));
    }
    j["ConsumerNodeIds"] = nlohmann::json::array();
    for (auto consumer : node->consumers) {
        j["ConsumerNodeIds"].emplace_back(nodes_.index(consumer));
    }
    j["Ops"] = nlohmann::json::array();
    for (auto op : node->ops) {
        j["Ops"].emplace_back(op->serialize());
    }
    return j;
}

std::string ModelGraph::Impl::serialize(int indent) const {
    nlohmann::ordered_json j;
    j["Nodes"] = nlohmann::json::array();
    for (const auto &node : nodes_) {
        j["Nodes"].emplace_back(this->to_json(node));
    }
    j["Tensors"] = nlohmann::json::array();
    for (const auto &tensor_and_op : tensor_to_producer_op_) {
        j["Tensors"].emplace_back(tensor_and_op.first->serialize());
    }
    return j.dump(indent);
}

std::vector<ModelNodeRef> ModelGraph::Impl::nodes() const {
    std::vector<ModelNodeRef> vec;
    vec.insert(vec.end(), nodes_.begin(), nodes_.end());
    return vec;
}

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
