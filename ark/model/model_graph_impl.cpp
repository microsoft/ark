// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model_graph_impl.hpp"

#include "logging.hpp"
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

ModelGraphContextStack::ModelGraphContextStack(
    const ModelGraphContextStack &other) {
    for (const auto &pair : other.storage_) {
        for (const auto &value : pair.second) {
            this->storage_[pair.first].push_back(value);
        }
    }
}

void ModelGraphContextStack::push(const std::string &key, const Json &value) {
    this->storage_[key].push_back(std::make_shared<Json>(value));
}

void ModelGraphContextStack::pop(const std::string &key) {
    auto it = this->storage_.find(key);
    if (it == this->storage_.end() || it->second.empty()) {
        ERR(InternalError, "context stack is empty");
    }
    it->second.pop_back();
}

Json ModelGraphContextStack::get_context(const std::string &key) const {
    if (this->storage_.find(key) == this->storage_.end() ||
        this->storage_.at(key).empty()) {
        return Json();
    }
    return *this->storage_.at(key).back();
}

std::map<std::string, Json> ModelGraphContextStack::get_context_all() const {
    std::map<std::string, Json> cur;
    for (const auto &pair : this->storage_) {
        if (!pair.second.empty()) {
            cur[pair.first] = *pair.second.back();
        }
    }
    return cur;
}

ModelGraph::Impl::Impl(const ModelGraph::Impl &other) { *this = other; }

ModelGraph::Impl &ModelGraph::Impl::operator=(const ModelGraph::Impl &other) {
    std::map<ModelNodeRef, ModelNodeRef> node_map;
    nodes_.clear();
    for (const auto &node : other.nodes_) {
        ModelNodeRef new_node = std::make_shared<ModelNode>();
        new_node->op = node->op;
        new_node->context = node->context;
        node_map.emplace(node, new_node);
        nodes_.push_back(new_node);
    }
    for (const auto &node : other.nodes_) {
        auto it = node_map.find(node);
        if (it == node_map.end()) {
            ERR(InternalError, "unexpected error");
        }
        ModelNodeRef new_node = it->second;
        for (auto &producer : node->producers) {
            auto it2 = node_map.find(producer);
            if (it2 == node_map.end()) {
                ERR(InternalError, "unexpected error");
            }
            new_node->producers.push_back(it2->second);
        }
        for (auto &consumer : node->consumers) {
            auto it2 = node_map.find(consumer);
            if (it2 == node_map.end()) {
                ERR(InternalError, "unexpected error");
            }
            new_node->consumers.push_back(it2->second);
        }
    }
    op_to_node_.clear();
    for (const auto &p : other.op_to_node_) {
        auto it = node_map.find(p.second);
        if (it == node_map.end()) {
            ERR(InternalError, "unexpected error");
        }
        op_to_node_[p.first] = it->second;
    }
    tensor_to_producer_op_ = other.tensor_to_producer_op_;
    rank_ = other.rank_;
    world_size_ = other.world_size_;
    compressed_ = other.compressed_;
    context_stack_ =
        std::make_shared<ModelGraphContextStack>(*(other.context_stack_));
    return *this;
}

void ModelGraph::Impl::compress_nodes() {
    if (!compressed_) {
        this->recursive_remove_virtual_nodes();
        compressed_ = true;
    }
}

bool ModelGraph::Impl::verify() const {
    for (auto &node : nodes_) {
        if (node->op == nullptr) {
            LOG(DEBUG, "node has no op");
            return false;
        }
        if (op_to_node_.find(node->op) == op_to_node_.end()) {
            LOG(DEBUG, "op has not been added to the graph");
            return false;
        }
        if (op_to_node_.at(node->op) != node) {
            LOG(DEBUG, "op is not in the correct node");
            return false;
        }
        node->op->verify();
        for (auto &tns : node->op->result_tensors()) {
            if (tensor_to_producer_op_.find(tns) ==
                tensor_to_producer_op_.end()) {
                LOG(DEBUG, "result tensor has not been produced by any op");
                return false;
            }
            if (tensor_to_producer_op_.at(tns) != node->op) {
                LOG(DEBUG, "result tensor has been produced by another op");
                return false;
            }
        }
        for (auto &tns : node->op->input_tensors()) {
            if (tensor_to_producer_op_.find(tns) ==
                tensor_to_producer_op_.end()) {
                LOG(DEBUG, "input tensor has not been produced by any op");
                return false;
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
    try {
        ModelJson j(Json::parse(this->serialize(-1)));
        // avoid compiler optimization
        if (j.is_null()) {
            return false;
        }
    } catch (const BaseError &e) {
        LOG(DEBUG, "failed to serialize/parse the model graph: ", e.what());
        return false;
    }
    return true;
}

Json ModelGraph::Impl::get_context(const std::string &key) const {
    return context_stack_->get_context(key);
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
            ERR(InternalError, "Tensor has already been produced by an op. ",
                tns->serialize().dump(), "; ",
                tensor_to_producer_op_.at(tns)->serialize().dump());
        }
        tensor_to_producer_op_.emplace(tns, op);
    }

    ModelNodeRef node = std::make_shared<ModelNode>();
    node->op = op;
    op_to_node_[op] = node;

    for (auto &tns : op->input_tensors()) {
        auto it = tensor_to_producer_op_.find(tns);
        if (it == tensor_to_producer_op_.end()) {
            ERR(InternalError, "Tensor has not been produced by any op. ",
                tns->serialize().dump(), " ", tns.get());
        }
        auto it2 = op_to_node_.find(it->second);
        if (it2 == op_to_node_.end()) {
            ERR(InternalError, "Op has not been added to the graph");
        }
        auto producer = it2->second;
        node->producers.push_back(producer);
        producer->consumers.push_back(node);
    }

    node->context = context_stack_->get_context_all();

    nodes_.push_back(node);
    return node;
}

void ModelGraph::Impl::remove_node(ModelNodeRef node) {
    auto it = nodes_.find(node);
    if (it == nodes_.end()) {
        ERR(InternalError,
            "attempted to remove a node that is not in the graph");
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
    auto it2 = op_to_node_.find(node->op);
    if (it2 == op_to_node_.end()) {
        ERR(InternalError, "unexpected error");
    }
    if (it2->second == node) {
        op_to_node_.erase(it2);
    }
    nodes_.erase(it);
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
        if (boundary_node->op == nullptr) {
            ERR(InternalError, "unexpected error: empty node");
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
                ERR(InternalError,
                    "circular dependency detected: ", to_json(producer).dump());
            }
            MODEL_GRAPH_DEBUG("      added to next boundary: ",
                              to_json(producer).dump());
            new_boundary_nodes.emplace_back(producer);
        }
        if (boundary_node->op->is_virtual()) {
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

Json ModelGraph::Impl::to_json(const ModelNodeRef &node) const {
    Json j;
    j["Id"] = nodes_.index(node);
    j["ProducerNodeIds"] = Json::array();
    for (auto producer : node->producers) {
        j["ProducerNodeIds"].emplace_back(nodes_.index(producer));
    }
    j["ConsumerNodeIds"] = Json::array();
    for (auto consumer : node->consumers) {
        j["ConsumerNodeIds"].emplace_back(nodes_.index(consumer));
    }
    j["Op"] = node->op->serialize();
    return j;
}

std::string ModelGraph::Impl::serialize(bool pretty) const {
    Json j;
    j["Rank"] = rank_;
    j["WorldSize"] = world_size_;
    j["Nodes"] = Json::array();
    for (const auto &node : nodes_) {
        j["Nodes"].emplace_back(this->to_json(node));
    }
    if (pretty) {
        return ModelJson(j).dump_pretty();
    }
    return j.dump(-1);
}

std::vector<ModelNodeRef> ModelGraph::Impl::nodes() const {
    std::vector<ModelNodeRef> vec;
    vec.insert(vec.end(), nodes_.begin(), nodes_.end());
    return vec;
}

}  // namespace ark
