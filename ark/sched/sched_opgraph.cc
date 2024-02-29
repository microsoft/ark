// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "sched/sched_opgraph.h"

#include <algorithm>

#include "logging.h"
#include "model.h"
#include "nlohmann/json.hpp"

#define DEBUG_OPGRAPH 0
#define OPGRAPH_DEBUG(...)           \
    do {                             \
        if (DEBUG_OPGRAPH) {         \
            LOG(DEBUG, __VA_ARGS__); \
        }                            \
    } while (0);

namespace ark {

void OpNode::remove_self() {
    // Remove self from users and producers.
    for (auto &user : this->users) {
        user->producers.erase(this);
    }
    for (auto &producer : this->producers) {
        producer->users.erase(this);
    }
    // Connect users and producers.
    for (auto &user : this->users) {
        for (auto &producer : this->producers) {
            user->producers.insert(producer);
            producer->users.insert(user);
        }
    }
}

std::string OpNode::get_name() const {
    std::stringstream name;
    for (auto &op : this->ops) {
        name << op->name << ";";
    }
    return name.str();
}

OpGraph::OpGraph(const Model &model) {
    if (!model.verify()) {
        ERR(ModelError, "Model verification failed");
    }
    this->create_nodes(model);
}

OpGraph::OpGraph(OpGraph &graph) {
    // Copy nodes_storage
    *this = graph;
}

OpGraph &OpGraph::operator=(const OpGraph &graph) {
    // Copy nodes_storage
    this->nodes_storage.clear();
    for (auto &node : graph.nodes_storage) {
        this->nodes_storage.emplace_back(std::make_unique<OpNode>());
        this->nodes_storage.back()->ops = node->ops;
        this->nodes_storage.back()->users = node->users;
        this->nodes_storage.back()->producers = node->producers;
    }
    return *this;
}

/// Traverse the model graph and merge Ops that one of them is the only
/// user of the other and the other is the only producer of the first.
///
/// @param model The @ref Model.
///
void OpGraph::create_nodes(const Model &model) {
    std::list<OpNode *> leaf_nodes;
    std::map<const Op *, OpNode *> op2node;
    // Initialize OpNode.
    OPGRAPH_DEBUG("initialize OpNode. ", model.impl->get_ops().size(), " ops");
    for (auto &op : model.impl->get_ops()) {
        this->nodes_storage.emplace_back(std::make_unique<OpNode>());
        this->nodes_storage.back()->ops.emplace_back(op);
        op2node[op] = this->nodes_storage.back().get();
        if (model.impl->get_user_ops(op).size() == 0) {
            leaf_nodes.emplace_back(this->nodes_storage.back().get());
        }
    }
    // Complete producers and users of OpNode.
    for (auto &node : this->nodes_storage) {
        // As nothing is merged yet, all OpNode should have only one Op.
        Op *op = node->ops[0];
        OPGRAPH_DEBUG("node ", op->name);
        for (auto &producer_op : model.impl->get_producer_ops(op)) {
            node->producers.insert(op2node[producer_op]);
            OPGRAPH_DEBUG("  producer ", producer_op->name);
        }
        for (auto &user_op : model.impl->get_user_ops(op)) {
            node->users.insert(op2node[user_op]);
            OPGRAPH_DEBUG("  user ", user_op->name);
        }
    }

    std::set<OpNode *> seen_nodes;

    // Remove virtual Ops.
    recursive_rm_virt(this->nodes_storage, seen_nodes, leaf_nodes);
    seen_nodes.clear();

    // Recreate leaf_nodes.
    leaf_nodes.clear();
    for (auto &node : this->nodes_storage) {
        if (node->users.empty()) {
            leaf_nodes.emplace_back(node.get());
        }
    }

    // Merge Ops.
    recursive_merge(this->nodes_storage, seen_nodes, leaf_nodes);
}

/// Helper of @ref create_nodes().
/// Traverse the model graph and remove virtual Ops that perform no computation.
///
/// @param nodes The list of @ref OpNode.
/// @param boundary_nodes The list of boundary @ref OpNode.
///
void OpGraph::recursive_rm_virt(std::list<std::unique_ptr<OpNode>> &nodes,
                                std::set<OpNode *> &seen_nodes,
                                const std::list<OpNode *> &boundary_nodes) {
    if (boundary_nodes.size() == 0) {
        return;
    }
    OPGRAPH_DEBUG("remove virtual ops");
    std::list<OpNode *> new_boundary_nodes;
    for (auto &boundary_node : boundary_nodes) {
        if (boundary_node->ops.size() == 0) {
            ERR(SchedulerError, "unexpected error: empty OpNode");
        } else if (boundary_node->ops.size() > 1) {
            ERR(SchedulerError, "unexpected error: multiple Ops in OpNode");
        }
        OPGRAPH_DEBUG("  boundary node");
        OPGRAPH_DEBUG("    op: ", boundary_node->get_name());
        for (auto &producer : boundary_node->producers) {
            // Exception: if any user of the producer (rather than the current
            // boundary_node) is unseen, we should not add the producer to the
            // next boundary.
            bool should_add = true;
            for (auto &user : producer->users) {
                if (user == boundary_node) {
                    continue;
                }
                if (seen_nodes.find(user) == seen_nodes.end()) {
                    should_add = false;
                    break;
                }
            }
            if (!should_add) {
                continue;
            }
            if (seen_nodes.find(producer) != seen_nodes.end()) {
                ERR(SchedulerError,
                    "unexpected error: circular dependency detected");
            }
            OPGRAPH_DEBUG("      added ", producer->get_name(),
                          " to next boundary");
            new_boundary_nodes.emplace_back(producer);
        }
        if (boundary_node->ops[0]->is_virtual()) {
            OPGRAPH_DEBUG("    remove op: ", boundary_node->get_name());
            // Remove this node from the graph.
            boundary_node->remove_self();
            // Remove this node from the list of nodes.
            auto it = std::find_if(
                nodes.begin(), nodes.end(),
                [boundary_node](const std::unique_ptr<OpNode> &node) {
                    return node.get() == boundary_node;
                });
            if (it == nodes.end()) {
                ERR(SchedulerError, "unexpected error");
            }
            nodes.erase(it);
            OPGRAPH_DEBUG("      nodes.size() ", nodes.size());
        } else {
            seen_nodes.insert(boundary_node);
        }
    }
    recursive_rm_virt(nodes, seen_nodes, new_boundary_nodes);
}

/// Helper of @ref create_nodes().
/// Traverse the model graph and merge pairs of Ops that are the only user
/// and producer of each other.
///
/// @param nodes The list of @ref OpNode.
/// @param seen_nodes The set of @ref OpNode that have been seen.
/// @param boundary_nodes The list of boundary @ref OpNode.
///
void OpGraph::recursive_merge(std::list<std::unique_ptr<OpNode>> &nodes,
                              std::set<OpNode *> &seen_nodes,
                              const std::list<OpNode *> &boundary_nodes) {
    if (boundary_nodes.size() == 0) {
        return;
    }
    OPGRAPH_DEBUG("merge ops");
    std::list<OpNode *> new_boundary_nodes;
    for (auto &boundary_node : boundary_nodes) {
        OPGRAPH_DEBUG("  boundary node");
        OPGRAPH_DEBUG("    op: ", boundary_node->get_name());
        if (boundary_node->producers.size() == 0) {
            // This node is a root.
            seen_nodes.insert(boundary_node);
            OPGRAPH_DEBUG("    root");
            continue;
        }
        // Add all producers of this node to the next boundary.
        for (auto &producer : boundary_node->producers) {
            // Exception: if any user of the producer (rather than the current
            // boundary_node) is unseen, we should not add the producer to the
            // next boundary.
            bool should_add = true;
            for (auto &user : producer->users) {
                if (user == boundary_node) {
                    continue;
                }
                if (seen_nodes.find(user) == seen_nodes.end()) {
                    should_add = false;
                    break;
                }
            }
            if (!should_add) {
                continue;
            }
            if (seen_nodes.find(producer) != seen_nodes.end()) {
                ERR(SchedulerError,
                    "unexpected error: circular dependency detected");
            }
            new_boundary_nodes.emplace_back(producer);
        }
        OpNode *merge_candidate = nullptr;
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
            if (merge_candidate == nullptr) {
                // At least one producer does not depend on others.
                // Cannot merge.
                seen_nodes.insert(boundary_node);
                OPGRAPH_DEBUG("    multiple producers");
                continue;
            }
        } else {
            // This node has only one producer.
            merge_candidate = *(boundary_node->producers.begin());
        }
        if (merge_candidate->users.size() == 0) {
            ERR(SchedulerError, "unexpected error: graph is incomplete");
        }
        if (merge_candidate->users.size() > 1) {
            // The candidate has multiple users. We can merge only if all
            // other users depend on the current boundary_node.
            bool depends_on_one = true;
            for (auto &user : merge_candidate->users) {
                if (user == boundary_node) {
                    continue;
                }
                if (!this->depends_on(user, boundary_node)) {
                    depends_on_one = false;
                    break;
                }
            }
            if (!depends_on_one) {
                // At least one user does not depend on the boundary_node.
                // Cannot merge.
                seen_nodes.insert(boundary_node);
                OPGRAPH_DEBUG("    multiple users");
                continue;
            }
        }
        // We can merge the two nodes.
        // Merge `boundary_node` into `merge_candidate`.
        OPGRAPH_DEBUG("  merge: ", merge_candidate->get_name(), " -> ",
                      boundary_node->get_name());
        auto &ops = boundary_node->ops;
        merge_candidate->ops.insert(merge_candidate->ops.end(), ops.begin(),
                                    ops.end());
        for (auto &user : boundary_node->users) {
            user->producers.erase(boundary_node);
            user->producers.insert(merge_candidate);
            merge_candidate->users.insert(user);
        }
        for (auto &producer : boundary_node->producers) {
            if (producer == merge_candidate) {
                continue;
            }
            producer->users.erase(boundary_node);
            producer->users.insert(merge_candidate);
            merge_candidate->producers.insert(producer);
        }
        merge_candidate->users.erase(boundary_node);

        // Remove `boundary_node` from `nodes`.
        auto it =
            std::find_if(nodes.begin(), nodes.end(),
                         [boundary_node](const std::unique_ptr<OpNode> &node) {
                             return node.get() == boundary_node;
                         });
        if (it == nodes.end()) {
            ERR(SchedulerError, "unexpected error");
        }
        nodes.erase(it);

        // Since producer is already in the next boundary and boundary_node is
        // merged into producer, we don't need to add anything to
        // seen_nodes here.
    }
    recursive_merge(nodes, seen_nodes, new_boundary_nodes);
}

OpNode *OpGraph::break_node(OpNode *node, int op_idx) {
    if (op_idx == 0) {
        return node;
    }
    if (op_idx < 0 || op_idx >= (int)node->ops.size()) {
        ERR(SchedulerError, "unexpected error: op_idx out of range");
    }
    this->nodes_storage.emplace_back(std::make_unique<OpNode>());
    OpNode *new_node = this->nodes_storage.back().get();
    new_node->ops.insert(new_node->ops.end(), node->ops.begin() + op_idx,
                         node->ops.end());
    new_node->users = node->users;
    new_node->producers.insert(node);
    for (auto &user : node->users) {
        user->producers.erase(node);
        user->producers.insert(new_node);
    }
    node->ops.erase(node->ops.begin() + op_idx, node->ops.end());
    node->users.clear();
    node->users.insert(new_node);
    return new_node;
}

/// Check dependencies between two @ref OpNode.
///
/// @param node1 The first @ref OpNode.
/// @param node2 The second @ref OpNode.
/// @return True if @p node1 depends on @p node2.
bool OpGraph::depends_on(OpNode *node1, OpNode *node2) const {
    if (node1 == node2) {
        return false;
    }
    std::set<OpNode *> seen_nodes;
    std::list<OpNode *> boundary_nodes;
    boundary_nodes.emplace_back(node1);
    while (boundary_nodes.size() > 0) {
        std::list<OpNode *> new_boundary_nodes;
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

nlohmann::json to_json(const TensorBuf &tensor_buf) {
    nlohmann::json j;
    j["Id"] = tensor_buf.id;
    j["Bytes"] = tensor_buf.bytes;
    return j;
}

nlohmann::json to_json(const Tensor &tensor) {
    nlohmann::json j;
    j["Id"] = tensor.id;
    j["TensorBuf"] = to_json(*(tensor.buf));
    j["TensorType"] = tensor.type.type_str();
    j["Shape"] = tensor.shape.serialize();
    j["Strides"] = tensor.ldims.serialize();
    j["Offsets"] = tensor.offs.serialize();
    if (tensor.imported_rank >= 0) {
        j["ImportedRank"] = tensor.imported_rank;
    }
    return j;
}

nlohmann::json to_json(const OpArg &op_arg) {
    nlohmann::json j;
    j["Type"] = op_arg.type.name;
    if (op_arg.type == OP_ARG_TENSOR) {
        Tensor *tns;
        op_arg.get(&tns);
        j["Value"] = tns->id;
    } else if (op_arg.type == OP_ARG_FLOAT) {
        float val;
        op_arg.get(&val);
        j["Value"] = val;
    } else if (op_arg.type == OP_ARG_INT) {
        int val;
        op_arg.get(&val);
        j["Value"] = val;
    } else if (op_arg.type == OP_ARG_BOOL) {
        bool val;
        op_arg.get(&val);
        j["Value"] = val;
    } else if (op_arg.type == OP_ARG_INT64) {
        long long int val;
        op_arg.get(&val);
        j["Value"] = val;
    } else if (op_arg.type == OP_ARG_UINT64) {
        uint64_t val;
        op_arg.get(&val);
        j["Value"] = val;
    } else if (op_arg.type == OP_ARG_DIMS) {
        Dims dims;
        op_arg.get(&dims);
        j["Value"] = dims.serialize();
    }
    else {
        throw std::runtime_error("unexpected OpArg: "+std::string(op_arg.type.name));
    }
    return j;
}

nlohmann::json to_json(const Op &op) {
    nlohmann::json j;
    j["Type"] = op.type.name;
    j["PrecisionType"] = op.prec_type;
    j["InputTensors"] = nlohmann::json();
    for (auto tensor : op.inputs) {
        j["InputTensors"].emplace_back(to_json(*tensor));
    }
    j["OutputTensors"] = nlohmann::json();
    for (auto tensor : op.inputs) {
        j["OutputTensors"].emplace_back(to_json(*tensor));
    }
    j["OutputRefTensors"] = nlohmann::json();
    for (auto tensor : op.inputs) {
        j["OutputRefTensors"].emplace_back(to_json(*tensor));
    }
    j["Args"] = nlohmann::json();
    for (auto arg : op.args.get_args()) {
        j["Args"].emplace_back(to_json(arg));
    }
    return j;
}

nlohmann::json to_json(const OpNode &node, const std::map<const OpNode *, size_t> &node2id) {
    nlohmann::json j;
    j["Id"] = node2id.at(&node);
    j["Ops"] = nlohmann::json();
    for (auto op : node.ops) {
        j["Ops"].emplace_back(to_json(*op));
    }
    j["ConsumerNodeIds"] = nlohmann::json();
    for (auto user : node.users) {
        j["ConsumerNodeIds"].emplace_back(node2id.at(user));
    }
    j["ProducerNodeIds"] = nlohmann::json();
    for (auto producer : node.producers) {
        j["ProducerNodeIds"].emplace_back(node2id.at(producer));
    }
    return j;
}

nlohmann::json to_json(const OpGraph &opgraph) {
    size_t id = 0;
    std::map<const OpNode *, size_t> node2id;
    for (const auto &node : opgraph.get_nodes()) {
        node2id[node.get()] = id++;
    }
    nlohmann::json j;
    j["Nodes"] = nlohmann::json();
    for (const auto &node : opgraph.get_nodes()) {
        j["Nodes"].emplace_back(to_json(*node, node2id));
    }
    return j;
}

std::string OpGraph::serialize(int indent) const {
    auto j = to_json(*this);
    return j.dump(indent);
}

}  // namespace ark
