// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "sched/sched_opgraph.h"
#include "logging.h"
#include "model.h"
#include <algorithm>

using namespace std;

#define DEBUG_OPGRAPH 0
#define OPGRAPH_DEBUG(...)                                                     \
    do {                                                                       \
        if (DEBUG_OPGRAPH) {                                                   \
            LOG(DEBUG, __VA_ARGS__);                                           \
        }                                                                      \
    } while (0);

namespace ark {

void OpNode::remove_self()
{
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

std::string OpNode::get_name() const
{
    std::stringstream name;
    for (auto &op : this->ops) {
        name << op->name << ";";
    }
    return name.str();
}

OpGraph::OpGraph(const Model &model)
{
    if (!model.verify()) {
        LOG(ERROR, "Model verification failed");
    }
    this->create_nodes(model);
}

OpGraph::OpGraph(OpGraph &graph)
{
    // Copy nodes_storage
    *this = graph;
}

OpGraph &OpGraph::operator=(const OpGraph &graph)
{
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
void OpGraph::create_nodes(const Model &model)
{
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
                                const std::list<OpNode *> &boundary_nodes)
{
    if (boundary_nodes.size() == 0) {
        return;
    }
    OPGRAPH_DEBUG("remove virtual ops");
    std::list<OpNode *> new_boundary_nodes;
    for (auto &boundary_node : boundary_nodes) {
        if (boundary_node->ops.size() != 1) {
            LOG(ERROR, "unexpected error");
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
                LOG(ERROR, "unexpected error: circular dependency detected");
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
                LOG(ERROR, "unexpected error");
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
                              const std::list<OpNode *> &boundary_nodes)
{
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
                LOG(ERROR, "unexpected error: circular dependency detected");
            }
            new_boundary_nodes.emplace_back(producer);
        }
        if (boundary_node->producers.size() > 1) {
            // This node has multiple producers. It cannot be merged.
            seen_nodes.insert(boundary_node);
            OPGRAPH_DEBUG("    multiple producers");
            continue;
        }
        // This node has only one producer.
        OpNode *producer = *(boundary_node->producers.begin());
        if (producer->users.size() == 0) {
            LOG(ERROR, "unexpected error: graph is incomplete");
        }
        if (producer->users.size() > 1) {
            // The producer has multiple users. It cannot be merged.
            seen_nodes.insert(boundary_node);
            OPGRAPH_DEBUG("    multiple users");
            continue;
        }
        // The producer has only one user. Merge the two nodes.

        // Merge `boundary_node` into `producer`.
        OPGRAPH_DEBUG("  merge ops: ", producer->get_name(), " -> ",
                      boundary_node->get_name());
        auto &ops = boundary_node->ops;
        producer->ops.insert(producer->ops.end(), ops.begin(), ops.end());
        producer->users = boundary_node->users;
        for (auto &user : producer->users) {
            user->producers.erase(boundary_node);
            user->producers.insert(producer);
        }

        // Remove `boundary_node` from `nodes`.
        auto it =
            std::find_if(nodes.begin(), nodes.end(),
                         [boundary_node](const std::unique_ptr<OpNode> &node) {
                             return node.get() == boundary_node;
                         });
        if (it == nodes.end()) {
            LOG(ERROR, "unexpected error");
        }
        nodes.erase(it);

        // Since producer is already in the next boundary and boundary_node is
        // merged into producer, we don't need to add anything to
        // seen_nodes here.
    }
    recursive_merge(nodes, seen_nodes, new_boundary_nodes);
}

OpNode *OpGraph::break_node(OpNode *node, int op_idx)
{
    if (op_idx == 0) {
        return node;
    }
    if (op_idx < 0 || op_idx >= (int)node->ops.size()) {
        LOG(ERROR, "unexpected error: op_idx out of range");
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

} // namespace ark
