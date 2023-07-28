// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "json.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <initializer_list>
#include <ostream>
#include <unistd.h>

#include "env.h"
#include "logging.h"
#include "math.h"
#include "model.h"
#include "sched/sched_opgraph.h"

using namespace std;

#define DEBUG_OPGRAPH 0
#define OPGRAPH_DEBUG(...)                                                     \
    do {                                                                       \
        if (DEBUG_OPGRAPH) {                                                   \
            LOG(DEBUG, __VA_ARGS__);                                           \
        }                                                                      \
    } while (0);

namespace ark {

void MergedOps::remove_self()
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

/// Helper of @ref merge_ops().
/// Traverse the model graph and remove virtual Ops that perform no computation.
///
/// @param merged_ops The list of @ref MergedOps.
/// @param boundary_merged_ops The list of boundary @ref MergedOps.
///
void OpGraph::recursive_rm_virt(
    std::list<std::unique_ptr<MergedOps>> &merged_ops,
    std::set<MergedOps *> &seen_merged_ops,
    const std::list<MergedOps *> &boundary_merged_ops)
{
    if (boundary_merged_ops.size() == 0) {
        return;
    }
    OPGRAPH_DEBUG("remove virtual ops");
    std::list<MergedOps *> new_boundary_merged_ops;
    for (auto &boundary_mop : boundary_merged_ops) {
        if (boundary_mop->ops.size() != 1) {
            LOG(ERROR, "unexpected error");
        }
        OPGRAPH_DEBUG("  boundary mop");
        OPGRAPH_DEBUG("    op: ", boundary_mop->ops[0]->name);
        for (auto &producer : boundary_mop->producers) {
            // Exception: if any user of the producer (rather than the current
            // boundary_mop) is unseen, we should not add the producer to the
            // next boundary.
            bool should_add = true;
            for (auto &user : producer->users) {
                if (user == boundary_mop) {
                    continue;
                }
                if (seen_merged_ops.find(user) == seen_merged_ops.end()) {
                    should_add = false;
                    break;
                }
            }
            if (!should_add) {
                continue;
            }
            if (seen_merged_ops.find(producer) != seen_merged_ops.end()) {
                LOG(ERROR, "unexpected error: circular dependency detected");
            }
            OPGRAPH_DEBUG("      added ", producer->ops[0]->name,
                          " to next boundary");
            new_boundary_merged_ops.emplace_back(producer);
        }
        if (boundary_mop->ops[0]->is_virtual()) {
            OPGRAPH_DEBUG("    remove op: ", boundary_mop->ops[0]->name);
            // Remove this node from the graph.
            boundary_mop->remove_self();
            // Remove this node from the list of merged_ops.
            auto it = std::find_if(
                merged_ops.begin(), merged_ops.end(),
                [boundary_mop](const std::unique_ptr<MergedOps> &mop) {
                    return mop.get() == boundary_mop;
                });
            if (it == merged_ops.end()) {
                LOG(ERROR, "unexpected error");
            }
            merged_ops.erase(it);
            OPGRAPH_DEBUG("      merged_ops.size() ", merged_ops.size());
        } else {
            seen_merged_ops.insert(boundary_mop);
        }
    }
    recursive_rm_virt(merged_ops, seen_merged_ops, new_boundary_merged_ops);
}

/// Helper of @ref merge_ops().
/// Traverse the model graph and merge pairs of Ops that are the only user
/// and producer of each other.
///
/// @param merged_ops The list of @ref MergedOps.
/// @param seen_merged_ops The set of @ref MergedOps that have been seen.
/// @param boundary_merged_ops The list of boundary @ref MergedOps.
///
void OpGraph::recursive_merge(std::list<std::unique_ptr<MergedOps>> &merged_ops,
                              std::set<MergedOps *> &seen_merged_ops,
                              const std::list<MergedOps *> &boundary_merged_ops)
{
    if (boundary_merged_ops.size() == 0) {
        return;
    }
    OPGRAPH_DEBUG("merge ops");
    std::list<MergedOps *> new_boundary_merged_ops;
    for (auto &boundary_mop : boundary_merged_ops) {
        OPGRAPH_DEBUG("  boundary mop");
        OPGRAPH_DEBUG("    op: ", boundary_mop->ops[0]->name);
        if (boundary_mop->producers.size() == 0) {
            // This node is a root.
            seen_merged_ops.insert(boundary_mop);
            OPGRAPH_DEBUG("    root");
            continue;
        }
        // Add all producers of this node to the next boundary.
        for (auto &producer : boundary_mop->producers) {
            // Exception: if any user of the producer (rather than the current
            // boundary_mop) is unseen, we should not add the producer to the
            // next boundary.
            bool should_add = true;
            for (auto &user : producer->users) {
                if (user == boundary_mop) {
                    continue;
                }
                if (seen_merged_ops.find(user) == seen_merged_ops.end()) {
                    should_add = false;
                    break;
                }
            }
            if (!should_add) {
                continue;
            }
            if (seen_merged_ops.find(producer) != seen_merged_ops.end()) {
                LOG(ERROR, "unexpected error: circular dependency detected");
            }
            new_boundary_merged_ops.emplace_back(producer);
        }
        if (boundary_mop->producers.size() > 1) {
            // This node has multiple producers. It cannot be merged.
            seen_merged_ops.insert(boundary_mop);
            OPGRAPH_DEBUG("    multiple producers");
            continue;
        }
        // This node has only one producer.
        MergedOps *producer = *(boundary_mop->producers.begin());
        if (producer->users.size() == 0) {
            LOG(ERROR, "unexpected error: graph is incomplete");
        }
        if (producer->users.size() > 1) {
            // The producer has multiple users. It cannot be merged.
            seen_merged_ops.insert(boundary_mop);
            OPGRAPH_DEBUG("    multiple users");
            continue;
        }
        // The producer has only one user. Merge the two nodes.

        // Merge `boundary_mop` into `producer`.
        OPGRAPH_DEBUG("  merge ops: ", producer->ops[0]->name, " -> ",
                      boundary_mop->ops[0]->name);
        auto &ops = boundary_mop->ops;
        producer->ops.insert(producer->ops.end(), ops.begin(), ops.end());
        producer->users = boundary_mop->users;

        // Remove `boundary_mop` from `merged_ops`.
        auto it =
            std::find_if(merged_ops.begin(), merged_ops.end(),
                         [boundary_mop](const std::unique_ptr<MergedOps> &mop) {
                             return mop.get() == boundary_mop;
                         });
        if (it == merged_ops.end()) {
            LOG(ERROR, "unexpected error");
        }
        merged_ops.erase(it);

        // Since producer is already in the next boundary and boundary_mop is
        // merged into producer, we don't need to add anything to
        // seen_merged_ops here.
    }
    recursive_merge(merged_ops, seen_merged_ops, new_boundary_merged_ops);
}

/// Traverse the model graph and merge Ops that one of them is the only
/// user of the other and the other is the only producer of the first.
///
/// @param model The @ref Model.
/// @return A list of unique_ptr of all @ref MergedOps.
///
std::list<std::unique_ptr<MergedOps>> OpGraph::merge_ops(const Model &model)
{
    std::list<std::unique_ptr<MergedOps>> merged_ops;
    std::list<MergedOps *> leaf_merged_ops;
    std::map<const Op *, MergedOps *> op2merged_ops;
    // Initialize MergedOps.
    OPGRAPH_DEBUG("initialize MergedOps. ", model.impl->get_ops().size(),
                  " ops");
    for (auto &op : model.impl->get_ops()) {
        merged_ops.emplace_back(std::make_unique<MergedOps>());
        merged_ops.back()->ops.emplace_back(op);
        op2merged_ops[op] = merged_ops.back().get();
        if (model.impl->get_user_ops(op).size() == 0) {
            leaf_merged_ops.emplace_back(merged_ops.back().get());
        }
    }
    // Complete producers and users of MergedOps.
    for (auto &mop : merged_ops) {
        // As nothing is merged yet, all MergedOps should have only one Op.
        Op *op = mop->ops[0];
        OPGRAPH_DEBUG("mop ", op->name);
        for (auto &producer_op : model.impl->get_producer_ops(op)) {
            mop->producers.insert(op2merged_ops[producer_op]);
            OPGRAPH_DEBUG("  producer ", producer_op->name);
        }
        for (auto &user_op : model.impl->get_user_ops(op)) {
            mop->users.insert(op2merged_ops[user_op]);
            OPGRAPH_DEBUG("  user ", user_op->name);
        }
    }

    std::set<MergedOps *> seen_merged_ops;

    // Remove virtual Ops.
    recursive_rm_virt(merged_ops, seen_merged_ops, leaf_merged_ops);
    seen_merged_ops.clear();

    // Recreate leaf_merged_ops.
    leaf_merged_ops.clear();
    for (auto &mop : merged_ops) {
        if (mop->users.empty()) {
            leaf_merged_ops.emplace_back(mop.get());
        }
    }

    // Merge Ops.
    recursive_merge(merged_ops, seen_merged_ops, leaf_merged_ops);
    return merged_ops;
}

/// Construct an @ref OpGraph from a @ref Model.
///
/// The @ref OpGraph is a DAG of operators, where each @ref OpGraphNode is a
/// node. The edges are the dependencies between @ref OpGraphNode.
///
/// @param model The @ref Model.
/// @param gpu_info @ref GpuInfo of the GPU to run the model on.
///
OpGraph::OpGraph(const Model &model, const GpuInfo &gpu_info)
{
    if (!model.verify()) {
        LOG(ERROR, "Model verification failed");
    }

    std::list<std::unique_ptr<MergedOps>> merged_ops = this->merge_ops(model);

    std::list<MergedOps *> root_merged_ops;
    for (auto &mop : merged_ops) {
        if (mop->producers.empty()) {
            root_merged_ops.emplace_back(mop.get());
        }
    }

    // Note that this function may modify MergedOps.
    recursive_create_opgraph(root_merged_ops, gpu_info, 0);
}

/// Recursively create the @ref OpGraph.
///
/// @param merged_ops The list of @ref MergedOps to create the @ref OpGraph
/// from.
/// @param gpu_info @ref GpuInfo of the GPU to run the model on.
/// @param depth The depth of the current @ref OpGraph.
///
void OpGraph::recursive_create_opgraph(std::list<MergedOps *> &merged_ops,
                                       const GpuInfo &gpu_info, int depth)
{
    if (merged_ops.empty()) {
        return;
    }
    this->depth_nodes.emplace_back();

    std::list<MergedOps *> next_merged_ops;
    for (auto &mop : merged_ops) {
        if (mop->ops.size() == 0) {
            LOG(ERROR, "unexpected error: empty MergedOps");
        }
        int id = this->nodes_storage.size();
        Op *op = mop->ops[0];
        const OpConfig *cfg = sched_op_config(op, gpu_info);
        this->nodes_storage.emplace_back(
            std::make_unique<OpGraphNode>(id, mop->ops[0], cfg, ""));
        OpGraphNode *node = this->nodes_storage.back().get();

        this->depth_nodes[depth].emplace_back(node);
        node->depth = depth;

        for (size_t i = 1; i < mop->ops.size(); i++) {
            // If there are multiple Ops, check if the Op configs allow merging.
            Op *next_op = mop->ops[i];
            const OpConfig *next_cfg = sched_op_config(next_op, gpu_info);
            if (cfg->sync_post || next_cfg->sync_pre) {
                // Cannot merge as we need a global sync between the two Ops.
                // Add remaining part of the MergedOps to next_merged_ops.
                std::vector<Op *> remaining_ops(mop->ops.begin() + i,
                                                mop->ops.end());
                mop->ops = std::move(remaining_ops);
                next_merged_ops.emplace_back(mop);
                break;
            } else {
                // Merge the next Op into the node.
                node->opseq.append(next_op, next_cfg);
            }
        }

        if (next_merged_ops.back() != mop) {
            // If MergedOp is completely merged, add its users to
            // next_merged_ops.
            for (auto &user_mop : mop->users) {
                next_merged_ops.emplace_back(user_mop);
            }
        }
    }
    recursive_create_opgraph(next_merged_ops, gpu_info, depth + 1);
}

} // namespace ark
