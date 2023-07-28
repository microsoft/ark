// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef _ARK_SCHED_OPGRAPH_H_
#define _ARK_SCHED_OPGRAPH_H_

#include "json.h"
#include "sched/sched_opseq.h"
#include <cassert>
#include <map>
#include <string>
#include <tuple>

namespace ark {

///
class OpGraphNode
{
  public:
    OpGraphNode(int id, const Op *op, const OpConfig *cfg,
                const std::string &name_)
        : opseq{id, op, cfg}, name{name_}
    {
    }
    // Inward dependencies.
    std::set<OpGraphNode *> inputs;
    // Outward dependencies.
    std::set<OpGraphNode *> outputs;
    // Scheduled depth.
    int depth = -1;
    //
    SchedOpSeq opseq;
    //
    const std::string name;
};

void to_json(nlohmann::json &j, const OpGraphNode &ogn);
void from_json(const nlohmann::json &j, OpGraphNode &ogn);

/// 
struct MergedOps
{
    std::vector<Op *> ops;
    std::set<MergedOps *> users;
    std::set<MergedOps *> producers;

    /// Remove this @ref MergedOps from the graph.
    void remove_self();
};

/// 
class OpGraph
{
  public:
    OpGraph(const Model &model, const GpuInfo &gpu_info);

    void recursive_create_opgraph(std::list<MergedOps *> &merged_ops,
                                  const GpuInfo &gpu_info, int depth);

    size_t get_num_depth() const
    {
        return this->depth_nodes.size();
    }
    const std::vector<OpGraphNode *> &get_depth(int depth) const
    {
        return this->depth_nodes[depth];
    }

    static std::list<std::unique_ptr<MergedOps>> merge_ops(const Model &model);
    static void recursive_rm_virt(
        std::list<std::unique_ptr<MergedOps>> &merged_ops,
        std::set<MergedOps *> &seen_merged_ops,
        const std::list<MergedOps *> &boundary_merged_ops);
    static void recursive_merge(
        std::list<std::unique_ptr<MergedOps>> &merged_ops,
        std::set<MergedOps *> &seen_merged_ops,
        const std::list<MergedOps *> &boundary_merged_ops);

  private:
    std::list<std::unique_ptr<OpGraphNode>> nodes_storage;
    std::vector<std::vector<OpGraphNode *>> depth_nodes;
};

} // namespace ark

#endif
