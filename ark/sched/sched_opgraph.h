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

class OpGraphNode
{
  public:
    OpGraphNode(int id, const Op *op, const OpConfig *cfg,
                const std::string &name_)
        : opseq{id, op, cfg}, name{name_}
    {
    }
    // Inward dependencies.
    std::set<OpGraphNode *> in_deps;
    // Outward dependencies.
    std::set<OpGraphNode *> out_deps;
    // Scheduled depth.
    int depth = -1;
    //
    SchedOpSeq opseq;
    //
    const std::string name;
};

void to_json(nlohmann::json &j, const OpGraphNode &ogn);
void from_json(const nlohmann::json &j, OpGraphNode &ogn);

//
class OpGraph
{
  public:
    OpGraph(const Model &model, const GpuInfo &gpu_info);

    size_t get_num_depth() const
    {
        return this->depth_nodes.size();
    }
    const std::vector<OpGraphNode *> &get_depth(int depth) const
    {
        return this->depth_nodes[depth];
    }

  private:
    OpGraphNode *get_or_create_node(int id, const Op *op, const OpConfig *cfg)
    {
        auto search = this->op_to_node_map.find(op);
        if (search != this->op_to_node_map.end()) {
            return search->second;
        }
        OpGraphNode *ogn = new OpGraphNode{id, op, cfg, ""};
        this->nodes_storage.emplace_back(ogn);
        this->op_to_node_map[op] = ogn;
        return ogn;
    }

    OpGraphNode *get_node(const Op *op) const
    {
        auto search = this->op_to_node_map.find(op);
        if (search != this->op_to_node_map.end()) {
            return search->second;
        }
        return nullptr;
    }

    std::list<std::unique_ptr<OpGraphNode>> nodes_storage;
    std::map<const ark::Op *, OpGraphNode *> op_to_node_map;
    std::vector<std::vector<OpGraphNode *>> depth_nodes;
};

// void to_json(nlohmann::json &j, const OpGraphNode &ogn)
// {
//     j = nlohmann::json{
//         {"opseq", ogn.opseq},
//         {"depth", ogn.depth},
//         {"in_deps", std::vector<int>{}},
//         {"out_deps", std::vector<int>{}},
//     };
//     for (OpGraphNode *in_dep : ogn.in_deps) {
//         j.at("in_deps").emplace_back(in_dep->opseq.get_id());
//     }
//     for (OpGraphNode *out_dep : ogn.out_deps) {
//         j.at("out_deps").emplace_back(out_dep->opseq.get_id());
//     }
// }
// void from_json(const nlohmann::json &j, OpGraphNode &ogn)
// {
// }

} // namespace ark

#endif
