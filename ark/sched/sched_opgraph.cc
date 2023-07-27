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

namespace ark {

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
    int opseq_id = 0;
    set<const Op *> seen;
    std::list<std::list<OpGraphNode *>> tmp_depth_nodes;
    tmp_depth_nodes.emplace_front();
    list<OpGraphNode *> *depth = &(tmp_depth_nodes.front());
    {
        set<Tensor *> final_outputs;
        for (auto &tns : model.impl->get_tensors()) {
            if (model.impl->is_no_ref(tns)) {
                final_outputs.emplace(tns);
            }
        }
        for (Tensor *out : final_outputs) {
            const Op *op = model.impl->get_gen_op(out);
            if (op != nullptr) {
                const OpConfig *cfg = sched_op_config(op, gpu_info);
                OpGraphNode *ogn =
                    this->get_or_create_node(opseq_id++, op, cfg);
                // LOG(INFO, "OGN: ", ogn);
                depth->emplace_back(ogn);
                seen.emplace(op);
                // LOG(DEBUG, "retrieve: final op ", op->type);
            }
        }
    }
    while (depth->size() > 0) {
        list<OpGraphNode *> *depth_prev = depth;
        tmp_depth_nodes.emplace_front();
        depth = &(tmp_depth_nodes.front());
        // LOG(INFO, "NEXT DEPTH -----------------------");

        set<const Op *> seen_tmp;
        auto it = depth_prev->begin();
        for (; it != depth_prev->end(); ++it) {
            SchedOpSeq &opseq = (*it)->opseq;
            // LOG(INFO, "retrieve: ", *it, ", last op ",
            //     (*it)->opseq.get_last_op()->type, " ",
            //     (*it)->opseq.get_last_op()->name);
            for (;;) {
                // Get all Ops of which results are used only by seen Ops
                // and at least one result is used by `opseq.back()`. If there
                // is only one such Op, it can be merged into `opseq` unless
                // this Op requires a global sync after execution or `opseq`
                // requires a global sync before execution.
                set<const Op *> dep_ops;
                for (Tensor *tns : opseq.get_last_op()->in_deps) {
                    const Op *op = model.impl->get_gen_op(tns);
                    if (op == nullptr) {
                        // No Op generates this tensor.
                        continue;
                    }
                    // Ignore if any output of `op` is referred by an unseen Op.
                    bool is_only = true;
                    for (auto &out_tns : op->out_deps) {
                        if (model.impl->is_no_ref(out_tns)) {
                            // `out_tns` is used nowhere. (pass)
                            continue;
                        }
                        for (auto &ref_op : model.impl->get_ref_ops(out_tns)) {
                            auto search = seen.find(ref_op);
                            if (search == seen.end()) {
                                // `out_tns` is used by an unseen Op. (fail)
                                is_only = false;
                                break;
                            }
                            // `out_tns` is used by a seen Op. (pass)
                        }
                        if (!is_only) {
                            break;
                        }
                    }
                    if (is_only) {
                        dep_ops.emplace(op);
                        // LOG(DEBUG, "retrieve: dep op ", op->name);
                    }
                }
                // If there is only one such Op, check if this can be
                // merged into `opseq`.
                if (dep_ops.size() == 1) {
                    const Op *op = *dep_ops.begin();
                    const OpConfig *cfg = sched_op_config(op, gpu_info);
                    // Cannot merge if `opseq` needs a global sync.
                    const OpConfig *prev_op_cfg =
                        opseq.get_sched_ops().back().get_cfg();
                    bool can_merge = true;
                    if (cfg != nullptr && cfg->sync_post) {
                        can_merge = false;
                    }
                    if (prev_op_cfg != nullptr && prev_op_cfg->sync_pre) {
                        can_merge = false;
                    }
                    if (can_merge) {
                        // Get all Ops which depends on results of `op`.
                        set<const Op *> out_dep_ops;
                        for (auto &out_tns : op->out_deps) {
                            if (model.impl->is_no_ref(out_tns)) {
                                continue;
                            }
                            for (auto &ref_op :
                                 model.impl->get_ref_ops(out_tns)) {
                                out_dep_ops.emplace(ref_op);
                            }
                        }
                        // If both `op` and `opseq` are not virtual,
                        // the batch size should be the same.
                        if ((cfg == nullptr) || (opseq.is_virtual()) ||
                            (opseq.get_tdim_z() == op->out_deps[0]->shape[0])) {
                            // Check if all Ops in `out_dep_ops` are in `opseq`.
                            // If so, we can merge `op` into `opseq`.
                            for (auto &sop : opseq.get_sched_ops()) {
                                auto search = out_dep_ops.find(sop.get_op());
                                if (search != out_dep_ops.end()) {
                                    out_dep_ops.erase(search);
                                }
                                if (out_dep_ops.size() == 0) {
                                    break;
                                }
                            }
                            if (out_dep_ops.size() == 0) {
                                // Try merge and continue if succeed.
                                if (opseq.append(
                                        op, sched_op_config(op, gpu_info))) {
                                    auto p = seen.emplace(op);
                                    if (!p.second) {
                                        LOG(ERROR, "unexpected error");
                                    }
                                    this->op_to_node_map[op] = *it;
                                    // LOG(DEBUG, "retrieve: merge");
                                    continue;
                                }
                            }
                        }
                    }
                } else if (dep_ops.size() > 1 && opseq.is_virtual()) {
                    // Ignore current `opseq` and continue on its dependencies
                    // instead.
                    for (const Op *op : dep_ops) {
                        auto p = seen.emplace(op);
                        if (!p.second) {
                            continue;
                        }
                        const OpConfig *cfg = sched_op_config(op, gpu_info);
                        OpGraphNode *ogn =
                            this->get_or_create_node(opseq_id++, op, cfg);
                        // stringstream ssod;
                        // for (auto &od : (*it)->out_deps) {
                        //     ssod << od << ",";
                        // }
                        // LOG(INFO, "OGN: ", ogn, " --> ", ssod.str());
                        depth_prev->emplace_back(ogn);
                        //
                        ogn->out_deps.insert((*it)->out_deps.begin(),
                                             (*it)->out_deps.end());
                        for (auto &od : (*it)->out_deps) {
                            od->in_deps.insert(ogn);
                        }
                    }
                    break;
                }
                // Put `dep_ops` into `depth` and `seen`, and break.
                for (const Op *op : dep_ops) {
                    auto p = seen_tmp.emplace(op);
                    const OpConfig *cfg = sched_op_config(op, gpu_info);
                    if (p.second) {
                        OpGraphNode *ogn =
                            this->get_or_create_node(opseq_id++, op, cfg);
                        depth->emplace_back(ogn);
                        //
                        (*it)->in_deps.emplace(ogn);

                        for (Tensor *tns : op->out_deps) {
                            for (const Op *out_dep :
                                 model.impl->get_ref_ops(tns)) {
                                OpGraphNode *out_dep_node =
                                    this->get_node(out_dep);
                                assert(out_dep_node != nullptr);
                                // LOG(INFO, "OGN: ", ogn, " --> ",
                                // out_dep_node);
                                out_dep_node->in_deps.insert(ogn);
                                ogn->out_deps.insert(out_dep_node);
                            }
                        }
                    } else {
                        OpGraphNode *ogn = this->get_node(op);
                        assert(ogn != nullptr);
                        // LOG(INFO, "OGN: ", ogn, " --> ", *it);
                        //
                        ogn->out_deps.emplace(*it);
                        (*it)->in_deps.emplace(ogn);
                    }
                }
                break;
            }
        }
        //
        seen.insert(seen_tmp.begin(), seen_tmp.end());
        if (depth_prev->size() == 0) {
            // this->depth_nodes.erase(next(this->depth_nodes.begin()));
            // LOG(DEBUG, "retrieve: ---------- next depth");
        }
    }

    for (auto &depth : tmp_depth_nodes) {
        // Remove virtual operations.
        for (auto it = depth.begin(); it != depth.end();) {
            if ((*it)->opseq.is_virtual()) {
                // LOG(INFO, "Remove OGN: ", *it);
                for (OpGraphNode *out_dep : (*it)->out_deps) {
                    for (OpGraphNode *in_dep : (*it)->in_deps) {
                        in_dep->out_deps.emplace(out_dep);
                        out_dep->in_deps.emplace(in_dep);
                    }
                }
                for (OpGraphNode *out_dep : (*it)->out_deps) {
                    out_dep->in_deps.erase(*it);
                }
                for (OpGraphNode *in_dep : (*it)->in_deps) {
                    in_dep->out_deps.erase(*it);
                }
                it = depth.erase(it);
            } else {
                ++it;
            }
        }
    }

    int depth_num = 0;
    for (auto &depth : tmp_depth_nodes) {
        // LOG(INFO, "Depth ", depth_num, " -------------------- ");
        for (OpGraphNode *ogn : depth) {
            ogn->depth = depth_num;
            // stringstream info;
            // for (OpGraphNode *x : ogn->in_deps) {
            //     info << x << ",";
            // }
            // info << " --> " << ogn << " --> ";
            // for (OpGraphNode *x : ogn->out_deps) {
            //     info << x << ",  ";
            // }
            // for (const SchedOp& sop : ogn->opseq.get_sched_ops()) {
            //     info << sop.func_string();
            // }
            // LOG(INFO, info.str());
        }
        ++depth_num;
    }

    int depth_to_migrate = 0;
    auto it1 = tmp_depth_nodes.begin();
    for (; it1 != tmp_depth_nodes.end(); ++it1) {
        for (auto it2 = next(it1); it2 != tmp_depth_nodes.end(); ++it2) {
            list<OpGraphNode *> &logn = *it2;
            for (auto oi = logn.begin(); oi != logn.end();) {
                OpGraphNode *ogn = *oi;
                bool can_migrate = true;
                for (OpGraphNode *in_dep : ogn->in_deps) {
                    if (in_dep->depth >= depth_to_migrate) {
                        can_migrate = false;
                        break;
                    }
                }
                if (can_migrate) {
                    // LOG(INFO, "migrate ", ogn, " to depth ",
                    // depth_to_migrate); for (OpGraphNode *in_dep :
                    // ogn->in_deps) {
                    //     LOG(INFO, "  in_dep: ", in_dep, ", depth ",
                    //     in_dep->depth);
                    // }
                    it1->emplace_back(ogn);
                    ogn->depth = depth_to_migrate;
                    oi = logn.erase(oi);
                } else {
                    ++oi;
                }
            }
        }
        ++depth_to_migrate;
    }
    auto it = tmp_depth_nodes.begin();
    for (; it != tmp_depth_nodes.end();) {
        if (it->size() == 0) {
            it = tmp_depth_nodes.erase(it);
        } else {
            ++it;
        }
    }

    depth_num = 0;
    for (auto &depth : tmp_depth_nodes) {
        // LOG(INFO, "Depth ", depth_num, " -------------------- ");
        for (OpGraphNode *ogn : depth) {
            ogn->depth = depth_num;
            // stringstream info;
            // for (OpGraphNode *x : ogn->in_deps) {
            //     info << x << ",";
            // }
            // info << " --> " << ogn << " --> ";
            // for (OpGraphNode *x : ogn->out_deps) {
            //     info << x << ",  ";
            // }
            // for (const SchedOp& sop : ogn->opseq.get_sched_ops()) {
            //     info << sop.func_string();
            // }
            // LOG(INFO, info.str());
        }
        ++depth_num;
    }

    //
    set<OpGraphNode *> seen2;
    vector<vector<OpGraphNode *>> new_depth_nodes;
    new_depth_nodes.emplace_back();

    bool comm_found = false;
    for (auto it = tmp_depth_nodes.begin(); it != tmp_depth_nodes.end(); ++it) {
        set<OpGraphNode *> seen2_tmp;
        vector<OpGraphNode *> others;
        for (OpGraphNode *ogn : *it) {
            //
            bool resolved = true;
            for (OpGraphNode *in_dep : ogn->in_deps) {
                if (seen2.find(in_dep) == seen2.end()) {
                    resolved = false;
                    break;
                }
            }
            if (!resolved) {
                if (next(it) == tmp_depth_nodes.end()) {
                    tmp_depth_nodes.emplace_back();
                }
                next(it)->emplace_back(ogn);
                continue;
            }
            //
            // if (ogn->opseq.is_send() || ogn->opseq.is_recv()) {
            //     comm_found = true;
            // }
            if (ogn->opseq.is_send() || ogn->opseq.is_recv() ||
                ogn->opseq.is_send_done() ||
                (ogn->opseq.get_last_op()->type != OP_MATMUL)) {
                new_depth_nodes.back().emplace_back(ogn);
                seen2_tmp.emplace(ogn);
            } else {
                others.emplace_back(ogn);
            }
        }
        if (comm_found) {
            if (others.size() > 0) {
                int idx = (int)others.size() - 1;
                for (auto it = others.rbegin(); it != others.rend(); ++it) {
                    if ((*it)->opseq.get_last_op()->type == OP_MATMUL) {
                        break;
                    }
                    --idx;
                }
                if (idx < 0) {
                    idx = (int)others.size() - 1;
                }

                new_depth_nodes.back().emplace_back(others[idx]);
                seen2_tmp.emplace(others[idx]);
                if (next(it) == tmp_depth_nodes.end()) {
                    tmp_depth_nodes.emplace_back();
                }
                for (int i = 0; i < (int)others.size(); ++i) {
                    if (i == idx) {
                        continue;
                    }
                    next(it)->emplace_back(others[i]);
                }
            }
            if (seen2_tmp.size() > 0) {
                seen2.insert(seen2_tmp.begin(), seen2_tmp.end());
                if (new_depth_nodes.back().size() > 0) {
                    new_depth_nodes.emplace_back();
                }
            }
        } else {
            seen2.insert(seen2_tmp.begin(), seen2_tmp.end());
            seen2.insert(others.begin(), others.end());
            new_depth_nodes.back().insert(new_depth_nodes.back().end(),
                                          others.begin(), others.end());
            new_depth_nodes.emplace_back();
        }
    }
    this->depth_nodes = new_depth_nodes;

    depth_num = 0;
    for (auto &depth : this->depth_nodes) {
        // LOG(INFO, "Depth ", depth_num, " -------------------- ");
        for (OpGraphNode *ogn : depth) {
            ogn->depth = depth_num;
            // stringstream info;
            // for (OpGraphNode *x : ogn->in_deps) {
            //     info << x << ",";
            // }
            // info << " --> " << ogn << " --> ";
            // for (OpGraphNode *x : ogn->out_deps) {
            //     info << x << ",  ";
            // }
            // for (const SchedOp &sop : ogn->opseq.get_sched_ops()) {
            //     info << sop.func_string();
            // }
            // LOG(INFO, info.str());
        }
        ++depth_num;
    }
}

////////////////////////////////////////////////////////////////////////////////
} // namespace ark
