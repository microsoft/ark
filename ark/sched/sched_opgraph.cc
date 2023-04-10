// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#include "third_party/json/json.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <initializer_list>
#include <ostream>
#include <unistd.h>

#include "ark/env.h"
#include "ark/logging.h"
#include "ark/math.h"
#include "ark/model_io.h"
#include "ark/sched/sched_opgraph.h"
using namespace std;

namespace ark {

void retreive_no_virt_dep_ops(const Model &model, const GpuInfo &gpu_info,
                              const Op *target, set<const Op *> &dep_ops)
{
    for (Tensor *tns : target->in_deps) {
        const Op *op = model.get_gen_op(tns);
        if (op == nullptr) {
            // No Op generates this tensor.
            continue;
        }
        const OpConfig *cfg = sched_op_config(op, gpu_info);
        if (cfg->num_warps == 0) {
            retreive_no_virt_dep_ops(model, gpu_info, op, dep_ops);
        } else {
            dep_ops.emplace(op);
        }
    }
}

OpGraph::OpGraph(const Model &model, const GpuInfo &gpu_info)
{
    int opseq_id = 0;
    set<const Op *> seen;
    std::list<std::list<OpGraphNode *>> tmp_depth_nodes;
    tmp_depth_nodes.emplace_front();
    list<OpGraphNode *> *depth = &(tmp_depth_nodes.front());
    {
        set<Tensor *> final_outputs;
        for (auto &tns : model.get_tensors()) {
            if (model.is_no_ref(tns.get())) {
                final_outputs.emplace(tns.get());
            }
        }
        for (Tensor *out : final_outputs) {
            const Op *op = model.get_gen_op(out);
            if (op != nullptr) {
                const OpConfig *cfg = sched_op_config(op, gpu_info);
                OpGraphNode *ogn =
                    this->get_or_create_node(opseq_id++, op, cfg);
                // LOG(INFO, "OGN: ", ogn);
                depth->emplace_back(ogn);
                seen.emplace(op);
                // LOG(DEBUG, "retreive: final op ", op->type);
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
            // LOG(INFO, "retreive: ", *it, ", last op ",
            // (*it)->opseq.get_last_op()->type, " ",
            // (*it)->opseq.get_last_op());
            for (;;) {
                // Get all Ops of which results are used by seen Ops only
                // and at least one result is used by `opseq.back()`.
                set<const Op *> dep_ops;
                for (Tensor *tns : opseq.get_last_op()->in_deps) {
                    const Op *op = model.get_gen_op(tns);
                    if (op == nullptr) {
                        // No Op generates this tensor.
                        continue;
                    }
                    // Ignore if any output of `op` is referred by an unseen Op.
                    bool is_only = true;
                    for (auto &out_tns : op->out_deps) {
                        if (model.is_no_ref(out_tns)) {
                            // `out_tns` is used nowhere. (pass)
                            continue;
                        }
                        for (auto &ref_op : model.get_ref_ops(out_tns)) {
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
                        // LOG(DEBUG, "retreive: dep op ", op->type);
                    }
                }
                // If there is only one such Op, check if this can be
                // merged into `opseq`.
                if (dep_ops.size() == 1) {
                    const Op *op = *dep_ops.begin();
                    const OpConfig *cfg = sched_op_config(op, gpu_info);
                    // Cannot merge if `opseq` needs a global sync.
                    if (!opseq.get_sched_ops().back().get_cfg()->sync_pre &&
                        !cfg->sync_post) {
                        // Get all Ops which depends on results of `op`.
                        set<const Op *> out_dep_ops;
                        for (auto &out_tns : op->out_deps) {
                            if (model.is_no_ref(out_tns)) {
                                continue;
                            }
                            for (auto &ref_op : model.get_ref_ops(out_tns)) {
                                out_dep_ops.emplace(ref_op);
                            }
                        }
                        // If both `op` and `opseq` are not virtual,
                        // the batch size should be the same.
                        if ((cfg->num_warps == 0) || (opseq.is_virtual()) ||
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
                                const OpConfig *cfg =
                                    sched_op_config(op, gpu_info);
                                // Try merge and continue if succeed.
                                if (opseq.append(op, cfg)) {
                                    auto p = seen.emplace(op);
                                    assert(p.second);
                                    this->nodes[op] = *it;
                                    // LOG(DEBUG, "retreive: merge");
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
                        assert(p.second);
                        const OpConfig *cfg = sched_op_config(op, gpu_info);
                        OpGraphNode *ogn =
                            this->get_or_create_node(opseq_id++, op, cfg);
                        stringstream ssod;
                        for (auto &od : (*it)->out_deps) {
                            ssod << od << ",";
                        }
                        // LOG(INFO, "OGN: ", ogn, " --> ", ssod.str());
                        depth_prev->emplace_back(ogn);
                        //
                        ogn->out_deps.insert(
                            // ogn->out_deps.end(),
                            (*it)->out_deps.begin(), (*it)->out_deps.end());
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
                            for (const Op *out_dep : model.get_ref_ops(tns)) {
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
            // LOG(DEBUG, "retreive: ---------- next depth");
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
            // for (const SchedOp& sop : ogn->opseq.get_sched_ops()) {
            //     info << sop.func_string();
            // }
            // LOG(INFO, info.str());
        }
        ++depth_num;
    }

    // for (auto& depth : this->depth_nodes) {
    //     for (OpGraphNode *ogn : depth) {
    //         if (ogn->opseq.is_send()) {
    //             stringstream info;
    //             for (OpGraphNode *x : ogn->in_deps) {
    //                 if (x->opseq.is_virtual()) {
    //                     continue;
    //                 }
    //                 info << x->depth << ",";
    //             }
    //             LOG(INFO, "SEND ", ogn->depth, " -- ", info.str());
    //         }
    //     }
    // }
    // for (auto& depth : this->depth_nodes) {
    //     for (auto it = depth.begin(); it != depth.end(); ) {
    //         OpGraphNode *ogn = *it;
    //         if (ogn->opseq.is_send()) {
    //             int dd;
    //             // LOG(INFO, "------------");
    //             for (OpGraphNode *in_dep : ogn->in_deps) {
    //                 if (in_dep->depth == -1) {
    //                     // LOG(INFO, "*OGN: ", ogn);
    //                     assert(false);
    //                 }
    //                 dd = in_dep->depth;
    //                 if (in_dep->in_deps.size() == 0) {
    //                     continue;
    //                 }
    //                 // LOG(INFO, "  ", ogn->depth, " ", in_dep->depth, " ",
    //                 (*in_dep->in_deps.begin())->depth);
    //             }
    //             assert(ogn->in_deps.size() == 1);
    //             if (ogn->depth - dd > 1) {
    //                 auto it2 = this->depth_nodes.begin();
    //                 for (int i = 0; i < dd + 1; ++i) {
    //                     ++it2;
    //                 }
    //                 it2->emplace_back(ogn);
    //                 it = depth.erase(it);
    //                 continue;
    //             }
    //         }
    //         ++it;
    //     }
    // }

    // for (auto& depth : this->depth_nodes) {
    //     for (auto it = depth.begin(); it != depth.end(); ) {
    //         OpGraphNode *ogn = *it;
    //         if (ogn->opseq.is_recv()) {
    //             int dd;
    //             LOG(INFO, "------------");
    //             for (OpGraphNode *in_dep : ogn->in_deps) {
    //                 if (in_dep->depth == -1) {
    //                     // LOG(INFO, "*OGN: ", ogn);
    //                     assert(false);
    //                 }
    //                 dd = in_dep->depth;
    //                 LOG(INFO, "  ", ogn->depth, " ", in_dep->depth);
    //             }
    //             assert(ogn->in_deps.size() == 1);
    //             if (ogn->depth - dd > 1) {
    //                 auto it2 = this->depth_nodes.begin();
    //                 for (int i = 0; i < dd + 1; ++i) {
    //                     ++it2;
    //                 }
    //                 it2->emplace_back(ogn);
    //                 it = depth.erase(it);
    //                 continue;
    //             }
    //         }
    //         ++it;
    //     }
    // }

    // for (auto it = this->depth_nodes.rbegin(); it !=
    // prev(this->depth_nodes.rend()); ++it) {
    //     list<OpGraphNode*>& depth = *it;
    //     for (auto it2 = depth.begin(); it2 != depth.end();) {
    //         set<const Op*> dep_ops;
    //         retreive_no_virt_dep_ops(model, gpu_info,
    //         (*it2)->opseq.get_last_op(), dep_ops); bool no_dep = true; for
    //         (auto& dep_op : dep_ops) {
    //             for (auto& nd_ogn : *next(it)) {
    //                 for (auto& nd_sched_op : nd_ogn->opseq.get_sched_ops()) {
    //                     if (nd_sched_op.op == dep_op) {
    //                         no_dep = false;
    //                         break;
    //                     }
    //                 }
    //                 if (!no_dep) {
    //                     break;
    //                 }
    //             }
    //             if (!no_dep) {
    //                 break;
    //             }
    //         }
    //         if (no_dep) {
    //             next(it)->emplace_back(*it2);
    //             it2 = depth.erase(it2);
    //         } else {
    //             ++it2;
    //         }
    //     }
    // }

    // return unique_ptr<OpGraph>{this};
}

// OpGraph::OpGraph(const Model& model, const GpuInfo& gpu_info)
// {
//     size_t num_ops = model.get_ops().size();
//     set<const Op*> seen;
//     // Retrieve the final depth.
//     int depth = 0;
//     this->depth_nodes.resize(1);
//     for (auto& tns : model.get_tensors()) {
//         Tensor *out = tns.get();
//         if (!model.is_no_ref(out)) {
//             continue;
//         }
//         const Op *op = model.get_gen_op(out);
//         if (op != nullptr) {
//             const OpConfig& cfg = sched_op_config(op, gpu_info, 0);
//             OpGraphNode *ogn = new OpGraphNode;
//             ogn->opseq.append(op, cfg);
//             ogn->depth = depth;

//             this->nodes_storage.emplace_back(ogn);
//             this->nodes[op] = ogn;
//             this->depth_nodes[depth].emplace_back(ogn);
//             seen.emplace(op);
//             // LOG(DEBUG, "retreive: final op ", op->type);
//         }
//     }
//     //
//     while (seen.size() < num_ops) {
//         ++depth;
//         this->depth_nodes.resize(depth + 1);
//         // Iterate over the previous depth.
//         auto it = this->depth_nodes[depth - 1].begin();
//         for (; it != this->depth_nodes[depth - 1].end(); ++it) {
//             OpGraphNode *ogn = *it;
//             for (;;) {
//                 // Get all Ops of which results are used by seen Ops only
//                 // and at least one result is used by `ogn->opseq`.
//                 set<const Op*> dep_ops;
//                 for (Tensor *tns : ogn->opseq.get_last_op()->in_deps) {
//                     const Op *op = model.get_gen_op(tns);
//                     if (op == nullptr) {
//                         // No Op generates this tensor.
//                         continue;
//                     }
//                     // Ignore if any output of `op` is referred by an unseen
//                     Op. bool is_only = true; for (auto& out_tns :
//                     op->out_deps) {
//                         if (model.is_no_ref(out_tns)) {
//                             continue;
//                         }
//                         for (auto& ref_op : model.get_ref_ops(out_tns)) {
//                             auto search = seen.find(ref_op);
//                             if (search == seen.end()) {
//                                 is_only = false;
//                                 break;
//                             }
//                         }
//                         if (!is_only) {
//                             break;
//                         }
//                     }
//                     if (is_only) {
//                         dep_ops.emplace(op);
//                         // LOG(DEBUG, "retreive: dep op ", op->type);
//                     }
//                 }
//                 //
//                 if (dep_ops.size() == 0) {
//                     break;
//                 } else if (dep_ops.size() > 1) {

//                 }
//                 // If there is only one such Op, check if this can be
//                 // merged into `opseq`.
//             }
//         }
//     }
// }

////////////////////////////////////////////////////////////////////////////////
} // namespace ark
