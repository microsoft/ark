// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "env.h"
#include "logging.h"
#include "math.h"
#include "model.h"
#include "sched/sched.h"

using namespace std;

namespace ark {

/// Calculate the number of tiles for a given op and a tile.
/// @param op input op
/// @param tile input tile
/// @return number of tiles
static int calc_num_tiles(const Op &op, const OpTile &tile)
{
    if (op.outputs.size() == 0) {
        // This op has no output.
        return 0;
    }
    assert(op.outputs[0] != nullptr);
    auto &s = op.outputs[0]->shape;
    int ndims = s.ndims();
    if (ndims == 0) {
        // The output has no element.
        return 0;
    }
    // tile.y corresponds to the last dimension.
    int num_tiles = math::div_up(s[ndims - 1], tile.y);
    // tile.x corresponds to the second last dimension.
    if (ndims > 1) {
        num_tiles *= math::div_up(s[ndims - 2], tile.x);
    } else if (tile.x != 1) {
        LOGERR("The tile is 2D, but the output is 1D.");
    }
    // The remaining dimensions are not tiled.
    int remain_dims = ndims - 2;
    while (remain_dims > 0) {
        num_tiles *= s[remain_dims - 1];
        remain_dims--;
    }
    return num_tiles;
}

/// Heuristic matmul optimization. Overwrite the input matmul op with an
/// optimized op.
/// @param model target model
/// @param matmul_op matmul op in the target model
/// @param gpu_info GPU info to optimize for
/// @param num_sm number of SMs to use for this op. This should be equal to or
/// less than the number of SMs on the GPU (`gpu_info.num_sm`).
void DefaultScheduler::heuristic_optimize_matmul(Model &model,
                                                 Model::Impl *model_impl,
                                                 Op &matmul_op,
                                                 const GpuInfo &gpu_info,
                                                 int num_sm)
{
    if (matmul_op.type != OP_MATMUL) {
        LOGERR("This is not a matmul op.");
    }
    if (matmul_op.gran_lev != -1) {
        // `gran_lev` is manually set. Do not optimize.
        return;
    }
    if (num_sm > gpu_info.num_sm) {
        LOGERR("The total number of SMs (%d) is less than the number of SMs "
               "requested (%d).",
               gpu_info.num_sm, num_sm);
    }
    const OpConfig *cfg = this->sched_op_config(&matmul_op);
    assert(cfg->output_tiles.size() == 1);
    int num_tiles = calc_num_tiles(matmul_op, cfg->output_tiles[0]);
    if (num_tiles == 0) {
        LOGERR("This matmul has no output tiles.");
    }

    // Heuristically select a split_k value. If split_k is larger than 1, split
    // the inner dimension of the matmul into split_k parts, where each part is
    // computed by a separate matmul op and the results are accumulated. Larger
    // split_k is preferred when the number of tiles is small and the inner
    // dimension is large.
    int split_k = 1;
    if (num_tiles < num_sm) {
        // If the number of tiles is less than the number of SMs, we can
        // potentially use more SMs to compute the matmul. We can split the
        // inner dimension into multiple parts and distribute them to different
        // SMs. We use a heuristic to determine the number of parts.

        // Calculate the maximum possible split_k according to the tile shape.
        const Dims &fst_input_shape = matmul_op.inputs[0]->shape;
        const OpTile &fst_input_tile = cfg->input_tiles[0];
        DimType inner_dim = fst_input_shape[fst_input_shape.ndims() - 1];
        DimType inner_dim_tile_len = fst_input_tile.y;
        size_t max_split_k = math::div_up(inner_dim, inner_dim_tile_len);

        // Calculate the max split_k to run two or less tiles per SM. Exceeding
        // this limit is heuristically bad for performance.
        size_t split_k_for_two_tiles_per_sm = num_sm * 2 / num_tiles;
        size_t tmp_split_k = min(max_split_k, split_k_for_two_tiles_per_sm);

        // Calculate the actual split_k if we can split the inner dimension
        // into tmp_split_k parts.
        size_t each_part_len =
            math::pad(math::div_up(inner_dim, tmp_split_k), inner_dim_tile_len);
        split_k = math::div_up(inner_dim, each_part_len);
        assert(split_k > 0);
    }
    if (split_k == 1) {
        // No optimization is needed.
        return;
    }
    LOG(DEBUG, "Optimize matmul %s with split_k=%d.", matmul_op.name, split_k);

    Tensor *input_a = matmul_op.inputs[0];
    Tensor *input_b = matmul_op.inputs[1];
    Tensor *output = matmul_op.outputs[0];
    bool is_column_a;
    bool is_column_b;
    bool is_relu;
    matmul_op.args.get(&is_column_a, 4);
    matmul_op.args.get(&is_column_b, 5);
    matmul_op.args.get(&is_relu, 6);
    std::string matmul_name = matmul_op.name;

    // Remove the original matmul op from the model.
    model_impl->delete_op(&matmul_op);

    // Create a new matmul op with the optimized split_k.
    model.matmul(input_a, input_b, output, split_k, is_column_a, is_column_b,
                 is_relu, matmul_name);
}

/// Heuristically optimize the model. Overwrite the model with an optimized
/// model.
/// @param model target model
/// @param gpu_info GPU info to optimize for
/// @param num_sm number of SMs to use for this op. This should be equal to or
/// less than the number of SMs on the GPU (`gpu_info.num_sm`).
void DefaultScheduler::heuristic_optimize_model(Model &model,
                                                Model::Impl *model_impl,
                                                const GpuInfo &gpu_info,
                                                int num_sm)
{
    if (get_env().disable_graph_opt) {
        LOG(INFO, "Graph optimization is disabled.");
        return;
    }
    // Make a copy of the ops because we will modify the model.
    std::vector<Op *> ops;
    for (auto &op : model_impl->get_ops()) {
        ops.push_back(op);
    }
    for (auto &op : ops) {
        if (op->type == OP_MATMUL) {
            heuristic_optimize_matmul(model, model_impl, *op, gpu_info, num_sm);
        }
    }
}

DefaultScheduler::DefaultScheduler(Model &model, int gpu_id, int rank_,
                                   int world_size_, int num_warps_per_sm_)
    : BaseScheduler(model, gpu_id, rank_, world_size_, num_warps_per_sm_)
{
    const GpuInfo &gpu_info = this->gpu_mgr->get_gpu_info();

    // Number of SMs to use for computation. The last SM is preserved for
    // communication only.
    int num_sm_calc = gpu_info.num_sm - 1;

    heuristic_optimize_model(model, model.impl.get(), gpu_info, num_sm_calc);

    this->op_graph = make_unique<OpGraph>(model);
}

void DefaultScheduler::schedule()
{
    LOG(DEBUG, "DefaultScheduler start scheduling");

    auto &nodes = this->op_graph->get_nodes();

    std::list<OpNode *> root_nodes;
    for (auto &node : nodes) {
        if (node->producers.empty()) {
            root_nodes.emplace_back(node.get());
        }
    }

    std::set<OpNode *> seen_nodes;
    recursive_schedule(root_nodes, seen_nodes);

    this->configure_gpu_buf(this->model->impl->get_tensors());

    if (this->comp_stream.size() != this->comm_stream.size()) {
        LOG(ERROR, "unexpected error");
    }
}

///
void DefaultScheduler::recursive_schedule(std::list<OpNode *> &nodes,
                                          std::set<OpNode *> &seen_nodes)
{
    if (nodes.empty()) {
        return;
    }
    const GpuInfo &gpu_info = this->gpu_mgr->get_gpu_info();

    std::list<OpNode *> next_nodes;
    std::vector<SchedItem> comp_items;
    std::vector<SchedItem> comm_items;
    bool sync_comm = false;
    bool sync_comp = false;
    for (auto &node : nodes) {
        if (node->ops.size() == 0) {
            LOG(ERROR, "unexpected error: empty OpNode");
        }
        Op *op = node->ops[0];
        const OpConfig *cfg = this->sched_op_config(op);
        int opseq_id = (int)this->opseqs.size();
        this->opseqs.emplace_back(make_unique<SchedOpSeq>(opseq_id, op, cfg));
        SchedOpSeq *opseq = this->opseqs.back().get();

        bool broke_node = false;
        for (size_t i = 1; i < node->ops.size(); i++) {
            // If there are multiple Ops, check if the Op configs allow merging.
            Op *next_op = node->ops[i];
            const OpConfig *next_cfg = this->sched_op_config(next_op);
            bool need_sync_between_ops = cfg->sync_post || next_cfg->sync_pre;
            bool comm_and_comp = (op->is_comm() && !next_op->is_comm()) ||
                                 (!op->is_comm() && next_op->is_comm());
            if (!need_sync_between_ops && !comm_and_comp) {
                if (opseq->append(next_op, next_cfg)) {
                    // Merge succeeded.
                    continue;
                }
            }
            // Cannot merge. Add remaining part of the OpNode to next_nodes.
            OpNode *next_node = this->op_graph->break_node(node, i);
            next_nodes.emplace_back(next_node);
            broke_node = true;
            break;
        }

        // Check if we need to sync between comp and comm.
        if (!sync_comm && opseq->is_comm()) {
            // Check if any producer is a computation Op.
            for (auto &producer : node->producers) {
                // As we do not merge computation Ops with communication Ops,
                // we only need to check the first Op.
                if (!producer->ops[0]->is_comm()) {
                    sync_comm = true;
                    break;
                }
            }
        } else if (!sync_comp && !opseq->is_comm()) {
            // Check if any producer is a communication Op.
            for (auto &producer : node->producers) {
                // As we do not merge computation Ops with communication Ops,
                // we only need to check the first Op.
                if (producer->ops[0]->is_comm()) {
                    sync_comp = true;
                    break;
                }
            }
        }

        auto p = seen_nodes.emplace(node);
        if (!p.second) {
            LOG(ERROR, "unexpected error: already seen node ", node->get_name(),
                " (", node->ops.size(), " ops)");
        }

        // Align shared memory size
        int smem_bytes = opseq->get_smem_bytes();
        int aligned_smem_bytes = math::pad(smem_bytes, gpu_info.smem_align);

        // Create a scheduling item.
        SchedItem item;
        item.opseq_id = opseq_id;
        item.num_uops = opseq->get_tdims_size();
        item.num_warps_per_uop = opseq->get_num_warps();
        item.smem_bytes_per_uop = aligned_smem_bytes;
        if (op->is_comm()) {
            comm_items.emplace_back(item);
        } else {
            comp_items.emplace_back(item);
        }

        if (!broke_node) {
            // If OpNode is completely merged, add its users to
            // next_nodes.
            for (auto &user_node : node->users) {
                // If any producer is unseen, skip the user.
                bool skip = false;
                for (auto &producer : user_node->producers) {
                    if (seen_nodes.find(producer) == seen_nodes.end()) {
                        skip = true;
                        break;
                    }
                }
                if (!skip) {
                    next_nodes.emplace_back(user_node);
                }
            }
        }
    }

    if (this->comp_stream.empty() || sync_comp || sync_comm) {
        // Create a new stream.
        this->comp_stream.emplace_back(make_unique<SchedStream>(
            0, gpu_info.num_sm - 1, this->num_warps_per_sm,
            gpu_info.smem_block_total));
    }
    if (this->comm_stream.empty() || sync_comp || sync_comm) {
        // Create a new stream.
        this->comm_stream.emplace_back(make_unique<SchedStream>(
            gpu_info.num_sm - 1, gpu_info.num_sm, this->num_warps_per_sm,
            gpu_info.smem_block_total));
    }

    // Schedule the Ops.
    this->comp_stream.back()->add_items(comp_items);
    this->comm_stream.back()->add_items(comm_items);

    LOG(DEBUG, "scheduled ", nodes.size(), " nodes");
    for (auto &item : comp_items) {
        LOG(DEBUG, "  comp: ", this->opseqs[item.opseq_id]->get_name());
    }
    for (auto &item : comm_items) {
        LOG(DEBUG, "  comm: ", this->opseqs[item.opseq_id]->get_name());
    }

    recursive_schedule(next_nodes, seen_nodes);
}

void DefaultScheduler::configure_gpu_buf(
    const std::list<Tensor *> &model_tensors)
{
    // A TensorBuf can be located on a local GPU or a remote GPU. If it is on
    // this rank's GPU, it should be allocated and might be exported to other
    // GPUs. If it is on a remote GPU (the gid is not equal to this rank), it
    // should be imported.
    // A TensorBuf can have multi tensors pointing to it. Different Tensor
    // represent a different sharding or view of the same TensorBuf.
    std::map<TensorBuf *, std::vector<Tensor *>> bufs;
    // export_tns_sids is a map of the TensorBuf that needed to be exported, and
    // the corresponding tensors and sids. A TensorBuf can have multiple tensors
    // pointing to it, and might be exported to multiple ranks as different
    // Tensor.
    std::map<TensorBuf *, std::vector<std::pair<Tensor *, int>>>
        export_tns_sids;

    for (auto &opseq : this->opseqs) {
        for (auto &sop : opseq->get_sched_ops()) {
            for (unsigned int i = 0; i < sop.get_op()->inputs.size(); ++i) {
                auto &tile = sop.get_cfg()->input_tiles[i];
                sop.get_op()->inputs[i]->update_pads({tile.x, tile.y});
            }
            for (unsigned int i = 0; i < sop.get_op()->outputs.size(); ++i) {
                auto &tile = sop.get_cfg()->output_tiles[i];
                sop.get_op()->outputs[i]->update_pads({tile.x, tile.y});
            }
        }
    }

    for (auto &opseq : this->opseqs) {
        for (auto &sop : opseq->get_sched_ops()) {
            const Op *op = sop.get_op();
            std::vector<Tensor *> tensors = op->inputs;
            tensors.insert(tensors.end(), op->outputs.begin(),
                           op->outputs.end());

            for (auto &tns : tensors) {
                // If the tensor is not imported, it should be allocated on this
                // GPU
                if (tns->imported_rank < 0) {
                    bufs[tns->buf].emplace_back(tns);
                }
            }

            const int send_ready_flag_sid_offset = 128;

            //
            if (op->type == OP_SEND) {
                //
                Tensor *in = op->inputs[0];
                int sid;
                int rank;
                int dst_rank;
                size_t bytes;
                op->args.get(&sid, 0);
                op->args.get(&rank, 1);
                op->args.get(&dst_rank, 2);
                op->args.get(&bytes, 3);
                size_t off = in->offset() * in->type_bytes();
                // TODO: generalize converting rank to GPU ID.
                int nrph = get_env().num_ranks_per_host;
                int dst_gpu_id = dst_rank % nrph;
                if ((dst_rank / nrph) == (this->rank / nrph)) {
                    // Same node.
                    this->buf_infos.emplace_back(dst_gpu_id, bytes, nullptr,
                                                 sid, off);
                }
                export_tns_sids[in->buf].emplace_back(in, sid);
                this->send_recv_ops.emplace_back(op);
            } else if (op->type == OP_RECV) {
                //
                Tensor *in = op->inputs[0];
                int sid;
                op->args.get(&sid, 0);
                export_tns_sids[in->buf].emplace_back(in, sid);
                this->send_recv_ops.emplace_back(op);
            } else if (op->type == OP_SEND_MM) {
                int sid;
                int dst_gid;
                sop.get_op()->args.get(&sid, 0);
                sop.get_op()->args.get(&dst_gid, 1);
                // import the recvbuf, the recvbuf should be allocated on the
                // receiver GPU
                Tensor *recvbuf = sop.get_op()->inputs[1];
                this->buf_infos.emplace_back(dst_gid, recvbuf->shape_bytes(),
                                             recvbuf->buf, sid, 0);

                // configure the send_ready_flag, the send_ready_flag needed to
                // be exported to the recv GPU, since the sid of the
                // send_ready_flag should not be the same as the recvBuf, so I
                // use the sid+128 as the sid of the send_ready_flag
                Tensor *send_ready_flag = sop.get_op()->inputs[2];
                export_tns_sids[send_ready_flag->buf].emplace_back(
                    send_ready_flag, sid + send_ready_flag_sid_offset);
            } else if (op->type == OP_RECV_MM) {
                int sid;
                int src_gid;
                sop.get_op()->args.get(&sid, 0);
                sop.get_op()->args.get(&src_gid, 1);
                // configure the recvbuf, the recvbuf needed to be export the to
                // the sender GPU, the sid is the same as the sid of the send_mm
                // op and the recv_mm op
                Tensor *recvbuf = sop.get_op()->inputs[1];
                export_tns_sids[recvbuf->buf].emplace_back(recvbuf, sid);

                // import the send_ready_flag, the send_ready_flag tensor should
                // be allocated on the sender GPU
                Tensor *send_ready_flag = sop.get_op()->inputs[2];
                this->buf_infos.emplace_back(
                    src_gid, send_ready_flag->shape_bytes(),
                    send_ready_flag->buf, sid + send_ready_flag_sid_offset, 0);
            }
        }
    }

    for (auto &tns : model_tensors) {
        auto search = bufs.find(tns->buf);
        if (search == bufs.end()) {
            bufs[tns->buf].emplace_back(tns);
        }
    }

    // Fix TensorBuf size.
    for (auto &el : bufs) {
        TensorBuf *buf = el.first;
        vector<Tensor *> &tensors = el.second;
        size_t max_bytes = 0;
        for (auto &tns : tensors) {
            size_t tns_bytes = tns->ldims_bytes();
            if (max_bytes < tns_bytes) {
                max_bytes = tns_bytes;
            }
            // TODO: more verficiations.
            auto &sh = tns->shape;
            auto &ld = tns->ldims;
            LOG(DEBUG, "Tensor buf ", tns->buf, " pads ", tns->pads,
                " padding ", sh, " -> ", ld, " exported ", tns->exported);
        }
        // Store the size.
        buf->bytes = max_bytes;
    }

    // Allocate all GPU buffers.
    for (auto &el : bufs) {
        TensorBuf *buf = el.first;
        int sid = -1;
        size_t off = 0;
        auto search = export_tns_sids.find(buf);
        if (search != export_tns_sids.end()) {
            for (auto &p : search->second) {
                Tensor *t = p.first;
                sid = p.second;
                off = t->offset() * t->type_bytes();
                this->buf_infos.emplace_back(this->gpu_mgr->gpu_id, buf->bytes,
                                             buf, sid, off);
            }
        } else {
            this->buf_infos.emplace_back(this->gpu_mgr->gpu_id, buf->bytes, buf,
                                         sid, off);
        }
    }
}

std::vector<std::string> DefaultScheduler::gen_code()
{
    std::stringstream code;

    std::set<int> imported_ranks;
    for (auto &tns : this->model->impl->get_tensors()) {
        if (tns->imported_rank >= 0) {
            imported_ranks.insert(tns->imported_rank);
        }
    }
    for (auto rank : imported_ranks) {
        this->codegen->def_remote_buf(code, rank);
    }

    this->codegen->def_sync_stream(code, 0);
    this->codegen->def_sync_stream(code, 1);

    std::map<std::string, int> uop_map;
    for (auto &opseq : this->opseqs) {
        for (auto &sop : opseq->get_sched_ops()) {
            int uop_id = (int)uop_map.size();
            std::string sop_func_str = sop.function_name();
            // Insert only if it does not exist
            auto p = uop_map.emplace(sop_func_str, uop_id);
            if (p.second) {
                // If this is a new function, define it.
                this->codegen->def_uop(code, sop, uop_id);
            }
        }
    }
    for (auto &opseq : this->opseqs) {
        this->codegen->opseq(code, "op" + std::to_string(opseq->get_id()),
                             *opseq, uop_map);
    }

    const GpuInfo &gpu_info = this->gpu_mgr->get_gpu_info();
    int num_sm_comp = gpu_info.num_sm - 1;
    int num_sm_comm = 1;

    code << "__device__ void ark_loop_body(int _iter) {\n";
    for (size_t i = 0; i < this->comp_stream.size(); ++i) {
        auto comp_streams = this->comp_stream[i]->get_streams();
        for (size_t j = 0; j < comp_streams.size(); ++j) {
            auto &stream = comp_streams[j];
            for (auto &branch : stream.branches) {
                this->codegen->branch(code, branch);
            }
            if (!stream.branches.empty() && j != comp_streams.size() - 1) {
                code << "  ";
                this->codegen->sync_stream(code, 0, 0, num_sm_comp);
            }
        }
        auto comm_streams = this->comm_stream[i]->get_streams();
        for (size_t j = 0; j < comm_streams.size(); ++j) {
            auto &stream = comm_streams[j];
            for (auto &branch : stream.branches) {
                this->codegen->branch(code, branch);
            }
            if (!stream.branches.empty() && j != comm_streams.size() - 1) {
                code << "  ";
                this->codegen->sync_stream(code, 1, num_sm_comp,
                                           num_sm_comp + num_sm_comm);
            }
        }
        if (i != this->comp_stream.size() - 1) {
            code << "  ";
            this->codegen->sync_gpu(code);
        }
    }
    code << "}\n";
    return {code.str()};
}

} // namespace ark
