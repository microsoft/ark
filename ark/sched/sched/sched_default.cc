// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/env.h"
#include "ark/logging.h"
#include "ark/math.h"
#include "ark/sched/sched.h"

using namespace std;

#define MATMUL_GRAPH_OPT 1
#define ALLOC_UNUSED_TENSORS 1
#define PRESERVE_WARP_FOR_COMM 1

namespace ark {

/// Calculate the number of tiles for a given op and a tile.
/// @param op input op
/// @param tile input tile
/// @return number of tiles
static int calc_num_tiles(const Op &op, const OpTile &tile)
{
    if (op.out_deps.size() == 0) {
        // This op has no output.
        return 0;
    }
    assert(op.out_deps[0] != nullptr);
    auto &s = op.out_deps[0]->shape;
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
static void heuristic_optimize_matmul(Model &model, Op &matmul_op,
                                      const GpuInfo &gpu_info, int num_sm)
{
    if (matmul_op.type != OP_MATMUL) {
        LOGERR("This is not a matmul op.");
    }
    if (num_sm > gpu_info.num_sm) {
        LOGERR("The total number of SMs (%d) is less than the number of SMs "
               "requested (%d).",
               gpu_info.num_sm, num_sm);
    }
    const OpConfig *cfg = sched_op_config(&matmul_op, gpu_info);
    assert(cfg->out_deps_tiles.size() == 1);
    int num_tiles = calc_num_tiles(matmul_op, cfg->out_deps_tiles[0]);
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
        const Dims &fst_input_shape = matmul_op.in_deps[0]->shape;
        const OpTile &fst_input_tile = cfg->in_deps_tiles[0];
        DimType inner_dim = fst_input_shape[fst_input_shape.ndims() - 1];
        DimType inner_dim_tile_len = fst_input_tile.y;
        size_t max_split_k = math::div_up(inner_dim, inner_dim_tile_len);

        // Calcualte the max split_k to run two or less tiles per SM. Exceeding
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

    Tensor *input_a = matmul_op.in_deps[0];
    Tensor *input_b = matmul_op.in_deps[1];
    Tensor *output = matmul_op.out_deps[0];
    bool trans_a = *(bool *)matmul_op.args[0].val;
    bool trans_b = *(bool *)matmul_op.args[1].val;
    bool is_relu = *(bool *)matmul_op.args[2].val;
    std::string matmul_name = matmul_op.name;

    // Remove the original matmul op from the model.
    model.delete_op(&matmul_op);

    // Create a new matmul op with the optimized split_k.
    model.matmul(input_a, input_b, output, split_k, trans_a, trans_b, is_relu,
                 matmul_name);
}

/// Heuristically optimize the model. Overwrite the model with an optimized
/// model.
/// @param model target model
/// @param gpu_info GPU info to optimize for
/// @param num_sm number of SMs to use for this op. This should be equal to or
/// less than the number of SMs on the GPU (`gpu_info.num_sm`).
static void heuristic_optimize_model(Model &model, const GpuInfo &gpu_info,
                                     int num_sm)
{
    // Make a copy of the ops because we will modify the model.
    std::vector<Op *> ops;
    for (auto &op : model.get_ops()) {
        ops.push_back(op.get());
    }
    for (auto &op : ops) {
        if (op->type == OP_MATMUL) {
            heuristic_optimize_matmul(model, *op, gpu_info, num_sm);
        }
    }
}

void DefaultScheduler::configure_gpu_buf(
    const std::list<std::unique_ptr<Tensor>> &model_tensors)
{
    //
    map<TensorBuf *, vector<Tensor *>> bufs;
    map<TensorBuf *, set<Tensor *>> buf_usage;
    map<TensorBuf *, vector<pair<Tensor *, int>>> tns_eids;
    // TODO:
    for (auto &depth : this->op_graph->depth_nodes) {
        for (auto &ogn : depth) {
            for (auto &sop : ogn->opseq.get_sched_ops()) {
                if (sop.get_op() == nullptr) {
                    continue;
                }
                if (sop.get_cfg()->num_warps == 0) {
                    continue;
                }
                for (unsigned int i = 0; i < sop.get_op()->in_deps.size();
                     ++i) {
                    auto &tile = sop.get_cfg()->in_deps_tiles[i];
                    sop.get_op()->in_deps[i]->update_pads({tile.x, tile.y});
                }
                for (unsigned int i = 0; i < sop.get_op()->out_deps.size();
                     ++i) {
                    auto &tile = sop.get_cfg()->out_deps_tiles[i];
                    sop.get_op()->out_deps[i]->update_pads({tile.x, tile.y});
                }
            }
        }
    }
    for (auto &depth : this->op_graph->depth_nodes) {
        for (auto &ogn : depth) {
            for (auto &sop : ogn->opseq.get_sched_ops()) {
                if (sop.get_cfg()->num_warps == 0) {
                    continue;
                }
                for (auto &tns : sop.get_op()->in_deps) {
                    bufs[tns->buf].emplace_back(tns);
                    if (!tns->buf->immutable) {
                        buf_usage[tns->buf].emplace(tns);
                    }
                }
                for (auto &tns : sop.get_op()->out_deps) {
                    bufs[tns->buf].emplace_back(tns);
                    if (!tns->buf->immutable) {
                        buf_usage[tns->buf].emplace(tns);
                    }
                }
                //
                if (sop.get_op()->type == OP_SEND) {
                    //
                    Tensor *in = sop.get_op()->in_deps[0];
                    int sid = *(int *)sop.get_op()->args[0].val;
                    int dst_rank = *(int *)sop.get_op()->args[1].val;
                    size_t bytes = *(size_t *)sop.get_op()->args[2].val;
                    size_t off = in->offset() * in->type_bytes();
                    // TODO: generalize converting rank to GPU ID.
                    int nrph = get_env().num_ranks_per_host;
                    int dst_gpu_id = dst_rank % nrph;
                    if ((dst_rank / nrph) == (this->rank / nrph)) {
                        // Same node.
                        this->buf_infos.emplace_back(dst_gpu_id, bytes, nullptr,
                                                     sid, off);
                    }
                    tns_eids[in->buf].emplace_back(in, sid);
                    this->send_recv_ops.emplace_back(sop.get_op());
                } else if (sop.get_op()->type == OP_RECV) {
                    //
                    Tensor *in = sop.get_op()->in_deps[0];
                    int eid = *(int *)sop.get_op()->args[0].val;
                    tns_eids[in->buf].emplace_back(in, eid);
                    this->send_recv_ops.emplace_back(sop.get_op());
                }
            }
        }
    }
#if (ALLOC_UNUSED_TENSORS)
    for (auto &tns : model_tensors) {
        Tensor *t = tns.get();
        auto search = bufs.find(t->buf);
        if (search == bufs.end()) {
            bufs[t->buf].emplace_back(t);
        }
    }
#endif // (ALLOC_UNUSED_TENSORS)
    struct GpuBufInfo
    {
        size_t bytes;
    };
    map<TensorBuf *, GpuBufInfo> binfs;
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
            stringstream ss;
            LOG(DEBUG, "Tensor buf ", tns->buf, " pads ", tns->pads,
                " padding ", sh, " -> ", ld, " exported ", tns->exported);
        }
        // Store the size.
        buf->bytes = max_bytes;
        GpuBufInfo &info = binfs[buf];
        info.bytes = max_bytes;
    }
#if (ALLOC_UNUSED_TENSORS)
    // Allocate all GPU buffers.
    for (auto &el : binfs) {
        TensorBuf *buf = el.first;
        GpuBufInfo &info = el.second;
        int sid = -1;
        size_t off = 0;
        auto search = tns_eids.find(buf);
        if (search != tns_eids.end()) {
            for (auto &p : search->second) {
                Tensor *t = p.first;
                sid = p.second;
                off = t->offset() * t->type_bytes();
                this->buf_infos.emplace_back(this->gpu_mgr->gpu_id, info.bytes,
                                             buf, sid, off);
            }
        } else {
            this->buf_infos.emplace_back(this->gpu_mgr->gpu_id, info.bytes, buf,
                                         sid, off);
        }
    }
#else
    //
    for (auto &depth : this->op_graph->depth_nodes) {
        vector<TensorBuf *> to_alloc;
        set<TensorBuf *> to_free;
        for (auto &ogn : depth) {
            for (auto &sop : ogn->opseq.get_sched_ops()) {
                for (auto &tns : sop.get_op()->in_deps) {
                    size_t num = bufs.erase(tns->buf);
                    if (num > 0) {
                        assert(num == 1);
                        to_alloc.emplace_back(tns->buf);
                    }
                    if (!tns->buf->immutable) {
                        buf_usage[tns->buf].erase(tns);
                        if (buf_usage[tns->buf].size() == 0) {
                            to_free.emplace(tns->buf);
                        }
                    }
                }
                for (auto &tns : sop.get_op()->out_deps) {
                    size_t num = bufs.erase(tns->buf);
                    if (num > 0) {
                        assert(num == 1);
                        to_alloc.emplace_back(tns->buf);
                    }
                    if (!tns->buf->immutable) {
                        buf_usage[tns->buf].erase(tns);
                        if (buf_usage[tns->buf].size() == 0) {
                            to_free.emplace(tns->buf);
                        }
                    }
                }
            }
        }
        // Allocate GPU buffers.
        for (auto &buf : to_alloc) {
            GpuBufInfo &info = binfs[buf];
            int sid = -1;
            size_t off = 0;
            auto search = tns_eids.find(buf);
            if (search != tns_eids.end()) {
                for (auto &p : search->second) {
                    Tensor *t = p.first;
                    sid = p.second;
                    off = t->offset() * t->type_bytes();
                    this->buf_infos.emplace_back(this->gpu_mgr->gpu_id,
                                                 info.bytes, buf, sid, off);
                }
            } else {
                this->buf_infos.emplace_back(this->gpu_mgr->gpu_id, info.bytes,
                                             buf, sid, off);
            }
        }
        // Free if it is no more used.
        // TODO: this incurs CUDA_ERROR_ILLEGAL_ADDRESS in computing kernels.
        // Enable this again when the issue is fixed.
        // for (auto& buf : to_free) {
        //     this->launcher->free_buffer(this->launcher->get_buf_trans()[buf]);
        // }
    }
#endif
}

DefaultScheduler::DefaultScheduler(const int gpu_id, int rank_, int world_size_,
                                   Model &model, int wps_)
    : SchedulerBase(gpu_id, rank_, world_size_, wps_), scg{buf_trans, 108, wps_,
                                                           world_size_}
{
    const GpuInfo &gpu_info = this->gpu_mgr->get_gpu_info();
    int min_wps = gpu_info.min_threads_per_block / gpu_info.threads_per_warp;
    this->wps = max(wps_, min_wps);

#ifdef PRESERVE_WARP_FOR_COMM
    // Number of SMs to use for computation. The last SM is preserved for
    // communication only.
    int num_sm_calc = gpu_info.num_sm - 1;
#else
    int num_sm_calc = gpu_info.num_sm;
#endif
    heuristic_optimize_model(model, gpu_info, num_sm_calc);

    this->op_graph = new OpGraph(model, gpu_info);
    this->configure_gpu_buf(model.get_tensors());
}

GpuMgrCtx *DefaultScheduler::create_context(const string &name)
{
    GpuMgrCtx *ctx =
        this->gpu_mgr->create_context(name, this->rank, this->world_size);
    for (BufInfo &bi : this->buf_infos) {
        GpuBuf *buf;
        if (bi.gpu_id == this->gpu_mgr->gpu_id) {
            auto search = this->buf_trans.find(bi.tbuf);
            if (search != this->buf_trans.end()) {
                // Already allocated.
                buf = search->second;
                if (bi.sid != -1) {
                    ctx->mem_export(this->buf_trans[bi.tbuf], bi.offset,
                                    bi.sid);
                }
            } else if (bi.sid == -1) {
                buf = ctx->mem_alloc(bi.bytes, 1);
            } else {
                // Align for RDMA performance.
                buf = ctx->mem_alloc(bi.bytes, 65536);
                ctx->mem_export(buf, bi.offset, bi.sid);
            }
        } else {
            buf = ctx->mem_import(bi.bytes, bi.sid, bi.gpu_id);
        }
        this->buf_trans[bi.tbuf] = buf;
    }
    for (auto &srop : this->send_recv_ops) {
        LOG(DEBUG, "reg_sendrecv: sid=", *(int *)srop->args[0].val,
            " remote=", *(int *)srop->args[1].val,
            " is_recv=", srop->type == OP_RECV);
        ctx->reg_sendrecv(*(int *)srop->args[0].val, *(int *)srop->args[1].val,
                          *(size_t *)srop->args[2].val, srop->type == OP_RECV);
    }
    ctx->freeze();
    this->ctx = ctx;
    return ctx;
}

Tensor *DefaultScheduler::get_tensor(Tensor *tns) const
{
    return tns;
}

GpuBuf *DefaultScheduler::get_gpu_buf(Tensor *tns) const
{
    Tensor *t = get_tensor(tns);
    if (t == nullptr) {
        return nullptr;
    }
    if (t->buf == nullptr) {
        return nullptr;
    }
    auto search = this->buf_trans.find(t->buf);
    if (search == this->buf_trans.end()) {
        return nullptr;
    }
    return search->second;
}

unsigned int DefaultScheduler::get_num_depths() const
{
    return this->op_graph->depth_nodes.size();
}

void DefaultScheduler::schedule_depth(vector<SchedOpSeq *> &depth,
                                      vector<Sched> &scheds)
{
    const GpuInfo &gpu_info = this->gpu_mgr->get_gpu_info();
#ifdef PRESERVE_WARP_FOR_COMM
    // Number of SMs to use for computation. The last SM is preserved for
    // communication only.
    int num_sm_calc = gpu_info.num_sm - 1;
#else
    int num_sm_calc = gpu_info.num_sm;
#endif
    struct
    {
        bool operator()(const SchedOpSeq *s0, const SchedOpSeq *s1) const
        {
            if (s0->get_num_warps() == s1->get_num_warps()) {
                return s0->get_tdims_size() > s1->get_tdims_size();
            }
            return s0->get_num_warps() > s1->get_num_warps();
        }
    } dec_num_warps;
    sort(depth.begin(), depth.end(), dec_num_warps);
    DimType warps_remain = 0;
    for (auto &opseq : depth) {
        warps_remain +=
            (DimType)opseq->get_tdims_size() * (DimType)opseq->get_num_warps();
    }
    LOG(DEBUG, warps_remain, " warps in depth");
    DimType sidx = 0;
    DimType widx = 0;
    for (auto &opseq : depth) {
        DimType tnum = opseq->get_tdims_size();
        DimType wnum = opseq->get_num_warps();
        DimType tidx = 0;
        LOG(DEBUG, "  op", opseq->get_id(), ": tnum ", tnum, " wnum ", wnum,
            " wrem ", warps_remain);
        while ((tidx < tnum) && (warps_remain > 0)) {
            DimType snum = num_sm_calc - sidx;
            DimType div = warps_remain / snum;
            DimType rem = warps_remain % snum;
            if (div >= this->wps) {
                div = this->wps;
                rem = 0;
            }
            if (widx > 0) {
                DimType cnt = 0;
                while (tidx < tnum) {
                    // An SM is not occupied enough by the previous opseq.
                    DimType wend = rem ? div + 1 : div;
                    if ((widx >= div) || (widx + wnum > wend)) {
                        widx = 0;
                        sidx = (sidx + 1) % num_sm_calc;
                        break;
                    }
                    DimType th_b = widx * 32;
                    DimType th_e = (widx + wnum) * 32;
                    LOG(DEBUG, "      sched ", sidx, " ", sidx + 1, " ", th_b,
                        " ", th_e);
                    scheds.emplace_back(opseq, sidx, sidx + 1, th_b, th_e, 0,
                                        tidx++);
                    widx += wnum;
                    warps_remain -= wnum;
                    ++cnt;
                }
                if (cnt > 0) {
                    LOG(DEBUG, "    div ", div, " rem ", rem, ": sm ", sidx,
                        " cnt ", cnt);
                }
                continue;
            }
            assert(widx == 0);
            DimType num;
            DimType sm_b;
            DimType sm_e;
            if (rem > 0) {
                num = math::div_up(div + 1, wnum);
                if (tnum - tidx < num) {
                    num = tnum - tidx;
                }
                sm_b = sidx;
                sm_e = sidx + min(rem, (tnum - tidx) / num);
                assert(sm_e < num_sm_calc);
            } else if (div > 0) {
                num = math::div_up(div, wnum);
                if (tnum - tidx < num) {
                    num = tnum - tidx;
                }
                sm_b = sidx;
                sm_e = min((DimType)num_sm_calc, sidx + (tnum - tidx) / num);
            } else {
                // Should not reach here.
                num = 0;
                assert(false);
            }
            for (DimType i = 0; i < num; ++i) {
#if 0
                LOG(DEBUG, "      sched ", sm_b, " ", sm_e, " ",
                    i * wnum * 32, " ", (i + 1) * wnum * 32);
#endif
                scheds.emplace_back(opseq, sm_b, sm_e, i * wnum * 32,
                                    (i + 1) * wnum * 32, num, tidx + i);
            }
            DimType cnt = (sm_e - sm_b) * num;
            // const SchedOp& so = opseq->get_sched_ops().back();
            // assert(&so.cfg != &VIRT_SCHED_OP_CONFIG);
            // const OpTile& tile = so.cfg.out_deps_tiles[0];
            // LOG(INFO, "    div ", div, " rem ", rem,
            //      ": sm ", sm_b, "-", sm_e - 1, " cnt ", cnt,
            //      " <", tile.x, ",", tile.y, ">");
            tidx += cnt;
            sidx = sm_e - 1;
            warps_remain -= cnt * wnum;
            widx = num * wnum;
            if (widx >= this->wps) {
                widx = 0;
                sidx = (sidx + 1) % num_sm_calc;
            }
        }
        assert(tidx == tnum);
    }
}

void DefaultScheduler::schedule_depth_comm(vector<SchedOpSeq *> &depth,
                                           vector<Sched> &scheds)
{
    const GpuInfo &gpu_info = this->gpu_mgr->get_gpu_info();
    vector<Sched> tmp_scheds;
    int sm_b = gpu_info.num_sm - 1;
    int sm_e = gpu_info.num_sm;
    DimType widx = 0;
    for (auto &opseq : depth) {
        DimType tnum = opseq->get_tdims_size();
        DimType wnum = opseq->get_num_warps();
        LOG(DEBUG, "  op", opseq->get_id(), ": tnum ", tnum, " wnum ", wnum);

        DimType th_b = widx * 32;
        DimType th_e = (widx + wnum) * 32;
        LOG(DEBUG, "sched ", sm_b, " ", sm_e, " ", th_b, " ", th_e);
        tmp_scheds.emplace_back(opseq, sm_b, sm_e, th_b, th_e, 0, 0);
        widx += wnum;
        if (widx >= this->wps) {
            widx = 0;
        }
    }
    // Sort the scheds by th_b.
    sort(tmp_scheds.begin(), tmp_scheds.end(),
         [](const Sched &a, const Sched &b) { return a.th_b < b.th_b; });
    // Emplace the tmp_scheds to the scheds.
    for (auto &sched : tmp_scheds) {
        scheds.emplace_back(sched);
    }
}

vector<string> DefaultScheduler::schedule()
{
    LOG(DEBUG, "DefaultScheduler start scheduling");

    vector<Sched> scheds;
    vector<GpuLoopKernel *> glks;
    for (auto &depth : this->op_graph->depth_nodes) {
        vector<Sched> ds;
        vector<SchedOpSeq *> calc_opseqs;
        vector<SchedOpSeq *> send_opseqs;
        vector<SchedOpSeq *> send_done_opseqs;
        vector<SchedOpSeq *> recv_opseqs;
        for (auto &ogn : depth) {
            if (ogn->opseq.is_send()) {
                send_opseqs.emplace_back(&(ogn->opseq));
            } else if (ogn->opseq.is_recv()) {
                recv_opseqs.emplace_back(&(ogn->opseq));
            } else if (ogn->opseq.is_send_done()) {
                send_done_opseqs.emplace_back(&(ogn->opseq));
            } else {
                calc_opseqs.emplace_back(&(ogn->opseq));
            }
        }
        LOG(DEBUG, "schedule depth");
        this->schedule_depth_comm(send_opseqs, scheds);
        this->schedule_depth(calc_opseqs, scheds);
        this->schedule_depth_comm(send_done_opseqs, scheds);
        this->schedule_depth_comm(recv_opseqs, scheds);
        // TODO: profile one depth
        // Global sync.
        scheds.emplace_back(nullptr, 0, 0, 0, 0, 0, 0);
    }
    return this->scg.codegen_codes_body(scheds);
}

} // namespace ark
