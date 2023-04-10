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

static int sched_op_num_tiles(const Op &op, const OpTile &tile)
{
    if (op.out_deps.size() == 0) {
        return 0;
    }
    assert(op.out_deps[0] != nullptr);
    auto &s = op.out_deps[0]->shape;
    return s[0] * math::div_up(s[1], tile.y) * math::div_up(s[3], tile.x);
}

void DefaultScheduler::configure_gpu_buf()
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
                    // TODO: generalize converting rank to GPU ID.
                    int nrph = get_env().num_ranks_per_host;
                    int dst_gpu_id = dst_rank % nrph;
                    if ((dst_rank / nrph) == (this->rank / nrph)) {
                        // Same node.
                        this->buf_infos.emplace_back(dst_gpu_id, bytes, nullptr,
                                                     sid, 0);
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
    for (auto &tns : this->opt_model->get_tensors()) {
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
                                   const Model &model, unsigned int wps_)
    : SchedulerBase(gpu_id, rank_, world_size_, wps_), scg{buf_trans, 108, wps_,
                                                           world_size_}
{
    const GpuInfo &gpu_info = this->gpu_mgr->get_gpu_info();
    unsigned int min_wps =
        gpu_info.min_threads_per_block / gpu_info.threads_per_warp;
    this->wps = max(wps_, min_wps);
    this->opt_model = this->optimize_model(model);
    this->op_graph =
        new OpGraph(*this->opt_model, this->gpu_mgr->get_gpu_info());
    this->configure_gpu_buf();
}

GpuMgrCtx *DefaultScheduler::create_context(const string &name)
{
    GpuMgrCtx *ctx =
        this->gpu_mgr->create_context(name, this->rank, this->world_size);
    for (BufInfo &bi : this->buf_infos) {
        GpuBuf *buf;
        if (bi.gpu_id == this->gpu_mgr->gpu_id) {
            if (bi.sid == -1) {
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
    auto search = this->tns_trans.find(tns);
    if (search == this->tns_trans.end()) {
        return nullptr;
    }
    return search->second;
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

static Tensor *copy_tensor_to_opt_model(
    Model *opt_model, Tensor *tns, map<Tensor *, Tensor *> &tns_trans,
    map<TensorBuf *, TensorBuf *> &tns_buf_trans)
{
    // Translate the original tensor address `tns` into the corresponding
    // tensor address of `opt_model` and return it.
    Tensor *t;
    auto search = tns_trans.find(tns);
    if (search != tns_trans.end()) {
        // already exists.
        t = search->second;
    } else {
        // Create a corresponding tensor. First, create a TensorBuf if needed.
        auto search2 = tns_buf_trans.find(tns->buf);
        TensorBuf *buf;
        if (search2 != tns_buf_trans.end()) {
            buf = search2->second;
        } else {
            // TensorBuf size will be determined later.
            assert(tns->buf != nullptr);
            buf = opt_model->create_tensor_buf();
            tns_buf_trans[tns->buf] = buf;
        }
        t = opt_model->tensor(tns->shape, tns->type, buf, tns->ldims, tns->offs,
                              tns->pads, {}, tns->exported, tns->imported);
        tns_trans[tns] = t;
        LOG(DEBUG, "translate: tns ", tns, " (tnsbuf ", tns->buf, ") -> tns ",
            t, " (tnsbuf ", t->buf, ')');
    }
    return t;
}

Model *DefaultScheduler::optimize_model(const Model &model)
{
    LOG(DEBUG, "Optimizing model...");
    const GpuInfo &gpu_info = this->gpu_mgr->get_gpu_info();
#ifdef PRESERVE_WARP_FOR_COMM
    // Number of SMs to use for computation. The last SM is preserved for
    // communication only.
    int num_sm_calc = gpu_info.num_sm - 1;
#else
    int num_sm_calc = gpu_info.num_sm;
#endif
    Model *opt_model = new Model;
    auto &tns_trans = this->tns_trans;
    map<TensorBuf *, TensorBuf *> tns_buf_trans;
    for (auto &op : model.get_ops()) {
        LOG(DEBUG, "Optimizing op: ", op->name);
        // Scheduling configuration.
        const OpConfig *cfg = sched_op_config(op.get(), gpu_info);
        // Create input tensors if those don't exist in `opt_model`.
        int idx = 0;
        std::vector<Tensor *> in_deps;
        for (auto &tns : op->in_deps) {
            Tensor *t = copy_tensor_to_opt_model(opt_model, tns, tns_trans,
                                                 tns_buf_trans);
            in_deps.emplace_back(t);
            if (cfg->num_warps == 0) {
                continue;
            }
            // Update padding of tensor.
            const OpTile &tile = cfg->in_deps_tiles[idx++];
            LOG(DEBUG, "Tensor buf ", t->buf, " update_pads ", tile.x, " ",
                tile.y);
            t->update_pads({tile.x, tile.y});
        }
        // Replace optimizable Op patterns.
        switch (op->type) {
        case OP_MATMUL: {
#if (MATMUL_GRAPH_OPT)
            assert(op->in_deps.size() == 2);
            assert(op->out_deps.size() == 1);
            assert(cfg->num_warps > 0);

            int num_tiles = sched_op_num_tiles(*op, cfg->out_deps_tiles[0]);
            int num_sm = num_sm_calc;
            int splitk = 1;
            bool do_split = false;
            if ((op->gran_lev == -1) && (num_tiles > 0) &&
                (num_tiles <= num_sm)) {
                // Split matmul.
                DimType inner_dim = op->in_deps[0]->shape[3];
                DimType inner_pad = cfg->in_deps_tiles[0].y;
                inner_pad = max(inner_pad, op->in_deps[0]->pads[3]);
                inner_pad = max(inner_pad, op->in_deps[1]->pads[1]);
                size_t max_splitk =
                    min((size_t)2 * num_sm / num_tiles,
                        math::div_up(inner_dim, cfg->in_deps_tiles[0].y));
                int unitk =
                    math::pad(math::div_up(inner_dim, max_splitk), inner_pad);
                splitk = math::div_up(inner_dim, unitk);
                assert(splitk > 0);
                if (splitk > 1) {
                    do_split = true;
                    LOG(DEBUG, "splitk ", splitk, " inner_dim ", inner_dim,
                        " unitk ", unitk);
                }
            }
            if (do_split) {
                Tensor *input = tns_trans[op->in_deps[0]];
                Tensor *other = tns_trans[op->in_deps[1]];
                Tensor *output = nullptr;
                auto search = tns_trans.find(op->out_deps[0]);
                if (search != tns_trans.end()) {
                    // If the output already exists, it means that the original
                    // model specified an existing tensor as the output.
                    // Thus we do the same here.
                    output = search->second;
                }
                bool trans_input = *(bool *)op->args[0].val;
                bool trans_other = *(bool *)op->args[1].val;
                Tensor *t =
                    opt_model->matmul(input, other, output, splitk, trans_input,
                                      trans_other, false, op->name);
                auto search2 = tns_buf_trans.find(t->buf);
                if (search2 == tns_buf_trans.end()) {
                    tns_buf_trans[op->out_deps[0]->buf] = t->buf;
                }
                auto search3 = tns_trans.find(t);
                if (search3 == tns_trans.end()) {
                    tns_trans[op->out_deps[0]] = t;
                    LOG(DEBUG, "translate: tns ", op->out_deps[0], " (tnsbuf ",
                        op->out_deps[0]->buf, ") -> tns ", t, " (tnsbuf ",
                        t->buf, ')');
                }
                const OpTile &tile = cfg->out_deps_tiles[0];
                // Update padding of each split tensor &
                // the final output tensor.
                const Op *latest_op = opt_model->get_gen_op(t);
                vector<Tensor *> in_deps;
                if (latest_op->type == OP_REDUCE) {
                    const Op *id_op =
                        opt_model->get_gen_op(latest_op->in_deps[0]);
                    assert(id_op->type == OP_REFER);
                    in_deps = id_op->in_deps;
                } else if (latest_op->type == OP_MATMUL) {
                    assert(false);
                    in_deps.emplace_back(t);
                }
                for (auto &in_dep : in_deps) {
                    in_dep->update_pads({tile.x, tile.y});
                    const Op *in_dep_op = opt_model->get_gen_op(in_dep);
                    if (in_dep_op == nullptr) {
                        continue;
                    }
                    Tensor *in_dep_in = in_dep_op->in_deps[0];
                    Tensor *in_dep_ot = in_dep_op->in_deps[1];
                    assert(in_dep_in != nullptr);
                    assert(in_dep_ot != nullptr);
                    auto &tile_in = cfg->in_deps_tiles[0];
                    auto &tile_ot = cfg->in_deps_tiles[1];
                    in_dep_in->update_pads({tile_in.x, tile_in.y});
                    in_dep_ot->update_pads({tile_ot.x, tile_ot.y});
                }
                break;
            }
            // Go to default.
#endif // (MATMUL_GRAPH_OPT)
        }
        default: {
            // By default, create the same Op in `opt_model`.
            idx = 0;
            std::vector<Tensor *> out_deps;
            for (auto &tns : op->out_deps) {
                Tensor *t = copy_tensor_to_opt_model(opt_model, tns, tns_trans,
                                                     tns_buf_trans);
                out_deps.emplace_back(t);
                if (cfg->num_warps == 0) {
                    continue;
                }
                // Update padding of tensor.
                const OpTile &tile = cfg->out_deps_tiles[idx++];
                LOG(DEBUG, "Tensor buf ", t->buf, " update_pads ", tile.x, " ",
                    tile.y);
                t->update_pads({tile.x, tile.y});
            }
            opt_model->create_op(op->type, op->prec_type, in_deps, out_deps,
                                 op->args, op->name, op->gran_lev);
        }
        }
    }
#if (ALLOC_UNUSED_TENSORS)
    // Allocate unused tensors.
    for (auto &tns : model.get_tensors()) {
        copy_tensor_to_opt_model(opt_model, tns.get(), tns_trans,
                                 tns_buf_trans);
    }
#endif // (ALLOC_UNUSED_TENSORS)
    LOG(DEBUG, "Model optimization finished");
    return opt_model;
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
        warps_remain += opseq->get_tdims_size() * opseq->get_num_warps();
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
        DimType tidx = 0;
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
        vector<SchedOpSeq *> recv_opseqs;
        for (auto &ogn : depth) {
            if (ogn->opseq.is_send()) {
                send_opseqs.emplace_back(&(ogn->opseq));
            } else if (ogn->opseq.is_recv()) {
                recv_opseqs.emplace_back(&(ogn->opseq));
            } else {
                calc_opseqs.emplace_back(&(ogn->opseq));
            }
        }
        LOG(DEBUG, "schedule depth");
        this->schedule_depth_comm(send_opseqs, scheds);
        this->schedule_depth(calc_opseqs, scheds);
        this->schedule_depth_comm(recv_opseqs, scheds);
        // TODO: profile one depth
        // Global sync.
        scheds.emplace_back(nullptr, 0, 0, 0, 0, 0, 0);
    }
    return this->scg.codegen_codes_body(scheds);
}

} // namespace ark
