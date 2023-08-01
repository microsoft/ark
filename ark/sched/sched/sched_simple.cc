// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "env.h"
#include "logging.h"
#include "math.h"
#include "model.h"
#include "sched/sched.h"

using namespace std;

namespace ark {

SimpleScheduler::SimpleScheduler(Model &model, int gpu_id, int rank_,
                                 int world_size_, int num_warps_per_sm_)
    : BaseScheduler(model, gpu_id, rank_, world_size_, num_warps_per_sm_)
{
    const GpuInfo &gpu_info = this->gpu_mgr->get_gpu_info();
    this->codegen = make_unique<SimpleCodeGenerator>(
        this->buf_trans, gpu_info, num_warps_per_sm_, this->world_size);
}

//
void SimpleScheduler::schedule()
{
    LOG(DEBUG, "SimpleScheduler start scheduling");
    const GpuInfo &gpu_info = this->gpu_mgr->get_gpu_info();

    int op_idx = 0;
    int opseq_idx = 0;
    std::vector<Op *> all_ops;
    for (auto &op : model->impl->get_ops()) {
        all_ops.push_back(op);
    }
    std::vector<Tensor *> finished_tensors;

    // get the input tensors of the model, and add them to the
    // finished_tensors vector
    for (auto &tns : model->impl->get_tensors()) {
        if (model->impl->get_producer(tns) == nullptr) {
            finished_tensors.push_back(tns);
        }
    }
    this->opseqs.emplace_back(std::make_unique<SchedOpSeq>(opseq_idx));
    opseq_idx++;

    while (!all_ops.empty()) {
        // find the next op to schedule
        Op *op = nullptr;
        for (size_t i = 0; i < all_ops.size(); i++) {
            bool input_ready = true;
            // check if all the input tensors of the op are ready
            for (Tensor *&tns : all_ops[i]->inputs) {
                if (std::find(finished_tensors.begin(), finished_tensors.end(),
                              tns) == finished_tensors.end()) {
                    input_ready = false;
                    break;
                }
            }
            if (input_ready) {
                op = all_ops[i];
                all_ops.erase(all_ops.begin() + i);
                // add the output tensors of the op to the
                // finished_tensors
                for (Tensor *tns : op->outputs) {
                    finished_tensors.push_back(tns);
                }
                break;
            }
        }
        if (op == nullptr) {
            LOG(INFO, "Cannot find next op to schedule");
            break;
        }
        // schedule model with sched_op_config and pad the tensors
        // by looking up the config table of the op in sched_op_config
        const OpConfig *cfg = sched_op_config(op, gpu_info);

        string sched_op_name("op_" + to_string(op_idx) + "_" + op->name);
        for (size_t i = 0; i < sched_op_name.size(); i++) {
            if (sched_op_name[i] == '/') {
                sched_op_name[i] = '_';
            }
        }
        LOG(DEBUG, "sched_op: ", sched_op_name);
        SchedOp sched_op(op, cfg, sched_op_name);
        // some virtual ops like ops_tensor are not scheduled, but we need to
        // allocated gpu_buf for them
        this->sched_ops.push_back(sched_op);
        if (cfg == nullptr) {
            continue;
        }
        // We create an opseq for each op for simplicity, we don't merge ops
        // into opseqs in SimpleScheduler
        this->opseqs.emplace_back(std::make_unique<SchedOpSeq>(op_idx));
        SchedOpSeq *opseq = this->opseqs.back().get();

        LOG(DEBUG, "get_sched_ops: ", opseq->get_sched_ops().size());
        opseq->append(op, cfg);
        op_idx++;
    }
    if (!all_ops.empty()) {
        LOGERR("Cannot schedule all ops");
    }

    this->configure_gpu_buf(model->impl->get_tensors());
}

void SimpleScheduler::schedule_sched_opseq(SchedOpSeq &seq, int max_wps,
                                           int max_sm_num,
                                           vector<Sched> &scheds)
{
    int seq_tile_num = seq.get_tdims_size();
    LOG(DEBUG, "opseq", seq.get_id(), " seq_tile_num: ", seq_tile_num);
    int warps_per_tile = seq.get_num_warps();

    int sched_tile_idx = 0;
    for (int seq_depth = 0; sched_tile_idx < seq_tile_num; seq_depth++) {
        int tile_num = seq_tile_num - sched_tile_idx;
        // calculate the number of tiles that can be scheduled in this
        // seq_depth
        int max_tiles_per_sm = max_wps / warps_per_tile;
        int max_tile_num = max_sm_num * max_wps / warps_per_tile;
        int sm_b;
        int sm_e;
        int warp_b;
        int warp_e;
        int tiles_per_sm;
        int sched_tile_num;
        int remained_tile_num = 0;
        bool have_remain = false;
        // if the seq_tile_num * warp_per_tile > max_sm_num *
        // max_wps, the seq can't be finished in one depth
        // we use all the sm resources to do the task
        if (tile_num > max_tile_num) {
            sched_tile_num = max_tile_num;
            tiles_per_sm = max_tiles_per_sm;
            sm_b = 0;
            sm_e = max_sm_num;
            warp_b = 0;
            warp_e = tiles_per_sm * warps_per_tile;
        } else {
            // if the tile_num is less than the number of the sm, we
            // only assign one tile to tile_num sm
            if (tile_num < max_sm_num) {
                sched_tile_num = tile_num;
                tiles_per_sm = 1;
                sm_b = 0;
                sm_e = tile_num;
                warp_b = 0;
                warp_e = tiles_per_sm * warps_per_tile;
            } else {
                // if the tile can be evenly distributed to all sm, we
                // assign each sm with the same number of tiles
                if (tile_num % max_sm_num == 0) {
                    sched_tile_num = tile_num;
                    tiles_per_sm = tile_num / max_sm_num;
                    sm_b = 0;
                    sm_e = max_sm_num;
                    warp_b = 0;
                    warp_e = tiles_per_sm * warps_per_tile;
                } else {
                    have_remain = true;
                }
            }
        }
        if (have_remain == false) {
            // Sched sched{};
            LOG(DEBUG, "sched: ", seq.get_id(), "sm_b: ", sm_b, " sm_e: ", sm_e,
                " warp_b: ", warp_b, " warp_e: ", warp_e,
                " tiles_per_sm: ", tiles_per_sm,
                " sched_tile_idx: ", sched_tile_idx);
            scheds.emplace_back(&seq, sm_b, sm_e, warp_b * 32, warp_e * 32,
                                tiles_per_sm, sched_tile_idx);
            sched_tile_idx += sched_tile_num;
        } else {
            // if the tile can't be evenly distributed to all
            // sm, we first use the first remained_tile_num sm
            // , each sm execute tiles_per_sm+1 tiles, and the
            // rest sm execute tiles_per_sm tiles
            // this is the first SchedTileTask
            tiles_per_sm = tile_num / max_sm_num + 1;
            remained_tile_num = tile_num % max_sm_num;
            sched_tile_num = remained_tile_num * tiles_per_sm;
            sm_b = 0;
            sm_e = remained_tile_num;
            warp_b = 0;
            warp_e = tiles_per_sm * warps_per_tile;
            scheds.emplace_back(&seq, sm_b, sm_e, warp_b * 32, warp_e * 32,
                                tiles_per_sm, sched_tile_idx);
            sched_tile_idx += sched_tile_num;

            // the second SchedTileTask
            tiles_per_sm = tile_num / max_sm_num;
            sched_tile_num = (max_sm_num - remained_tile_num) * tiles_per_sm;
            sm_b = remained_tile_num;
            sm_e = max_sm_num;
            warp_b = 0;
            warp_e = tiles_per_sm * warps_per_tile;
            scheds.emplace_back(&seq, sm_b, sm_e, warp_b * 32, warp_e * 32,
                                tiles_per_sm, sched_tile_idx);
            sched_tile_idx += sched_tile_num;
        }
        LOG(DEBUG, "sched_tile_idx: ", sched_tile_idx);
    }
    if (seq_tile_num != -1 && sched_tile_idx != seq_tile_num) {
        LOGERR("only ", sched_tile_idx, " tiles are scheduled, but ",
               seq_tile_num, " tiles are needed to be scheduled");
    }
}

vector<string> SimpleScheduler::gen_code()
{
    LOG(DEBUG, "SimpleScheduler start scheduling");
    int num_sm = this->gpu_mgr->get_gpu_info().num_sm;
    vector<Sched> scheds;
    for (auto &seq : this->opseqs) {
        this->schedule_sched_opseq(*seq, this->num_warps_per_sm, num_sm,
                                   scheds);
    }
    return this->codegen->codegen_codes_body(scheds);
}

void SimpleScheduler::configure_gpu_buf(const std::list<Tensor *> &)
{
    // A TensorBuf can be located on a local GPU or a remote GPU. If it is on
    // this rank's GPU, it should be allocated and might be exported to other
    // GPUs. If it is on a remote GPU (the gid is not equal to this rank), it
    // should be imported.
    // A TensorBuf can have multi tensors pointing to it. Different Tensor
    // represent a different sharding or view of the same TensorBuf.
    map<TensorBuf *, vector<Tensor *>> bufs;
    // export_tns_sids is a map of the TensorBuf that needed to be exported, and
    // the corresponding tensors and sids. A TensorBuf can have multiple tensors
    // pointing to it, and might be exported to multiple ranks as different
    // Tensor.
    map<TensorBuf *, vector<pair<Tensor *, int>>> export_tns_sids;
    // pad the tensors according to the tile size of the op
    for (auto &sop : this->sched_ops) {
        const int send_ready_flag_sid_offset = 128;

        LOG(DEBUG, "configure_gpu_buf: ", sop.get_op()->name);
        if (sop.get_op()->type == OP_SEND_MM) {
            int sid;
            int dst_gid;
            sop.get_op()->args.get(&sid, 0);
            sop.get_op()->args.get(&dst_gid, 1);
            // import the recvbuf, the recvbuf should be allocated on the
            // receiver GPU
            Tensor *recvbuf = sop.get_op()->inputs[1];
            this->buf_infos.emplace_back(dst_gid, recvbuf->shape_bytes(),
                                         recvbuf->buf, sid, 0);

            // configure the send_ready_flag, the send_ready_flag needed to be
            // exported to the recv GPU, since the sid of the send_ready_flag
            // should not be the same as the recvBuf, so I use the sid+128 as
            // the sid of the send_ready_flag
            Tensor *send_ready_flag = sop.get_op()->inputs[2];
            export_tns_sids[send_ready_flag->buf].emplace_back(
                send_ready_flag, sid + send_ready_flag_sid_offset);
        } else if (sop.get_op()->type == OP_RECV_MM) {
            int sid;
            int src_gid;
            sop.get_op()->args.get(&sid, 0);
            sop.get_op()->args.get(&src_gid, 1);
            // configure the recvbuf, the recvbuf needed to be export the to the
            // sender GPU, the sid is the same as the sid of the send_mm op and
            // the recv_mm op
            Tensor *recvbuf = sop.get_op()->inputs[1];
            export_tns_sids[recvbuf->buf].emplace_back(recvbuf, sid);

            // import the send_ready_flag, the send_ready_flag tensor should be
            // allocated on the sender GPU
            Tensor *send_ready_flag = sop.get_op()->inputs[2];
            this->buf_infos.emplace_back(
                src_gid, send_ready_flag->shape_bytes(), send_ready_flag->buf,
                sid + send_ready_flag_sid_offset, 0);
        }

        if (sop.get_op()->type == OP_SEND) {
            Tensor *in = sop.get_op()->inputs[0];
            int sid;
            int rank;
            int dst_rank;
            size_t bytes;
            sop.get_op()->args.get(&sid, 0);
            sop.get_op()->args.get(&rank, 1);
            sop.get_op()->args.get(&dst_rank, 2);
            sop.get_op()->args.get(&bytes, 3);
            // TODO: generalize converting rank to GPU ID.
            int nrph = get_env().num_ranks_per_host;
            int dst_gpu_id = dst_rank % nrph;
            if ((dst_rank / nrph) == (this->rank / nrph)) {
                // Same node.
                this->buf_infos.emplace_back(dst_gpu_id, bytes, nullptr, sid,
                                             0);
            }
            export_tns_sids[in->buf].emplace_back(in, sid);
            this->send_recv_ops.emplace_back(sop.get_op());
        } else if (sop.get_op()->type == OP_RECV) {
            Tensor *in = sop.get_op()->inputs[0];
            int sid;
            sop.get_op()->args.get(&sid, 0);
            export_tns_sids[in->buf].emplace_back(in, sid);
            this->send_recv_ops.emplace_back(sop.get_op());
        }
        for (auto &tns : sop.get_op()->inputs) {
            // if the tensor is not imported, it should be allocated on this GPU
            if (tns->imported == false)
                bufs[tns->buf].emplace_back(tns);
        }
        // TODO: print warning if the tensor is not used by any real computation
        for (auto &tns : sop.get_op()->outputs) {
            if (tns->imported == false)
                bufs[tns->buf].emplace_back(tns);
        }
    }
    // Fix TensorBuf size. The size of the TensorBuf is the max size of its
    // tensors.
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
            LOG(DEBUG, "Tensor buf ", tns->buf, " pads ", tns->pads, " shape ",
                sh, " ldims ", ld, " offs ", tns->offs, " exported ",
                tns->exported, " max_bytes ", max_bytes);
        }
        // Store the size.
        buf->bytes = max_bytes;
    }
    LOG(DEBUG, "bufs.size(): ", bufs.size());
    // the tensor that needed to be allocated
    vector<TensorBuf *> to_alloc;
    for (auto &sop : this->sched_ops) {
        for (auto &tns : sop.get_op()->inputs) {
            size_t buf_num = bufs.erase(tns->buf);
            if (buf_num > 0) {
                assert(buf_num == 1);
                to_alloc.emplace_back(tns->buf);
            }
        }
        for (auto &tns : sop.get_op()->outputs) {
            size_t buf_num = bufs.erase(tns->buf);
            if (buf_num > 0) {
                assert(buf_num == 1);
                to_alloc.emplace_back(tns->buf);
            }
        }
    }
    LOG(DEBUG, "alloc: ", to_alloc.size(), " TensorBufs");
    // Allocate GPU buffers.
    for (auto &buf : to_alloc) {
        int sid = -1;
        size_t off = 0;
        auto search = export_tns_sids.find(buf);
        if (search != export_tns_sids.end()) {
            for (auto &p : search->second) {
                Tensor *t = p.first;
                sid = p.second;
                off = t->offset() * t->type_bytes();
                // the TensorBuf that needed to be allocated and exported
                this->buf_infos.emplace_back(this->gpu_mgr->gpu_id, buf->bytes,
                                             buf, sid, off);
            }
        } else {
            // the TensorBuf that needed to be allocated
            this->buf_infos.emplace_back(this->gpu_mgr->gpu_id, buf->bytes, buf,
                                         sid, off);
        }
    }
}

} // namespace ark
