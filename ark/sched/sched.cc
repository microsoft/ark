// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "sched/sched.h"
#include "logging.h"
#include "math.h"

using namespace std;

namespace ark {

BaseScheduler::BaseScheduler(Model &model, int gpu_id, int rank_,
                             int world_size_, int num_warps_per_sm_)
    : model{&model}, gpu_mgr{get_gpu_mgr(gpu_id)}, rank{rank_},
      world_size{world_size_}, num_warps_per_sm{num_warps_per_sm_}
{
    const GpuInfo &gpu_info = this->gpu_mgr->get_gpu_info();
    int max_warps_per_sm =
        (int)(gpu_info.max_threads_per_block / gpu_info.threads_per_warp);
    this->num_warps_per_sm = std::min(num_warps_per_sm_, max_warps_per_sm);
    this->codegen = std::make_unique<CodeGenerator>(this->buf_trans, gpu_info,
                                                    num_warps_per_sm_);
}

// create context on gpu for the model
GpuMgrCtx *BaseScheduler::create_context(const std::string &name)
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
        int sid;
        int remote_rank;
        size_t bytes;
        srop->args.get(&sid, 0);
        srop->args.get(&remote_rank, 2);
        srop->args.get(&bytes, 3);

        LOG(DEBUG, "reg_sendrecv: sid=", sid, " remote=", remote_rank,
            " bytes=", bytes, " is_recv=", srop->type == OP_RECV);
        ctx->reg_sendrecv(sid, remote_rank, bytes, srop->type == OP_RECV);
    }
    ctx->freeze();
    this->ctx = ctx;
    return ctx;
}

const OpConfig *BaseScheduler::sched_op_config(const Op *op)
{
    if (op == nullptr || op->outputs.size() == 0) {
        LOG(ERROR, "unexpected error");
    }
    Tensor *output = op->outputs[0];
    if (output == nullptr || op->cfg_map == nullptr) {
        return nullptr;
    }
    const GpuInfo &gpu_info = this->gpu_mgr->get_gpu_info();
    OpArchType arch_type;
    if (gpu_info.arch == GPU_ARCH_CUDA_70) {
        arch_type = OP_ARCH_CUDA_70;
    } else if (gpu_info.arch == GPU_ARCH_CUDA_80) {
        arch_type = OP_ARCH_CUDA_80;
    } else {
        LOGERR("unsupported GPU architecture: ", gpu_info.arch);
    }
    auto search = op->cfg_map->find({arch_type, op->prec_type});
    if (search == op->cfg_map->end()) {
        return nullptr;
    } else if (op->gran_lev >= 0) {
        if (search->second.size() > (unsigned int)op->gran_lev) {
            return &search->second[op->gran_lev];
        }
        LOGERR("invalid granularity level: ", op->gran_lev);
    }
    std::vector<const OpConfig *> feasible_configs;
    for (auto &cfg : search->second) {
        if (cfg.num_warps <= this->num_warps_per_sm) {
            feasible_configs.push_back(&cfg);
        }
    }
    // Heuristic auto-selection of granularity level
    int gran_lev = 0;
    int ndims = output->shape.ndims();
    unsigned int min_wps =
        gpu_info.min_threads_per_block / gpu_info.threads_per_warp;
    for (auto &cfg : feasible_configs) {
        assert(cfg->output_tiles.size() > 0);
        const OpTile &ot = cfg->output_tiles[0];
        DimType num_tiles;
        DimType dim_0;
        DimType dim_1;
        if (ndims == 1) {
            if (ot.x != 1) {
                ++gran_lev;
                continue;
            }
            dim_0 = output->shape[0];
            dim_1 = 1;
            num_tiles = math::div_up(dim_0, ot.y);
        } else {
            num_tiles = 1;
            for (int i = 0; i < ndims - 2; ++i) {
                num_tiles *= output->shape[i];
            }
            dim_0 = output->shape[ndims - 1];
            dim_1 = output->shape[ndims - 2];
            num_tiles *= math::div_up(dim_0, ot.y);
            num_tiles *= math::div_up(dim_1, ot.x);
        }
        if (gran_lev == (int)feasible_configs.size() - 1) {
            // no more option, just use the finest-grained config
            break;
        }
        // magic condition
        if ((dim_0 * 2 > ot.y) && (dim_1 * 2 > ot.x) &&
            ((num_tiles * cfg->num_warps) >= (min_wps * gpu_info.num_sm / 2))) {
            break;
        }
        ++gran_lev;
    }
    if (gran_lev == (int)feasible_configs.size()) {
        stringstream configs_str;
        if (feasible_configs.size() > 0) {
            const OpTile &ot = feasible_configs[0]->output_tiles[0];
            configs_str << "{ " << ot.x << ", " << ot.y << " }";
        }
        for (int i = 1; i < (int)feasible_configs.size(); ++i) {
            const OpTile &ot = feasible_configs[i]->output_tiles[0];
            configs_str << ", { " << ot.x << ", " << ot.y << " }";
        }
        configs_str << ".";
        LOGERR("no valid tile configuration found. Output shape ",
               output->shape, ", available tiles: ", configs_str.str());
    }
    const OpConfig *cfg = feasible_configs[gran_lev];
    OpConfig *cfg_new = new OpConfig(*cfg);
    // TODO: remove this hack way to set the output_tiles[0].y. Probable
    // solution: Split the layernorm and softmax into two ops, one for
    // calulating the reduction of the mean and variance, the other for the
    // normalization the input.
    if (op->type == OP_LAYERNORM || op->type == OP_SOFTMAX) {
        // The output_tiles[0].y of the original config is 1, we need to make
        // output_tiles[0].y equal to the output last dimension size, which is
        // also the dimension that the layer norm or softmax is performed.
        cfg_new->output_tiles[0].y = output->shape[ndims - 1];
    }
    return cfg_new;
}

GpuBuf *BaseScheduler::get_gpu_buf(Tensor *tns) const
{
    if (tns == nullptr) {
        return nullptr;
    }
    if (tns->buf == nullptr) {
        return nullptr;
    }
    auto search = this->buf_trans.find(tns->buf);
    if (search == this->buf_trans.end()) {
        return nullptr;
    }
    return search->second;
}

} // namespace ark