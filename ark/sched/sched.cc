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
    this->codegen =
        std::make_unique<CodeGenerator>(gpu_info, num_warps_per_sm_);
}

// create context on gpu for the model
GpuMgrCtx *BaseScheduler::create_context(const std::string &name)
{
    GpuMgrCtx *ctx =
        this->gpu_mgr->create_context(name, this->rank, this->world_size);
    for (BufInfo &bi : this->buf_infos) {
        GpuBuf *buf;
        if (bi.gpu_id == this->gpu_mgr->gpu_id) {
            if (bi.tbuf->buf != nullptr) {
                // Already allocated.
                buf = static_cast<GpuBuf *>(bi.tbuf->buf);
                if (bi.sid != -1) {
                    ctx->mem_export(buf, bi.offset, bi.sid);
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
        if (bi.tbuf != nullptr) {
            bi.tbuf->buf = buf;
        }
    }
    for (auto &srop : this->send_recv_ops) {
        int sid;
        int remote_rank;
        size_t bytes;
        srop->args.get(&sid, 0);
        srop->args.get(&remote_rank, 2);
        srop->args.get(&bytes, 3);

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
    OpArchType arch_type = OP_ARCH_CUDA_60;
    if (gpu_info.arch == GPU_ARCH_CUDA_60) {
        arch_type = OP_ARCH_CUDA_60;
    } else if (gpu_info.arch == GPU_ARCH_CUDA_70) {
        arch_type = OP_ARCH_CUDA_70;
    } else if (gpu_info.arch == GPU_ARCH_CUDA_80) {
        arch_type = OP_ARCH_CUDA_80;
    } else if (gpu_info.arch == GPU_ARCH_CUDA_90) {
        arch_type = OP_ARCH_CUDA_90;
    } else {
        LOG(ERROR, "unsupported GPU architecture: ", gpu_info.arch);
    }
    auto &configs = op->cfg_map->get({arch_type, op->prec_type});
    if (configs.empty()) {
        LOG(ERROR, "no config found for op: ", op->name,
            ", arch_type: ", arch_type, ", prec_type: ", op->prec_type);
    }
    if (op->gran_lev >= 0) {
        if (configs.size() > (unsigned int)op->gran_lev) {
            return &configs[op->gran_lev];
        }
        LOG(ERROR, "invalid granularity level: ", op->gran_lev);
    }
    std::vector<const OpConfig *> feasible_configs;
    for (auto &cfg : configs) {
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
        DimType ot_x = (ot.x == -1) ? output->ldims[ndims - 2] : ot.x;
        DimType ot_y = (ot.y == -1) ? output->ldims[ndims - 1] : ot.y;
        DimType num_tiles;
        DimType dim_0;
        DimType dim_1;
        if (ndims == 1) {
            if (ot_x != 1) {
                ++gran_lev;
                continue;
            }
            dim_0 = output->shape[0];
            dim_1 = 1;
            num_tiles = math::div_up(dim_0, ot_y);
        } else {
            num_tiles = 1;
            for (int i = 0; i < ndims - 2; ++i) {
                num_tiles *= output->shape[i];
            }
            dim_0 = output->shape[ndims - 1];
            dim_1 = output->shape[ndims - 2];
            num_tiles *= math::div_up(dim_0, ot_y);
            num_tiles *= math::div_up(dim_1, ot_x);
        }
        if (gran_lev == (int)feasible_configs.size() - 1) {
            // no more option, just use the finest-grained config
            break;
        }
        // magic condition
        if ((dim_0 * 2 > ot_y) && (dim_1 * 2 > ot_x) &&
            ((num_tiles * cfg->num_warps) >= (min_wps * gpu_info.num_sm / 2))) {
            break;
        }
        ++gran_lev;
    }
    if (gran_lev == (int)feasible_configs.size()) {
        stringstream configs_str;
        if (feasible_configs.size() > 0) {
            const OpTile &ot = feasible_configs[0]->output_tiles[0];
            DimType ot_x = (ot.x == -1) ? output->ldims[ndims - 2] : ot.x;
            DimType ot_y = (ot.y == -1) ? output->ldims[ndims - 1] : ot.y;
            configs_str << "{ " << ot_x << ", " << ot_y << " }";
        }
        for (int i = 1; i < (int)feasible_configs.size(); ++i) {
            const OpTile &ot = feasible_configs[i]->output_tiles[0];
            DimType ot_x = (ot.x == -1) ? output->ldims[ndims - 2] : ot.x;
            DimType ot_y = (ot.y == -1) ? output->ldims[ndims - 1] : ot.y;
            configs_str << ", { " << ot_x << ", " << ot_y << " }";
        }
        configs_str << ".";
        LOG(ERROR, "no valid tile configuration found. Output shape ",
            output->shape, ", available tiles: ", configs_str.str());
    }
    const OpConfig *cfg = feasible_configs[gran_lev];
    OpConfig *cfg_new = new OpConfig(*cfg);
    OpTile &op_tile = cfg_new->output_tiles[0];
    if (op_tile.x == -1) {
        op_tile.x = output->ldims[ndims - 2];
    }
    if (op_tile.y == -1) {
        op_tile.y = output->ldims[ndims - 1];
    }
    return cfg_new;
}

} // namespace ark
