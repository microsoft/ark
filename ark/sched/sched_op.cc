// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "sched/sched_op.h"
#include "logging.h"
#include "math.h"

using namespace std;
#define COM ", "

namespace ark {

const OpConfig *sched_op_config(const Op *op, const GpuInfo &gpu_info)
{
    assert(op != nullptr);
    assert(op->out_deps.size() > 0);
    Tensor *output = op->out_deps[0];
    if (output == nullptr || op->cfg_map == nullptr) {
        return nullptr;
    }
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
    // Heuristic auto-selection of granularity level
    int gran_lev = 0;
    int ndims = output->shape.ndims();
    unsigned int min_wps =
        gpu_info.min_threads_per_block / gpu_info.threads_per_warp;
    for (auto &cfg : search->second) {
        assert(cfg.out_deps_tiles.size() > 0);
        const OpTile &ot = cfg.out_deps_tiles[0];
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
        if (gran_lev == (int)search->second.size() - 1) {
            // no more option, just use the finest-grained config
            break;
        }
        // magic condition
        if ((dim_0 * 2 > ot.y) && (dim_1 * 2 > ot.x) &&
            ((num_tiles * cfg.num_warps) >= (min_wps * gpu_info.num_sm / 2))) {
            break;
        }
        ++gran_lev;
    }
    if (gran_lev == (int)search->second.size()) {
        stringstream configs_str;
        if (search->second.size() > 0) {
            const OpTile &ot = search->second[0].out_deps_tiles[0];
            configs_str << "{ " << ot.x << ", " << ot.y << " }";
        }
        for (int i = 1; i < (int)search->second.size(); ++i) {
            const OpTile &ot = search->second[i].out_deps_tiles[0];
            configs_str << ", { " << ot.x << ", " << ot.y << " }";
        }
        configs_str << ".";
        LOGERR("no valid tile configuration found. Output shape ",
               output->shape, ", available tiles: ", configs_str.str());
    }
    const OpConfig *cfg = &search->second[gran_lev];
    OpConfig *cfg_new = new OpConfig(*cfg);
    // TODO: remove this hack way to set the out_deps_tiles[0].y. Probable
    // solution: Split the layernorm and softmax into two ops, one for
    // calulating the reduction of the mean and variance, the other for the
    // normalization the input.
    if (op->type == OP_LAYERNORM || op->type == OP_SOFTMAX) {
        // The out_deps_tiles[0].y of the original config is 1, we need to make
        // out_deps_tiles[0].y equal to the output last dimension size, which is
        // also the dimension that the layer norm or softmax is performed.
        cfg_new->out_deps_tiles[0].y = output->shape[ndims - 1];
        LOG(DEBUG, "op cfg: ", cfg_new->out_deps_tiles[0].x, COM,
            cfg_new->out_deps_tiles[0].y);
    }
    return cfg_new;
}

SchedOp::SchedOp(const Op *op_, const OpConfig *cfg_, const string name)
    : op{op_}, cfg{cfg_}, name{name}, tnums{}
{
    if (op_ == nullptr) {
        return;
    }
    if (cfg_ == nullptr) {
        LOG(DEBUG, "virtual op: ", op_->name);
        return;
    }
    LOG(DEBUG, "op: ", op_->name, ", cfg: num_warps ", cfg_->num_warps,
        " smem_bytes ", cfg_->smem_bytes, " #in_deps ",
        cfg_->in_deps_tiles.size(), " #out_deps ", cfg_->out_deps_tiles.size(),
        " sync_pre ", cfg_->sync_pre, " sync_post ", cfg_->sync_post);
    // pad the tensor of the SchedOp
    for (unsigned int i = 0; i < this->op->in_deps.size(); ++i) {
        if (i >= this->cfg->in_deps_tiles.size()) {
            LOG(DEBUG, "input tensor can not be all padded");
            break;
        }
        // Update pads based on the tile shape. The tiling is applied to the
        // last two dimensions of the tensor. If the tensor is 1D, the first
        // dimension of the tile shape should be 1.
        auto &tile = this->cfg->in_deps_tiles[i];
        int ndims = this->op->in_deps[i]->ndims();
        vector<DimType> pads;
        if (ndims == 1) {
            if (tile.x != 1) {
                LOGERR("invalid tile shape for 1D tensor: {", tile.x, ", ",
                       tile.y, "}");
            }
            pads.emplace_back(tile.y);
        } else {
            for (int j = 0; j < ndims - 2; ++j) {
                pads.emplace_back(1);
            }
            pads.emplace_back(tile.x);
            pads.emplace_back(tile.y);
        }
        this->op->in_deps[i]->update_pads(pads);
    }
    for (unsigned int i = 0; i < this->op->out_deps.size(); ++i) {
        auto &tile = this->cfg->out_deps_tiles[i];
        int ndims = this->op->out_deps[i]->ndims();
        vector<DimType> pads;
        if (ndims == 1) {
            if (tile.x != 1) {
                LOGERR("invalid tile shape for 1D tensor: {", tile.x, ", ",
                       tile.y, "}");
            }
            pads.emplace_back(tile.y);
        } else {
            for (int j = 0; j < ndims - 2; ++j) {
                pads.emplace_back(1);
            }
            pads.emplace_back(tile.x);
            pads.emplace_back(tile.y);
        }
        this->op->out_deps[i]->update_pads(pads);
    }
    // claculate the tile size for the SchedOp
    if ((this->op->out_deps.size() == 1) && (this->cfg != nullptr)) {
        const OpTile &tile = this->cfg->out_deps_tiles[0];
        const Dims &s = this->op->out_deps[0]->shape;
        int ndims = s.ndims();
        vector<DimType> vec;
        if (ndims == 1) {
            vec.emplace_back((DimType)math::div_up(s[0], tile.y));
        } else {
            int i = 0;
            for (; i < ndims - 2; ++i) {
                vec.emplace_back(s[i]);
            }
            vec.emplace_back((DimType)math::div_up(s[i], tile.x));
            vec.emplace_back((DimType)math::div_up(s[i + 1], tile.y));
        }
        this->tnums = Dims{vec};
        LOG(DEBUG, "SchedOp: ", name, " tile num: ", this->tnums,
            " tile size: {", tile.x, ", ", tile.y, "}");
    }
}

const string SchedOp::func_string() const
{
    if (this->cfg == nullptr) {
        return "";
    }
    return this->op->function_name(*this->cfg);
}

bool SchedOp::is_virtual() const
{
    return this->cfg == nullptr;
}

} // namespace ark
