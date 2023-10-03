// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "sched/sched_op.h"

#include "logging.h"
#include "math.h"

using namespace std;

namespace ark {

SchedOp::SchedOp(const Op *op_, const OpConfig *cfg_, const string name)
    : op{op_}, cfg{cfg_}, name{name}, tnums{} {
    if (op_ == nullptr) {
        return;
    }
    if (cfg_ == nullptr) {
        LOG(DEBUG, "virtual op: ", op_->name);
        return;
    }
    // LOG(DEBUG, "op: ", op_->name, ", cfg: num_warps ", cfg_->num_warps,
    //     " smem_bytes ", cfg_->smem_bytes, " #inputs ",
    //     cfg_->input_tiles.size(), " #outputs ", cfg_->output_tiles.size(), "
    //     sync_pre ", cfg_->sync_pre, " sync_post ", cfg_->sync_post);

    // pad the tensor of the SchedOp
    for (unsigned int i = 0; i < this->op->inputs.size(); ++i) {
        if (i >= this->cfg->input_tiles.size()) {
            LOG(DEBUG, "input tensor can not be all padded");
            break;
        }
        // Update pads based on the tile shape. The tiling is applied to the
        // last two dimensions of the tensor. If the tensor is 1D, the first
        // dimension of the tile shape should be 1.
        auto tile = this->cfg->input_tiles[i];
        if (tile.x < 0) {
            tile.x = 1;
        }
        if (tile.y < 0) {
            tile.y = 1;
        }
        int ndims = this->op->inputs[i]->ndims();
        vector<DimType> pads;
        if (ndims == 1) {
            if (tile.x != 1) {
                LOG(ERROR, "invalid tile shape for 1D tensor: {", tile.x, ", ",
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
        this->op->inputs[i]->update_pads(pads);
    }
    for (unsigned int i = 0; i < this->op->outputs.size(); ++i) {
        auto tile = this->cfg->output_tiles[i];
        if (tile.x < 0) {
            tile.x = 1;
        }
        if (tile.y < 0) {
            tile.y = 1;
        }
        int ndims = this->op->outputs[i]->ndims();
        vector<DimType> pads;
        if (ndims == 1) {
            if (tile.x != 1) {
                LOG(ERROR, "invalid tile shape for 1D tensor: {", tile.x, ", ",
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
        this->op->outputs[i]->update_pads(pads);
    }
    // claculate the tile size for the SchedOp
    if ((this->op->outputs.size() == 1) && (this->cfg != nullptr)) {
        const OpTile &tile = this->cfg->output_tiles[0];
        const Dims &s = this->op->outputs[0]->shape;
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
    }
}

const string SchedOp::function_name() const {
    if (this->cfg == nullptr) {
        return "";
    }
    return this->op->function_name(*this->cfg);
}

const string SchedOp::serialize() const {
    // Serialize sop definition as a string.
    std::stringstream ss;
    ss << this->function_name() << ",";

    OpArgs call_args = this->get_op()->function_call_args(*(this->get_cfg()));
    for (const OpArg &arg : call_args.get_args()) {
        if (arg.type == OP_ARG_TENSOR) {
            Tensor *tns;
            arg.get(&tns);
            ss << tns->type.type_str() << " *";
        } else if (arg.type == OP_ARG_FLOAT) {
            ss << "float";
        } else if (arg.type == OP_ARG_INT) {
            ss << "int";
        } else if (arg.type == OP_ARG_BOOL) {
            ss << "bool";
        } else if (arg.type == OP_ARG_INT64) {
            ss << "long long int";
        } else if (arg.type == OP_ARG_UINT64) {
            ss << "uint64_t";
        } else {
            LOG(ERROR, "Not implemented");
        }
        ss << ",";
    }
    return ss.str();
}

bool SchedOp::is_virtual() const { return this->cfg == nullptr; }

}  // namespace ark
