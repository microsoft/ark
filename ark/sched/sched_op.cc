// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "sched/sched_op.h"

#include "logging.h"
#include "math.h"

using namespace std;

namespace ark {

SchedOp::SchedOp(const Op *op_, const OpConfig *cfg_, const string name)
    : op{op_}, cfg{cfg_}, name{name} {
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
            ERR(SchedulerError, "Not implemented");
        }
        ss << ",";
    }
    return ss.str();
}

bool SchedOp::is_virtual() const { return this->cfg == nullptr; }

}  // namespace ark
