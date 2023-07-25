// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_common.h"
#include "include/ark.h"
#include "json.h"
#include "logging.h"
#include <algorithm>
#include <ostream>

using namespace std;

namespace ark {

Dims broadcast(const Dims &dims1, const Dims &dims2)
{
    std::vector<DimType> output_dims_reversed;
    int ndims = std::max(dims1.ndims(), dims2.ndims());
    for (int i = 1; i < ndims + 1; ++i) {
        int d1 = (i - 1 < dims1.ndims()) ? dims1[-i] : 1;
        int d2 = (i - 1 < dims2.ndims()) ? dims2[-i] : 1;
        if (d1 == d2) {
            output_dims_reversed.push_back(d1);
        } else if (d1 == 1) {
            output_dims_reversed.push_back(d2);
        } else if (d2 == 1) {
            output_dims_reversed.push_back(d1);
        } else {
            LOGERR("input and other cannot be broadcasted: ", dims1, ", ",
                   dims2);
        }
    }
    std::reverse(output_dims_reversed.begin(), output_dims_reversed.end());
    return Dims{output_dims_reversed};
}

ostream &operator<<(ostream &os, const OpType &s)
{
    // clang-format off
    switch (s) {
    case OP_UNKNOWN:       os << "OP_UNKNOWN";       break;
    case OP_TENSOR:        os << "OP_TENSOR";        break;
    case OP_REFER:         os << "OP_REFER";         break;
    case OP_RESHAPE:       os << "OP_RESHAPE";       break;
    case OP_MERGE:         os << "OP_MERGE";         break;
    case OP_REDUCE_E_SUM:  os << "OP_REDUCE_E_SUM";  break;
    case OP_REDUCE_E_MEAN: os << "OP_REDUCE_E_MEAN"; break;
    case OP_REDUCE_E_MAX:  os << "OP_REDUCE_E_MAX";  break;
    case OP_REDUCE_W_SUM:  os << "OP_REDUCE_W_SUM";  break;
    case OP_REDUCE_W_MEAN: os << "OP_REDUCE_W_MEAN"; break;
    case OP_REDUCE_W_MAX:  os << "OP_REDUCE_W_MAX";  break;
    case OP_SCALE:         os << "OP_SCALE";         break;
    case OP_MATMUL:        os << "OP_MATMUL";        break;
    case OP_MAX_POOL:      os << "OP_MAX_POOL";      break;
    case OP_ADD:           os << "OP_ADD";           break;
    case OP_MUL:           os << "OP_MUL";           break;
    case OP_IM2COL:        os << "OP_IM2COL";        break;
    case OP_TRANSPOSE:     os << "OP_TRANSPOSE";     break;
    case OP_SEND:          os << "OP_SEND";          break;
    case OP_SEND_DONE:     os << "OP_SEND_DONE";     break;
    case OP_SEND_MM:       os << "OP_SEND_MM";       break;
    case OP_RECV:          os << "OP_RECV";          break;
    case OP_RECV_MM:       os << "OP_RECV_MM";       break;
    case OP_LAYERNORM:     os << "OP_LAYERNORM";     break;
    case OP_SOFTMAX:       os << "OP_SOFTMAX";       break;
    case OP_RELU:          os << "OP_RELU";          break;
    case OP_GELU:          os << "OP_GELU";          break;
    }
    // clang-format on
    return os;
}

OpArg::OpArg(int arg) : type{OP_ARG_INT}, val{new int{arg}}
{
    assert(this->val != nullptr);
}
OpArg::OpArg(DimType arg) : type{OP_ARG_INT64}, val{new DimType{arg}}
{
    assert(this->val != nullptr);
}
OpArg::OpArg(uint64_t arg) : type{OP_ARG_UINT64}, val{new uint64_t{arg}}
{
    assert(this->val != nullptr);
}
OpArg::OpArg(bool arg) : type{OP_ARG_BOOL}, val{new bool{arg}}
{
    assert(this->val != nullptr);
}
OpArg::OpArg(float arg) : type{OP_ARG_FLOAT}, val{new float{arg}}
{
    assert(this->val != nullptr);
}
OpArg::OpArg(const Dims &arg) : type{OP_ARG_DIMS}, val{new Dims{arg}}
{
    assert(this->val != nullptr);
}
OpArg::OpArg(Tensor *arg) : type{OP_ARG_TENSOR}, val{arg}
{
    assert(this->val != nullptr);
}
OpArg::OpArg(const OpArg &arg) : type{arg.type}
{
    if (this->type == OP_ARG_INT) {
        this->val = new int{*(int *)arg.val};
    } else if (this->type == OP_ARG_INT64) {
        this->val = new DimType{*(DimType *)arg.val};
    } else if (this->type == OP_ARG_UINT64) {
        this->val = new uint64_t{*(uint64_t *)arg.val};
    } else if (this->type == OP_ARG_BOOL) {
        this->val = new bool{*(bool *)arg.val};
    } else if (this->type == OP_ARG_FLOAT) {
        this->val = new float{*(float *)arg.val};
    } else if (this->type == OP_ARG_DIMS) {
        this->val = new Dims{*(Dims *)arg.val};
    } else if (this->type == OP_ARG_TENSOR) {
        this->val = arg.val;
    } else {
        LOGERR("invalid argument type ", this->type);
    }
}
OpArg::~OpArg()
{
    if (this->type == OP_ARG_INT) {
        delete static_cast<int *>(this->val);
    } else if (this->type == OP_ARG_INT64) {
        delete static_cast<DimType *>(this->val);
    } else if (this->type == OP_ARG_UINT64) {
        delete static_cast<uint64_t *>(this->val);
    } else if (this->type == OP_ARG_BOOL) {
        delete static_cast<bool *>(this->val);
    } else if (this->type == OP_ARG_FLOAT) {
        delete static_cast<float *>(this->val);
    } else if (this->type == OP_ARG_DIMS) {
        delete static_cast<Dims *>(this->val);
    } else if (this->type == OP_ARG_TENSOR) {
        // Do nothing
    } else {
        LOGERR("invalid argument type ", this->type);
    }
}
void OpArg::get(int *arg) const
{
    if (this->type != OP_ARG_INT) {
        LOGERR("invalid argument type ", this->type);
    }
    *arg = *static_cast<int *>(this->val);
}

void OpArg::get(long long int *arg) const
{
    if (this->type != OP_ARG_INT64) {
        LOGERR("invalid argument type ", this->type);
    }
    *arg = *static_cast<long long int *>(this->val);
}

void OpArg::get(uint64_t *arg) const
{
    if (this->type != OP_ARG_UINT64) {
        LOGERR("invalid argument type ", this->type);
    }
    *arg = *static_cast<uint64_t *>(this->val);
}

void OpArg::get(bool *arg) const
{
    if (this->type != OP_ARG_BOOL) {
        LOGERR("invalid argument type ", this->type);
    }
    *arg = *static_cast<bool *>(this->val);
}

void OpArg::get(float *arg) const
{
    if (this->type != OP_ARG_FLOAT) {
        LOGERR("invalid argument type ", this->type);
    }
    *arg = *static_cast<float *>(this->val);
}

void OpArg::get(Dims *arg) const
{
    if (this->type != OP_ARG_DIMS) {
        LOGERR("invalid argument type ", this->type);
    }
    *arg = *static_cast<Dims *>(this->val);
}

void OpArg::get(Tensor **arg) const
{
    if (this->type != OP_ARG_TENSOR) {
        LOGERR("invalid argument type ", this->type);
    }
    *arg = static_cast<Tensor *>(this->val);
}

bool operator<(const OpArg &oa1, const OpArg &oa2)
{
    if (oa1.type != oa2.type) {
        return oa1.type < oa2.type;
    }
    assert(oa1.val != nullptr);
    assert(oa2.val != nullptr);
    switch (oa1.type) {
    case OP_ARG_INT:
        return *(int *)oa1.val < *(int *)oa2.val;
    case OP_ARG_INT64:
        return *(DimType *)oa1.val < *(DimType *)oa2.val;
    case OP_ARG_UINT64:
        return *(uint64_t *)oa1.val < *(uint64_t *)oa2.val;
    case OP_ARG_BOOL:
        return *(bool *)oa1.val < *(bool *)oa2.val;
    case OP_ARG_FLOAT:
        return *(float *)oa1.val < *(float *)oa2.val;
    case OP_ARG_DIMS:
        return *(Dims *)oa1.val < *(Dims *)oa2.val;
    case OP_ARG_TENSOR:
        return (uintptr_t)oa1.val < (uintptr_t)oa2.val;
    }
    assert(false);
    return false;
}
bool operator==(const OpArg &oa1, const OpArg &oa2)
{
    if (oa1.type != oa2.type) {
        return false;
    }
    assert(oa1.val != nullptr);
    assert(oa2.val != nullptr);
    switch (oa1.type) {
    case OP_ARG_INT:
        return *(int *)oa1.val == *(int *)oa2.val;
    case OP_ARG_INT64:
        return *(DimType *)oa1.val == *(DimType *)oa2.val;
    case OP_ARG_UINT64:
        return *(uint64_t *)oa1.val == *(uint64_t *)oa2.val;
    case OP_ARG_BOOL:
        return *(bool *)oa1.val == *(bool *)oa2.val;
    case OP_ARG_FLOAT:
        return *(float *)oa1.val == *(float *)oa2.val;
    case OP_ARG_DIMS:
        return *(Dims *)oa1.val == *(Dims *)oa2.val;
    case OP_ARG_TENSOR:
        return oa1.val == oa2.val;
    }
    assert(false);
    return false;
}

OpArgs::OpArgs(const std::vector<OpArg> &args) : args{args}
{
}

OpArgs &OpArgs::operator=(const OpArgs &opargs)
{
    if (this != &opargs) {
        this->args = opargs.args;
    }
    return *this;
}

void OpArgs::put(const OpArg &arg)
{
    this->args.emplace_back(arg);
}

void OpArgs::get(int *arg, size_t idx) const
{
    if (this->args.size() <= idx) {
        LOGERR("invalid argument index ", idx, " size ", this->args.size());
    }
    if (this->args[idx].type != OP_ARG_INT) {
        LOGERR("invalid argument type ", this->args[idx].type);
    }
    *arg = *static_cast<int *>(this->args[idx].val);
}

void OpArgs::get(long long int *arg, size_t idx) const
{
    if (this->args.size() <= idx) {
        LOGERR("invalid argument index ", idx, " size ", this->args.size());
    }
    if (this->args[idx].type != OP_ARG_INT64) {
        LOGERR("invalid argument type ", this->args[idx].type);
    }
    *arg = *static_cast<long long int *>(this->args[idx].val);
}

void OpArgs::get(uint64_t *arg, size_t idx) const
{
    if (this->args.size() <= idx) {
        LOGERR("invalid argument index ", idx, " size ", this->args.size());
    }
    if (this->args[idx].type != OP_ARG_UINT64) {
        LOGERR("invalid argument type ", this->args[idx].type);
    }
    *arg = *static_cast<uint64_t *>(this->args[idx].val);
}

void OpArgs::get(bool *arg, size_t idx) const
{
    if (this->args.size() <= idx) {
        LOGERR("invalid argument index ", idx, " size ", this->args.size());
    }
    if (this->args[idx].type != OP_ARG_BOOL) {
        LOGERR("invalid argument type ", this->args[idx].type);
    }
    *arg = *static_cast<bool *>(this->args[idx].val);
}

void OpArgs::get(float *arg, size_t idx) const
{
    if (this->args.size() <= idx) {
        LOGERR("invalid argument index ", idx, " size ", this->args.size());
    }
    if (this->args[idx].type != OP_ARG_FLOAT) {
        LOGERR("invalid argument type ", this->args[idx].type);
    }
    *arg = *static_cast<float *>(this->args[idx].val);
}

void OpArgs::get(Dims *arg, size_t idx) const
{
    if (this->args.size() <= idx) {
        LOGERR("invalid argument index ", idx, " size ", this->args.size());
    }
    if (this->args[idx].type != OP_ARG_DIMS) {
        LOGERR("invalid argument type ", this->args[idx].type);
    }
    *arg = *static_cast<Dims *>(this->args[idx].val);
}

void OpArgs::get(Tensor **arg, size_t idx) const
{
    if (this->args.size() <= idx) {
        LOGERR("invalid argument index ", idx, " size ", this->args.size());
    }
    if (this->args[idx].type != OP_ARG_TENSOR) {
        LOGERR("invalid argument type ", this->args[idx].type);
    }
    *arg = static_cast<Tensor *>(this->args[idx].val);
}

const std::vector<OpArg> &OpArgs::get_args() const
{
    return this->args;
}

bool operator<(const OpArgs &opargs1, const OpArgs &opargs2)
{
    for (size_t i = 0; i < opargs1.args.size(); ++i) {
        if (opargs1.args[i] == opargs2.args[i]) {
            continue;
        }
        return opargs1.args[i] < opargs2.args[i];
    }
    return false;
}

bool operator==(const OpArgs &opargs1, const OpArgs &opargs2)
{
    for (size_t i = 0; i < opargs1.args.size(); ++i) {
        if (opargs1.args[i] == opargs2.args[i]) {
            continue;
        }
        return false;
    }
    return true;
}

bool operator!=(const OpArgs &opargs1, const OpArgs &opargs2)
{
    return !(opargs1 == opargs2);
}

Op::Op(const OpType &type_, const OpPrecType &prec_type_,
       const vector<Tensor *> &in_deps_, const vector<Tensor *> &out_deps_,
       const OpArgs &args_, const string &name_, int gran_lev_,
       bool force_inline_)
    : type{type_}, prec_type{prec_type_}, in_deps{in_deps_},
      out_deps{out_deps_}, args{args_}, name{name_}, gran_lev{gran_lev_},
      force_inline{force_inline_}
{
    for (auto &tns : in_deps_) {
        assert(tns != nullptr);
    }
    for (auto &tns : out_deps_) {
        assert(tns != nullptr);
    }
}

std::string Op::function_name(const OpConfig &cfg) const
{
    switch (this->type) {
    case OP_REDUCE_E_SUM:
        return static_cast<const ReduceESumOp *>(this)->function_name(cfg);
    case OP_REDUCE_E_MEAN:
        return static_cast<const ReduceEMeanOp *>(this)->function_name(cfg);
    case OP_REDUCE_E_MAX:
        return static_cast<const ReduceEMaxOp *>(this)->function_name(cfg);
    case OP_REDUCE_W_SUM:
        return static_cast<const ReduceWSumOp *>(this)->function_name(cfg);
    case OP_REDUCE_W_MEAN:
        return static_cast<const ReduceWMeanOp *>(this)->function_name(cfg);
    case OP_REDUCE_W_MAX:
        return static_cast<const ReduceWMaxOp *>(this)->function_name(cfg);
    case OP_SCALE:
        return static_cast<const ScaleOp *>(this)->function_name(cfg);
    case OP_MATMUL:
        return static_cast<const MatmulOp *>(this)->function_name(cfg);
    case OP_MAX_POOL:
        return static_cast<const MaxPoolOp *>(this)->function_name(cfg);
    case OP_ADD:
        return static_cast<const AddOp *>(this)->function_name(cfg);
    case OP_MUL:
        return static_cast<const MulOp *>(this)->function_name(cfg);
    case OP_IM2COL:
        return static_cast<const Im2colOp *>(this)->function_name(cfg);
    case OP_TRANSPOSE:
        return static_cast<const TransposeOp *>(this)->function_name(cfg);
    case OP_SEND:
        return static_cast<const SendOp *>(this)->function_name(cfg);
    case OP_SEND_DONE:
        return static_cast<const SendDoneOp *>(this)->function_name(cfg);
    case OP_SEND_MM:
        return static_cast<const SendMMOp *>(this)->function_name(cfg);
    case OP_RECV:
        return static_cast<const RecvOp *>(this)->function_name(cfg);
    case OP_RECV_MM:
        return static_cast<const RecvMMOp *>(this)->function_name(cfg);
    case OP_LAYERNORM:
        return static_cast<const LayernormOp *>(this)->function_name(cfg);
    case OP_SOFTMAX:
        return static_cast<const SoftmaxOp *>(this)->function_name(cfg);
    case OP_RELU:
        return static_cast<const ReluOp *>(this)->function_name(cfg);
    case OP_GELU:
        return static_cast<const GeluOp *>(this)->function_name(cfg);
    default:
        return "";
    }
    // Never reach here.
    return "";
}

OpArgs Op::function_call_args(const OpConfig &cfg) const
{
    switch (this->type) {
    case OP_SCALE:
        return static_cast<const ScaleOp *>(this)->function_call_args(cfg);
    case OP_SEND:
        return static_cast<const SendOp *>(this)->function_call_args(cfg);
    case OP_SEND_DONE:
        return static_cast<const SendDoneOp *>(this)->function_call_args(cfg);
    // case OP_SEND_MM:
    //     return static_cast<const SendMMOp *>(this)->function_call_args(cfg);
    case OP_RECV:
        return static_cast<const RecvOp *>(this)->function_call_args(cfg);
    // case OP_RECV_MM:
    //     return static_cast<const RecvMMOp *>(this)->function_call_args(cfg);
    default:
        OpArgs opargs;
        std::vector<Tensor *> deps = this->out_deps;
        deps.insert(deps.end(), this->in_deps.begin(), this->in_deps.end());
        for (Tensor *tns : deps) {
            opargs.put(tns);
        }
        return opargs;
    }
    // Never reach here.
    return {};
}

std::string Op::function_name(const std::string &kernel_name,
                              const OpArgs &template_args)
{
    std::stringstream ss;
    ss << kernel_name;
    size_t num_args = template_args.args.size();
    if (num_args == 0) {
        return ss.str();
    }
    ss << "<";
    for (size_t i = 0; i < num_args; ++i) {
        auto &arg = template_args.args[i];
        if (arg.type == OP_ARG_INT) {
            int val;
            template_args.get(&val, i);
            ss << val;
        } else if (arg.type == OP_ARG_INT64) {
            long long int val;
            template_args.get(&val, i);
            ss << val;
        } else if (arg.type == OP_ARG_UINT64) {
            uint64_t val;
            template_args.get(&val, i);
            ss << val;
        } else if (arg.type == OP_ARG_BOOL) {
            bool val;
            template_args.get(&val, i);
            ss << val;
        } else if (arg.type == OP_ARG_FLOAT) {
            LOGERR("float template args are not supported");
        } else if (arg.type == OP_ARG_DIMS) {
            Dims val;
            template_args.get(&val, i);
            ss << "ark::Vec" << val;
        }
        if (i < num_args - 1) {
            ss << ", ";
        }
    }
    ss << ">";
    return ss.str();
}

bool operator<(const Op &op1, const Op &op2)
{
    if (op1.type < op2.type) {
        return true;
    }
    if (op1.prec_type < op2.prec_type) {
        return true;
    }
    if (op1.args < op2.args) {
        return true;
    }
    return false;
}

bool operator==(const Op &op1, const Op &op2)
{
    if (op1.type != op2.type) {
        return false;
    }
    if (op1.prec_type != op2.prec_type) {
        return false;
    }
    if (op1.args != op2.args) {
        return false;
    }
    return true;
}
// void to_json(nlohmann::json &j, const Op &op)
// {
//     j = nlohmann::json{
//         {"type", op.type},          {"prec_type", op.prec_type},
//         {"in_deps", vector<int>{}}, {"out_deps", vector<int>{}},
//         {"args", op.args},          {"name", op.name},
//     };
//     for (Tensor *pt : op.in_deps) {
//         j.at("in_deps").emplace_back(pt->id);
//     }
//     for (Tensor *pt : op.out_deps) {
//         j.at("out_deps").emplace_back(pt->id);
//     }
// }
// void from_json(const nlohmann::json &j, Op &op)
// {
// }

} // namespace ark
