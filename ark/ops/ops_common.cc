// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_common.h"

#include <algorithm>
#include <cassert>
#include <ostream>

#include "include/ark.h"
#include "logging.h"

using namespace std;

namespace ark {

Dims broadcast(const Dims &dims1, const Dims &dims2) {
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
            ERR(InvalidUsageError,
                "input and other cannot be broadcasted: ", dims1, ", ", dims2);
        }
    }
    std::reverse(output_dims_reversed.begin(), output_dims_reversed.end());
    return Dims{output_dims_reversed};
}

OpArchType op_arch_from_string(const std::string &arch) {
    if (arch == "cuda_60") {
        return OP_ARCH_CUDA_60;
    } else if (arch == "cuda_70") {
        return OP_ARCH_CUDA_70;
    } else if (arch == "cuda_80") {
        return OP_ARCH_CUDA_80;
    } else if (arch == "cuda_90") {
        return OP_ARCH_CUDA_90;
    } else if (arch == "rocm_90a") {
        return OP_ARCH_ROCM_90A;
    }
    return OP_ARCH_UNKNOWN;
}

bool operator<(const OpConfigKey &ops1, const OpConfigKey &ops2) {
    if (ops1.arch_type != ops2.arch_type) {
        return ops1.arch_type < ops2.arch_type;
    } else {
        return ops1.prec_type < ops2.prec_type;
    }
}

bool operator==(const OpConfigKey &ops1, const OpConfigKey &ops2) {
    return ops1.arch_type == ops2.arch_type && ops1.prec_type == ops2.prec_type;
}

OpConfigMap::OpConfigMap(
    std::initializer_list<
        std::pair<const OpConfigKey, const std::vector<OpConfig>>>
        ilist)
    : cfg_map{ilist} {}

// A dummy OpConfig vector to return when no config is found
static const std::vector<OpConfig> NoneConfigs;

const std::vector<OpConfig> &OpConfigMap::get(const OpConfigKey &key) const {
    auto search = this->cfg_map.find(key);
    if (search != this->cfg_map.end()) {
        return search->second;
    }
    search = this->cfg_map.find({key.arch_type, "any"});
    if (search != this->cfg_map.end()) {
        return search->second;
    }
#if defined(ARK_CUDA)
    search = this->cfg_map.find({OP_ARCH_CUDA_ANY, key.prec_type});
    if (search != this->cfg_map.end()) {
        return search->second;
    }
    search = this->cfg_map.find({OP_ARCH_CUDA_ANY, "any"});
    if (search != this->cfg_map.end()) {
        return search->second;
    }
#elif defined(ARK_ROCM)
    search = this->cfg_map.find({OP_ARCH_ROCM_ANY, key.prec_type});
    if (search != this->cfg_map.end()) {
        return search->second;
    }
    search = this->cfg_map.find({OP_ARCH_ROCM_ANY, "any"});
    if (search != this->cfg_map.end()) {
        return search->second;
    }
#endif
    search = this->cfg_map.find({OP_ARCH_ANY, key.prec_type});
    if (search != this->cfg_map.end()) {
        return search->second;
    }
    search = this->cfg_map.find({OP_ARCH_ANY, "any"});
    if (search != this->cfg_map.end()) {
        return search->second;
    }
    return NoneConfigs;
}

OpArg::OpArg(int arg) : type{OP_ARG_INT}, val{new int{arg}} {
    assert(this->val != nullptr);
}
OpArg::OpArg(DimType arg) : type{OP_ARG_INT64}, val{new DimType{arg}} {
    assert(this->val != nullptr);
}
OpArg::OpArg(uint64_t arg) : type{OP_ARG_UINT64}, val{new uint64_t{arg}} {
    assert(this->val != nullptr);
}
OpArg::OpArg(bool arg) : type{OP_ARG_BOOL}, val{new bool{arg}} {
    assert(this->val != nullptr);
}
OpArg::OpArg(float arg) : type{OP_ARG_FLOAT}, val{new float{arg}} {
    assert(this->val != nullptr);
}
OpArg::OpArg(const Dims &arg) : type{OP_ARG_DIMS}, val{new Dims{arg}} {
    assert(this->val != nullptr);
}
OpArg::OpArg(Tensor *arg) : type{OP_ARG_TENSOR}, val{arg} {
    assert(this->val != nullptr);
}
OpArg::OpArg(const OpArg &arg) : type{arg.type} {
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
        ERR(InvalidUsageError, "invalid argument type ", this->type);
    }
}
OpArg::~OpArg() {
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
    }
}
void OpArg::get(int *arg) const {
    if (this->type != OP_ARG_INT) {
        ERR(InvalidUsageError, "invalid argument type ", this->type);
    }
    *arg = *static_cast<int *>(this->val);
}

void OpArg::get(long long int *arg) const {
    if (this->type != OP_ARG_INT64) {
        ERR(InvalidUsageError, "invalid argument type ", this->type);
    }
    *arg = *static_cast<long long int *>(this->val);
}

void OpArg::get(uint64_t *arg) const {
    if (this->type != OP_ARG_UINT64) {
        ERR(InvalidUsageError, "invalid argument type ", this->type);
    }
    *arg = *static_cast<uint64_t *>(this->val);
}

void OpArg::get(bool *arg) const {
    if (this->type != OP_ARG_BOOL) {
        ERR(InvalidUsageError, "invalid argument type ", this->type);
    }
    *arg = *static_cast<bool *>(this->val);
}

void OpArg::get(float *arg) const {
    if (this->type != OP_ARG_FLOAT) {
        ERR(InvalidUsageError, "invalid argument type ", this->type);
    }
    *arg = *static_cast<float *>(this->val);
}

void OpArg::get(Dims *arg) const {
    if (this->type != OP_ARG_DIMS) {
        ERR(InvalidUsageError, "invalid argument type ", this->type);
    }
    *arg = *static_cast<Dims *>(this->val);
}

void OpArg::get(Tensor **arg) const {
    if (this->type != OP_ARG_TENSOR) {
        ERR(InvalidUsageError, "invalid argument type ", this->type);
    }
    *arg = static_cast<Tensor *>(this->val);
}

bool operator<(const OpArg &oa1, const OpArg &oa2) {
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
bool operator==(const OpArg &oa1, const OpArg &oa2) {
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

OpArgs::OpArgs(const std::vector<OpArg> &args) : args{args} {}

OpArgs &OpArgs::operator=(const OpArgs &opargs) {
    if (this != &opargs) {
        this->args = opargs.args;
    }
    return *this;
}

void OpArgs::put(const OpArg &arg) { this->args.emplace_back(arg); }

void OpArgs::get(int *arg, size_t idx) const {
    if (this->args.size() <= idx) {
        ERR(InvalidUsageError, "invalid argument index ", idx, " size ",
            this->args.size());
    }
    if (this->args[idx].type != OP_ARG_INT) {
        ERR(InvalidUsageError, "invalid argument type ", this->args[idx].type);
    }
    *arg = *static_cast<int *>(this->args[idx].val);
}

void OpArgs::get(long long int *arg, size_t idx) const {
    if (this->args.size() <= idx) {
        ERR(InvalidUsageError, "invalid argument index ", idx, " size ",
            this->args.size());
    }
    if (this->args[idx].type != OP_ARG_INT64) {
        ERR(InvalidUsageError, "invalid argument type ", this->args[idx].type);
    }
    *arg = *static_cast<long long int *>(this->args[idx].val);
}

void OpArgs::get(uint64_t *arg, size_t idx) const {
    if (this->args.size() <= idx) {
        ERR(InvalidUsageError, "invalid argument index ", idx, " size ",
            this->args.size());
    }
    if (this->args[idx].type != OP_ARG_UINT64) {
        ERR(InvalidUsageError, "invalid argument type ", this->args[idx].type);
    }
    *arg = *static_cast<uint64_t *>(this->args[idx].val);
}

void OpArgs::get(bool *arg, size_t idx) const {
    if (this->args.size() <= idx) {
        ERR(InvalidUsageError, "invalid argument index ", idx, " size ",
            this->args.size());
    }
    if (this->args[idx].type != OP_ARG_BOOL) {
        ERR(InvalidUsageError, "invalid argument type ", this->args[idx].type);
    }
    *arg = *static_cast<bool *>(this->args[idx].val);
}

void OpArgs::get(float *arg, size_t idx) const {
    if (this->args.size() <= idx) {
        ERR(InvalidUsageError, "invalid argument index ", idx, " size ",
            this->args.size());
    }
    if (this->args[idx].type != OP_ARG_FLOAT) {
        ERR(InvalidUsageError, "invalid argument type ", this->args[idx].type);
    }
    *arg = *static_cast<float *>(this->args[idx].val);
}

void OpArgs::get(Dims *arg, size_t idx) const {
    if (this->args.size() <= idx) {
        ERR(InvalidUsageError, "invalid argument index ", idx, " size ",
            this->args.size());
    }
    if (this->args[idx].type != OP_ARG_DIMS) {
        ERR(InvalidUsageError, "invalid argument type ", this->args[idx].type);
    }
    *arg = *static_cast<Dims *>(this->args[idx].val);
}

void OpArgs::get(Tensor **arg, size_t idx) const {
    if (this->args.size() <= idx) {
        ERR(InvalidUsageError, "invalid argument index ", idx, " size ",
            this->args.size());
    }
    if (this->args[idx].type != OP_ARG_TENSOR) {
        ERR(InvalidUsageError, "invalid argument type ", this->args[idx].type);
    }
    *arg = static_cast<Tensor *>(this->args[idx].val);
}

const std::vector<OpArg> &OpArgs::get_args() const { return this->args; }

bool operator<(const OpArgs &opargs1, const OpArgs &opargs2) {
    for (size_t i = 0; i < opargs1.args.size(); ++i) {
        if (opargs1.args[i] == opargs2.args[i]) {
            continue;
        }
        return opargs1.args[i] < opargs2.args[i];
    }
    return false;
}

bool operator==(const OpArgs &opargs1, const OpArgs &opargs2) {
    for (size_t i = 0; i < opargs1.args.size(); ++i) {
        if (opargs1.args[i] == opargs2.args[i]) {
            continue;
        }
        return false;
    }
    return true;
}

bool operator!=(const OpArgs &opargs1, const OpArgs &opargs2) {
    return !(opargs1 == opargs2);
}

Op::Op(const OpType &type_, const std::string &prec_type_,
       const vector<Tensor *> &inputs_, const vector<Tensor *> &output_refs_,
       const OpArgs &args_, const string &name_, const OpConfigMap *cfg_map_,
       int gran_lev_, bool force_inline_)
    : type{type_},
      prec_type{prec_type_},
      inputs{inputs_},
      output_refs{output_refs_},
      args{args_},
      name{name_},
      cfg_map{cfg_map_},
      gran_lev{gran_lev_},
      force_inline{force_inline_} {
    for (auto &tns : inputs_) {
        if (tns == nullptr) {
            ERR(ModelError, "input tensor is null");
        }
    }
    for (auto &tns : output_refs_) {
        if (tns == nullptr) {
            ERR(ModelError, "output reference tensor is null");
        }
    }
}

std::string Op::function_name(const OpConfig &cfg) const {
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
        case OP_SUB:
            return static_cast<const SubOp *>(this)->function_name(cfg);
        case OP_MUL:
            return static_cast<const MulOp *>(this)->function_name(cfg);
        case OP_DIV:
            return static_cast<const DivOp *>(this)->function_name(cfg);
        case OP_IM2COL:
            return static_cast<const Im2colOp *>(this)->function_name(cfg);
        case OP_TRANSPOSE:
            return static_cast<const TransposeOp *>(this)->function_name(cfg);
        case OP_SEND:
            return static_cast<const SendOp *>(this)->function_name(cfg);
        case OP_SEND_DONE:
            return static_cast<const SendDoneOp *>(this)->function_name(cfg);
        case OP_RECV:
            return static_cast<const RecvOp *>(this)->function_name(cfg);
        case OP_LAYERNORM:
            return static_cast<const LayernormOp *>(this)->function_name(cfg);
        case OP_RMSNORM:
            return static_cast<const RMSnormOp *>(this)->function_name(cfg);
        case OP_SOFTMAX:
            return static_cast<const SoftmaxOp *>(this)->function_name(cfg);
        case OP_RELU:
            return static_cast<const ReluOp *>(this)->function_name(cfg);
        case OP_COPY:
            return static_cast<const CopyOp *>(this)->function_name(cfg);
        case OP_SIGMOID:
            return static_cast<const SigmoidOp *>(this)->function_name(cfg);
        case OP_GELU:
            return static_cast<const GeluOp *>(this)->function_name(cfg);
        case OP_EXP:
            return static_cast<const ExpOp *>(this)->function_name(cfg);
        case OP_SQRT:
            return static_cast<const SqrtOp *>(this)->function_name(cfg);
        case OP_ROPE:
            return static_cast<const RopeOp *>(this)->function_name(cfg);
        case OP_EMBEDDING:
            return static_cast<const EmbeddingOp *>(this)->function_name(cfg);
        case OP_CAST:
            return static_cast<const CastOp *>(this)->function_name(cfg);
        case OP_DEVICE_SYNC:
            return static_cast<const DeviceSyncOp *>(this)->function_name(cfg);
        case OP_READ_AND_REDUCE:
            return static_cast<const ReadAndReduceOp *>(this)->function_name(
                cfg);
        case OP_GATHER_FROM_PEERS:
            return static_cast<const GatherFromPeersOp *>(this)->function_name(
                cfg);
        case OP_PUT_PACKET:
            return static_cast<const PutPacketOp *>(this)->function_name(cfg);
        case OP_REDUCE_AND_WRITE_PACKET:
            return static_cast<const ReduceAndWritePacketOp *>(this)
                ->function_name(cfg);
        case OP_GET_FROM_PACKET:
            return static_cast<const GetFromPacketOp *>(this)->function_name(
                cfg);
        default:
            ERR(ModelError, "invalid op type ", this->type);
            return "";
    }
    // Never reach here.
    return "";
}

OpArgs Op::function_call_args(const OpConfig &cfg) const {
    switch (this->type) {
        case OP_SCALE:
            return static_cast<const ScaleOp *>(this)->function_call_args(cfg);
        case OP_SEND:
            return static_cast<const SendOp *>(this)->function_call_args(cfg);
        case OP_SEND_DONE:
            return static_cast<const SendDoneOp *>(this)->function_call_args(
                cfg);
        case OP_RECV:
            return static_cast<const RecvOp *>(this)->function_call_args(cfg);
        case OP_DEVICE_SYNC:
            return static_cast<const DeviceSyncOp *>(this)->function_call_args(
                cfg);
        case OP_READ_AND_REDUCE:
            return static_cast<const ReadAndReduceOp *>(this)
                ->function_call_args(cfg);
        case OP_GATHER_FROM_PEERS:
            return static_cast<const GatherFromPeersOp *>(this)
                ->function_call_args(cfg);
        case OP_PUT_PACKET:
            return static_cast<const PutPacketOp *>(this)->function_call_args(
                cfg);
        case OP_REDUCE_AND_WRITE_PACKET:
            return static_cast<const ReduceAndWritePacketOp *>(this)
                ->function_call_args(cfg);
        case OP_GET_FROM_PACKET:
            return static_cast<const GetFromPacketOp *>(this)
                ->function_call_args(cfg);
        default:
            OpArgs opargs;
            std::vector<Tensor *> deps = this->outputs;
            deps.insert(deps.end(), this->inputs.begin(), this->inputs.end());
            for (Tensor *tns : deps) {
                opargs.put(tns);
            }
            return opargs;
    }
    // Never reach here.
    return {};
}

std::string Op::function_name(const std::string &kernel_name,
                              const OpArgs &template_args) {
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
            ss << (val ? "true" : "false");
        } else if (arg.type == OP_ARG_FLOAT) {
            ERR(ModelError, "float template args are not supported");
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

bool Op::is_virtual() const { return this->cfg_map == nullptr; }

bool Op::is_comm() const {
    return this->type == OP_SEND || this->type == OP_SEND_DONE ||
           this->type == OP_RECV || this->type == OP_DEVICE_SYNC;
}

bool operator<(const Op &op1, const Op &op2) {
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

bool operator==(const Op &op1, const Op &op2) {
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

}  // namespace ark
