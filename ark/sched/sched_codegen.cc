// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>
#include <fstream>
#include <initializer_list>
#include <ostream>
#include <unistd.h>

#include "env.h"
#include "logging.h"
#include "math.h"
#include "sched/sched_codegen.h"

using namespace std;

#define COM ", "
#define OP_PREFIX "op"
#define UNIT_OP_PREFIX "uop"

namespace ark {

CodeGenerator::CodeGenerator(const std::map<TensorBuf *, GpuBuf *> &buf_trans,
                             const GpuInfo &gpu_info_, int num_warps_per_sm_)
    : buf_trans{buf_trans}, gpu_info{gpu_info_}, sm_num{gpu_info_.num_sm},
      num_warps_per_sm{num_warps_per_sm_}, num_indent{0}
{
}

size_t CodeGenerator::get_tensor_offset(const Tensor *tensor) const
{
    size_t off = this->buf_trans.find(tensor->buf)->second->get_offset();
    assert(off % 8 == 0);
    return off + tensor->offset_bytes();
}

std::ostream &CodeGenerator::def_remote_buf(std::ostream &os,
                                            int remote_rank) const
{
    os << "__device__ char *" ARK_BUF_NAME << remote_rank << ";\n";
    return os;
}

std::ostream &CodeGenerator::sync_gpu(std::ostream &os) const
{
    os << "ark::sync_gpu<" << this->sm_num << ">(" ARK_LSS_NAME ");\n";
    return os;
}

std::ostream &CodeGenerator::def_sync_stream(std::ostream &os,
                                             int stream_id) const
{
    os << "__device__ ark::sync::State " ARK_LSS_NAME "_" << stream_id << ";\n";
    return os;
}

std::ostream &CodeGenerator::sync_stream(std::ostream &os, int stream_id,
                                         int sm_id_begin, int sm_id_end) const
{
    if (sm_id_begin >= sm_id_end) {
        LOG(ERROR, "invalid SM range");
    }
    if (sm_id_begin == 0) {
        os << "if (blockIdx.x < " << sm_id_end << ") {";
    } else if (sm_id_begin + 1 == sm_id_end) {
        os << "if (blockIdx.x == " << sm_id_begin << ") {";
    } else {
        os << "if (blockIdx.x >= " << sm_id_begin << " && blockIdx.x < "
           << sm_id_end << ") {";
    }
    os << " ark::sync_gpu<" << sm_id_end - sm_id_begin << ">(" ARK_LSS_NAME "_"
       << stream_id << "); }\n";
    return os;
}

ostream &CodeGenerator::tensor(ostream &os, const Tensor *tensor) const
{
    size_t off = this->get_tensor_offset(tensor);
    if (tensor->type == FP16) {
        os << "(ark::half *)";
    } else if (tensor->type == FP32) {
        os << "(float *)";
    } else if (tensor->type == INT32) {
        os << "(int *)";
    } else if (tensor->type == BYTE) {
        os << "(void *)";
    } else {
        LOGERR("unknown tensor type");
    }
    std::string buf_name = ARK_BUF_NAME;
    if (tensor->imported_rank >= 0) {
        buf_name += std::to_string(tensor->imported_rank);
    }
    os << "&" << buf_name << "[" << off << "]";
    return os;
}

std::ostream &CodeGenerator::def_oparg(std::ostream &os, const OpArg &arg,
                                       const std::string &name) const
{
    if (arg.type == OP_ARG_TENSOR) {
        Tensor *tns;
        arg.get(&tns);
        switch (tns->type) {
        case FP16:
            os << "ark::half *" << name;
            break;
        case FP32:
            os << "float *" << name;
            break;
        case INT32:
            os << "int *" << name;
            break;
        case BYTE:
            os << "void *" << name;
            break;
        default:
            LOGERR("Not implemented");
            break;
        }
    } else if (arg.type == OP_ARG_FLOAT) {
        os << "float " << name;
    } else if (arg.type == OP_ARG_INT) {
        os << "int " << name;
    } else if (arg.type == OP_ARG_BOOL) {
        os << "bool " << name;
    } else if (arg.type == OP_ARG_INT64) {
        os << "long long int " << name;
    } else if (arg.type == OP_ARG_UINT64) {
        os << "uint64_t " << name;
    } else {
        LOGERR("Not implemented");
    }
    return os;
}

std::ostream &CodeGenerator::oparg(std::ostream &os, const OpArg &arg) const
{
    if (arg.type == OP_ARG_TENSOR) {
        Tensor *tns;
        arg.get(&tns);
        this->tensor(os, tns);
    } else if (arg.type == OP_ARG_FLOAT) {
        float val;
        arg.get(&val);
        os << val;
    } else if (arg.type == OP_ARG_INT) {
        int val;
        arg.get(&val);
        os << val;
    } else if (arg.type == OP_ARG_BOOL) {
        bool val;
        arg.get(&val);
        os << val;
    } else if (arg.type == OP_ARG_INT64) {
        long long int val;
        arg.get(&val);
        os << val;
    } else if (arg.type == OP_ARG_UINT64) {
        uint64_t val;
        arg.get(&val);
        os << val;
    } else {
        LOGERR("Not implemented");
    }
    return os;
}

std::ostream &CodeGenerator::branch(std::ostream &os, const Branch &br,
                                    int prev_sm_id_end) const
{
    if (br.warp_branches.empty()) {
        return os;
    }
    if (prev_sm_id_end < 0) {
        prev_sm_id_end = this->sm_num;
    }
    if (br.sm_id_begin == 0) {
        if (br.sm_id_end == this->sm_num) {
            os << "\n  { // for all SMs";
        } else {
            os << "\n  if (blockIdx.x < " << br.sm_id_end << ") {";
        }
    } else if (br.sm_id_begin == prev_sm_id_end) {
        if (br.sm_id_end == this->sm_num) {
            os << " else {";
        } else {
            os << " else if (blockIdx.x < " << br.sm_id_end << ") {";
        }
    } else if (br.sm_id_begin < prev_sm_id_end) {
        if (br.sm_id_begin == br.sm_id_end) {
            os << "\n  if (blockIdx.x == " << br.sm_id_begin << ") {";
        } else {
            os << "\n  if (blockIdx.x >= " << br.sm_id_begin
               << " && blockIdx.x < " << br.sm_id_end << ") {";
        }
    } else {
        if (br.sm_id_begin == br.sm_id_end) {
            os << " else if (blockIdx.x == " << br.sm_id_begin << ") {";
        } else {
            os << " else if (blockIdx.x >= " << br.sm_id_begin
               << " && blockIdx.x < " << br.sm_id_end << ") {";
        }
    }

    int tpw = this->gpu_info.threads_per_warp;

    for (auto &warp_branch : br.warp_branches) {
        int thread_begin = warp_branch.warp_id_begin * tpw;
        int thread_end = warp_branch.warp_id_end * tpw;
        if (warp_branch.warp_id_begin == 0) {
            if (warp_branch.warp_id_end == this->num_warps_per_sm) {
                os << "\n    { // for all threads\n";
            } else {
                os << "\n    if (threadIdx.x < " << thread_end << ") {\n";
            }
        } else {
            os << "\n    if (threadIdx.x >= " << thread_begin
               << " && threadIdx.x < " << thread_end << ") {\n";
        }

        for (auto &branch_op : warp_branch.branch_ops) {
            os << "      " << OP_PREFIX << branch_op.opseq_id << '(';
            // num_uops = (warp_id_end - warp_id_begin) / num_warps_per_uop;
            // warp_idx = warp_id - warp_id_begin;
            // sm_idx = sm_id - sm_id_begin;
            // uop = uop_id_diff * (warp_idx / num_warps_per_uop +
            //                      num_uops * sm_idx) + uop_id_begin;
            int num_warps = warp_branch.warp_id_end - warp_branch.warp_id_begin;
            int num_uops = num_warps / branch_op.num_warps_per_uop;
            int num_threads_per_uop = branch_op.num_warps_per_uop * tpw;
            if (branch_op.uop_id_diff != 0) {
                std::stringstream thread_indexing;
                std::stringstream sm_indexing;
                if (thread_end - thread_begin > num_threads_per_uop) {
                    if (thread_begin > 0) {
                        thread_indexing << "((threadIdx.x - " << thread_begin
                                        << ")";
                    } else {
                        thread_indexing << "(threadIdx.x";
                    }
                    if (math::is_pow2(num_threads_per_uop)) {
                        thread_indexing << " >> "
                                        << math::ilog2(num_threads_per_uop)
                                        << ")";
                    } else {
                        thread_indexing << " / " << num_threads_per_uop << ")";
                    }
                }
                if (br.sm_id_end - br.sm_id_begin > 1) {
                    if (br.sm_id_begin > 0) {
                        sm_indexing << "((blockIdx.x - " << br.sm_id_begin
                                    << ")";
                    } else {
                        sm_indexing << "(blockIdx.x";
                    }
                    if (num_uops > 1) {
                        sm_indexing << " * " << num_uops;
                    }
                    sm_indexing << ")";
                }
                std::string indexing;
                if (thread_indexing.str().empty()) {
                    indexing = sm_indexing.str();
                } else if (sm_indexing.str().empty()) {
                    indexing = thread_indexing.str();
                } else {
                    indexing = "(" + sm_indexing.str() + " + " +
                               thread_indexing.str() + ")";
                }
                if (!indexing.empty()) {
                    if (branch_op.uop_id_diff != 1) {
                        os << branch_op.uop_id_diff << " * ";
                    }
                    os << indexing << " + ";
                }
            }
            os << branch_op.uop_id_begin << ");\n";
        }
        os << "    }\n";
    }
    os << "  }\n";
    return os;
}

ostream &CodeGenerator::def_uop(ostream &os, const SchedOp &sop,
                                int uop_id) const
{
    std::string uop_name = UNIT_OP_PREFIX + std::to_string(uop_id);
    std::string func_name = sop.function_name();
    assert(!func_name.empty());

    const Op *op = sop.get_op();
    if (op->force_inline) {
        os << "DEVICE ";
    } else {
        os << "__noinline__ __device__ ";
    }
    os << "void " << uop_name << "(";

    OpArgs call_args = op->function_call_args(*sop.get_cfg());
    int cnt_param = 0;
    for (const OpArg &arg : call_args.get_args()) {
        this->def_oparg(os, arg, "_" + std::to_string(cnt_param)) << ", ";
        ++cnt_param;
    }

    os << "int _uop_idx) {\n";
    os << "  " << func_name << "(";

    for (int i = 0; i < cnt_param; ++i) {
        os << '_' << i << ", ";
    }
    os << "_uop_idx);\n}\n";
    return os;
}

std::ostream &CodeGenerator::uop(std::ostream &os, int uop_id) const
{
    os << UNIT_OP_PREFIX << uop_id;
    return os;
}

//
ostream &CodeGenerator::opseq(ostream &os, const string &name,
                              const SchedOpSeq &opseq,
                              map<string, int> &uop_map) const
{
    auto &sched_ops = opseq.get_sched_ops();
    unsigned int idx = sched_ops.size();
    auto it = sched_ops.rbegin();
    for (; it != sched_ops.rend(); ++it) {
        auto &sop = *it;
        if (sop.is_virtual()) {
            continue;
        }
        if (idx == sched_ops.size()) {
            os << "// tile dims: (" << opseq.get_tdims()[0] << COM
               << opseq.get_tdims()[1] << COM << opseq.get_tdims()[2] << ")\n"
               << "__noinline__ __device__ void " << name
               << "(int _uop_idx) {\n";
        }
        --idx;
        os << "  ";
        auto uop_map_it = uop_map.find(sop.function_name());
        if (uop_map_it != uop_map.end()) {
            this->uop(os, uop_map_it->second);
        } else {
            os << sop.function_name();
        }
        os << '(';

        OpArgs call_args = sop.get_op()->function_call_args(*sop.get_cfg());
        for (const OpArg &arg : call_args.get_args()) {
            this->oparg(os, arg) << ", ";
        }

        os << "_uop_idx);\n";
    }
    if (idx != sched_ops.size()) {
        os << "}\n";
    }
    return os;
}

std::ostream &CodeGenerator::sched(std::ostream &os, Sched &sched) const
{

    os << "if(";
    if (sched.sm_b > 0)
        os << "blockIdx.x >= " << sched.sm_b << "&& ";
    os << "blockIdx.x < " << sched.sm_e << "){\n";
    os << "  if(";
    if (sched.th_b > 0)
        os << "threadIdx.x >= " << sched.th_b << " && ";
    os << "threadIdx.x<" << sched.th_e << "){\n";
    SchedOpSeq *opseq = sched.opseq;
    os << "    opseq_" << opseq->get_id() << "_tile_task"
       << "(" << sched.alpha << "*(blockIdx.x - " << sched.sm_b << ") + "
       << "threadIdx.x / " << opseq->get_num_warps() * 32 << " + " << sched.beta
       << ");\n";
    os << "    "
       << "ark::sync_warps<" << opseq->get_num_warps() * 32 << ">();\n";
    os << "  }\n"
       << " }\n";
    return os;
}

} // namespace ark
