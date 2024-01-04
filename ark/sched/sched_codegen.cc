// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "sched/sched_codegen.h"

#include <unistd.h>

#include <cassert>
#include <fstream>
#include <initializer_list>
#include <ostream>

#include "env.h"
#include "logging.h"
#include "math_utils.h"

#define OP_PREFIX "op"
#define UNIT_OP_PREFIX "uop"
#define ALLOW_FOR_LOOP 1

namespace ark {

CodeGenerator::CodeGenerator(const GpuManager::Info &gpu_info_,
                             int num_warps_per_sm_)
    : gpu_info{gpu_info_},
      sm_num{gpu_info_.num_sm},
      num_warps_per_sm{num_warps_per_sm_},
      num_indent{0} {}

size_t CodeGenerator::get_tensor_offset(const Tensor *tensor) const {
    size_t off = tensor->buf->get_buf_offset();
    assert(off % 8 == 0);
    return off + tensor->offset_bytes();
}

std::ostream &CodeGenerator::def_remote_buf(std::ostream &os,
                                            int remote_rank) const {
    os << "__device__ char *" ARK_BUF_NAME << remote_rank << ";\n";
    return os;
}

std::ostream &CodeGenerator::sync_gpu(std::ostream &os) const {
    os << "ark::sync_gpu<" << this->sm_num << ">(" ARK_LSS_NAME ");\n";
    return os;
}

std::ostream &CodeGenerator::def_sync_stream(std::ostream &os,
                                             int stream_id) const {
    os << "__device__ ark::sync::State " ARK_LSS_NAME "_" << stream_id << ";\n";
    return os;
}

std::ostream &CodeGenerator::sync_stream(std::ostream &os, int stream_id,
                                         int sm_id_begin, int sm_id_end) const {
    if (sm_id_begin >= sm_id_end) {
        ERR(SchedulerError, "invalid SM range");
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

std::ostream &CodeGenerator::tensor(std::ostream &os,
                                    const Tensor *tensor) const {
    size_t off = this->get_tensor_offset(tensor);
    os << "(" << tensor->type.type_str() << " *)";
    std::string buf_name;
    if (tensor->imported_rank >= 0) {
        buf_name = ARK_BUF_NAME + std::to_string(tensor->imported_rank);
    } else {
        buf_name = "_buf";
    }
    os << "&" << buf_name << "[" << off << "]";
    return os;
}

std::ostream &CodeGenerator::def_oparg(std::ostream &os, const OpArg &arg,
                                       const std::string &name) const {
    if (arg.type == OP_ARG_TENSOR) {
        Tensor *tns;
        arg.get(&tns);
        os << tns->type.type_str() << " *" << name;
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
        ERR(SchedulerError, "Not implemented");
    }
    return os;
}

std::ostream &CodeGenerator::oparg(std::ostream &os, const OpArg &arg) const {
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
        ERR(SchedulerError, "Not implemented");
    }
    return os;
}

std::ostream &CodeGenerator::branch(std::ostream &os, const Branch &br,
                                    int prev_sm_id_end) const {
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
            os << "  else {";
        } else {
            os << "  else if (blockIdx.x < " << br.sm_id_end << ") {";
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
            os << "  else if (blockIdx.x == " << br.sm_id_begin << ") {";
        } else {
            os << "  else if (blockIdx.x >= " << br.sm_id_begin
               << " && blockIdx.x < " << br.sm_id_end << ") {";
        }
    }

    int tpw = this->gpu_info.threads_per_warp;

    for (auto &warp_branch : br.warp_branches) {
        if (warp_branch.branch_ops.empty()) continue;
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

        int num_warps = warp_branch.warp_id_end - warp_branch.warp_id_begin;

        auto get_indexing = [&](int num_warps_per_uop) -> std::string {
            int num_uops = num_warps / num_warps_per_uop;
            int num_threads_per_uop = num_warps_per_uop * tpw;
            std::stringstream thread_indexing;
            if (thread_end - thread_begin > num_threads_per_uop) {
                if (thread_begin > 0) {
                    thread_indexing << "((threadIdx.x - " << thread_begin
                                    << ")";
                } else {
                    thread_indexing << "(threadIdx.x";
                }
                if (math::is_pow2(num_threads_per_uop)) {
                    thread_indexing << " >> "
                                    << math::ilog2(num_threads_per_uop) << ")";
                } else {
                    thread_indexing << " / " << num_threads_per_uop << ")";
                }
            }
            auto thread_indexing_str = thread_indexing.str();

            std::stringstream sm_indexing;
            if (br.sm_id_end - br.sm_id_begin > 1) {
                if (br.sm_id_begin > 0) {
                    sm_indexing << "((blockIdx.x - " << br.sm_id_begin << ")";
                } else {
                    sm_indexing << "(blockIdx.x";
                }
                if (num_uops > 1) {
                    sm_indexing << " * " << num_uops;
                }
                sm_indexing << ")";
            }
            auto sm_indexing_str = sm_indexing.str();

            std::string indexing;
            if (thread_indexing_str.empty()) {
                indexing = sm_indexing_str;
            } else if (sm_indexing_str.empty()) {
                indexing = thread_indexing_str;
            } else {
                indexing =
                    "(" + sm_indexing_str + " + " + thread_indexing_str + ")";
            }
            return indexing;
        };

        auto uop_code = [&](int opseq_id, int uop_id_diff,
                            int num_warps_per_uop,
                            const std::string &uop_id_begin) -> std::string {
            // num_uops = (warp_id_end - warp_id_begin) / num_warps_per_uop;
            // warp_idx = warp_id - warp_id_begin;
            // sm_idx = sm_id - sm_id_begin;
            // uop = uop_id_diff * (warp_idx / num_warps_per_uop +
            //                      num_uops * sm_idx) + uop_id_begin;
            std::stringstream ss;
            ss << OP_PREFIX << opseq_id << "(_buf, ";
            if (uop_id_diff != 0) {
                auto indexing = get_indexing(num_warps_per_uop);
                if (!indexing.empty()) {
                    if (uop_id_diff != 1) {
                        ss << uop_id_diff << " * ";
                    }
                    ss << indexing << " + ";
                }
            }
            ss << uop_id_begin << ", " << br.smem_bytes_per_warp << ");";
            return ss.str();
        };

        if (ALLOW_FOR_LOOP == 0 || warp_branch.branch_ops.size() < 3) {
            for (auto &branch_op : warp_branch.branch_ops) {
                os << "      "
                   << uop_code(branch_op.opseq_id, branch_op.uop_id_diff,
                               branch_op.num_warps_per_uop,
                               std::to_string(branch_op.uop_id_begin))
                   << "\n";
            }
        } else {
            size_t idx = 0;
            while (idx < warp_branch.branch_ops.size() - 1) {
                int opseq_id = warp_branch.branch_ops[idx].opseq_id;
                int num_warps_per_uop =
                    warp_branch.branch_ops[idx].num_warps_per_uop;
                int uop_id_diff = warp_branch.branch_ops[idx].uop_id_diff;
                int uop_id_begin = warp_branch.branch_ops[idx].uop_id_begin;
                int uop_id_begin_diff =
                    warp_branch.branch_ops[idx + 1].uop_id_begin -
                    warp_branch.branch_ops[idx].uop_id_begin;
                size_t idx2 = idx + 1;
                for (; idx2 < warp_branch.branch_ops.size(); ++idx2) {
                    auto &branch_op = warp_branch.branch_ops[idx2];
                    if (branch_op.opseq_id != opseq_id ||
                        branch_op.num_warps_per_uop != num_warps_per_uop ||
                        branch_op.uop_id_diff != uop_id_diff ||
                        branch_op.uop_id_begin !=
                            (int)(uop_id_begin +
                                  uop_id_begin_diff * (idx2 - idx))) {
                        break;
                    }
                }
                if (idx2 - idx > 2) {
                    os << "      for (int _i = " << uop_id_begin << "; _i < "
                       << uop_id_begin + (idx2 - idx) * uop_id_begin_diff
                       << "; _i += " << uop_id_begin_diff << ") { "
                       << uop_code(opseq_id, uop_id_diff, num_warps_per_uop,
                                   "_i")
                       << " }\n";
                    idx = idx2;
                } else {
                    os << "      "
                       << uop_code(opseq_id, uop_id_diff, num_warps_per_uop,
                                   std::to_string(uop_id_begin))
                       << "\n";
                    ++idx;
                }
            }
            if (idx < warp_branch.branch_ops.size()) {
                auto &branch_op = warp_branch.branch_ops[idx];
                os << "      "
                   << uop_code(branch_op.opseq_id, branch_op.uop_id_diff,
                               branch_op.num_warps_per_uop,
                               std::to_string(branch_op.uop_id_begin))
                   << "\n";
            }
        }
        os << "    }\n";
    }
    os << "  }\n";
    return os;
}

std::ostream &CodeGenerator::def_uop(std::ostream &os, const SchedOp &sop,
                                     int uop_id) const {
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

    os << "int _uop_idx, int _smem_per_warp) {\n";
    os << "  " << func_name << "(";

    for (int i = 0; i < cnt_param; ++i) {
        os << '_' << i << ", ";
    }
    os << "_uop_idx, _smem_per_warp);\n}\n";
    return os;
}

std::ostream &CodeGenerator::uop(std::ostream &os, int uop_id) const {
    os << UNIT_OP_PREFIX << uop_id;
    return os;
}

//
std::ostream &CodeGenerator::opseq(std::ostream &os, const std::string &name,
                                   const SchedOpSeq &opseq,
                                   std::map<std::string, int> &uop_map) const {
    auto &sched_ops = opseq.get_sched_ops();
    unsigned int idx = sched_ops.size();
    for (auto &sop : sched_ops) {
        if (sop.is_virtual()) {
            continue;
        }
        if (idx == sched_ops.size()) {
            os << "// tile dims: (" << opseq.get_tdims()[0] << ", "
               << opseq.get_tdims()[1] << ", " << opseq.get_tdims()[2] << ")\n"
               << "__noinline__ __device__ void " << name
               << "(char *_buf, int _uop_idx, int _smem_per_warp) {\n";
        }
        --idx;
        os << "  ";
        auto uop_map_it = uop_map.find(sop.serialize());
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

        os << "_uop_idx, _smem_per_warp);\n";
    }
    if (idx != sched_ops.size()) {
        os << "}\n";
    }
    return os;
}

std::ostream &CodeGenerator::def_proxy_channels(std::ostream &os,
                                                size_t num_channels) const {
    if (num_channels == 0) {
        return os;
    }
    os << "#include <mscclpp/proxy_channel_device.hpp>\n"
          "__constant__ mscclpp::SimpleProxyChannelDeviceHandle "
          "_ARK_PROXY_CHANS["
       << num_channels << "];\n";
    return os;
}

std::ostream &CodeGenerator::def_sm_channels(std::ostream &os,
                                             size_t num_channels) const {
    if (num_channels == 0) {
        return os;
    }
    os << "#include <mscclpp/sm_channel_device.hpp>\n"
          "__constant__ mscclpp::SmChannelDeviceHandle "
          "_ARK_SM_CHANS["
       << num_channels << "];\n";
    return os;
}

}  // namespace ark
