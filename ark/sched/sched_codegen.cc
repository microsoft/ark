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

std::ostream &BaseCodeGenerator::codegen_sync_gpu(std::ostream &os)
{
    os << "ark::sync_gpu<" << this->sm_num << ">(" ARK_LSS_NAME ");\n";
    return os;
}

size_t SimpleCodeGenerator::get_tensor_offset(const Tensor *tensor)
{
    size_t off = this->buf_trans.find(tensor->buf)->second->get_offset();
    assert(off % 8 == 0);
    return off + tensor->offset_bytes();
}

ostream &SimpleCodeGenerator::codegen_tensor(ostream &os, const Tensor *tensor)
{
    size_t off = this->get_tensor_offset(tensor);
    if (tensor->type == FP16) {
        os << "(ark::half *)";
    } else if (tensor->type == FP32) {
        os << "(float *)";
    } else if (tensor->type == INT32) {
        os << "(int *)";
    } else {
        LOGERR("unknown tensor type");
    }
    os << "&" ARK_BUF_NAME "[" << off << "]";
    return os;
}

std::ostream &SimpleCodeGenerator::codegen_opseq(std::ostream &os,
                                                 SchedOpSeq *sopseq)
{
    LOG(DEBUG, "Schedopseq", sopseq->get_id());
    os << "__noinline__ __device__ void "
       << "opseq_" << sopseq->get_id() << "_tile_task"
       << "(int tile_idx) {\n";
    for (SchedOp sop : sopseq->get_sched_ops()) {
        if (sop.get_cfg() == nullptr) {
            continue;
        }
        os << "// tile nums: " << sop.get_tnums() << "\n";

        os << sop.function_name();
        // communication op does not need to be tiled
        // if (!is_tiled_op(sop)) {
        //     continue;
        // }
        os << "(";
        if (sop.get_op()->type == OP_SEND_MM) {
            // the first arg is the src_data tensor, the second arg is the
            // recvbuf tensor, the third arg is the send_ready_flag tensor
            int gid;
            sop.get_op()->args.get(&gid, 1);
            // using the recvbuf to find the tensor offset
            size_t off = this->get_tensor_offset(sop.get_op()->in_deps[1]);
            os << "(ark::comm::DataPacketLL *)&" << ARK_BUF_NAME << gid << "["
               << off << "]" << COM;
            this->codegen_tensor(os, sop.get_op()->in_deps[0]) << COM;
            os << "(int *)&" << ARK_BUF_NAME << "["
               << this->get_tensor_offset(sop.get_op()->in_deps[2]) << "]"
               << COM;
            os << "tile_idx" << COM;
        } else if (sop.get_op()->type == OP_RECV_MM) {
            // the first arg is the recvbuf tensor, the second arg is the
            // dst_data, the third arg is the send_ready_flag tensor
            int gid;
            sop.get_op()->args.get(&gid, 1);

            Tensor *recvbuf = sop.get_op()->in_deps[1];
            os << "(ark::comm::DataPacketLL *)&" << ARK_BUF_NAME << "["
               << this->get_tensor_offset(recvbuf) << "]" << COM;
            this->codegen_tensor(os, sop.get_op()->in_deps[0]) << COM;
            os << "(int *)&" << ARK_BUF_NAME << gid << "["
               << this->get_tensor_offset(sop.get_op()->in_deps[2]) << "]"
               << COM;
            os << "tile_idx" << COM;
        } else {
            for (Tensor *tns : sop.get_op()->out_deps) {
                this->codegen_tensor(os, tns) << COM;
            }
            for (Tensor *tns : sop.get_op()->in_deps) {
                this->codegen_tensor(os, tns) << COM;
            }
        }
        if (sop.get_op()->type == OP_SCALE) {
            float val;
            sop.get_op()->args.get(&val, 0);
            os << val << COM;
        }

        // the tile index
        os << "tile_idx);\n";
    }
    os << "}\n";
    return os;
}

std::ostream &SimpleCodeGenerator::codegen_sched(std::ostream &os, Sched &sched)
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

vector<string> SimpleCodeGenerator::codegen_codes_body(vector<Sched> &scheds)
{
    stringstream loop_body_code, sched_opseq_code, data_buf_code;
    for (int i = 0; i < this->world_size; i++) {
        data_buf_code << "__device__ char *" << ARK_BUF_NAME << i << ";\n";
    }
    loop_body_code << "__device__ void ark_loop_body(int _iter) {\n";
    // to avoid the same opseq code being generated multiple times
    set<int> opseq_ids;
    for (Sched &sched : scheds) {
        bool virt_opseq = false;
        // for the baseline scheduler, one opseq is a depth and have a global
        // sync between each opseq
        SchedOpSeq *opseq = sched.opseq;
        if (opseq_ids.find(opseq->get_id()) != opseq_ids.end()) {
        } else {
            opseq_ids.insert(opseq->get_id());
            this->codegen_opseq(sched_opseq_code, opseq);
        }
        if (virt_opseq)
            continue;
        this->codegen_sched(loop_body_code, sched);
        loop_body_code << "  ";
        this->codegen_sync_gpu(loop_body_code);
    }
    loop_body_code << "}\n";
    vector<string> ret;
    ret.emplace_back(data_buf_code.str() + sched_opseq_code.str() +
                     loop_body_code.str());
    return ret;
}

void Brancher::add(const Sched &sc)
{
    assert(sc.opseq != nullptr);
    assert(sc.sm_b < sc.sm_e);
    assert(sc.th_b < sc.th_e);
    SmBranch &sb = sbs.back();
    if ((sb.sm_b == sc.sm_b) && (sb.sm_e == sc.sm_e)) {
        ThBranch &tb = sb.tbs.back();
        if ((tb.th_b == sc.th_b) && (tb.th_e == sc.th_e)) {
            tb.ops.emplace_back(sc.opseq, sc.alpha, sc.beta);
        } else if ((tb.th_e <= sc.th_b) || (sc.th_b == 0)) {
            sb.tbs.emplace_back();
            ThBranch &new_tb = sb.tbs.back();
            new_tb.th_b = sc.th_b;
            new_tb.th_e = sc.th_e;
            new_tb.ops.emplace_back(sc.opseq, sc.alpha, sc.beta);
        } else {
            LOGERR(OP_PREFIX, sc.opseq->get_id(), " ", tb.th_e, " ", sc.th_b);
        }
    } else if ((sb.sm_e <= sc.sm_b) || (sc.sm_b == 0)) {
        sbs.emplace_back();
        SmBranch &new_sb = sbs.back();
        new_sb.sm_b = sc.sm_b;
        new_sb.sm_e = sc.sm_e;
        new_sb.tbs.emplace_back();
        ThBranch &new_tb = new_sb.tbs.back();
        new_tb.th_b = sc.th_b;
        new_tb.th_e = sc.th_e;
        new_tb.ops.emplace_back(sc.opseq, sc.alpha, sc.beta);
    } else {
        assert(false);
    }
}

ostream &Brancher::codegen(ostream &os)
{
    int prev_sm_e = sm_num;
    for (auto &sb : sbs) {
        if (sb.sm_b == 0) {
            if (sb.sm_e == sm_num) {
                os << "\n  { // for all SMs";
            } else {
                os << "\n  if (blockIdx.x < " << sb.sm_e << ") {";
            }
        } else if (sb.sm_b == prev_sm_e) {
            if (sb.sm_e == sm_num) {
                os << " else {";
            } else {
                os << " else if (blockIdx.x < " << sb.sm_e << ") {";
            }
        } else if (sb.sm_b < prev_sm_e) {
            os << "\n  if (blockIdx.x >= " << sb.sm_b << " && blockIdx.x < "
               << sb.sm_e << ") {";
        } else if (sb.sm_b > prev_sm_e) {
            os << " else if (blockIdx.x >= " << sb.sm_b << " && blockIdx.x < "
               << sb.sm_e << ") {";
        }
        prev_sm_e = sb.sm_e;
        int prev_th_e = th_num;
        //
        auto it_b = sb.tbs.begin();
        if (it_b == sb.tbs.end()) {
            os << "\n  }";
            continue;
        }
        vector<int> beta_diffs;
        int th_sz = it_b->th_e - it_b->th_b;
        auto it_e = next(it_b);
        while (it_b != sb.tbs.end()) {
            bool compress = true;
            if ((it_e == sb.tbs.end()) ||
                (it_b->ops.size() != it_e->ops.size()) ||
                ((it_e->th_e - it_e->th_b) != th_sz)) {
                compress = false;
            } else {
                SchedOpSeq *opseq_0;
                SchedOpSeq *opseq_1;
                int alpha_0;
                int alpha_1;
                int beta_0;
                int beta_1;
                for (unsigned int i = 0; i < it_b->ops.size(); ++i) {
                    tie(opseq_0, alpha_0, beta_0) = it_b->ops[i];
                    tie(opseq_1, alpha_1, beta_1) = it_e->ops[i];
                    if ((opseq_0->get_id() != opseq_1->get_id()) ||
                        (alpha_0 != alpha_1)) {
                        compress = false;
                        break;
                    }
                    int d = distance(it_b, it_e);
                    if (d == 1) {
                        beta_diffs.emplace_back(beta_1 - beta_0);
                        continue;
                    } else if ((d * beta_diffs[i]) != (beta_1 - beta_0)) {
                        compress = false;
                        break;
                    }
                }
            }
            if (compress) {
                it_e = next(it_e);
                continue;
            }
            auto it_l = prev(it_e);
            //
            if (it_b->th_b == 0) {
                if (it_l->th_e == th_num) {
                    os << "\n    { // for all threads\n";
                } else {
                    os << "\n    if (threadIdx.x < " << it_l->th_e << ") {\n";
                }
            } else if (it_b->th_b == prev_th_e) {
                if (it_l->th_e == th_num) {
                    os << " else {\n";
                } else {
                    os << " else if (threadIdx.x < " << it_l->th_e << ") {\n";
                }
            } else if (it_b->th_b < prev_th_e) {
                os << "\n    if (threadIdx.x >= " << it_b->th_b
                   << " && threadIdx.x < " << it_l->th_e << ") {\n";
            } else if (it_b->th_b > prev_th_e) {
                os << " else if (threadIdx.x >= " << it_b->th_b
                   << " && threadIdx.x < " << it_l->th_e << ") {\n";
            }
            prev_th_e = it_l->th_e;
            SchedOpSeq *opseq;
            int alpha;
            int beta;
            for (unsigned int i = 0; i < it_b->ops.size(); ++i) {
                tie(opseq, alpha, beta) = it_b->ops[i];
                os << "      " << OP_PREFIX << opseq->get_id() << '(';
                if (alpha != 0) {
                    if (alpha != 1) {
                        os << alpha << " * ";
                    }
                    if (sb.sm_b == 0) {
                        os << "blockIdx.x";
                    } else {
                        os << "(blockIdx.x - " << sb.sm_b << ')';
                    }
                }
                if (beta_diffs.size() > 0) {
                    assert(beta_diffs.size() == it_b->ops.size());
                    if (alpha != 0) {
                        if (beta_diffs[i] > 0)
                            os << " + ";
                    }
                    if (beta_diffs[i] == 1) {
                        os << "((threadIdx.x - " << it_b->th_b << ") >> ";
                    } else {
                        os << beta_diffs[i] << " * ((threadIdx.x - "
                           << it_b->th_b << ") >> ";
                    }
                    os << math::ilog2(th_sz) << ")";
                    if (beta != 0) {
                        os << " + " << beta;
                    }
                } else if ((alpha != 0) && (beta != 0)) {
                    os << " + " << beta;
                } else if (alpha == 0) {
                    os << beta;
                }
                os << ");\n";
            }
            os << "    }";
            if (it_e == sb.tbs.end()) {
                break;
            } else {
                it_b = it_e;
                it_e = next(it_b);
                th_sz = it_b->th_e - it_b->th_b;
                beta_diffs.clear();
            }
        }
        os << "\n  }";
    }
    os << '\n';
    return os;
}

ostream &DefaultCodeGenerator::codegen_tensor(ostream &os, const Tensor &tensor)
{
    auto search = this->buf_trans.find(tensor.buf);
    if (search == this->buf_trans.end()) {
        LOGERR("unknown tensor buffer");
    }
    size_t off = search->second->get_offset();
    assert(off % 8 == 0);
    size_t data_size;
    if (tensor.type == FP16) {
        os << "(ark::half *)";
        data_size = 2;
    } else if (tensor.type == FP32) {
        os << "(float *)";
        data_size = 4;
    } else if (tensor.type == INT32) {
        os << "(int *)";
        data_size = 4;
    } else {
        LOGERR("unknown tensor type");
    }
    off += tensor.offset() * data_size;
    os << "&" ARK_BUF_NAME "[" << off << "]";
    return os;
}

std::ostream &DefaultCodeGenerator::codegen_arg(std::ostream &os,
                                                const OpArg &arg)
{
    if (arg.type == OP_ARG_TENSOR) {
        Tensor *tns;
        arg.get(&tns);
        this->codegen_tensor(os, *tns);
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

std::ostream &DefaultCodeGenerator::codegen_arg_def(std::ostream &os,
                                                    const OpArg &arg,
                                                    const std::string &name)
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

//
ostream &DefaultCodeGenerator::codegen_opseq(ostream &os, const string &name,
                                             const SchedOpSeq &opseq,
                                             map<string, int> &uop_map)
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
        auto uop_map_it = uop_map.find(sop.function_name());
        assert(uop_map_it != uop_map.end());
        os << "  " << UNIT_OP_PREFIX << uop_map_it->second << '(';

        OpArgs call_args = sop.get_op()->function_call_args(*sop.get_cfg());
        for (const OpArg &arg : call_args.get_args()) {
            this->codegen_arg(os, arg) << ", ";
        }

        os << "_uop_idx);\n";
    }
    if (idx != sched_ops.size()) {
        os << "}\n";
    }
    return os;
}

ostream &DefaultCodeGenerator::codegen_uop_def(ostream &os, const SchedOp &sop,
                                               int uop_id)
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
        this->codegen_arg_def(os, arg, "_" + std::to_string(cnt_param)) << ", ";
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

ostream &DefaultCodeGenerator::codegen_depth(ostream &os, const string &name,
                                             Brancher *brc,
                                             set<SchedOpSeq *> &opseqs,
                                             map<string, int> &uop_map)
{
    for (auto &opseq : opseqs) {
        for (auto &sop : opseq->get_sched_ops()) {
            if (sop.is_virtual()) {
                continue;
            }
            int uop_id = uop_map.size();
            // Insert only if it does not exist
            // TODO: check if runtime args are the same
            string sop_func_str = sop.function_name();
            auto p = uop_map.emplace(sop_func_str, uop_id);
            if (p.second) {
                // If this is a new function, define it.
                this->codegen_uop_def(os, sop, uop_id);
            }
        }
    }
    for (auto &opseq : opseqs) {
        if (opseq->is_virtual()) {
            continue;
        }
        this->codegen_opseq(os, OP_PREFIX + to_string(opseq->get_id()), *opseq,
                            uop_map);
    }
    os << "DEVICE void " << name << "() {";
    brc->codegen(os);
    os << "}\n";
    return os;
}

vector<string> DefaultCodeGenerator::codegen_codes_body(vector<Sched> &scheds)
{
    const int sm_num = this->sm_num;
    const int th_num = this->wps * 32;
    stringstream body;
    body << "__device__ void ark_loop_body(int _iter) {\n";
    stringstream depths;
    set<SchedOpSeq *> opseqs;
    Brancher *brc = new Brancher{sm_num, th_num};
    int depth_idx = 0;
    map<string, int> uop_map;
    for (auto it = scheds.begin(); it != scheds.end(); ++it) {
        if (it->opseq != nullptr) {
            opseqs.emplace(it->opseq);
            brc->add(*it);
        }
        if ((it->opseq == nullptr) ||
            (!brc->is_empty() && (it == (scheds.end() - 1)))) {
            if (!brc->is_empty()) {
                string name = "depth" + to_string(depth_idx);
                //
                this->codegen_depth(depths, name, brc, opseqs, uop_map);
                //
                if (depth_idx != 0) {
                    body << "  ";
                    this->codegen_sync_gpu(body);
                }
                body << "  " << name << "();\n";
            }
            //
            ++depth_idx;
            opseqs.clear();
            delete brc;
            brc = new Brancher{sm_num, th_num};
        }
    }
    body << "}\n";
    vector<string> ret;
    ret.emplace_back(depths.str() + body.str());
    return ret;
}
} // namespace ark