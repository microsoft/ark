// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <sstream>

#include "ark/model_io.h"
#include "third_party/json/json.h"

using namespace std;
using namespace nlohmann;

namespace ark {

static json to_json(const TensorBuf &tns_buf)
{
    return tns_buf.bytes;
}

static json to_json(const Tensor &tns,
                    const map<TensorBuf *, int> &tns_buf_trans)
{
    json j;
    j.emplace_back(tns_buf_trans.at(tns.buf));
    j.emplace_back(tns.type);
    j.emplace_back(json(tns.shape.data));
    j.emplace_back(json(tns.ldims.data));
    j.emplace_back(json(tns.offs.data));
    j.emplace_back(json(tns.pads.data));
    return j;
}

static json to_json(const OpArg &oa)
{
    json j;
    j.emplace_back(oa.type);
    if (oa.type == OP_ARG_INT)
        j.emplace_back(*static_cast<int *>(oa.val));
    else if (oa.type == OP_ARG_UINT64)
        j.emplace_back(*static_cast<uint64_t *>(oa.val));
    else if (oa.type == OP_ARG_BOOL)
        j.emplace_back(*static_cast<bool *>(oa.val));
    else if (oa.type == OP_ARG_FLOAT)
        j.emplace_back(*static_cast<float *>(oa.val));
    return j;
}

static json to_json(const Op &op, const map<Tensor *, int> &tns_trans)
{
    json j;
    j.emplace_back(op.type);
    j.emplace_back(op.prec_type);
    json in_deps;
    for (auto &tns : op.in_deps)
        in_deps.emplace_back(tns_trans.at(tns));
    j.emplace_back(in_deps);
    json out_deps;
    for (auto &tns : op.out_deps)
        out_deps.emplace_back(tns_trans.at(tns));
    j.emplace_back(out_deps);
    json args;
    for (auto &oa : op.args)
        args.emplace_back(to_json(oa));
    j.emplace_back(args);
    return j;
}

ostream &operator<<(ostream &os, const Model &m)
{
    map<TensorBuf *, int> tns_buf_trans;
    json json_tns_bufs;
    for (auto &tns_buf : m.get_tensor_bufs()) {
        json_tns_bufs.emplace_back(to_json(*tns_buf));
        tns_buf_trans.emplace(tns_buf.get(), (int)tns_buf_trans.size());
    }
    map<Tensor *, int> tns_trans;
    json json_tns;
    for (auto &tns : m.get_tensors()) {
        json_tns.emplace_back(to_json(*tns, tns_buf_trans));
        tns_trans.emplace(tns.get(), (int)tns_trans.size());
    }
    json json_ops;
    for (auto &op : m.get_ops())
        json_ops.emplace_back(to_json(*op, tns_trans));
    json j;
    j.emplace_back(json_tns_bufs);
    j.emplace_back(json_tns);
    j.emplace_back(json_ops);
    os << j;
    return os;
}

////////////////////////////////////////////////////////////////////////////////

static TensorBuf *add_tensor_buf(const json &obj, Model &m)
{
    return m.create_tensor_buf(obj.get<DimType>());
}

static Tensor *add_tensor(const json &obj, Model &m,
                          const map<int, TensorBuf *> &tns_buf_trans)
{
    const int &id = obj[0].get<int>();
    const TensorType &type = obj[1].get<TensorType>();
    Tensor *tns =
        m.tensor(obj[2].get<vector<DimType>>(), type, tns_buf_trans.at(id),
                 obj[3].get<vector<DimType>>(), obj[4].get<vector<DimType>>(),
                 obj[5].get<vector<DimType>>());
    return tns;
}

static OpArg json_to_op_arg(const json &obj)
{
    OpArgType type = obj[0].get<OpArgType>();
    if (type == OP_ARG_INT) {
        return OpArg{obj[1].get<int>()};
    } else if (type == OP_ARG_UINT64) {
        return OpArg{obj[1].get<uint64_t>()};
    } else if (type == OP_ARG_BOOL) {
        return OpArg{obj[1].get<bool>()};
    } else {
        assert(type == OP_ARG_FLOAT);
        return OpArg{obj[1].get<float>()};
    }
}

static Op *add_op(const json &obj, Model &m,
                  const map<int, Tensor *> &tns_trans)
{
    auto &json_op_type = obj[0];
    auto &json_op_prec_type = obj[1];
    auto &json_op_in_deps = obj[2];
    auto &json_op_out_deps = obj[3];
    auto &json_op_args = obj[4];

    vector<Tensor *> in_deps;
    for (auto &json_in_dep : json_op_in_deps)
        in_deps.emplace_back(tns_trans.at((int)json_in_dep));
    vector<Tensor *> out_deps;
    for (auto &json_out_dep : json_op_out_deps)
        out_deps.emplace_back(tns_trans.at((int)json_out_dep));
    // args.
    vector<OpArg> args;
    for (auto &json_oa : json_op_args) {
        args.emplace_back(json_to_op_arg(json_oa));
    }
    return m.create_op(json_op_type.get<OpType>(),
                       json_op_prec_type.get<OpPrecType>(), in_deps, out_deps,
                       args, "");
}

istream &operator>>(istream &is, Model &m)
{
    json j;
    is >> j;
    auto &json_tns_bufs = j[0];
    auto &json_tns = j[1];
    auto &json_ops = j[2];
    map<int, TensorBuf *> tns_buf_trans;
    for (auto &json_tns_buf : json_tns_bufs) {
        TensorBuf *buf = add_tensor_buf(json_tns_buf, m);
        tns_buf_trans[tns_buf_trans.size()] = buf;
    }
    map<int, Tensor *> tns_trans;
    for (auto &json_tensor : json_tns) {
        Tensor *tns = add_tensor(json_tensor, m, tns_buf_trans);
        tns_trans[tns_trans.size()] = tns;
    }
    for (auto &json_op : json_ops) {
        add_op(json_op, m, tns_trans);
    }
    return is;
}

const string type_str(const TensorType &type)
{
    if (type == FP16)
        return "fp16";
    else if (type == FP32)
        return "fp32";
    return "none";
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
    case OP_GELU:          os << "OP_GELU";          break;
    }
    // clang-format on
    return os;
}

const string op_str(const Op &op)
{
    stringstream ss;
    if (op.type == OP_MATMUL) {
        ss << "OP_MATMUL[in: " << op.in_deps[0]->shape
           << " ot: " << op.in_deps[1]->shape << ']';
    } else {
        ss << op.type;
    }
    return ss.str();
}

} // namespace ark
