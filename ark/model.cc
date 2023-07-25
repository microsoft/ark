// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>

#include "logging.h"
#include "model.h"

using namespace std;

namespace ark {

// Create a new TensorBuf object with `bytes` bytes.
// A common usage is setting `bytes` to 0 during declaring a model and let the
// scheduler determine the value after the model is completely defined.
TensorBuf *Model::Impl::create_tensor_buf(const DimType bytes)
{
    TensorBuf *buf = new TensorBuf{bytes, (int)this->tns_bufs_storage.size()};
    assert(buf != nullptr);
    this->tns_bufs_storage.emplace_back(buf);
    return buf;
}

// Remove a TensorBuf object from the model.
void Model::Impl::destroy_tensor_buf(const TensorBuf *buf)
{
    for (auto &tns : this->tns_storage) {
        if (tns->buf == buf) {
            LOGERR("dangling tensor detected");
        }
    }
    bool is_found = false;
    auto it = this->tns_bufs_storage.begin();
    for (; it != this->tns_bufs_storage.end(); ++it) {
        if (it->get() == buf) {
            this->tns_bufs_storage.erase(it);
            is_found = true;
            break;
        }
    }
    if (!is_found) {
        LOGERR("the given TensorBuf is not found");
    }
}

// Add a new operator to the model. This is a helper function for operator APIs.
//
// `type`: the type of the operator
// `prec_type`: the precision type of the operator
// `in_deps`:
//     The input tensors of the operator, including execution dependencies.
// `out_deps`:
//     The output tensors of the operator, including execution dependencies.
// `args`:
//     The arguments of the operator.
// `name`: the name of the operator
// `gran_lev`:
//      The granularity level of the operator. Larger values should indicate
//      finer-grained operators. If it is -1, the granularity level will be
//      automatically determined by the scheduler.
//
Op *Model::Impl::add_op(const OpType type, const OpPrecType prec_type,
                        const vector<Tensor *> &in_deps,
                        const vector<Tensor *> &out_deps, const OpArgs &args,
                        const string &name, const OpConfigMap *cfg_map, int gran_lev)
{
    string suffix_str;
    auto p = this->name_cnts.emplace(name, 1);
    if (!p.second) {
        int suffix_num = p.first->second;
        this->name_cnts[name] = suffix_num + 1;
        suffix_str = "_" + to_string(suffix_num);
    } else {
        suffix_str = "";
    }
    Op *op = new Op{type, prec_type,         in_deps, out_deps,
                    args, name + suffix_str, cfg_map, gran_lev};
    assert(op != nullptr);
    this->ops_storage.emplace_back(op);
    for (auto &tns : in_deps) {
        // If an input tensor is not generated by another op,
        // the buffer for this tensor may store external input data,
        // so we set the buffer to be immutable.
        if (this->get_gen_op(tns) == nullptr) {
            tns->buf->immutable = true;
        }
        this->ref_ops[tns].insert(op);
    }
    for (auto &tns : out_deps) {
        this->gen_op[tns] = op;
    }
    return op;
}

Op *Model::Impl::add_op(Op &op)
{
    string suffix_str;
    auto p = this->name_cnts.emplace(op.name, 1);
    if (!p.second) {
        int suffix_num = p.first->second;
        this->name_cnts[op.name] = suffix_num + 1;
        suffix_str = "_" + to_string(suffix_num);
    } else {
        suffix_str = "";
    }
    op.name = op.name + suffix_str;
    this->ops_storage.emplace_back(make_unique<Op>(op));

    Op *op_ptr = this->ops_storage.back().get();
    for (auto &tns : op.in_deps) {
        // If an input tensor is not generated by another op,
        // the buffer for this tensor may store external input data,
        // so we set the buffer to be immutable.
        if (this->get_gen_op(tns) == nullptr) {
            tns->buf->immutable = true;
        }
        this->ref_ops[tns].insert(op_ptr);
    }
    for (auto &tns : op.out_deps) {
        this->gen_op[tns] = op_ptr;
    }
    return op_ptr;
}

/// Delete an existing operator from the model.
/// @param op the existing op to be deleted.
void Model::Impl::delete_op(Op *op)
{
    // Remove the operator from the set of operators that have the given tensor
    // as one of their inputs.
    for (auto &tns : op->in_deps) {
        auto search = this->ref_ops.find(tns);
        if (search == this->ref_ops.end()) {
            LOGERR("Not an existing tensor.");
        }
        search->second.erase(op);
    }
    // Remove the operator from the set of operators that have the given tensor
    // as its output.
    for (auto &tns : op->out_deps) {
        auto search = this->gen_op.find(tns);
        if (search == this->gen_op.end()) {
            LOGERR("Not an existing tensor.");
        }
        this->gen_op.erase(search);
    }
    // Remove the operator from the model.
    bool is_found = false;
    auto it = this->ops_storage.begin();
    for (; it != this->ops_storage.end(); ++it) {
        if (it->get() == op) {
            this->ops_storage.erase(it);
            is_found = true;
            break;
        }
    }
    if (!is_found) {
        LOGERR("the given Op is not found");
    }
    auto search = this->name_cnts.find(op->name);
    if (search != this->name_cnts.end()) {
        if (search->second == 1) {
            this->name_cnts.erase(search);
        }
        // If there are multiple operators with the same name, we do not
        // decrease the counter to avoid conflicts when creating new operators
        // with the same name.
    }
}

std::list<TensorBuf *> Model::Impl::get_tensor_bufs() const
{
    std::list<TensorBuf *> tns_buf_list;
    for (auto &tns_buf : this->tns_bufs_storage) {
        tns_buf_list.emplace_back(tns_buf.get());
    }
    return tns_buf_list;
};

std::list<Tensor *> Model::Impl::get_tensors() const
{
    std::list<Tensor *> tns_list;
    for (auto &tns : this->tns_storage) {
        tns_list.emplace_back(tns.get());
    }
    return tns_list;
};

std::list<Op *> Model::Impl::get_ops() const
{
    std::list<Op *> ops;
    for (auto &op : ops_storage) {
        ops.emplace_back(op.get());
    }
    return ops;
};

// Returns the latest-declared operator that has the given tensor as its output.
const Op *Model::Impl::get_gen_op(Tensor *tns) const
{
    auto search = this->gen_op.find(tns);
    if (search == this->gen_op.end()) {
        return nullptr;
    }
    return search->second;
}

// Returns the set of operators that have the given tensor as one of their
// inputs.
const std::set<Op *> &Model::Impl::get_ref_ops(Tensor *tns) const
{
    auto search = this->ref_ops.find(tns);
    if (search == this->ref_ops.end()) {
        LOGERR("Not an existing tensor.");
    }
    return search->second;
}

// Returns true if the given tensor is not an input of any operator.
bool Model::Impl::is_no_ref(Tensor *tns) const
{
    auto search = this->ref_ops.find(tns);
    if (search == this->ref_ops.end()) {
        return true;
    }
    return false;
}

// void to_json(nlohmann::json &j, const Model &model)
// {
//     j = nlohmann::json{
//         {"tensors", vector<Tensor>{}},
//         {"ops", vector<Op>{}},
//     };
//     for (auto &pt : model.get_tensors()) {
//         j.at("tensors").emplace_back(*pt);
//     }
//     for (auto &po : model.get_ops()) {
//         j.at("ops").emplace_back(*po);
//     }
// }
// void from_json(const nlohmann::json &j, Model &model)
// {
// }

//
Model::Model(int rank_) : impl{make_unique<Model::Impl>()}
{
    this->impl->rank = rank_;
}

Model::~Model() = default;

} // namespace ark
