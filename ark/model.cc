// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model.h"

#include <stack>

#include "logging.h"

using namespace std;

namespace ark {

// Create a new TensorBuf object with `bytes` bytes.
// A common usage is setting `bytes` to 0 during declaring a model and let the
// scheduler determine the value after the model is completely defined.
TensorBuf *Model::Impl::create_tensor_buf(const DimType bytes) {
    this->tns_bufs_storage.emplace_back(
        make_unique<TensorBuf>(bytes, (int)this->tns_bufs_storage.size()));
    return this->tns_bufs_storage.back().get();
}

// Remove a TensorBuf object from the model.
void Model::Impl::destroy_tensor_buf(const TensorBuf *buf) {
    for (auto &tns : this->tns_storage) {
        if (tns->buf == buf) {
            ERR(ModelError, "dangling tensor detected");
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
        ERR(ModelError, "the given TensorBuf is not found");
    }
}

std::vector<Tensor *> Model::Impl::add_op(
    const OpType type, const std::string &prec_type,
    const vector<Tensor *> &inputs, const vector<Tensor *> &outputs,
    const OpArgs &args, const string &name, const OpConfigMap *cfg_map,
    int gran_lev) {
    Op op{type, prec_type, inputs, outputs, args, name, cfg_map, gran_lev};
    return this->add_op(op);
}

std::string Model::Impl::append_name_postfix(const std::string &name) {
    string suffix_str;
    auto p = this->name_cnts.emplace(name, 1);
    if (!p.second) {
        int suffix_num = p.first->second;
        this->name_cnts[name] = suffix_num + 1;
        suffix_str = "_" + to_string(suffix_num);
    } else {
        suffix_str = "";
    }
    return name + suffix_str;
}

std::vector<Tensor *> Model::Impl::add_op(Op &op) {
    op.name = append_name_postfix(op.name);
    this->ops_storage.emplace_back(make_unique<Op>(op));

    Op *op_ptr = this->ops_storage.back().get();
    for (auto &tns : op_ptr->inputs) {
        this->tns_to_users[tns].insert(op_ptr);
    }
    if (op_ptr->type == OP_TENSOR) {
        // Only for TensorOps, output references directly become outputs.
        for (auto &tns : op_ptr->output_refs) {
            this->tns_to_producer[tns] = op_ptr;
            if (this->tns_to_users.find(tns) == this->tns_to_users.end()) {
                this->tns_to_users[tns] = {};
            }
        }
        op_ptr->outputs = op_ptr->output_refs;

        // Clear references.
        op_ptr->output_refs.clear();
        return op_ptr->outputs;
    }
    std::vector<Tensor *> output_tensors;
    for (auto &tns : op_ptr->output_refs) {
        // If Op type is not OP_TENSOR and if we set the producer of the
        // given output reference to be the current Op, then the TensorOp that
        // has produced this output reference will be forgotten (because there
        // can be only one producer Op for each tensor). To avoid this,
        // we create an identical tensor and set the producer of the new tensor
        // to be the current Op.
        this->tns_storage.emplace_back(make_unique<Tensor>(
            tns->shape, tns->type, tns->buf, tns->ldims, tns->offs, tns->pads,
            tns->exported, tns->imported_rank, (int)this->tns_storage.size(),
            tns->name));
        Tensor *output_tensor = this->tns_storage.back().get();
        output_tensor->buf->immutable = false;

        this->tns_to_producer[output_tensor] = op_ptr;
        this->tns_to_users[output_tensor] = {};

        // The current Op becomes a user (not producer) of the given output
        // tensor.
        this->tns_to_users[tns].insert(op_ptr);

        output_tensors.push_back(output_tensor);
    }
    op_ptr->outputs = output_tensors;
    return output_tensors;
}

/// Delete an existing operator from the model.
/// @param op the existing op to be deleted.
void Model::Impl::delete_op(Op *op) {
    // Remove the operator from the set of operators that have the given tensor
    // as one of their inputs.
    for (auto &tns : op->inputs) {
        auto search = this->tns_to_users.find(tns);
        if (search == this->tns_to_users.end()) {
            ERR(ModelError, "Not an existing tensor.");
        }
        search->second.erase(op);
    }
    for (auto &tns : op->output_refs) {
        auto search = this->tns_to_users.find(tns);
        if (search == this->tns_to_users.end()) {
            ERR(ModelError, "Not an existing tensor.");
        }
        search->second.erase(op);
    }
    // Remove the operator from the set of operators that have the given tensor
    // as its output.
    for (auto &tns : op->outputs) {
        auto search = this->tns_to_producer.find(tns);
        if (search == this->tns_to_producer.end()) {
            ERR(ModelError, "Not an existing tensor.");
        }
        this->tns_to_producer.erase(search);
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
        ERR(ModelError, "the given Op is not found");
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

std::list<TensorBuf *> Model::Impl::get_tensor_bufs() const {
    std::list<TensorBuf *> tns_buf_list;
    for (auto &tns_buf : this->tns_bufs_storage) {
        tns_buf_list.emplace_back(tns_buf.get());
    }
    return tns_buf_list;
};

std::list<Tensor *> Model::Impl::get_tensors() const {
    std::list<Tensor *> tns_list;
    for (auto &tns : this->tns_storage) {
        tns_list.emplace_back(tns.get());
    }
    return tns_list;
};

std::list<Op *> Model::Impl::get_ops() const {
    std::list<Op *> ops;
    for (auto &op : this->ops_storage) {
        ops.emplace_back(op.get());
    }
    return ops;
};

// Returns the latest-declared operator that has the given tensor as its output.
const Op *Model::Impl::get_producer(Tensor *tns) const {
    auto search = this->tns_to_producer.find(tns);
    if (search == this->tns_to_producer.end()) {
        return nullptr;
    }
    return search->second;
}

// Returns the set of operators that have the given tensor as one of their
// inputs.
const std::set<Op *> &Model::Impl::get_users(Tensor *tns) const {
    auto search = this->tns_to_users.find(tns);
    if (search == this->tns_to_users.end()) {
        ERR(ModelError, "Not an existing tensor.");
    }
    return search->second;
}

// Returns true if the given tensor is not an input of any operator.
bool Model::Impl::is_no_user(Tensor *tns) const {
    auto search = this->tns_to_users.find(tns);
    if (search == this->tns_to_users.end()) {
        return true;
    }
    return false;
}

std::list<const Op *> Model::Impl::get_leaf_ops() const {
    std::list<const Op *> leaf_ops;
    for (auto &op : this->ops_storage) {
        bool is_leaf = true;
        for (auto &tns : op->outputs) {
            if (!this->is_no_user(tns)) {
                is_leaf = false;
                break;
            }
        }
        if (is_leaf) {
            leaf_ops.emplace_back(op.get());
        }
    }
    return leaf_ops;
}

std::list<const Op *> Model::Impl::get_producer_ops(const Op *op) const {
    // Input tensors and output reference tensors are all producer tensors.
    std::vector<Tensor *> producer_tensors = op->inputs;
    for (auto &tns : op->output_refs) {
        producer_tensors.push_back(tns);
    }

    std::list<const Op *> producer_ops;
    for (auto &tns : producer_tensors) {
        const Op *producer_op = this->get_producer(tns);
        if (producer_op != nullptr) {
            producer_ops.emplace_back(producer_op);
        }
    }
    return producer_ops;
}

/// Returns the set of Ops that are user of the given Op's output.
std::list<const Op *> Model::Impl::get_user_ops(const Op *op) const {
    std::list<const Op *> user_ops;
    for (auto &tns : op->outputs) {
        const std::set<Op *> &user_op_set = this->get_users(tns);
        for (auto &user_op : user_op_set) {
            user_ops.emplace_back(user_op);
        }
    }
    return user_ops;
}

const Op *Model::Impl::get_cyclic_op() const {
    std::list<const Op *> leaf_ops = this->get_leaf_ops();
    std::set<const Op *> visited_ops;
    std::stack<const Op *> op_stack;
    for (auto &op : leaf_ops) {
        op_stack.push(op);
    }
    while (!op_stack.empty()) {
        const Op *op = op_stack.top();
        op_stack.pop();
        if (visited_ops.find(op) != visited_ops.end()) {
            return op;
        }
        visited_ops.insert(op);
        std::list<const Op *> producer_ops = this->get_producer_ops(op);
        for (auto &producer_op : producer_ops) {
            op_stack.push(producer_op);
        }
    }
    return nullptr;
}

//
Model::Model(int rank_) : impl{make_unique<Model::Impl>()} {
    this->impl->rank = rank_;
}

Model::~Model() = default;

bool Model::verify() const {
    const Op *cyclic_op = this->impl->get_cyclic_op();
    if (cyclic_op != nullptr) {
        LOG(WARN, "Cyclic dependency detected around Op ",
            cyclic_op->name.c_str(), " and its inputs.");
        return false;
    }
    return true;
}

}  // namespace ark
