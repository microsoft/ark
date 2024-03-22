// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model.h"

#include <algorithm>
#include <stack>

#include "logging.h"
#include "nlohmann/json.hpp"

#define DEBUG_OPGRAPH 0
#define OPGRAPH_DEBUG(...)           \
    do {                             \
        if (DEBUG_OPGRAPH) {         \
            LOG(DEBUG, __VA_ARGS__); \
        }                            \
    } while (0);

namespace ark {

// Create a new TensorBuf object with `bytes` bytes.
// A common usage is setting `bytes` to 0 during declaring a model and let the
// scheduler determine the value after the model is completely defined.
TensorBuf *Model::Impl::create_tensor_buf(const DimType bytes) {
    this->tns_bufs_storage.emplace_back(
        std::make_unique<TensorBuf>(bytes, (int)this->tns_bufs_storage.size()));
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
    const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
    const OpArgs &args, const std::string &name, const OpConfigMap *cfg_map,
    int gran_lev) {
    Op op{type, prec_type, inputs, outputs, args, name, cfg_map, gran_lev};
    return this->add_op(op);
}

std::string Model::Impl::append_name_postfix(const std::string &name) {
    std::string suffix_str;
    auto p = this->name_cnts.emplace(name, 1);
    if (!p.second) {
        int suffix_num = p.first->second;
        this->name_cnts[name] = suffix_num + 1;
        suffix_str = "_" + std::to_string(suffix_num);
    } else {
        suffix_str = "";
    }
    return name + suffix_str;
}

std::vector<Tensor *> Model::Impl::add_op(Op &op) {
    op.name = append_name_postfix(op.name);
    this->ops_storage.emplace_back(std::make_unique<Op>(op));

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
        this->tns_storage.emplace_back(std::make_unique<Tensor>(
            tns->shape, tns->type, tns->buf, tns->ldims, tns->offs, tns->pads,
            tns->exported, tns->imported_rank, (int)this->tns_storage.size(),
            tns->name));
        Tensor *output_tensor = this->tns_storage.back().get();
        output_tensor->buf->immutable = false;

        this->tns_to_producer[output_tensor] = op_ptr;
        this->tns_to_users[output_tensor] = {};

        // The current Op becomes a user (not producer) of the given output
        // reference tensor.
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

/// Replace a @ref Tensor with another @ref Tensor.
/// @param tns the @ref Tensor to be replaced.
/// @param new_tns the new @ref Tensor.
void Model::Impl::replace_tensor(Tensor *tns, Tensor *new_tns) {
    for (auto &user : this->tns_to_users[tns]) {
        for (auto &input : user->inputs) {
            if (input == tns) {
                input = new_tns;
            }
        }
        for (auto &output_ref : user->output_refs) {
            if (output_ref == tns) {
                output_ref = new_tns;
            }
        }
        this->tns_to_users[new_tns].insert(user);
    }
    this->tns_to_users.erase(tns);

    if (this->tns_to_producer.find(tns) != this->tns_to_producer.end()) {
        Op *producer = this->tns_to_producer[tns];
        for (auto &output : producer->outputs) {
            if (output == tns) {
                output = new_tns;
            }
        }
        this->tns_to_producer[new_tns] = producer;
        this->tns_to_producer.erase(tns);
    }
}

/// Delete a @ref Tensor from the model.
/// @param tns the @ref Tensor to be deleted.
void Model::Impl::delete_tensor(Tensor *tns) {
    // Should not delete if there is any user of this tensor.
    if (!this->is_no_user(tns)) {
        ERR(ModelError,
            "Cannot delete a tensor that has users. Use "
            "replace_tensor() first to replace the tensor with another one.");
    }
    // Should not delete if the producer still exists.
    if (this->get_producer(tns) != nullptr) {
        ERR(ModelError,
            "Cannot delete a tensor that has a producer. Use "
            "replace_tensor() or delete_op() first to delete the producer.");
    }
    // Remove the tensor from the model.
    bool is_found = false;
    auto it = this->tns_storage.begin();
    for (; it != this->tns_storage.end(); ++it) {
        if (it->get() == tns) {
            this->tns_storage.erase(it);
            is_found = true;
            break;
        }
    }
    if (!is_found) {
        ERR(ModelError, "the given Tensor is not found");
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
Model::Model(int rank_) : impl{std::make_unique<Model::Impl>()} {
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

void OpNode::remove_self() {
    // Remove self from users and producers.
    for (auto &user : this->users) {
        user->producers.erase(this);
    }
    for (auto &producer : this->producers) {
        producer->users.erase(this);
    }
    // Connect users and producers.
    for (auto &user : this->users) {
        for (auto &producer : this->producers) {
            user->producers.insert(producer);
            producer->users.insert(user);
        }
    }
}

std::string OpNode::get_name() const {
    std::stringstream name;
    for (auto &op : this->ops) {
        name << op->name << ";";
    }
    return name.str();
}

OpGraph::OpGraph(const Model &model) {
    if (!model.verify()) {
        ERR(ModelError, "Model verification failed");
    }
    this->create_nodes(model);
}

OpGraph::OpGraph(OpGraph &graph) {
    // Copy nodes_storage
    *this = graph;
}

OpGraph &OpGraph::operator=(const OpGraph &graph) {
    // Copy nodes_storage
    this->nodes_storage.clear();
    for (auto &node : graph.nodes_storage) {
        this->nodes_storage.emplace_back(std::make_unique<OpNode>());
        this->nodes_storage.back()->ops = node->ops;
        this->nodes_storage.back()->users = node->users;
        this->nodes_storage.back()->producers = node->producers;
    }
    return *this;
}

/// Traverse the model graph and merge Ops that one of them is the only
/// user of the other and the other is the only producer of the first.
///
/// @param model The @ref Model.
///
void OpGraph::create_nodes(const Model &model) {
    std::list<OpNode *> leaf_nodes;
    std::map<const Op *, OpNode *> op2node;
    // Initialize OpNode.
    OPGRAPH_DEBUG("initialize OpNode. ", model.impl->get_ops().size(), " ops");
    for (auto &op : model.impl->get_ops()) {
        this->nodes_storage.emplace_back(std::make_unique<OpNode>());
        this->nodes_storage.back()->ops.emplace_back(op);
        op2node[op] = this->nodes_storage.back().get();
        if (model.impl->get_user_ops(op).size() == 0) {
            leaf_nodes.emplace_back(this->nodes_storage.back().get());
        }
    }
    // Complete producers and users of OpNode.
    for (auto &node : this->nodes_storage) {
        // As nothing is merged yet, all OpNode should have only one Op.
        Op *op = node->ops[0];
        OPGRAPH_DEBUG("node ", op->name);
        for (auto &producer_op : model.impl->get_producer_ops(op)) {
            node->producers.insert(op2node[producer_op]);
            OPGRAPH_DEBUG("  producer ", producer_op->name);
        }
        for (auto &user_op : model.impl->get_user_ops(op)) {
            node->users.insert(op2node[user_op]);
            OPGRAPH_DEBUG("  user ", user_op->name);
        }
    }

    std::set<OpNode *> seen_nodes;

    // Remove virtual Ops.
    recursive_rm_virt(this->nodes_storage, seen_nodes, leaf_nodes);
    seen_nodes.clear();

    // Recreate leaf_nodes.
    leaf_nodes.clear();
    for (auto &node : this->nodes_storage) {
        if (node->users.empty()) {
            leaf_nodes.emplace_back(node.get());
        }
    }

    // Merge Ops.
    recursive_merge(this->nodes_storage, seen_nodes, leaf_nodes);
}

/// Helper of @ref create_nodes().
/// Traverse the model graph and remove virtual Ops that perform no computation.
///
/// @param nodes The list of @ref OpNode.
/// @param boundary_nodes The list of boundary @ref OpNode.
///
void OpGraph::recursive_rm_virt(std::list<std::unique_ptr<OpNode>> &nodes,
                                std::set<OpNode *> &seen_nodes,
                                const std::list<OpNode *> &boundary_nodes) {
    if (boundary_nodes.size() == 0) {
        return;
    }
    OPGRAPH_DEBUG("remove virtual ops");
    std::list<OpNode *> new_boundary_nodes;
    for (auto &boundary_node : boundary_nodes) {
        if (boundary_node->ops.size() == 0) {
            ERR(SchedulerError, "unexpected error: empty OpNode");
        } else if (boundary_node->ops.size() > 1) {
            ERR(SchedulerError, "unexpected error: multiple Ops in OpNode");
        }
        OPGRAPH_DEBUG("  boundary node");
        OPGRAPH_DEBUG("    op: ", boundary_node->get_name());
        for (auto &producer : boundary_node->producers) {
            // Exception: if any user of the producer (rather than the current
            // boundary_node) is unseen, we should not add the producer to the
            // next boundary.
            bool should_add = true;
            for (auto &user : producer->users) {
                if (user == boundary_node) {
                    continue;
                }
                if (seen_nodes.find(user) == seen_nodes.end()) {
                    should_add = false;
                    break;
                }
            }
            if (!should_add) {
                continue;
            }
            if (seen_nodes.find(producer) != seen_nodes.end()) {
                ERR(SchedulerError,
                    "unexpected error: circular dependency detected");
            }
            OPGRAPH_DEBUG("      added ", producer->get_name(),
                          " to next boundary");
            new_boundary_nodes.emplace_back(producer);
        }
        if (boundary_node->ops[0]->is_virtual()) {
            OPGRAPH_DEBUG("    remove op: ", boundary_node->get_name());
            // Remove this node from the graph.
            boundary_node->remove_self();
            // Remove this node from the list of nodes.
            auto it = std::find_if(
                nodes.begin(), nodes.end(),
                [boundary_node](const std::unique_ptr<OpNode> &node) {
                    return node.get() == boundary_node;
                });
            if (it == nodes.end()) {
                ERR(SchedulerError, "unexpected error");
            }
            nodes.erase(it);
            OPGRAPH_DEBUG("      nodes.size() ", nodes.size());
        } else {
            seen_nodes.insert(boundary_node);
        }
    }
    recursive_rm_virt(nodes, seen_nodes, new_boundary_nodes);
}

/// Helper of @ref create_nodes().
/// Traverse the model graph and merge pairs of Ops that are the only user
/// and producer of each other.
///
/// @param nodes The list of @ref OpNode.
/// @param seen_nodes The set of @ref OpNode that have been seen.
/// @param boundary_nodes The list of boundary @ref OpNode.
///
void OpGraph::recursive_merge(std::list<std::unique_ptr<OpNode>> &nodes,
                              std::set<OpNode *> &seen_nodes,
                              const std::list<OpNode *> &boundary_nodes) {
    if (boundary_nodes.size() == 0) {
        return;
    }
    OPGRAPH_DEBUG("merge ops");
    std::list<OpNode *> new_boundary_nodes;
    for (auto &boundary_node : boundary_nodes) {
        OPGRAPH_DEBUG("  boundary node");
        OPGRAPH_DEBUG("    op: ", boundary_node->get_name());
        if (boundary_node->producers.size() == 0) {
            // This node is a root.
            seen_nodes.insert(boundary_node);
            OPGRAPH_DEBUG("    root");
            continue;
        }
        // Add all producers of this node to the next boundary.
        for (auto &producer : boundary_node->producers) {
            // Exception: if any user of the producer (rather than the current
            // boundary_node) is unseen, we should not add the producer to the
            // next boundary.
            bool should_add = true;
            for (auto &user : producer->users) {
                if (user == boundary_node) {
                    continue;
                }
                if (seen_nodes.find(user) == seen_nodes.end()) {
                    should_add = false;
                    break;
                }
            }
            if (!should_add) {
                continue;
            }
            if (seen_nodes.find(producer) != seen_nodes.end()) {
                ERR(SchedulerError,
                    "unexpected error: circular dependency detected");
            }
            new_boundary_nodes.emplace_back(producer);
        }
        OpNode *merge_candidate = nullptr;
        if (boundary_node->producers.size() > 1) {
            // This node has multiple producers. We can merge only if one
            // producer depends on all other producers.
            for (auto &producer : boundary_node->producers) {
                bool depends_on_all = true;
                for (auto &other_producer : boundary_node->producers) {
                    if (other_producer == producer) {
                        continue;
                    }
                    if (!this->depends_on(producer, other_producer)) {
                        depends_on_all = false;
                        break;
                    }
                }
                if (depends_on_all) {
                    merge_candidate = producer;
                    break;
                }
            }
            if (merge_candidate == nullptr) {
                // At least one producer does not depend on others.
                // Cannot merge.
                seen_nodes.insert(boundary_node);
                OPGRAPH_DEBUG("    multiple producers");
                continue;
            }
        } else {
            // This node has only one producer.
            merge_candidate = *(boundary_node->producers.begin());
        }
        if (merge_candidate->users.size() == 0) {
            ERR(SchedulerError, "unexpected error: graph is incomplete");
        }
        if (merge_candidate->users.size() > 1) {
            // The candidate has multiple users. We can merge only if all
            // other users depend on the current boundary_node.
            bool depends_on_one = true;
            for (auto &user : merge_candidate->users) {
                if (user == boundary_node) {
                    continue;
                }
                if (!this->depends_on(user, boundary_node)) {
                    depends_on_one = false;
                    break;
                }
            }
            if (!depends_on_one) {
                // At least one user does not depend on the boundary_node.
                // Cannot merge.
                seen_nodes.insert(boundary_node);
                OPGRAPH_DEBUG("    multiple users");
                continue;
            }
        }
        // We can merge the two nodes.
        // Merge `boundary_node` into `merge_candidate`.
        OPGRAPH_DEBUG("  merge: ", merge_candidate->get_name(), " -> ",
                      boundary_node->get_name());
        auto &ops = boundary_node->ops;
        merge_candidate->ops.insert(merge_candidate->ops.end(), ops.begin(),
                                    ops.end());
        for (auto &user : boundary_node->users) {
            user->producers.erase(boundary_node);
            user->producers.insert(merge_candidate);
            merge_candidate->users.insert(user);
        }
        for (auto &producer : boundary_node->producers) {
            if (producer == merge_candidate) {
                continue;
            }
            producer->users.erase(boundary_node);
            producer->users.insert(merge_candidate);
            merge_candidate->producers.insert(producer);
        }
        merge_candidate->users.erase(boundary_node);

        // Remove `boundary_node` from `nodes`.
        auto it =
            std::find_if(nodes.begin(), nodes.end(),
                         [boundary_node](const std::unique_ptr<OpNode> &node) {
                             return node.get() == boundary_node;
                         });
        if (it == nodes.end()) {
            ERR(SchedulerError, "unexpected error");
        }
        nodes.erase(it);

        // Since producer is already in the next boundary and boundary_node is
        // merged into producer, we don't need to add anything to
        // seen_nodes here.
    }
    recursive_merge(nodes, seen_nodes, new_boundary_nodes);
}

OpNode *OpGraph::break_node(OpNode *node, int op_idx) {
    if (op_idx == 0) {
        return node;
    }
    if (op_idx < 0 || op_idx >= (int)node->ops.size()) {
        ERR(SchedulerError, "unexpected error: op_idx out of range");
    }
    this->nodes_storage.emplace_back(std::make_unique<OpNode>());
    OpNode *new_node = this->nodes_storage.back().get();
    new_node->ops.insert(new_node->ops.end(), node->ops.begin() + op_idx,
                         node->ops.end());
    new_node->users = node->users;
    new_node->producers.insert(node);
    for (auto &user : node->users) {
        user->producers.erase(node);
        user->producers.insert(new_node);
    }
    node->ops.erase(node->ops.begin() + op_idx, node->ops.end());
    node->users.clear();
    node->users.insert(new_node);
    return new_node;
}

/// Check dependencies between two @ref OpNode.
///
/// @param node1 The first @ref OpNode.
/// @param node2 The second @ref OpNode.
/// @return True if @p node1 depends on @p node2.
bool OpGraph::depends_on(OpNode *node1, OpNode *node2) const {
    if (node1 == node2) {
        return false;
    }
    std::set<OpNode *> seen_nodes;
    std::list<OpNode *> boundary_nodes;
    boundary_nodes.emplace_back(node1);
    while (boundary_nodes.size() > 0) {
        std::list<OpNode *> new_boundary_nodes;
        for (auto &boundary_node : boundary_nodes) {
            if (boundary_node == node2) {
                return true;
            }
            for (auto &producer : boundary_node->producers) {
                if (seen_nodes.find(producer) != seen_nodes.end()) {
                    continue;
                }
                new_boundary_nodes.emplace_back(producer);
            }
        }
        boundary_nodes = new_boundary_nodes;
    }
    return false;
}

nlohmann::json to_json(const TensorBuf &tensor_buf) {
    nlohmann::json j;
    j["Id"] = tensor_buf.id;
    j["Bytes"] = tensor_buf.bytes;
    return j;
}

nlohmann::json to_json(const Tensor &tensor) {
    nlohmann::json j;
    j["Id"] = tensor.id;
    j["TensorBuf"] = to_json(*(tensor.buf));
    j["TensorType"] = tensor.type.type_str();
    j["Shape"] = tensor.shape.serialize();
    j["Strides"] = tensor.ldims.serialize();
    j["Offsets"] = tensor.offs.serialize();
    if (tensor.imported_rank >= 0) {
        j["ImportedRank"] = tensor.imported_rank;
    }
    return j;
}

nlohmann::json to_json(const OpArg &op_arg) {
    nlohmann::json j;
    j["Type"] = op_arg.type.name;
    if (op_arg.type == OP_ARG_TENSOR) {
        Tensor *tns;
        op_arg.get(&tns);
        j["Value"] = tns->id;
    } else if (op_arg.type == OP_ARG_FLOAT) {
        float val;
        op_arg.get(&val);
        j["Value"] = val;
    } else if (op_arg.type == OP_ARG_INT) {
        int val;
        op_arg.get(&val);
        j["Value"] = val;
    } else if (op_arg.type == OP_ARG_BOOL) {
        bool val;
        op_arg.get(&val);
        j["Value"] = val;
    } else if (op_arg.type == OP_ARG_INT64) {
        long long int val;
        op_arg.get(&val);
        j["Value"] = val;
    } else if (op_arg.type == OP_ARG_UINT64) {
        uint64_t val;
        op_arg.get(&val);
        j["Value"] = val;
    } else if (op_arg.type == OP_ARG_DIMS) {
        Dims dims;
        op_arg.get(&dims);
        j["Value"] = dims.serialize();
    } else {
        throw std::runtime_error("unexpected OpArg: " +
                                 std::string(op_arg.type.name));
    }
    return j;
}

nlohmann::json to_json(const Op &op) {
    nlohmann::json j;
    j["Type"] = op.type.name;
    j["PrecisionType"] = op.prec_type;
    j["InputTensors"] = nlohmann::json();
    for (auto tensor : op.inputs) {
        j["InputTensors"].emplace_back(to_json(*tensor));
    }
    j["OutputTensors"] = nlohmann::json();
    for (auto tensor : op.inputs) {
        j["OutputTensors"].emplace_back(to_json(*tensor));
    }
    j["OutputRefTensors"] = nlohmann::json();
    for (auto tensor : op.inputs) {
        j["OutputRefTensors"].emplace_back(to_json(*tensor));
    }
    j["Args"] = nlohmann::json();
    for (auto arg : op.args.get_args()) {
        j["Args"].emplace_back(to_json(arg));
    }
    return j;
}

nlohmann::json to_json(const OpNode &node,
                       const std::map<const OpNode *, size_t> &node2id) {
    nlohmann::json j;
    j["Id"] = node2id.at(&node);
    j["Ops"] = nlohmann::json();
    for (auto op : node.ops) {
        j["Ops"].emplace_back(to_json(*op));
    }
    j["ConsumerNodeIds"] = nlohmann::json();
    for (auto user : node.users) {
        j["ConsumerNodeIds"].emplace_back(node2id.at(user));
    }
    j["ProducerNodeIds"] = nlohmann::json();
    for (auto producer : node.producers) {
        j["ProducerNodeIds"].emplace_back(node2id.at(producer));
    }
    return j;
}

nlohmann::json to_json(const OpGraph &opgraph) {
    size_t id = 0;
    std::map<const OpNode *, size_t> node2id;
    for (const auto &node : opgraph.get_nodes()) {
        node2id[node.get()] = id++;
    }
    nlohmann::json j;
    j["Nodes"] = nlohmann::json();
    for (const auto &node : opgraph.get_nodes()) {
        j["Nodes"].emplace_back(to_json(*node, node2id));
    }
    return j;
}

std::string OpGraph::serialize(int indent) const {
    auto j = to_json(*this);
    return j.dump(indent);
}

}  // namespace ark
