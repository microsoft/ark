// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_GRAPH_IMPL_HPP_
#define ARK_MODEL_GRAPH_IMPL_HPP_

#include <list>
#include <map>
#include <set>
#include <tuple>
#include <vector>

#include "ark/dims.hpp"
#include "ark/model_graph.hpp"
#include "model_json.hpp"
#include "model_op.hpp"
#include "unique_list.hpp"
#include "utils/utils_string.hpp"

namespace ark {

class ModelGraphContextStack {
   private:
    std::map<std::string, std::list<std::shared_ptr<Json>>> storage_;

   public:
    ModelGraphContextStack() = default;

    ModelGraphContextStack(const ModelGraphContextStack &other);

    ~ModelGraphContextStack() = default;

    void push(const std::string &key, const Json &value);

    void pop(const std::string &key);

    Json get_context(const std::string &key) const;

    std::map<std::string, Json> get_context_all() const;
};

class ModelGraph::Impl {
   public:
    Impl(int rank, int world_size)
        : rank_(rank),
          world_size_(world_size),
          compressed_(false),
          context_stack_(std::make_shared<ModelGraphContextStack>()){};

    Impl(const Impl &other);

    Impl &operator=(const Impl &other);

    template <typename T, typename... Args>
    ModelOpRef create_op(const std::string &name, Args &&... args) {
        ModelOpRef op = std::make_shared<T>(std::forward<Args>(args)...);
        std::string name_copy;
        if (name.empty()) {
            name_copy = pascal_to_snake(op->type()->type_name());
        } else {
            name_copy = name;
        }
        size_t count = op_names_.count(name_copy);
        op_names_.insert(name_copy);
        if (count > 0) {
            name_copy += "_" + std::to_string(count);
        }
        op->set_name(name_copy);
        add_op(op);
        return op;
    }

    int rank() const { return rank_; }

    int world_size() const { return world_size_; }

    void compress_nodes();

    bool compressed() const { return compressed_; }

    bool verify() const;

    Json get_context(const std::string &key) const;

    std::string serialize(bool pretty = true) const;

    std::vector<ModelNodeRef> nodes() const;

   private:
    ModelNodeRef add_op(ModelOpRef op);

    void remove_node(ModelNodeRef node);

    void recursive_remove_virtual_nodes();

    void recursive_remove_virtual_nodes(
        UniqueList<ModelNodeRef> &seen_nodes,
        const std::vector<ModelNodeRef> &boundary_nodes);

    Json to_json(const ModelNodeRef &node) const;

    /// The list of @ref ModelNode in the graph.
    UniqueList<ModelNodeRef> nodes_;

    /// The set of used names of @ref ModelOp.
    std::multiset<std::string> op_names_;

    /// The mapping from @ref ModelTensor to the @ref ModelOp that produces it.
    std::map<ModelTensorRef, ModelOpRef> tensor_to_producer_op_;

    /// The mapping from @ref ModelOp to the @ref ModelNode that contains it.
    std::map<ModelOpRef, ModelNodeRef> op_to_node_;

    /// Rank
    int rank_;

    /// World size
    int world_size_;

    /// True if `compress_nodes` has been called.
    bool compressed_;

   protected:
    friend class ModelContextManager;

    /// Graph context stack.
    std::shared_ptr<ModelGraphContextStack> context_stack_;
};

}  // namespace ark

#endif  // ARK_MODEL_GRAPH_IMPL_HPP_
