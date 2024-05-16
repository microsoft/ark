// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_GRAPH_IMPL_HPP_
#define ARK_MODEL_GRAPH_IMPL_HPP_

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

class ModelGraph::Impl {
   public:
    Impl() : compressed_(false) {};

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
        if (count > 0) {
            name_copy += "_" + std::to_string(count);
        }
        op_names_.insert(name_copy);
        op->set_name(name_copy);
        add_op(op);
        return op;
    }

    void compress_nodes();

    bool compressed() const { return compressed_; }

    bool verify() const;

    std::string serialize(bool pretty = true) const;

    std::vector<ModelNodeRef> nodes() const;

   private:
    ModelNodeRef add_op(ModelOpRef op);

    void remove_node(ModelNodeRef node);

    bool depends_on(ModelNodeRef node1, ModelNodeRef node2) const;

    void recursive_remove_virtual_nodes();

    void recursive_remove_virtual_nodes(
        UniqueList<ModelNodeRef> &seen_nodes,
        const std::vector<ModelNodeRef> &boundary_nodes);

    void recursive_merge_nodes();

    void recursive_merge_nodes(UniqueList<ModelNodeRef> &seen_nodes,
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

    /// True if `compress_nodes` has been called.
    bool compressed_;
};

}  // namespace ark

#endif  // ARK_MODEL_GRAPH_IMPL_HPP_
