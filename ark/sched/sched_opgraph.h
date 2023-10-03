// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef _ARK_SCHED_OPGRAPH_H_
#define _ARK_SCHED_OPGRAPH_H_

#include <list>
#include <memory>
#include <set>
#include <vector>

#include "ops/ops_common.h"

namespace ark {

class Model;

/// A node in the @ref OpGraph.
class OpNode {
   public:
    /// Construct an empty @ref OpNode.
    OpNode(){};

    /// Destruct an @ref OpNode.
    ~OpNode(){};

    /// The list of @ref Op that this @ref OpNode contains. Sorted in the
    /// execution order.
    std::vector<Op *> ops;

    /// The list of @ref OpNode that depends on this @ref OpNode.
    std::set<OpNode *> users;

    /// The list of @ref OpNode that this @ref OpNode depends on.
    std::set<OpNode *> producers;

    /// Remove this @ref OpNode from the graph.
    void remove_self();

    /// Get the name of this @ref OpNode.
    std::string get_name() const;
};

/// A directed acyclic graph of operators.
///
/// The @ref OpGraph is a DAG of operators, where each @ref OpNode is a
/// node. The edges are the dependencies between @ref OpNode.
///
class OpGraph {
   public:
    /// Construct an @ref OpGraph from a @ref Model.
    ///
    /// The @ref OpGraph is a DAG of operators, where each @ref OpNode is a
    /// node. The edges are the dependencies between @ref OpNode.
    ///
    /// @param model The @ref Model.
    ///
    OpGraph(const Model &model);

    /// Construct an @ref OpGraph from another @ref OpGraph.
    OpGraph(OpGraph &graph);

    /// Construct an empty @ref OpGraph.
    OpGraph(){};

    /// Destruct an @ref OpGraph.
    ~OpGraph(){};

    OpGraph &operator=(const OpGraph &);

    /// Get the @ref OpNode list.
    /// @return The @ref OpNode list.
    const std::list<std::unique_ptr<OpNode>> &get_nodes() const {
        return this->nodes_storage;
    }

    /// Break a @ref OpNode into two @ref OpNode.
    ///
    /// The original node will have the first @p op_idx ops, and the new node
    /// will have the rest.
    ///
    /// @param node The @ref OpNode to break.
    /// @param op_idx The index of the first op in the new @ref OpNode.
    /// @return The new @ref OpNode.
    OpNode *break_node(OpNode *node, int op_idx);

   private:
    std::list<std::unique_ptr<OpNode>> nodes_storage;

    void create_nodes(const Model &model);
    static void recursive_rm_virt(std::list<std::unique_ptr<OpNode>> &nodes,
                                  std::set<OpNode *> &seen_nodes,
                                  const std::list<OpNode *> &boundary_nodes);
    static void recursive_merge(std::list<std::unique_ptr<OpNode>> &nodes,
                                std::set<OpNode *> &seen_nodes,
                                const std::list<OpNode *> &boundary_nodes);
};

}  // namespace ark

#endif  // _ARK_SCHED_OPGRAPH_H_
