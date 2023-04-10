// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#ifndef ARK_KAHYPAR_H_
#define ARK_KAHYPAR_H_

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "ark/logging.h"
#include "third_party/kahypar/include/libkahypar.h"

namespace ark {

template <typename ItemType> class KahyparGraph
{
  public:
    // Constructor.
    KahyparGraph()
    {
        char *ark_root = getenv("ARK_ROOT");
        if (ark_root == 0) {
            LOGERR("ARK_ROOT is not set.");
            throw;
        }
        // Get the context file path.
        ctx_file_path =
            std::string(ark_root) + "/cut_kKaHyPar_dissertation.ini";
    };

    // Copy constructors.
    KahyparGraph(const KahyparGraph &) = delete;
    KahyparGraph &operator=(const KahyparGraph &) = delete;

    // Add nodes from an initializer list.
    void add_nodes(int weight, std::initializer_list<ItemType *> init)
    {
        for (ItemType *item : init) {
            add_node(weight, item);
        }
        resize_wm();
    }

    // Add nodes from a generator.
    void add_nodes(int weight, std::function<ItemType *()> gen)
    {
        ItemType *item;
        while ((item = gen()) != nullptr) {
            add_node(weight, item);
        }
        resize_wm();
    }

    // Add an edge which consists of nodes from an initializer list.
    void add_edge(int weight, std::initializer_list<ItemType *> init)
    {
        std::vector<size_t> nodes;
        for (ItemType *item : init) {
            nodes.push_back(item2node.at(*item));
        }
        if (nodes.size() > 0) {
            add_nodes_into_edge(weight, nodes);
        }
    }

    // Add an edge which consists of nodes from a generator.
    void add_edge(int weight, std::function<ItemType *()> gen)
    {
        std::vector<size_t> nodes;
        ItemType *item;
        while ((item = gen()) != nullptr) {
            nodes.push_back(item2node.at(*item));
        }
        if (nodes.size() > 0) {
            add_nodes_into_edge(weight, nodes);
        }
    }

    // Finalize the graph and partition it into `num_part` parts.
    std::vector<std::vector<std::pair<ItemType *, ItemType *>>> &partition(
        int num_part)
    {
        // Verify the graph.
        size_t num_nodes = nws.size();
        if (num_nodes != items.size() || num_nodes != wm.size()) {
            LOGERR("Unexpected error.");
            throw;
        }
        for (auto wv : wm) {
            if (num_nodes != wv.size()) {
                LOGERR("Unexpected error.");
                throw;
            }
        }
        if (ews.size() == 0) {
            // If there is no edges, add a virtual edge with weight 0.
            eis.push_back(edges.size());
            ews.push_back(0);
            for (size_t i = 0; i < num_nodes; ++i) {
                edges.push_back(i);
            }
        }
        // Finalize hypergraph.
        eis.push_back(edges.size());
        // Run partition.
        kahypar_context_t *ctx = kahypar_context_new();
        kahypar_configure_context_from_file(ctx, ctx_file_path.c_str());
        kahypar_hyperedge_weight_t objective = 0;
        std::vector<kahypar_partition_id_t> partition(num_nodes, -1);
        kahypar_partition(num_nodes, ews.size(), 0.03, num_part,
                          (kahypar_hypernode_weight_t *)&nws[0],
                          (kahypar_hyperedge_weight_t *)&ews[0],
                          (size_t *)&eis[0],
                          (kahypar_hyperedge_id_t *)&edges[0], &objective, ctx,
                          partition.data());
        // Group nodes in the same part.
        std::vector<std::vector<size_t>> nparts(num_part);
        std::set<size_t> dup_chk;
        for (size_t i = 0; i < partition.size(); ++i) {
            nparts[partition[i]].push_back(i);
            dup_chk.insert(i);
        }
        // Correctness check: duplicated nodes.
        if (partition.size() != dup_chk.size()) {
            LOGERR("Found duplicated nodes.");
            throw;
        }
        // Correctness check: node count.
        if (partition.size() != num_nodes) {
            LOGERR("Node count mismatch.");
            throw;
        }
        //
        parts.resize(nparts.size());
        std::vector<std::pair<size_t, size_t>> cands;
        for (size_t i = 0; i < nparts.size(); ++i) {
            auto &npart = nparts.at(i);
            auto &part = parts.at(i);
            // Sort `npart` in increasing order of items.
            std::sort(npart.begin(), npart.end(), [&](size_t a, size_t b) {
                return *items[a] < *items[b];
            });
            //
            while (npart.size() > 1) {
                // Get candidate nodes which have the maximum weight.
                cands.clear();
                int mw = INT_MIN;
                for (auto it0 = npart.begin(); it0 != npart.end(); ++it0) {
                    for (auto it1 = next(it0); it1 != npart.end(); ++it1) {
                        int mvw = wm[*it0][*it1];
                        if (mvw > mw) {
                            mw = mvw;
                            cands.clear();
                        }
                        if (mvw >= mw) {
                            cands.emplace_back(*it0, *it1);
                        }
                    }
                }
                // If there are multiple candidates, select the smallest one.
                auto &select = cands[0];
                // Put corresponding items of the selected nodes into `part`.
                part.emplace_back(items[select.first].get(),
                                  items[select.second].get());
                // Exclude selected nodes from `npart`.
                auto it = npart.begin();
                int cnt = 0;
                while (it != npart.end()) {
                    if (*it == select.first || *it == select.second) {
                        it = npart.erase(it);
                        ++cnt;
                        if (cnt == 2) {
                            break;
                        }
                    } else {
                        ++it;
                    }
                }
            }
            // Put the last single node into `part` without pair.
            if (npart.size() == 1) {
                part.emplace_back(items[npart[0]].get(), nullptr);
            }
        }
        return parts;
    }

    int get_num_nodes()
    {
        return nws.size();
    }

  private:
    // Add a new node for `item`.
    void add_node(int weight, ItemType *item)
    {
        item2node[*item] = nws.size();
        items.push_back(std::unique_ptr<ItemType>(item));
        nws.push_back(weight);
    }

    // Add `nodes` as a new edge.
    void add_nodes_into_edge(int weight, std::vector<size_t> &nodes)
    {
        eis.push_back(edges.size());
        ews.push_back(weight);
        edges.insert(edges.end(), nodes.begin(), nodes.end());
        // Update the weight matrix.
        for (auto it0 = nodes.begin(); it0 != nodes.end(); ++it0) {
            for (auto it1 = next(it0); it1 != nodes.end(); ++it1) {
                wm[*it0][*it1] += weight;
                wm[*it1][*it0] += weight;
            }
        }
    }

    // Resize the weight matrix when the number of nodes is changed.
    void resize_wm()
    {
        for (auto &wv : wm) {
            wv.resize(nws.size(), 0);
        }
        size_t nws_size = nws.size();
        size_t wm_size = wm.size();
        for (size_t i = 0; i < nws_size - wm_size; ++i) {
            wm.emplace_back(nws_size, 0);
        }
    }

    static const int INT_MIN = std::numeric_limits<int>::min();

    // KaHyPar context file path.
    std::string ctx_file_path;
    // KaHyPar hypergraph representation.
    std::vector<int> nws;
    std::vector<int> ews;
    std::vector<size_t> eis;
    std::vector<unsigned int> edges;
    std::vector<std::unique_ptr<ItemType>> items;
    std::map<ItemType, size_t> item2node;
    // Edge weight matrix.
    std::vector<std::vector<int>> wm;
    // Partition result.
    std::vector<std::vector<std::pair<ItemType *, ItemType *>>> parts;
};

} // namespace ark

#endif // ARK_KAHYPAR_H_