// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "sched_branch.h"
#include "logging.h"
#include <algorithm>
#include <unordered_set>
#include <vector>

namespace ark {

class SchedBranch::Impl
{
  private:
    struct Tile
    {
        int opseq_id;
        int tile_id;
        int sm_id;
        int warp_id_begin;
        int warp_id_end;
    };

    static bool cmp_tile(const Tile &a, const Tile &b);

    std::vector<Tile> tiles;

  public:
    Impl();
    ~Impl();

  protected:
    void add_tile(int opseq_id, int tile_id, int sm_id, int warp_id_begin,
                  int warp_id_end);
    std::vector<Branch> get_branches();

    friend class SchedBranch;
};

SchedBranch::Impl::Impl()
{
}

SchedBranch::Impl::~Impl()
{
}

void SchedBranch::Impl::add_tile(int opseq_id, int tile_id, int sm_id,
                                 int warp_id_begin, int warp_id_end)
{
    Tile tile{opseq_id, tile_id, sm_id, warp_id_begin, warp_id_end};
    tiles.emplace_back(tile);
}

bool SchedBranch::Impl::cmp_tile(const Tile &a, const Tile &b)
{
    if (a.opseq_id != b.opseq_id) {
        return a.opseq_id < b.opseq_id;
    } else if (a.sm_id != b.sm_id) {
        return a.sm_id < b.sm_id;
    } else if (a.warp_id_begin != b.warp_id_begin) {
        return a.warp_id_begin < b.warp_id_begin;
    } else if (a.warp_id_end != b.warp_id_end) {
        return a.warp_id_end < b.warp_id_end;
    } else {
        return a.tile_id < b.tile_id;
    }
}

std::vector<Branch> SchedBranch::Impl::get_branches()
{
    std::vector<Branch> branches;

    std::sort(tiles.begin(), tiles.end(), cmp_tile);

    std::unordered_set<size_t> merged_tile_indices;

    for (size_t i = 0; i < tiles.size(); ++i) {
        if (merged_tile_indices.find(i) != merged_tile_indices.end()) {
            continue;
        }
        Tile current_tile = tiles[i];
        Branch new_branch;
        new_branch.opseq_id = current_tile.opseq_id;
        new_branch.sm_id_begin = current_tile.sm_id;
        new_branch.sm_id_end = current_tile.sm_id + 1;
        new_branch.warp_id_begin = current_tile.warp_id_begin;
        new_branch.warp_id_end = current_tile.warp_id_end;
        new_branch.tile_id_begin = current_tile.tile_id;
        new_branch.tile_id_last = current_tile.tile_id;
        new_branch.tile_id_diff = 0;
        new_branch.num_warps_per_tile =
            current_tile.warp_id_end - current_tile.warp_id_begin;

        merged_tile_indices.emplace(i);

        int current_warp_id_end = new_branch.warp_id_end;

        for (size_t j = i + 1; j < tiles.size(); j++) {
            if (merged_tile_indices.find(j) != merged_tile_indices.end()) {
                continue;
            }
            Tile next_tile = tiles[j];
            if (next_tile.opseq_id != current_tile.opseq_id) {
                // Scheduling another opseq. There is no more tile to merge.
                // Break.
                break;
            } else {
                // Scheduling the same opseq.
                if (next_tile.warp_id_end - next_tile.warp_id_begin !=
                    new_branch.num_warps_per_tile) {
                    // The same opseq should have the same number of warps per
                    // tile.
                    LOGERR("invalid num_warps_per_tile: ",
                           next_tile.warp_id_end - next_tile.warp_id_begin,
                           ", expected: ", new_branch.num_warps_per_tile);
                }
                if (next_tile.sm_id == new_branch.sm_id_end - 1) {
                    // Scheduling the same opseq on the same SM as the previous
                    // tile.
                    if (next_tile.warp_id_begin >= new_branch.warp_id_begin &&
                        next_tile.warp_id_begin < current_warp_id_end) {
                        // Scheduling another tile from the same opseq on the
                        // same SM and warp. This should be handled in another
                        // branch. Skip here.
                        continue;
                    } else if (next_tile.warp_id_begin == current_warp_id_end) {
                        // Contiguous warps. Try merge.
                        if (new_branch.tile_id_diff == 0) {
                            // Diff is undetermined yet. Set it and merge.
                            new_branch.tile_id_diff =
                                next_tile.tile_id - new_branch.tile_id_last;
                            current_warp_id_end = next_tile.warp_id_end;
                            new_branch.tile_id_last = next_tile.tile_id;
                            if (new_branch.warp_id_end < current_warp_id_end) {
                                new_branch.warp_id_end = current_warp_id_end;
                            }
                        } else if (new_branch.tile_id_diff ==
                                   next_tile.tile_id -
                                       new_branch.tile_id_last) {
                            // Diff is the same as the previous tile. Merge.
                            current_warp_id_end = next_tile.warp_id_end;
                            new_branch.tile_id_last = next_tile.tile_id;
                            if (new_branch.warp_id_end < current_warp_id_end) {
                                new_branch.warp_id_end = current_warp_id_end;
                            }
                        } else {
                            // Diff is different from the previous tile. Break.
                            break;
                        }
                        merged_tile_indices.emplace(j);
                        continue;
                    } else {
                        // Non-contiguous warps. Break.
                        break;
                    }
                } else if (next_tile.sm_id == new_branch.sm_id_end) {
                    // Scheduling the same opseq on the next SM.
                    if (next_tile.warp_id_begin != new_branch.warp_id_begin) {
                        // Using different warp IDs from the next SM. Break.
                        break;
                    } else {
                        // Contiguous SMs and using the same warp IDs. Try
                        // merge.
                        if (new_branch.tile_id_diff == 0) {
                            // Diff is undetermined yet. Set it and merge.
                            new_branch.tile_id_diff =
                                next_tile.tile_id - new_branch.tile_id_last;
                            current_warp_id_end = next_tile.warp_id_end;
                            new_branch.tile_id_last = next_tile.tile_id;
                            new_branch.sm_id_end = next_tile.sm_id + 1;
                            if (new_branch.warp_id_end < current_warp_id_end) {
                                new_branch.warp_id_end = current_warp_id_end;
                            }
                        } else if (new_branch.tile_id_diff ==
                                   next_tile.tile_id -
                                       new_branch.tile_id_last) {
                            // Diff is the same as the previous tile. Merge.
                            current_warp_id_end = next_tile.warp_id_end;
                            new_branch.tile_id_last = next_tile.tile_id;
                            new_branch.sm_id_end = next_tile.sm_id + 1;
                            if (new_branch.warp_id_end < current_warp_id_end) {
                                new_branch.warp_id_end = current_warp_id_end;
                            }
                        } else {
                            // Diff is different from the previous tile. Break.
                            break;
                        }
                        merged_tile_indices.emplace(j);
                        continue;
                    }
                } else {
                    // Non-contiguous SMs. Break.
                    break;
                }
            }
        }
        branches.emplace_back(new_branch);
    }

    return branches;
}

SchedBranch::SchedBranch()
{
    this->impl = std::make_unique<Impl>();
}

SchedBranch::~SchedBranch()
{
}

void SchedBranch::add(int opseq_id, int tile_id, int sm_id, int warp_id_begin,
                      int warp_id_end)
{
    this->impl->add_tile(opseq_id, tile_id, sm_id, warp_id_begin, warp_id_end);
}

std::vector<Branch> SchedBranch::get_branches()
{
    return this->impl->get_branches();
}

} // namespace ark
