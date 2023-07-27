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
    struct UnitOp
    {
        int opseq_id;
        int uop_id;
        int sm_id;
        int warp_id_begin;
        int warp_id_end;
    };

    static bool cmp_uop(const UnitOp &a, const UnitOp &b);

    std::vector<UnitOp> uops;

  public:
    Impl();
    ~Impl();

  protected:
    void add(int opseq_id, int uop_id, int sm_id, int warp_id_begin,
             int warp_id_end);
    void clear();
    std::vector<Branch> get_branches();

    friend class SchedBranch;
};

SchedBranch::Impl::Impl()
{
}

SchedBranch::Impl::~Impl()
{
}

void SchedBranch::Impl::add(int opseq_id, int uop_id, int sm_id,
                            int warp_id_begin, int warp_id_end)
{
    UnitOp uop{opseq_id, uop_id, sm_id, warp_id_begin, warp_id_end};
    uops.emplace_back(uop);
}

void SchedBranch::Impl::clear()
{
    uops.clear();
}

bool SchedBranch::Impl::cmp_uop(const UnitOp &a, const UnitOp &b)
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
        return a.uop_id < b.uop_id;
    }
}

std::vector<Branch> SchedBranch::Impl::get_branches()
{
    std::vector<Branch> branches;

    std::sort(uops.begin(), uops.end(), cmp_uop);

    std::unordered_set<size_t> merged_uop_indices;

    for (size_t i = 0; i < uops.size(); ++i) {
        if (merged_uop_indices.find(i) != merged_uop_indices.end()) {
            continue;
        }
        UnitOp current_uop = uops[i];
        Branch new_branch;
        new_branch.opseq_id = current_uop.opseq_id;
        new_branch.sm_id_begin = current_uop.sm_id;
        new_branch.sm_id_end = current_uop.sm_id + 1;
        new_branch.warp_id_begin = current_uop.warp_id_begin;
        new_branch.warp_id_end = current_uop.warp_id_end;
        new_branch.uop_id_begin = current_uop.uop_id;
        new_branch.uop_id_last = current_uop.uop_id;
        new_branch.uop_id_diff = 0;
        new_branch.num_warps_per_uop =
            current_uop.warp_id_end - current_uop.warp_id_begin;

        merged_uop_indices.emplace(i);

        int current_warp_id_end = new_branch.warp_id_end;

        for (size_t j = i + 1; j < uops.size(); j++) {
            if (merged_uop_indices.find(j) != merged_uop_indices.end()) {
                continue;
            }
            UnitOp next_uop = uops[j];
            if (next_uop.opseq_id != current_uop.opseq_id) {
                // Scheduling another opseq. There is no more uop to merge.
                // Break.
                break;
            } else {
                // Scheduling the same opseq.
                if (next_uop.warp_id_end - next_uop.warp_id_begin !=
                    new_branch.num_warps_per_uop) {
                    // The same opseq should have the same number of warps per
                    // uop.
                    LOGERR("invalid num_warps_per_uop: ",
                           next_uop.warp_id_end - next_uop.warp_id_begin,
                           ", expected: ", new_branch.num_warps_per_uop);
                }
                if (next_uop.sm_id == new_branch.sm_id_end - 1) {
                    // Scheduling the same opseq on the same SM as the previous
                    // uop.
                    if (next_uop.warp_id_begin >= new_branch.warp_id_begin &&
                        next_uop.warp_id_begin < current_warp_id_end) {
                        // Scheduling another uop from the same opseq on the
                        // same SM and warp. This should be handled in another
                        // branch. Skip here.
                        continue;
                    } else if (next_uop.warp_id_begin == current_warp_id_end) {
                        // Contiguous warps. Try merge.
                        if (new_branch.uop_id_diff == 0) {
                            // Diff is undetermined yet. Set it and merge.
                            new_branch.uop_id_diff =
                                next_uop.uop_id - new_branch.uop_id_last;
                            current_warp_id_end = next_uop.warp_id_end;
                            new_branch.uop_id_last = next_uop.uop_id;
                            if (new_branch.warp_id_end < current_warp_id_end) {
                                new_branch.warp_id_end = current_warp_id_end;
                            }
                        } else if (new_branch.uop_id_diff ==
                                   next_uop.uop_id - new_branch.uop_id_last) {
                            // Diff is the same as the previous uop. Merge.
                            current_warp_id_end = next_uop.warp_id_end;
                            new_branch.uop_id_last = next_uop.uop_id;
                            if (new_branch.warp_id_end < current_warp_id_end) {
                                new_branch.warp_id_end = current_warp_id_end;
                            }
                        } else {
                            // Diff is different from the previous uop. Break.
                            break;
                        }
                        merged_uop_indices.emplace(j);
                        continue;
                    } else {
                        // Non-contiguous warps. Break.
                        break;
                    }
                } else if (next_uop.sm_id == new_branch.sm_id_end) {
                    // Scheduling the same opseq on the next SM.
                    if (next_uop.warp_id_begin != new_branch.warp_id_begin) {
                        // Using different warp IDs from the next SM. Break.
                        break;
                    } else {
                        // Contiguous SMs and using the same warp IDs. Try
                        // merge.
                        if (new_branch.uop_id_diff == 0) {
                            // Diff is undetermined yet. Set it and merge.
                            new_branch.uop_id_diff =
                                next_uop.uop_id - new_branch.uop_id_last;
                            current_warp_id_end = next_uop.warp_id_end;
                            new_branch.uop_id_last = next_uop.uop_id;
                            new_branch.sm_id_end = next_uop.sm_id + 1;
                            if (new_branch.warp_id_end < current_warp_id_end) {
                                new_branch.warp_id_end = current_warp_id_end;
                            }
                        } else if (new_branch.uop_id_diff ==
                                   next_uop.uop_id - new_branch.uop_id_last) {
                            // Diff is the same as the previous uop. Merge.
                            current_warp_id_end = next_uop.warp_id_end;
                            new_branch.uop_id_last = next_uop.uop_id;
                            new_branch.sm_id_end = next_uop.sm_id + 1;
                            if (new_branch.warp_id_end < current_warp_id_end) {
                                new_branch.warp_id_end = current_warp_id_end;
                            }
                        } else {
                            // Diff is different from the previous uop. Break.
                            break;
                        }
                        merged_uop_indices.emplace(j);
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

void SchedBranch::add(int opseq_id, int uop_id, int sm_id, int warp_id_begin,
                      int warp_id_end)
{
    this->impl->add(opseq_id, uop_id, sm_id, warp_id_begin, warp_id_end);
}

void SchedBranch::clear()
{
    this->impl->clear();
}

std::vector<Branch> SchedBranch::get_branches()
{
    return this->impl->get_branches();
}

} // namespace ark
