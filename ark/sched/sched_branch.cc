// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "sched_branch.h"
#include "logging.h"
#include <algorithm>
#include <unordered_set>
#include <vector>

namespace ark {

/// Contains information on which SMs and warps execute a given opseq.
///
/// Indicates:
/// ```
///   if (sm_id_begin <= sm_id < sm_id_end &&
///       warp_id_begin <= warp_id < warp_id_end) {
///     num_uops_per_sm = (warp_id_end - warp_id_begin) / num_warps_per_uop;
///     warp_idx = warp_id - warp_id_begin;
///     sm_idx = sm_id - sm_id_begin;
///     op = opseq_id;
///     uop = uop_id_diff * (warp_idx / num_warps_per_uop + num_uops_per_sm *
///                          sm_idx) + uop_id_begin;
///   }
/// ```
struct OpBranchInfo
{
    /// The opseq ID.
    int opseq_id;
    /// The SM ID range [sm_id_begin, sm_id_end).
    int sm_id_begin;
    /// The SM ID range [sm_id_begin, sm_id_end).
    int sm_id_end;
    /// The warp ID range [warp_id_begin, warp_id_end).
    int warp_id_begin;
    /// The warp ID range [warp_id_begin, warp_id_end).
    int warp_id_end;
    /// The uop ID that the first warp in the range executes.
    int uop_id_begin;
    /// The uop ID that the last warp in the range executes.
    int uop_id_last;
    /// The difference between the uop ID of the first warp and the uop ID of
    /// the `num_warps_per_uop`-th warp.
    int uop_id_diff;
    /// The number of warps per uop.
    int num_warps_per_uop;
};

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
    std::vector<OpBranchInfo> get_op_branch_info();

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
    if (uop_id < 0) {
        LOG(ERROR, "uop_id ", uop_id, " out of range [0, inf)");
    }
    if (sm_id < 0) {
        LOG(ERROR, "sm_id ", sm_id, " out of range [0, inf)");
    }
    if (warp_id_begin < 0) {
        LOG(ERROR, "warp_id_begin ", warp_id_begin, " out of range [0, inf)");
    }
    if (warp_id_end <= warp_id_begin) {
        LOG(ERROR, "warp_id_end ", warp_id_end, " <= warp_id_begin ",
            warp_id_begin);
    }
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

std::vector<OpBranchInfo> SchedBranch::Impl::get_op_branch_info()
{
    std::vector<OpBranchInfo> infos;

    std::sort(uops.begin(), uops.end(), cmp_uop);

    std::unordered_set<size_t> merged_uop_indices;

    for (size_t i = 0; i < uops.size(); ++i) {
        if (merged_uop_indices.find(i) != merged_uop_indices.end()) {
            continue;
        }
        UnitOp current_uop = uops[i];
        OpBranchInfo info;
        info.opseq_id = current_uop.opseq_id;
        info.sm_id_begin = current_uop.sm_id;
        info.sm_id_end = current_uop.sm_id + 1;
        info.warp_id_begin = current_uop.warp_id_begin;
        info.warp_id_end = current_uop.warp_id_end;
        info.uop_id_begin = current_uop.uop_id;
        info.uop_id_last = current_uop.uop_id;
        info.uop_id_diff = 0;
        info.num_warps_per_uop =
            current_uop.warp_id_end - current_uop.warp_id_begin;

        merged_uop_indices.emplace(i);

        int current_warp_id_end = info.warp_id_end;

        for (size_t j = i + 1; j < uops.size(); j++) {
            if (merged_uop_indices.find(j) != merged_uop_indices.end()) {
                continue;
            }
            UnitOp next_uop = uops[j];
            if (next_uop.opseq_id != current_uop.opseq_id) {
                // Scheduling another opseq. There is no more uop to merge.
                // Break.
                break;
            }
            // Scheduling the same opseq.
            if (next_uop.warp_id_end - next_uop.warp_id_begin !=
                info.num_warps_per_uop) {
                // The same opseq should have the same number of warps per
                // uop.
                LOG(ERROR, "invalid num_warps_per_uop: ",
                    next_uop.warp_id_end - next_uop.warp_id_begin,
                    ", expected: ", info.num_warps_per_uop);
            }
            if (next_uop.sm_id == info.sm_id_end - 1) {
                // Scheduling the same opseq on the same SM as the previous
                // uop.
                if (next_uop.warp_id_begin >= info.warp_id_begin &&
                    next_uop.warp_id_begin < current_warp_id_end) {
                    // Scheduling another uop from the same opseq on the
                    // same SM and warp. This should be handled in another
                    // branch. Skip here.
                    continue;
                } else if (next_uop.warp_id_begin == current_warp_id_end) {
                    // Contiguous warps. Try merge.
                    if (info.uop_id_diff == 0) {
                        // Diff is undetermined yet. Set it and merge.
                        info.uop_id_diff = next_uop.uop_id - info.uop_id_last;
                        current_warp_id_end = next_uop.warp_id_end;
                        info.uop_id_last = next_uop.uop_id;
                        if (info.warp_id_end < current_warp_id_end) {
                            info.warp_id_end = current_warp_id_end;
                        }
                    } else if (info.uop_id_diff ==
                               next_uop.uop_id - info.uop_id_last) {
                        // Diff is the same as the previous uop. Merge.
                        current_warp_id_end = next_uop.warp_id_end;
                        info.uop_id_last = next_uop.uop_id;
                        if (info.warp_id_end < current_warp_id_end) {
                            info.warp_id_end = current_warp_id_end;
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
            } else if (next_uop.sm_id == info.sm_id_end) {
                // Scheduling the same opseq on the next SM.
                if (next_uop.warp_id_begin != info.warp_id_begin) {
                    // Using different warp IDs from the next SM. Break.
                    break;
                } else {
                    // Contiguous SMs and using the same warp IDs. Try
                    // merge.
                    if (info.uop_id_diff == 0) {
                        // Diff is undetermined yet. Set it and merge.
                        info.uop_id_diff = next_uop.uop_id - info.uop_id_last;
                        current_warp_id_end = next_uop.warp_id_end;
                        info.uop_id_last = next_uop.uop_id;
                        info.sm_id_end = next_uop.sm_id + 1;
                        if (info.warp_id_end < current_warp_id_end) {
                            info.warp_id_end = current_warp_id_end;
                        }
                    } else if (info.uop_id_diff ==
                               next_uop.uop_id - info.uop_id_last) {
                        // Diff is the same as the previous uop. Merge.
                        current_warp_id_end = next_uop.warp_id_end;
                        info.uop_id_last = next_uop.uop_id;
                        info.sm_id_end = next_uop.sm_id + 1;
                        if (info.warp_id_end < current_warp_id_end) {
                            info.warp_id_end = current_warp_id_end;
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
        infos.emplace_back(info);
    }
    return infos;
}

std::vector<Branch> SchedBranch::Impl::get_branches()
{
    std::vector<OpBranchInfo> op_branch_info = this->get_op_branch_info();
    std::vector<Branch> branches;
    for (const OpBranchInfo &info : op_branch_info) {
        if (!branches.empty() &&
            branches.back().sm_id_begin == info.sm_id_begin &&
            branches.back().sm_id_end == info.sm_id_end) {
            // Merge with the previous branch.
            BranchOp branch_op;
            branch_op.opseq_id = info.opseq_id;
            branch_op.uop_id_begin = info.uop_id_begin;
            branch_op.uop_id_diff = info.uop_id_diff;
            branch_op.num_warps_per_uop = info.num_warps_per_uop;
            branch_op.num_uops_per_sm =
                (info.warp_id_end - info.warp_id_begin) /
                info.num_warps_per_uop;
            WarpBranch &warp_branch = branches.back().warp_branches.back();
            if (warp_branch.warp_id_begin == info.warp_id_begin &&
                warp_branch.warp_id_end == info.warp_id_end) {
                // Merge with the previous warp branch.
                warp_branch.branch_ops.emplace_back(std::move(branch_op));
            } else if (warp_branch.warp_id_end <= info.warp_id_begin ||
                       info.warp_id_begin == 0) {
                // Add a new warp branch.
                WarpBranch warp_branch;
                warp_branch.warp_id_begin = info.warp_id_begin;
                warp_branch.warp_id_end = info.warp_id_end;
                warp_branch.branch_ops.emplace_back(std::move(branch_op));
                branches.back().warp_branches.emplace_back(
                    std::move(warp_branch));
            } else {
                // This may be not possible.
                LOG(ERROR, "unexpected error");
            }
        } else {
            // Add a new branch.
            Branch branch;
            branch.sm_id_begin = info.sm_id_begin;
            branch.sm_id_end = info.sm_id_end;
            WarpBranch warp_branch;
            warp_branch.warp_id_begin = info.warp_id_begin;
            warp_branch.warp_id_end = info.warp_id_end;
            BranchOp branch_op;
            branch_op.opseq_id = info.opseq_id;
            branch_op.uop_id_begin = info.uop_id_begin;
            branch_op.uop_id_diff = info.uop_id_diff;
            branch_op.num_warps_per_uop = info.num_warps_per_uop;
            branch_op.num_uops_per_sm =
                (info.warp_id_end - info.warp_id_begin) /
                info.num_warps_per_uop;
            warp_branch.branch_ops.emplace_back(std::move(branch_op));
            branch.warp_branches.emplace_back(std::move(warp_branch));
            branches.emplace_back(std::move(branch));
        }
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
