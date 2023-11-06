// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "sched_branch.h"

#include <algorithm>
#include <map>
#include <set>
#include <unordered_set>
#include <vector>

#include "include/ark.h"
#include "logging.h"

#define DEBUG_BRANCH 0
#define BRANCH_DEBUG(...)            \
    do {                             \
        if (DEBUG_BRANCH) {          \
            LOG(DEBUG, __VA_ARGS__); \
        }                            \
    } while (0);

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
struct OpBranchInfo {
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
    /// Bytes of shared memory allowed per warp.
    int smem_bytes_per_warp;

    int get_num_uops() const {
        int uops_per_sm = (warp_id_end - warp_id_begin) / num_warps_per_uop;
        int num_sms = sm_id_end - sm_id_begin;
        return uops_per_sm * num_sms;
    }
};

class SchedBranch::Impl {
   private:
    struct UnitOp {
        int opseq_id;
        int uop_id;
        int sm_id;
        int warp_id_begin;
        int warp_id_end;
    };

    static bool cmp_uop(const UnitOp &a, const UnitOp &b);
    std::vector<OpBranchInfo> get_op_branch_info(
        const std::map<int, int> &sm_id_to_smem_per_warp);

    std::vector<UnitOp> uops;
    std::map<int, std::set<int>> opseq_to_uop_ids;

   public:
    Impl();
    ~Impl();

   protected:
    void add(int opseq_id, int uop_id, int sm_id, int warp_id_begin,
             int warp_id_end);
    void clear();
    std::vector<Branch> get_branches(
        const std::map<int, int> &sm_id_to_smem_per_warp);

    friend class SchedBranch;
};

SchedBranch::Impl::Impl() {}

SchedBranch::Impl::~Impl() {}

void SchedBranch::Impl::add(int opseq_id, int uop_id, int sm_id,
                            int warp_id_begin, int warp_id_end) {
    if (uop_id < 0) {
        ERR(SchedulerError, "uop_id ", uop_id, " out of range [0, inf)");
    }
    if (sm_id < 0) {
        ERR(SchedulerError, "sm_id ", sm_id, " out of range [0, inf)");
    }
    if (warp_id_begin < 0) {
        ERR(SchedulerError, "warp_id_begin ", warp_id_begin,
            " out of range [0, inf)");
    }
    if (warp_id_end <= warp_id_begin) {
        ERR(SchedulerError, "warp_id_end ", warp_id_end, " <= warp_id_begin ",
            warp_id_begin);
    }
    auto p = this->opseq_to_uop_ids[opseq_id].insert(uop_id);
    if (!p.second) {
        ERR(SchedulerError, "opseq_id ", opseq_id, " uop_id ", uop_id,
            " already exists");
    }
    UnitOp uop{opseq_id, uop_id, sm_id, warp_id_begin, warp_id_end};
    this->uops.emplace_back(uop);
}

void SchedBranch::Impl::clear() { this->uops.clear(); }

bool SchedBranch::Impl::cmp_uop(const UnitOp &a, const UnitOp &b) {
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

std::vector<OpBranchInfo> SchedBranch::Impl::get_op_branch_info(
    const std::map<int, int> &sm_id_to_smem_per_warp) {
    std::vector<OpBranchInfo> infos;

    std::sort(this->uops.begin(), this->uops.end(), cmp_uop);

    std::unordered_set<size_t> merged_uop_indices;

    for (size_t i = 0; i < this->uops.size(); ++i) {
        if (merged_uop_indices.find(i) != merged_uop_indices.end()) {
            continue;
        }
        size_t num_merged_uops = merged_uop_indices.size();

        UnitOp current_uop = this->uops[i];
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

        auto search = sm_id_to_smem_per_warp.find(current_uop.sm_id);
        if (search == sm_id_to_smem_per_warp.end()) {
            info.smem_bytes_per_warp = 0;
        } else {
            info.smem_bytes_per_warp = search->second;
        }

        merged_uop_indices.emplace(i);
        BRANCH_DEBUG("merged uop id ", current_uop.uop_id, " sm_id ",
                     current_uop.sm_id, " warp_id_begin ",
                     current_uop.warp_id_begin, " warp_id_end ",
                     current_uop.warp_id_end);

        int current_warp_id_end = info.warp_id_end;

        for (size_t j = i + 1; j < this->uops.size(); j++) {
            if (merged_uop_indices.find(j) != merged_uop_indices.end()) {
                continue;
            }
            UnitOp next_uop = this->uops[j];
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
                ERR(SchedulerError, "invalid num_warps_per_uop: ",
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
                } else if (next_uop.warp_id_begin != current_warp_id_end) {
                    // Non-contiguous warps. Break.
                    break;
                }
                // Contiguous warps. Try merge.
            } else if (next_uop.sm_id == info.sm_id_end) {
                // Scheduling the same opseq on the next SM.
                if (next_uop.warp_id_begin != info.warp_id_begin) {
                    // Using different warp IDs from the next SM. Break.
                    break;
                }

                search = sm_id_to_smem_per_warp.find(next_uop.sm_id);
                if (search != sm_id_to_smem_per_warp.end()) {
                    if (info.smem_bytes_per_warp != search->second) {
                        // Different SMs have different shared memory bytes
                        // per warp. Break.
                        break;
                    }
                } else {
                    // The next SM is supposed to not use shared memory.
                    // We can merge it. Do nothing.
                }
                // Contiguous SMs and using the same warp IDs. Try merge.
            } else {
                // Non-contiguous SMs. Break.
                break;
            }

            // Try merge.

            if (info.uop_id_diff != 0 &&
                info.uop_id_diff != next_uop.uop_id - info.uop_id_last) {
                // Diff is different from the previous uop. Break.
                break;
            }
            if (info.sm_id_end - info.sm_id_begin > 1 &&
                (next_uop.warp_id_end > info.warp_id_end ||
                 next_uop.warp_id_begin < info.warp_id_begin)) {
                // This branch is scheduling multiple SMs and next_uop
                // uses warp IDs that are not used by previous SMs.
                // Break.
                break;
            }

            // Merge.
            if (info.uop_id_diff == 0) {
                // Diff is undetermined yet. Set it.
                info.uop_id_diff = next_uop.uop_id - info.uop_id_last;
            } else {
                // Diff is the same as the previous uop. Do nothing.
            }
            if (next_uop.sm_id == info.sm_id_end) {
                // Scheduling the same opseq on the next SM.
                info.sm_id_end = next_uop.sm_id + 1;
            }
            current_warp_id_end = next_uop.warp_id_end;
            info.uop_id_last = next_uop.uop_id;
            if (info.warp_id_end < current_warp_id_end) {
                info.warp_id_end = current_warp_id_end;
            }
            merged_uop_indices.emplace(j);
            BRANCH_DEBUG("merged uop id ", next_uop.uop_id, " sm_id ",
                         next_uop.sm_id, " warp_id_begin ",
                         next_uop.warp_id_begin, " warp_id_end ",
                         next_uop.warp_id_end);
        }

        if (current_warp_id_end < info.warp_id_end) {
            // The last scheduled SM uses less warps than the previous
            // scheduled SMs. Break the info into two.

            int num_uops_in_last_sm =
                (current_warp_id_end - info.warp_id_begin) /
                info.num_warps_per_uop;
            int first_uop_id_in_last_sm =
                info.uop_id_last - (num_uops_in_last_sm - 1) * info.uop_id_diff;

            OpBranchInfo new_info;
            new_info.opseq_id = info.opseq_id;
            new_info.sm_id_begin = info.sm_id_end - 1;
            new_info.sm_id_end = info.sm_id_end;
            new_info.warp_id_begin = info.warp_id_begin;
            new_info.warp_id_end = current_warp_id_end;
            new_info.uop_id_begin = first_uop_id_in_last_sm;
            new_info.uop_id_last = info.uop_id_last;
            new_info.uop_id_diff = info.uop_id_diff;
            new_info.num_warps_per_uop = info.num_warps_per_uop;

            search = sm_id_to_smem_per_warp.find(new_info.sm_id_begin);
            if (search == sm_id_to_smem_per_warp.end()) {
                new_info.smem_bytes_per_warp = 0;
            } else {
                new_info.smem_bytes_per_warp = search->second;
            }

            info.sm_id_end -= 1;
            info.uop_id_last = first_uop_id_in_last_sm - info.uop_id_diff;

            if (merged_uop_indices.size() - num_merged_uops !=
                (size_t)(info.get_num_uops() + new_info.get_num_uops())) {
                ERR(SchedulerError,
                    "unexpected error: numbers of newly merged uops mismatch (",
                    merged_uop_indices.size() - num_merged_uops, " vs ",
                    info.get_num_uops() + new_info.get_num_uops(), ")");
            }

            infos.emplace_back(info);
            infos.emplace_back(new_info);
        } else if (current_warp_id_end != info.warp_id_end) {
            ERR(SchedulerError, "unexpected error");
        } else {
            if (merged_uop_indices.size() - num_merged_uops !=
                (size_t)info.get_num_uops()) {
                ERR(SchedulerError,
                    "unexpected error: numbers of newly merged uops mismatch (",
                    merged_uop_indices.size() - num_merged_uops, " vs ",
                    info.get_num_uops(), ")");
            }
            infos.emplace_back(info);
        }
    }

    // Verify if there is a missing uop.
    for (auto &p : this->opseq_to_uop_ids) {
        int expected_id = 0;
        for (int uop_id : p.second) {
            if (uop_id != expected_id) {
                ERR(SchedulerError, "missing uop ", expected_id, " in opseq ",
                    p.first);
            }
            expected_id += 1;
        }
    }

    // Verify if the number of uops matches.
    int num_uops = 0;
    for (const OpBranchInfo &info : infos) {
        num_uops += info.get_num_uops();
    }
    size_t compare_num_uops = 0;
    for (auto &p : this->opseq_to_uop_ids) {
        compare_num_uops += p.second.size();
    }
    if ((size_t)num_uops != compare_num_uops) {
        ERR(SchedulerError, "invalid number of uops: ", num_uops,
            ", expected: ", compare_num_uops,
            " merged_uop_indices.size(): ", merged_uop_indices.size());
    }

    return infos;
}

std::vector<Branch> SchedBranch::Impl::get_branches(
    const std::map<int, int> &sm_id_to_smem_per_warp) {
    std::vector<OpBranchInfo> op_branch_info =
        this->get_op_branch_info(sm_id_to_smem_per_warp);
    std::vector<Branch> branches;
    for (const OpBranchInfo &info : op_branch_info) {
        if (!branches.empty() &&
            branches.back().sm_id_begin == info.sm_id_begin &&
            branches.back().sm_id_end == info.sm_id_end &&
            (branches.back().smem_bytes_per_warp == info.smem_bytes_per_warp ||
             branches.back().smem_bytes_per_warp * info.smem_bytes_per_warp ==
                 0)) {
            Branch &branch = branches.back();
            // Merge with the previous branch.
            BranchOp branch_op;
            branch_op.opseq_id = info.opseq_id;
            branch_op.uop_id_begin = info.uop_id_begin;
            branch_op.uop_id_diff = info.uop_id_diff;
            branch_op.num_warps_per_uop = info.num_warps_per_uop;
            branch_op.num_uops_per_sm =
                (info.warp_id_end - info.warp_id_begin) /
                info.num_warps_per_uop;
            if (branch.smem_bytes_per_warp != info.smem_bytes_per_warp) {
                branch.smem_bytes_per_warp = std::max(
                    branch.smem_bytes_per_warp, info.smem_bytes_per_warp);
            }
            WarpBranch &warp_branch = branch.warp_branches.back();
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
                branch.warp_branches.emplace_back(std::move(warp_branch));
            } else {
                // This may be not possible.
                ERR(SchedulerError, "unexpected error");
            }
        } else {
            // Add a new branch.
            Branch branch;
            branch.sm_id_begin = info.sm_id_begin;
            branch.sm_id_end = info.sm_id_end;
            branch.smem_bytes_per_warp = info.smem_bytes_per_warp;
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

SchedBranch::SchedBranch() { this->impl = std::make_unique<Impl>(); }

SchedBranch::~SchedBranch() {}

void SchedBranch::add(int opseq_id, int uop_id, int sm_id, int warp_id_begin,
                      int warp_id_end) {
    this->impl->add(opseq_id, uop_id, sm_id, warp_id_begin, warp_id_end);
}

void SchedBranch::clear() { this->impl->clear(); }

std::vector<Branch> SchedBranch::get_branches(
    const std::map<int, int> &sm_id_to_smem_per_warp) {
    return this->impl->get_branches(sm_id_to_smem_per_warp);
}

}  // namespace ark
