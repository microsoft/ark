// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_SCHED_BRANCH_H_
#define ARK_SCHED_BRANCH_H_

#include <memory>
#include <vector>

namespace ark {

/// Information of an operator that runs on a branch.
struct BranchOp
{
    /// The opseq ID.
    int opseq_id;
    /// The uop ID that runs on the first warp among the entire @ref Branch.
    int uop_id_begin;
    /// The difference between the uop ID of the first warp and the uop ID of
    /// the `num_warps_per_uop`-th warp.
    int uop_id_diff;
    /// The number of warps that run the same uop.
    int num_warps_per_uop;
    /// The number of uops that runs in a SM at the same time.
    int num_uops_per_sm;
};

/// A branch of execution that runs on a SM.
struct WarpBranch
{
    /// The warp ID range of this branch [warp_id_begin, warp_id_end).
    int warp_id_begin;
    /// The warp ID range of this branch [warp_id_begin, warp_id_end).
    int warp_id_end;
    /// The list of operators that run on this branch.
    std::vector<BranchOp> branch_ops;
};

/// A branch of execution that runs over multiple SMs.
struct Branch
{
    /// The SM ID range of this branch [sm_id_begin, sm_id_end).
    int sm_id_begin;
    /// The SM ID range of this branch [sm_id_begin, sm_id_end).
    int sm_id_end;
    /// The list of warp branches that run on every SMs in the range.
    std::vector<WarpBranch> warp_branches;
};

/// A class that records the branches of execution.
class SchedBranch
{
  public:
    /// Construct a @ref SchedBranch.
    SchedBranch();

    /// Destruct a @ref SchedBranch.
    ~SchedBranch();

    /// Add an execution branch.
    /// @param opseq_id The opseq ID.
    /// @param uop_id The uop ID.
    /// @param sm_id The SM ID to run the uop.
    /// @param warp_id_begin The warp ID range [warp_id_begin, warp_id_end) to
    /// run the uop.
    /// @param warp_id_end The warp ID range [warp_id_begin, warp_id_end) to run
    /// the uop.
    void add(int opseq_id, int uop_id, int sm_id, int warp_id_begin,
             int warp_id_end);

    /// Clear all the branches.
    void clear();

    /// Get the branches.
    std::vector<Branch> get_branches();

  private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace ark

#endif // ARK_SCHED_BRANCH_H_
