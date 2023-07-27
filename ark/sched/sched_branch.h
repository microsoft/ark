// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_SCHED_BRANCH_H_
#define ARK_SCHED_BRANCH_H_

#include <memory>
#include <vector>

namespace ark {

struct Branch
{
    int opseq_id;
    int sm_id_begin;
    int sm_id_end;
    int warp_id_begin;
    int warp_id_end;
    int uop_id_begin;
    int uop_id_last;
    int uop_id_diff;
    int num_warps_per_uop;
};

class SchedBranch
{
  public:
    SchedBranch();
    ~SchedBranch();

    void add(int opseq_id, int uop_id, int sm_id, int warp_id_begin,
             int warp_id_end);
    void clear();
    std::vector<Branch> get_branches();

  private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace ark

#endif // ARK_SCHED_BRANCH_H_
