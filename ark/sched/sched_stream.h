// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_SCHED_STREAM_H_
#define ARK_SCHED_STREAM_H_

#include "sched_branch.h"
#include <memory>
#include <vector>

namespace ark {

struct SchedItem
{
    int opseq_id;
    int num_uops;
    int num_warps_per_uop;
    int smem_bytes_per_uop;
};

class SchedStream
{
  public:
    SchedStream(int sm_id_begin, int sm_id_end, int num_warps_per_sm,
                int smem_bytes_per_sm);
    ~SchedStream();

    void add_items(const std::vector<SchedItem> &items);
    void sync();
    void clear();
    std::vector<std::vector<Branch>> get_branches();

    int get_num_sm() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace ark

#endif // ARK_SCHED_STREAM_H_
