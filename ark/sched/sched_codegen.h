// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_SCHED_CODEGEN_H_
#define ARK_SCHED_CODEGEN_H_

#include "gpu/gpu_kernel.h"
#include "sched/sched_op.h"
#include "sched/sched_opseq.h"
#include "sched_branch.h"
#include <map>

namespace ark {

class CodeGenerator
{
  public:
    CodeGenerator(const GpuInfo &gpu_info_, int num_warps_per_sm_);

    std::ostream &def_remote_buf(std::ostream &os, int remote_rank) const;

    std::ostream &sync_gpu(std::ostream &os) const;

    std::ostream &def_sync_stream(std::ostream &os, int stream_id) const;
    std::ostream &sync_stream(std::ostream &os, int stream_id, int sm_id_begin,
                              int sm_id_end) const;

    std::ostream &tensor(std::ostream &os, const Tensor *tensor) const;

    std::ostream &def_oparg(std::ostream &os, const OpArg &arg,
                            const std::string &name) const;
    std::ostream &oparg(std::ostream &os, const OpArg &arg) const;

    std::ostream &branch(std::ostream &os, const Branch &branch,
                         int prev_sm_id_end = -1) const;

    std::ostream &def_uop(std::ostream &os, const SchedOp &sop,
                          int uop_id) const;

    std::ostream &uop(std::ostream &os, int uop_id) const;

    std::ostream &opseq(std::ostream &os, const std::string &name,
                        const SchedOpSeq &opseq,
                        std::map<std::string, int> &uop_map) const;

    std::ostream &sched(std::ostream &os, Sched &sched) const;

    std::ostream &def_proxy_channels(std::ostream &os,
                                     size_t num_channels) const;

    std::ostream &def_sm_channels(std::ostream &os,
                                          size_t num_channels) const;

  protected:
    size_t get_tensor_offset(const Tensor *tensor) const;

    const GpuInfo &gpu_info;
    int sm_num;
    int num_warps_per_sm;
    int world_size;
    int num_indent;
};

} // namespace ark

#endif // ARK_SCHED_CODEGEN_H_
