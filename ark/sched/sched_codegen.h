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

class BaseCodeGenerator
{
  public:
    BaseCodeGenerator(const std::map<TensorBuf *, GpuBuf *> &buf_trans,
                      const GpuInfo &gpu_info_, int num_warps_per_sm_,
                      int world_size)
        : buf_trans{buf_trans}, gpu_info{gpu_info_}, sm_num{gpu_info_.num_sm},
          num_warps_per_sm{num_warps_per_sm_}, world_size{world_size},
          num_indent{0}
    {
    }
    virtual std::vector<std::string> codegen_codes_body(
        std::vector<Sched> &scheds) = 0;

    std::ostream &sync_stream(std::ostream &os, int stream_id, int sm_id_begin, int sm_id_end);
    std::ostream &sync_stream_state(std::ostream &os, int stream_id);

    std::ostream &codegen_sync_gpu(std::ostream &os);
    std::ostream &codegen_branch(std::ostream &os, const Branch &branch,
                                 int prev_sm_id_end = -1);

  protected:
    const std::map<TensorBuf *, GpuBuf *> &buf_trans;
    const GpuInfo &gpu_info;
    int sm_num;
    int num_warps_per_sm;
    int world_size;
    int num_indent;
};

class SimpleCodeGenerator : public BaseCodeGenerator
{
  public:
    SimpleCodeGenerator(const std::map<TensorBuf *, GpuBuf *> &buf_trans,
                        const GpuInfo &gpu_info, int num_warps_per_sm,
                        int world_size)
        : BaseCodeGenerator(buf_trans, gpu_info, num_warps_per_sm, world_size)
    {
    }
    std::vector<std::string> codegen_codes_body(std::vector<Sched> &scheds);

  private:
    size_t get_tensor_offset(const Tensor *tensor);
    std::ostream &codegen_tensor(std::ostream &os, const Tensor *tensor);
    std::ostream &codegen_opseq(std::ostream &os, SchedOpSeq *sopseq);
    std::ostream &codegen_sched(std::ostream &os, Sched &sched);
};

struct ThBranch
{
    int th_b;
    int th_e;
    std::vector<std::tuple<SchedOpSeq *, int, int>> ops;
};

struct SmBranch
{
    int sm_b;
    int sm_e;
    std::list<ThBranch> tbs;
};

class Brancher
{
  public:
    Brancher(int sm_num_, int th_num_) : sm_num{sm_num_}, th_num{th_num_}
    {
    }
    void add(const Sched &sc);

    std::ostream &codegen(std::ostream &os);

    bool is_empty() const
    {
        return sbs.size() == 0;
    }

  private:
    const int sm_num;
    const int th_num;
    std::list<SmBranch> sbs;
};

class DefaultCodeGenerator : public BaseCodeGenerator
{
  public:
    DefaultCodeGenerator(const std::map<TensorBuf *, GpuBuf *> &buf_trans,
                         const GpuInfo &gpu_info, int num_warps_per_sm,
                         int world_size)
        : BaseCodeGenerator(buf_trans, gpu_info, num_warps_per_sm, world_size)
    {
    }
    std::vector<std::string> codegen_codes_body(std::vector<Sched> &scheds);

    void add(SchedOpSeq *sopseq);

    std::ostream &codegen_tensor(std::ostream &os, const Tensor &tensor);
    std::ostream &codegen_arg(std::ostream &os, const OpArg &arg);
    std::ostream &codegen_arg_def(std::ostream &os, const OpArg &arg,
                                  const std::string &name);
    std::ostream &codegen_opseq(std::ostream &os, const std::string &name,
                                const SchedOpSeq &opseq,
                                std::map<std::string, int> &uop_map);
    std::ostream &codegen_uop_def(std::ostream &os, const SchedOp &sop,
                                  int uop_id);
    std::ostream &codegen_depth(std::ostream &os, const std::string &name,
                                Brancher *brc, std::set<SchedOpSeq *> &opseqs,
                                std::map<std::string, int> &uop_map);

  private:
    SchedBranch branch;
};

} // namespace ark
#endif // ARK_SCHED_CODEGEN_H_