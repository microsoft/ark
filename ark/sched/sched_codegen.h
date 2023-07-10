// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_SCHED_CODEGEN_H_
#define ARK_SCHED_CODEGEN_H_
#include "ark/gpu/gpu_kernel.h"
#include "ark/sched/sched_op.h"
#include "ark/sched/sched_opseq.h"
#include <map>

namespace ark {

class BaseCodeGenerator
{
  public:
    BaseCodeGenerator(const std::map<TensorBuf *, GpuBuf *> &buf_trans,
                      const GpuInfo &gpu_info, int wps, int world_size)
        : buf_trans{buf_trans}, sm_num{gpu_info.num_sm}, wps{wps},
          world_size{world_size}
    {
    }
    virtual std::vector<std::string> codegen_codes_body(
        std::vector<Sched> &scheds) = 0;

  protected:
    const std::map<TensorBuf *, GpuBuf *> &buf_trans;
    int sm_num;
    int wps;
    int world_size;
};

class SimpleCodeGenerator : BaseCodeGenerator
{
  public:
    SimpleCodeGenerator(const std::map<TensorBuf *, GpuBuf *> &buf_trans,
                        const GpuInfo &gpu_info, int wps, int world_size)
        : BaseCodeGenerator(buf_trans, gpu_info, wps, world_size)
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

class DefaultCodeGenerator : BaseCodeGenerator
{
  public:
    DefaultCodeGenerator(const std::map<TensorBuf *, GpuBuf *> &buf_trans,
                         const GpuInfo &gpu_info, int wps, int world_size)
        : BaseCodeGenerator(buf_trans, gpu_info, wps, world_size)
    {
    }
    std::vector<std::string> codegen_codes_body(std::vector<Sched> &scheds);

  private:
    std::ostream &codegen_tensor(std::ostream &os, const Tensor &tensor);
    std::ostream &codegen_sched_op(std::ostream &os, const SchedOp &sop,
                                   const std::pair<int, int> &fdims);
    std::ostream &codegen_opseq(std::ostream &os, const std::string &name,
                                const SchedOpSeq &opseq,
                                std::map<std::string, int> &uop_map);
    // std::ostream &codegen_loop_body(
    //     std::ostream &os, const std::vector<SchedTileDepth *> &tile_depths);
    std::ostream &codegen_depth(std::ostream &os, const std::string &name,
                                Brancher *brc, std::set<SchedOpSeq *> &opseqs,
                                std::map<std::string, int> &sropseq_map,
                                std::map<std::string, int> &uop_map);
};

} // namespace ark
#endif // ARK_SCHED_CODEGEN_H_