// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef _ARK_SCHED_PROFILER_H_
#define _ARK_SCHED_PROFILER_H_

#include "ark/gpu/gpu_kernel.h"
#include "ark/gpu/gpu_mgr.h"
#include "ark/include/ark.h"
#include "ark/sched/sched_codegen.h"
#include "ark/sched/sched_op.h"
#include "ark/sched/sched_opgraph.h"
#include "ark/sched/sched_opseq.h"
#include "ark/sched/sched_tile.h"
namespace ark {

struct SchedPerf
{
    SchedPerf()
    {
    }
    SchedPerf(std::tuple<float, int, int> perf)
        : elapsed{std::get<0>(perf)}, regs_num{std::get<1>(perf)},
          score{std::get<2>(perf)}
    {
        assert(this->elapsed > 0);
        assert(this->regs_num > 0);
    }

    void set(float e, int r, int s = -1)
    {
        assert(e > 0);
        assert(r > 0);
        elapsed = e;
        regs_num = r;
        score = s;
    }
    bool is_set() const
    {
        return elapsed > 0 && regs_num > 0;
    }
    //
    float elapsed = -1;
    int regs_num = -1;
    int score = -1;
};

struct SchedOpSeqPerf
{
    SchedPerf s;
    SchedPerf x;
    SchedPerf y;
    SchedPerf xy;
    std::map<const SchedOpSeq *, SchedPerf> mixed;
};

std::vector<Sched> gen_sched(SchedTileDepth *tile_depths, int wps);

class SchedProfiler
{
  public:
    SchedProfiler(GpuMgr *gpu_mgr, int wps) : gpu_mgr{gpu_mgr}, wps{wps}
    {
    }
    void profile(OpGraph *op_graph, DefaultCodeGenerator &codegen,
                 GpuMgrCtx *ctx);
    float profile_routine(GpuLoopKernel *glk, GpuMgrCtx *ctx);

    GpuMgr *gpu_mgr;
    int wps;
    const std::string wps_prof_cache_path = "wps_prof_cache.json";
    std::map<int, std::map<const SchedOpSeq *, SchedOpSeqPerf>>
        wps_prof_results;
};

}; // namespace ark

#endif
