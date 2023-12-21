// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef _ARK_SCHED_PROFILER_H_
#define _ARK_SCHED_PROFILER_H_

#include "gpu/gpu_context.h"
#include "gpu/gpu_loop_kernel.h"
#include "include/ark.h"
#include "sched/sched_codegen.h"
#include "sched/sched_op.h"
#include "sched/sched_opgraph.h"
#include "sched/sched_opseq.h"
#include "sched/sched_tile.h"
namespace ark {

struct SchedPerf {
    SchedPerf() {}
    SchedPerf(std::tuple<float, int, int> perf)
        : elapsed{std::get<0>(perf)},
          regs_num{std::get<1>(perf)},
          score{std::get<2>(perf)} {
        assert(this->elapsed > 0);
        assert(this->regs_num > 0);
    }

    void set(float e, int r, int s = -1) {
        assert(e > 0);
        assert(r > 0);
        elapsed = e;
        regs_num = r;
        score = s;
    }
    bool is_set() const { return elapsed > 0 && regs_num > 0; }
    //
    float elapsed = -1;
    int regs_num = -1;
    int score = -1;
};

struct SchedOpSeqPerf {
    SchedPerf s;
    SchedPerf x;
    SchedPerf y;
    SchedPerf xy;
    std::map<const SchedOpSeq *, SchedPerf> mixed;
};

std::vector<Sched> gen_sched(SchedTileDepth *tile_depths, int num_warps_per_sm);

class SchedProfiler {
   public:
    SchedProfiler(std::shared_ptr<GpuContext> ctx, int num_warps_per_sm_)
        : ctx_{ctx}, num_warps_per_sm_{num_warps_per_sm_} {}
    void profile(OpGraph *op_graph, CodeGenerator &codegen,
                 std::shared_ptr<GpuContext> ctx);
    float profile_routine(std::shared_ptr<GpuLoopKernelV2> glk,
                          std::shared_ptr<GpuContext> ctx);

    std::shared_ptr<GpuContext> ctx_;
    int num_warps_per_sm_;
    const std::string wps_prof_cache_path_ = "wps_prof_cache.json";
    std::map<int, std::map<const SchedOpSeq *, SchedOpSeqPerf>>
        wps_prof_results_;
};

};  // namespace ark

#endif
