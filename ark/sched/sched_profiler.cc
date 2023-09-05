// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "json.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <initializer_list>
#include <ostream>
#include <unistd.h>

#include "logging.h"
#include "math.h"
#include "sched/sched_opseq.h"
#include "sched/sched_profiler.h"
#include "sched/sched_tile.h"

using namespace std;

namespace ark {

float SchedProfiler::profile_routine(GpuLoopKernel *glk, GpuMgrCtx *ctx)
{
    const int probe_iter = 10;

    glk->load();
    GpuState ret = glk->launch(ctx->create_stream(), false);
    if (ret != CUDA_SUCCESS) {
        LOG(ERROR, "launch() failed with error code ", ret);
    }

    int test_iter = probe_iter * 1e2 / 1;
    if (test_iter == 0)
        test_iter = 1;
    // LOG(DEBUG, "test_iter", test_iter);
    glk->run(test_iter);
    glk->stop();
    // msec->usec. Drop values below usec.
    return (float)((glk->get_elapsed_msec() / test_iter * 1e3));
}

struct ProfInfo
{
    ProfInfo(const SchedTileSetType &type_, const SchedOpSeq *opseq0_,
             const SchedOpSeq *opseq1_ = nullptr)
        : type{type_}, opseq0{opseq0_}, opseq1{opseq1_}
    {
        if (type == SCHED_TILE_SET_MIXED) {
            assert(opseq1_ != nullptr);
        } else {
            assert(opseq1_ == nullptr);
        }
    }

    const string get_name() const
    {
        if (type == SCHED_TILE_SET_S) {
            return "prof_" + to_string(opseq0->get_id()) + "_s";
        } else if (type == SCHED_TILE_SET_X) {
            return "prof_" + to_string(opseq0->get_id()) + "_x";
        } else if (type == SCHED_TILE_SET_Y) {
            return "prof_" + to_string(opseq0->get_id()) + "_y";
        } else if (type == SCHED_TILE_SET_XY) {
            return "prof_" + to_string(opseq0->get_id()) + "_xy";
        } else if (type == SCHED_TILE_SET_MIXED) {
            return "prof_" + to_string(opseq0->get_id()) + "_" +
                   to_string(opseq1->get_id());
        }
    }

    const SchedTileSetType type;
    const SchedOpSeq *opseq0;
    const SchedOpSeq *opseq1;
};

bool operator<(const ProfInfo &info0, const ProfInfo &info1)
{
    if (!(*info0.opseq0 == *info1.opseq0)) {
        return *info0.opseq0 < *info1.opseq0;
    } else if (info0.opseq1 != nullptr && info1.opseq1 == nullptr) {
        return true;
    } else if (info0.opseq1 == nullptr && info1.opseq1 != nullptr) {
        return false;
    } else if (info0.opseq1 == nullptr && info1.opseq1 == nullptr) {
        return info0.type < info1.type;
    } else if (!(*info0.opseq1 == *info1.opseq1)) {
        return *info0.opseq1 < *info1.opseq1;
    }
    return info0.type < info1.type;
}

// static string prof_name(const SchedTileSet &ts)
// {
//     const int &id0 = ts.tiles[0].opseq->get_id();
//     if (ts.type == SCHED_TILE_SET_MIXED) {
//         const int &id1 = ts.tiles[1].opseq->get_id();
//         return "prof_" + to_string(id0) + "_" + to_string(id1);
//     } else if (ts.type == SCHED_TILE_SET_S) {
//         return "prof_" + to_string(id0) + "_s";
//     } else if (ts.type == SCHED_TILE_SET_X) {
//         return "prof_" + to_string(id0) + "_x";
//     } else if (ts.type == SCHED_TILE_SET_Y) {
//         return "prof_" + to_string(id0) + "_y";
//     } else {
//         assert(ts.type == SCHED_TILE_SET_XY);
//         return "prof_" + to_string(id0) + "_xy";
//     }
// }

// convert SchedTileDepth to Sched
vector<Sched> gen_sched(SchedTileDepth *tile_depths, int num_warps_per_sm)
{
    vector<Sched> scheds;
    int sm_b = 0;
    int sm_e = 1;
    int th_b = 0;
    int th_e = 0;
    for (auto &sm : tile_depths->sms) {
        th_e = 0;
        th_b = 0;
        // execute all tiles in the sm sm_b
        vector<Sched> tmp_sched;
        for (auto &tileset : sm) {
            // execute all tiles in the tileset
            for (auto &tile : tileset.tiles) {
                int id = tile.id;
                SchedOpSeq *opseq = const_cast<ark::SchedOpSeq *>(tile.opseq);
                // DimType tnum = opseq->get_tdims_size();
                DimType wnum = opseq->get_num_warps();
                th_e = th_b + wnum * 32;
                if (th_e > num_warps_per_sm * 32) {
                    th_b = 0;
                    th_e = wnum * 32;
                }
                // LOG(DEBUG, "  op", opseq->get_id(), ": tnum ", tnum, " wnum",
                //     wnum, " sm_b ", sm_b, " sm_e ", sm_e, " th_b ", th_b,
                //     " th_e ", th_e, " id ", id);
                tmp_sched.emplace_back(opseq, sm_b, sm_e, th_b, th_e, 1, id);
                th_b = th_e;
            }
        }
        // order the tmp_sched by the th_b
        std::sort(tmp_sched.begin(), tmp_sched.end(),
                  [](const Sched &sched1, const Sched &sched2) {
                      return sched1.th_b < sched2.th_b;
                  });
        // push the tmp_sched to scheds
        for (auto &sched : tmp_sched) {
            // LOG(DEBUG, "  op", sched.opseq->get_id(), ": tnum ",
            //     sched.opseq->get_tdims_size(), " wnum",
            //     sched.opseq->get_num_warps(), " sm_b ", sched.sm_b, " sm_e ",
            //     sched.sm_e, " th_b ", sched.th_b, " th_e ", sched.th_e, " id
            //     ", sched.beta);
            scheds.push_back(sched);
        }
        sm_b++;
        sm_e++;
    }
    return scheds;
}

void SchedProfiler::profile(OpGraph *, CodeGenerator &, GpuMgrCtx *)
{
#if 0
    using ProfCallback = function<void(float, int)>;
    const GpuInfo &gpu_info = this->gpu_mgr->get_gpu_info();
    // Get or create entry of profile results.
    auto &res = this->wps_prof_results[this->num_warps_per_sm];
    //
    map<ProfInfo, unsigned int> info2id;
    vector<vector<tuple<SchedTileDepth *, ProfCallback>>> to_prof;
    //
    size_t num_depth = op_graph->get_num_depth();
    for (size_t depth = 0; depth < num_depth; ++depth) {
        auto &depth_nodes = op_graph->get_depth(depth);
        for (auto &ogn : depth_nodes) {
            auto &opseq = ogn->opseq;
            if (opseq.is_send() || opseq.is_recv() || opseq.is_send_done() ||
                opseq.is_virtual())
                continue;
            SchedOpSeqPerf &perf = res[&opseq];
            // Skip if this already has profiled results.
            if (perf.s.is_set())
                continue;
            assert(opseq.get_num_warps() <= this->num_warps_per_sm);
            //
            {
                auto p =
                    info2id.emplace(ProfInfo{SCHED_TILE_SET_S, &opseq, nullptr},
                                    info2id.size());
                if (p.second)
                    to_prof.emplace_back();
                assert(to_prof.size() > p.first->second);
                SchedTileDepth *sd = new SchedTileDepth{1};
                sd->append_tiles(0, {{&opseq, 0}}, SCHED_TILE_SET_S);
                to_prof[p.first->second].emplace_back(
                    sd, [&](float e, int r) { perf.s.set(e, r); });
            }
            if (this->num_warps_per_sm < opseq.get_num_warps() * 2 ||
                gpu_info.smem_block_total < opseq.get_smem_bytes() * 2)
                // Cannot run two tiles concurrently in a SM.
                continue;
            //
            if (opseq.get_tdim_x() > 1) {
                auto p =
                    info2id.emplace(ProfInfo{SCHED_TILE_SET_X, &opseq, nullptr},
                                    info2id.size());
                if (p.second)
                    to_prof.emplace_back();
                assert(to_prof.size() > p.first->second);
                SchedTileDepth *sd = new SchedTileDepth{1};
                sd->append_tiles(0, {{&opseq, 0, 0, 0}, {&opseq, 1, 0, 0}},
                                 SCHED_TILE_SET_X);
                to_prof[p.first->second].emplace_back(sd, [&](float e, int r) {
                    perf.x.set(e, r, (perf.s.elapsed * 2 - e) * 1e3);
                });
            }
            if (opseq.get_tdim_y() > 1) {
                auto p =
                    info2id.emplace(ProfInfo{SCHED_TILE_SET_Y, &opseq, nullptr},
                                    info2id.size());
                if (p.second)
                    to_prof.emplace_back();
                assert(to_prof.size() > p.first->second);
                SchedTileDepth *sd = new SchedTileDepth{1};
                sd->append_tiles(0, {{&opseq, 0, 0, 0}, {&opseq, 0, 1, 0}},
                                 SCHED_TILE_SET_Y);
                to_prof[p.first->second].emplace_back(sd, [&](float e, int r) {
                    perf.y.set(e, r, (perf.s.elapsed * 2 - e) * 1e3);
                });
            }
            if (opseq.get_tdim_x() > 1 && opseq.get_tdim_y() > 1) {
                auto p = info2id.emplace(
                    ProfInfo{SCHED_TILE_SET_XY, &opseq, nullptr},
                    info2id.size());
                if (p.second)
                    to_prof.emplace_back();
                assert(to_prof.size() > p.first->second);
                SchedTileDepth *sd = new SchedTileDepth{1};
                sd->append_tiles(0, {{&opseq, 0, 0, 0}, {&opseq, 1, 1, 0}},
                                 SCHED_TILE_SET_XY);
                to_prof[p.first->second].emplace_back(sd, [&](float e, int r) {
                    perf.xy.set(e, r, (perf.s.elapsed * 2 - e) * 1e3);
                });
            }
        }
        // Inter-Op profiling
        for (auto it0 = depth_nodes.begin(); it0 != depth_nodes.end(); ++it0) {
            const SchedOpSeq &opseq0 = (*it0)->opseq;
            if (opseq0.is_send() || opseq0.is_recv() || opseq0.is_send_done() ||
                opseq0.is_virtual())
                continue;
            if (opseq0.get_num_warps() >= this->num_warps_per_sm)
                continue;
            SchedOpSeqPerf &perf0 = res[&opseq0];
            for (auto it1 = next(it0); it1 != depth_nodes.end(); ++it1) {
                const SchedOpSeq &opseq1 = (*it1)->opseq;
                if (perf0.mixed[&opseq1].is_set())
                    continue;
                if (opseq0.get_num_warps() + opseq1.get_num_warps() > this->num_warps_per_sm)
                    continue;
                //
                auto p = info2id.emplace(
                    ProfInfo{SCHED_TILE_SET_MIXED, &opseq0, &opseq1},
                    info2id.size());
                if (p.second)
                    to_prof.emplace_back();
                assert(to_prof.size() > p.first->second);
                SchedTileDepth *sd = new SchedTileDepth{1};
                sd->append_tiles(0, {{&opseq0, 0, 0, 0}, {&opseq1, 0, 0, 0}},
                                 SCHED_TILE_SET_MIXED);
                to_prof[p.first->second].emplace_back(sd, [&](float e, int r) {
                    SchedOpSeqPerf &perf1 = res[&opseq1];
                    assert(perf0.s.is_set());
                    assert(perf1.s.is_set());
                    float &e0 = perf0.s.elapsed;
                    float &e1 = perf1.s.elapsed;
                    float emin = e0 > e1 ? e1 : e0;
                    float emax = e0 > e1 ? e0 : e1;
                    int score = (e0 + e1 - e) / emin * emax * 1000;
                    perf0.mixed[&opseq1].set(e, r, score);
                    perf1.mixed[&opseq0].set(e, r, score);
                });
            }
        }
    }

    bool cache_exists = (access(this->wps_prof_cache_path.c_str(), F_OK) != -1);
    nlohmann::json cache_json;
    nlohmann::json wps_cache_json;
    if (cache_exists) {
        try {
            ifstream cache_file_stream(this->wps_prof_cache_path);
            cache_file_stream >> cache_json;
            wps_cache_json = cache_json.at(to_string(this->num_warps_per_sm));
        } catch (...) {
            wps_cache_json = {};
        }
    }

    for (auto &tp : to_prof) {
        assert(tp.size() > 0);
        auto &pf = tp[0];
        SchedTileDepth *tile_depth = get<0>(pf);
        string name = prof_name(tile_depth->sms[0][0]);
        auto search = wps_cache_json.find(name);
        if (search != wps_cache_json.end()) {
            float e = wps_cache_json[name][0].get<float>();
            int r = wps_cache_json[name][1].get<int>();
            // Call all callbacks.
            for (auto &p : tp)
                get<1>(p)(e, r);
            return;
        }
        assert(tile_depth != nullptr);
        assert(tile_depth->get_num_warps() <= this->num_warps_per_sm);
        // `gpu_loop_kernel()` is supposed to be thread-safe
        // as long as `glk` is excessed by only a single thread.
        auto scheds = gen_sched(tile_depth, this->num_warps_per_sm);
        auto codes = codegen.codegen_codes_body(scheds);

        GpuLoopKernel *glk =
            new GpuLoopKernel(name, codes, gpu_info.num_sm, this->num_warps_per_sm,
                              gpu_info.smem_block_total, "", ctx, 1);
        glk->compile(gpu_info);
        float e = this->profile_routine(glk, ctx);
        int r = glk->get_function_attribute(CU_FUNC_ATTRIBUTE_NUM_REGS);
        // Store profile results.
        // Call all callbacks & cache results.
        LOG(INFO, name, ' ', e, "us ", r, "regs");

        for (auto &p : tp) {
            get<1>(p)(e, r);
            wps_cache_json[prof_name(get<0>(p)->sms[0][0])] = {e, r};
        }
        // Performance summary.
    }
    // write the profiling results as json to the cache file
    ofstream cache_file_stream(this->wps_prof_cache_path);
    cache_json[to_string(this->num_warps_per_sm)] = wps_cache_json;
    cache_file_stream << cache_json;
#endif
}

} // namespace ark