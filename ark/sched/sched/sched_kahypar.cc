// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#include "ark/kahypar.h"
#include "ark/logging.h"
#include "ark/math.h"
#include "ark/sched/sched.h"

using namespace std;

#define PRESERVE_WARP_FOR_COMM 1

namespace ark {

KahyparScheduler::KahyparScheduler(const int gpu_id, int rank_, int world_size_,
                                   const Model &model, unsigned int wps_)
    : DefaultScheduler(gpu_id, rank_, world_size_, model, wps_),
      profiler{get_gpu_mgr(gpu_id), wps_}
{
}

vector<Sched> KahyparScheduler::simplify_sched(vector<Sched> &original_scheds)
{
    vector<Sched> merged_scheds;
    SchedOpSeq *opseq = NULL;
    int sm_b = 0;
    int sm_e = 0;
    int th_b = 0;
    int th_e = 0;
    int alpha = 1;
    int beta = 0;
    int id = 0;
    // merge the scheds that are in the sequential order
    for (auto &sched : original_scheds) {
        if (opseq == sched.opseq && sm_e == sched.sm_b && th_e == sched.th_e &&
            th_b == sched.th_b && id + 1 == sched.beta) {
            // we can merge the sched into current scheds
            LOG(DEBUG, " merged op", opseq->get_id(), " sm_b ", sm_b, " sm_e",
                sm_e, " th_b ", th_b, " th_e ", th_e, " id ", beta);
            sm_e = sched.sm_e;
            id++;
        } else {
            // we cannot merge the sched into current scheds
            if (opseq != NULL) {
                LOG(DEBUG, "  op", opseq->get_id(), " sm_b ", sm_b, " sm_e ",
                    sm_e, " th_b ", th_b, " th_e ", th_e, " id ", beta);
                LOG(DEBUG, "  op", sched.opseq->get_id(), " sm_b ", sched.sm_b,
                    " sm_e ", sched.sm_e, " th_b ", sched.th_b, " th_e ",
                    sched.th_e, " id ", sched.beta);
            }
            if (opseq != NULL)
                merged_scheds.emplace_back(opseq, sm_b, sm_e, th_b, th_e, alpha,
                                           beta);
            opseq = sched.opseq;
            sm_b = sched.sm_b;
            sm_e = sched.sm_e;
            th_b = sched.th_b;
            th_e = sched.th_e;
            // alpha = sched.alpha;
            beta = sched.beta;
            id = beta;
        }
    }
}

function<SchedTile *()> gen_tile_nodes(const SchedOpSeq *opseq, int xz_min = -1,
                                       int xz_max = -1, int y_min = -1,
                                       int y_max = -1)
{
    int xz_b = (xz_min == -1) ? 0 : xz_min;
    int xz_e = (xz_max == -1) ? opseq->get_tdim_xz() : xz_max + 1;
    int y_b = (y_min == -1) ? 0 : y_min;
    int y_e = (y_max == -1) ? opseq->get_tdim_y() : y_max + 1;
    return [opseq, xz_b, xz_e, y_b, y_e] {
        const int ydim = opseq->get_tdim_y();
        int xzidx = xz_b;
        int yidx = y_b;
        int id = yidx + xzidx * ydim;
        return [=]() mutable {
            if (xzidx >= xz_e) {
                return (SchedTile *)nullptr;
            }
            SchedTile *ret = new SchedTile{opseq, id++};
            assert(ret != nullptr);
            if (++yidx == y_e) {
                ++xzidx;
                yidx = y_b;
                id = yidx + xzidx * ydim;
            }
            return ret;
        };
    }();
}

function<SchedTile *()> gen_tile_nodes(const SchedOpSeq *opseq0,
                                       const SchedOpSeq *opseq1)
{
    return [=] {
        SchedOpSeq *opseq = (SchedOpSeq *)opseq0;
        int max_id = opseq0->get_tdims_size();
        int id = 0;
        return [=]() mutable {
            SchedTile *ret = new SchedTile{opseq, id++};
            assert(ret != nullptr);
            if (id == max_id) {
                if (opseq == opseq1) {
                    return (SchedTile *)nullptr;
                }
                opseq = (SchedOpSeq *)opseq1;
                max_id = opseq1->get_tdims_size();
                id = 0;
            }
            return ret;
        };
    }();
}

int KahyparScheduler::kahypar_schedule_depth(vector<SchedOpSeq *> &depth,
                                             vector<Sched> &scheds)
{
    const GpuInfo &gpu_info = this->gpu_mgr->get_gpu_info();
#ifdef PRESERVE_WARP_FOR_COMM
    // Number of SMs to use for computation. The last SM is preserved for
    // communication only.
    int num_sm_calc = gpu_info.num_sm - 1;
#else
    int num_sm_calc = gpu_info.num_sm;
#endif
    KahyparGraph<SchedTile> kg;
    auto search = this->profiler.wps_prof_results.find(this->wps);
    if (search == this->profiler.wps_prof_results.end()) {
        LOGERR("Unexpected error.");
    }
    auto &opseq_perf = search->second;
    for (auto &opseq : depth) {
        auto &perf = opseq_perf[opseq];
        // Add tile nodes.
        int vw = perf.s.elapsed * 1e3;
        kg.add_nodes(vw, gen_tile_nodes(opseq));
        // Check the minimum scored inter-tile edge.
        int min_score = -1;
        if (perf.xy.is_set()) {
            min_score = perf.xy.score;
        }
        if (perf.x.is_set() && min_score > perf.x.score) {
            min_score = perf.x.score;
        }
        if (perf.y.is_set() && min_score > perf.x.score) {
            min_score = perf.x.score;
        }
        if (min_score == -1) {
            // No inter-tile edges.
            continue;
        }
        // Add an edge connecting all vertices.
        int ew = perf.x.score - min_score;
        kg.add_edge(ew, gen_tile_nodes(opseq));
        if (perf.x.is_set() && perf.x.score > min_score * 1.03) {
            // Add edges connecting vertices with the same y-axis, i.e.
            // intra-x-axis correlation.
            ew = perf.x.score - min_score;
            for (int yidx = 0; yidx < opseq->get_tdim_y(); ++yidx) {
                // Weak overall correlations.
                kg.add_edge(ew * 0.05,
                            gen_tile_nodes(opseq, -1, -1, yidx, yidx));
                // Strong adjacent correlations.
                for (int xzidx = 1; xzidx < opseq->get_tdim_xz(); ++xzidx) {
                    kg.add_edge(ew, gen_tile_nodes(opseq, xzidx - 1, xzidx,
                                                   yidx, yidx));
                }
            }
        }
        if (perf.y.is_set() && (perf.y.score > min_score * 1.03)) {
            // Add edges connecting vertices with the same x-axis, i.e.
            // intra-y-axis correlation.
            ew = perf.y.score - min_score;
            for (int xzidx = 0; xzidx < opseq->get_tdim_xz(); ++xzidx) {
                // Weak overall correlations.
                kg.add_edge(ew * 0.05,
                            gen_tile_nodes(opseq, xzidx, xzidx, -1, -1));
                // Strong adjacent correlations.
                for (int yidx = 1; yidx < opseq->get_tdim_y(); ++yidx) {
                    kg.add_edge(ew, gen_tile_nodes(opseq, xzidx, xzidx,
                                                   yidx - 1, yidx));
                }
            }
        }
    }
    // Add inter-taskset edges.
    set<const SchedOpSeq *> seen;
    for (auto &opseq0 : depth) {
        auto &perf = opseq_perf[opseq0];
        seen.insert(opseq0);
        for (auto &el : perf.mixed) {
            auto &opseq1 = el.first;
            auto search = seen.find(opseq1);
            if (search != seen.end()) {
                continue;
            }
            int ew = el.second.score;
            kg.add_edge(ew, gen_tile_nodes(opseq0, opseq1));
            kg.add_edge(-ew, gen_tile_nodes(opseq0));
            kg.add_edge(-ew, gen_tile_nodes(opseq1));
        }
    }
    //
    if (kg.get_num_nodes() > num_sm_calc) {
        LOG(DEBUG, "Partitioning graph with ", kg.get_num_nodes(), " nodes");
        auto &parts = kg.partition(num_sm_calc);
        SchedTileDepth *tile_depth = new SchedTileDepth(num_sm_calc);
        for (int i = 0; i < num_sm_calc; ++i) {
            auto &sm = tile_depth->sms[i];
            auto &part = parts[i];
            if (part.size() > 0)
                sm.emplace_back();
            for (auto &p : part) {
                if (sm.back().get_num_warps() >= this->wps)
                    sm.emplace_back();
                sm.back().tiles.emplace_back(move(*p.first));
                if (p.second == nullptr)
                    continue;
                if (sm.back().get_num_warps() >= this->wps)
                    sm.emplace_back();
                sm.back().tiles.emplace_back(move(*p.second));
            }
        }
        std::vector<std::vector<SchedTileSet>> &sms = tile_depth->sms;
        // LOG(DEBUG, "Sorting tiles");
        // order the tiles in each SM by their id
        // sort(sms.begin(), sms.end(),
        //      [](const std::vector<SchedTileSet> &a,
        //         const std::vector<SchedTileSet> &b) {
        //          return a[0].tiles[0] < b[0].tiles[0];
        //      });
        // LOG(DEBUG, "Generating tiles");
        auto scheds_ = gen_sched(tile_depth, this->wps);
        LOG(DEBUG, "Generated ", scheds_.size(), " scheds");
        for (auto sched : scheds_) {
            scheds.push_back(sched);
        }
        return 0;
    } else {
        LOG(DEBUG, "Can't generate kahypar scheds from graph with ",
            kg.get_num_nodes(), " nodes");
        return -1;
    }
}

vector<string> KahyparScheduler::schedule()
{
    LOG(DEBUG, "KahyparScheduler start scheduling");

    LOG(DEBUG, "profiling the ops ...");
    this->profiler.profile(this->op_graph, this->scg, this->ctx);
    vector<Sched> scheds;
    vector<GpuLoopKernel *> glks;
    for (auto &depth : this->op_graph->depth_nodes) {
        vector<Sched> ds;
        vector<SchedOpSeq *> calc_opseqs;
        vector<SchedOpSeq *> send_opseqs;
        vector<SchedOpSeq *> recv_opseqs;
        for (auto &ogn : depth) {
            if (ogn->opseq.is_send()) {
                send_opseqs.emplace_back(&(ogn->opseq));
            } else if (ogn->opseq.is_recv()) {
                recv_opseqs.emplace_back(&(ogn->opseq));
            } else {
                calc_opseqs.emplace_back(&(ogn->opseq));
            }
        }
        LOG(DEBUG, "schedule depth");
        this->schedule_depth_comm(send_opseqs, scheds);
        // The kahypar schedule algorithm only works for the calculation ops. If
        // the tile number is less than sm_num, we will use the original
        // schedule algorithm instead.
        if (this->kahypar_schedule_depth(calc_opseqs, scheds) != 0) {
            LOG(DEBUG, "schedule depth calc ops failed");
            this->schedule_depth(calc_opseqs, scheds);
        }
        this->schedule_depth_comm(recv_opseqs, scheds);
        // TODO: profile one depth
        // Global sync.
        scheds.emplace_back(nullptr, 0, 0, 0, 0, 0, 0);
    }
    return this->scg.codegen_codes_body(scheds);
}

} // namespace ark
