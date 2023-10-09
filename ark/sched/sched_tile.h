// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef _ARK_SCHED_TILE_H_
#define _ARK_SCHED_TILE_H_
#include "sched/sched_opseq.h"
// #include "sched/sched/sched_profiler.h"
namespace ark {

struct SchedTile {
    SchedTile(const SchedOpSeq *opseq, int x, int y, int z);
    SchedTile(const SchedOpSeq *opseq, int id);
    //
    bool operator<(const SchedTile &rhs) const {
        const int &o_id = opseq->get_id();
        const int &o_rhs_id = rhs.opseq->get_id();
        return o_id == o_rhs_id ? id < rhs.id : o_id < o_rhs_id;
    }
    //
    int id;
    const SchedOpSeq *opseq;
};

typedef enum {
    SCHED_TILE_SET_S,
    SCHED_TILE_SET_X,
    SCHED_TILE_SET_Y,
    SCHED_TILE_SET_XY,
    SCHED_TILE_SET_MIXED,
} SchedTileSetType;

struct SchedTileSet {
    SchedTileSet() {}
    SchedTileSet(std::initializer_list<SchedTile> tiles, SchedTileSetType type);

    int get_num_warps() const {
        int sum = 0;
        for (auto &t : tiles) sum += t.opseq->get_num_warps();
        return sum;
    }
    //
    std::vector<SchedTile> tiles;
    SchedTileSetType type;
    // SchedPerf perf;
};

struct SchedTileDepth {
    SchedTileDepth(int num_sm);

    void append_tiles(int sm_id, std::initializer_list<SchedTile> tiles,
                      SchedTileSetType type);
    void clear();

    int get_num_sm() const { return (int)sms.size(); }
    int get_num_warps() const {
        int max_num_warps = 0;
        for (auto &sm : sms) {
            for (auto &ts : sm) {
                int nw = ts.get_num_warps();
                if (nw > max_num_warps) max_num_warps = nw;
            }
        }
        return max_num_warps;
    }
    bool is_full() const {
        for (auto &sm : sms) {
            if (sm.size() == 0) return false;
        }
        return true;
    }
    std::vector<std::vector<SchedTileSet>> sms;
};

}  // namespace ark

#endif  // _ARK_SCHED_TILE_H_
