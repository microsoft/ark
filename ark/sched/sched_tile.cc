// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "sched/sched_tile.h"

#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <fstream>
#include <initializer_list>
#include <ostream>

#include "json.h"
#include "logging.h"
#include "math.h"

using namespace std;

namespace ark {

SchedTile::SchedTile(const SchedOpSeq *opseq_, int x, int y, int z)
    : opseq{opseq_} {
    auto &tdims = opseq_->get_tdims();
    this->id = x + (y * tdims[2]) + (z * tdims[2] * tdims[1]);
}
SchedTile::SchedTile(const SchedOpSeq *opseq_, int id_)
    : id{id_}, opseq{opseq_} {}

SchedTileSet::SchedTileSet(std::initializer_list<SchedTile> tiles_,
                           SchedTileSetType type_)
    : tiles{tiles_}, type{type_} {}

SchedTileDepth::SchedTileDepth(int num_sm) { this->sms.resize(num_sm); }
void SchedTileDepth::append_tiles(int sm_id, initializer_list<SchedTile> tiles,
                                  SchedTileSetType type) {
    assert((size_t)sm_id < this->sms.size());
    this->sms[sm_id].emplace_back(tiles, type);
}

void SchedTileDepth::clear() {
    for (auto &sm : this->sms) {
        sm.clear();
    }
}

}  // namespace ark