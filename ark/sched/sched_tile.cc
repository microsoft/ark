#include "third_party/json/json.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <initializer_list>
#include <ostream>
#include <unistd.h>

#include "ark/logging.h"
#include "ark/math.h"
#include "ark/sched/sched_tile.h"

using namespace std;

#define COM ", "
#define OP_PREFIX "op"

#define USE_KAHYPAR 0
#define EVAL_DEPTHS 1
#define COMPRESS_BRANCH 1
#define MATMUL_GRAPH_OPT 1
#define ALLOC_UNUSED_TENSORS 1

namespace ark {

SchedTile::SchedTile(const SchedOpSeq *opseq_, int x, int y, int z)
    : opseq{opseq_}
{
    auto &tdims = opseq_->get_tdims();
    this->id = x + (y * tdims[2]) + (z * tdims[2] * tdims[1]);
}
SchedTile::SchedTile(const SchedOpSeq *opseq_, int id_) : id{id_}, opseq{opseq_}
{
}

SchedTileSet::SchedTileSet(std::initializer_list<SchedTile> tiles_,
                           SchedTileSetType type_)
    : tiles{tiles_}, type{type_}
{
}

SchedTileDepth::SchedTileDepth(int num_sm)
{
    this->sms.resize(num_sm);
}
void SchedTileDepth::append_tiles(int sm_id, initializer_list<SchedTile> tiles,
                                  SchedTileSetType type)
{
    assert((size_t)sm_id < this->sms.size());
    this->sms[sm_id].emplace_back(tiles, type);
}

void SchedTileDepth::clear()
{
    for (auto &sm : this->sms) {
        sm.clear();
    }
}

} // namespace ark