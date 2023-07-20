// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// #include <algorithm>

#include "sched/sched_opseq.h"
#include "env.h"
#include "logging.h"
#include "math.h"

using namespace std;

namespace ark {

SchedOpSeq::SchedOpSeq(int id_) : id{id_}
{
    LOG(DEBUG, "create SchedOpSeq", id);
}

SchedOpSeq::SchedOpSeq(int id_, const Op *op, const OpConfig *cfg) : id{id_}
{
    this->append(op, cfg);
}

bool SchedOpSeq::is_send() const
{
    for (auto &sop : this->seq) {
        if (sop.get_cfg()->num_warps == 0) {
            continue;
        }
        const OpType &ot = sop.get_op()->type;
        if (ot == OP_SEND) {
            continue;
        }
        return false;
    }
    return true;
}

bool SchedOpSeq::is_send_done() const
{
    for (auto &sop : this->seq) {
        if (sop.get_cfg()->num_warps == 0) {
            continue;
        }
        const OpType &ot = sop.get_op()->type;
        if (ot == OP_SEND_DONE) {
            continue;
        }
        return false;
    }
    return true;
}

bool SchedOpSeq::is_recv() const
{
    for (auto &sop : this->seq) {
        if (sop.get_cfg()->num_warps == 0) {
            continue;
        }
        const OpType &ot = sop.get_op()->type;
        if (ot == OP_RECV) {
            continue;
        }
        return false;
    }
    return true;
}

bool SchedOpSeq::append(const Op *op, const OpConfig *cfg)
{
    assert(op != nullptr);
    assert(cfg != nullptr);
    int wn = cfg->num_warps;
    int sb = cfg->smem_bytes;
    int dx = 0;
    int dy = 0;
    int dz = 0;
    if ((op->out_deps.size() > 0) && (cfg->num_warps > 0)) {
        const Dims &s = op->out_deps[0]->shape;
        const OpTile &tile = cfg->out_deps_tiles[0];
        // TODO: temporal.
        int ndims = s.ndims();
        assert(ndims != 0);
        if (ndims == 1) {
            dx = math::div_up(s[0], tile.y);
            dy = 1;
            dz = 1;
        } else {
            dx = math::div_up(s[ndims - 1], tile.y);
            dy = math::div_up(s[ndims - 2], tile.x);
            dz = s.size() / s[ndims - 1] / s[ndims - 2];
        }
        // LOG(INFO, ">>>> ", s, ", ", dx, ", ", dy, ", ", dz);
    }
    if (dz == 0) {
        this->seq_fdims.emplace_back(1, 1);
    } else if ((this->seq.size() == 0) || (this->tdims[0] == 0)) {
        this->num_warps = wn;
        this->smem_bytes = sb;
        this->tdims[0] = dz;
        this->tdims[1] = dy;
        this->tdims[2] = dx;
        this->seq_fdims.emplace_back(1, 1);
    } else {
        // Cannot merge if proportion of the # warps of `op` to that of `opseq`
        // is not the same as proportion of the # tiles of `op` to that of
        // `opseq`.
        if ((wn * this->tdims[0] * this->tdims[1] * this->tdims[2]) !=
            (this->num_warps * dx * dy * dz)) {
            return false;
        }
        LOG(DEBUG, "Merging Ops: (", dz, ", ", dy, ", ", dx, "), #warps ", wn,
            "; + (", this->tdims[0], ", ", this->tdims[1], ", ", this->tdims[2],
            "), #warps ", this->num_warps, ";");
        if (this->tdims[0] != dz) {
            LOGERR("Tried to merge Ops with different batch sizes: ",
                   this->tdims[0], " (latest OpType ",
                   this->seq.back().get_op()->type, "), ", dz, " (OpType ",
                   op->type, ")");
        }
        int fdimsx = 1;
        int fdimsy = 1;
        if (this->tdims[2] < dx) {
            assert((dx % this->tdims[2]) == 0);
            fdimsx = dx / this->tdims[2];
            wn *= fdimsx;
            sb *= fdimsx;
        } else if (this->tdims[2] > dx) {
            assert((this->tdims[2] % dx) == 0);
            int scale = this->tdims[2] / dx;
            for (auto &fdims : this->seq_fdims) {
                fdims.first *= scale;
            }
            this->num_warps *= scale;
            this->smem_bytes *= scale;
            this->tdims[2] = dx;
        }
        if (this->tdims[1] < dy) {
            assert((dy % this->tdims[1]) == 0);
            fdimsy = dy / this->tdims[1];
            wn *= fdimsy;
            sb *= fdimsy;
        } else if (this->tdims[1] > dy) {
            assert((this->tdims[1] % dy) == 0);
            int scale = this->tdims[1] / dy;
            for (auto &fdims : this->seq_fdims) {
                fdims.second *= scale;
            }
            this->num_warps *= scale;
            this->smem_bytes *= scale;
            this->tdims[1] = dy;
        }
        LOG(DEBUG, "Merged Ops: now (", this->tdims[0], ", ", this->tdims[1],
            ", ", this->tdims[2], "), #warps ", this->num_warps);
        // if (wn == 2048) {
        //     const OpTile& tile = cfg->out_deps_tiles[0];
        //     LOG(INFO, "??????? ", this->tdims[0], ",", this->tdims[1], ",",
        //     this->tdims[2], ",", this->num_warps, ", ",
        //     this->seq_fdims[0].first, ",", this->seq_fdims[0].second);
        //     LOG(INFO, "??????? ", op->out_deps[0]->shape, ", ", tile.x,
        //     ", ", tile.y); assert(false);
        // }
        this->num_warps = max(this->num_warps, wn);
        this->smem_bytes = max(this->smem_bytes, sb);
        this->seq_fdims.emplace_back(fdimsx, fdimsy);
    }
    // LOG(INFO, "######## ", this->tdims[0], ",", this->tdims[1], ",",
    // this->tdims[2]);
    assert(this->num_warps <= 16);
    this->seq.emplace_back(op, cfg, "");
    return true;
}
bool operator<(const SchedOpSeq &ops1, const SchedOpSeq &ops2)
{
    auto &seq1 = ops1.get_sched_ops();
    auto &seq2 = ops2.get_sched_ops();
    for (size_t i = 0; i < seq1.size(); ++i) {
        if (seq2.size() <= i) {
            return true;
        } else if (seq1[i].get_cfg() != seq2[i].get_cfg()) {
            return seq1[i].get_cfg() < seq2[i].get_cfg();
        } else if (seq1[i].get_op() == nullptr) {
            return false;
        } else if (seq2[i].get_op() == nullptr) {
            return true;
        } else if (*seq1[i].get_op() == *seq2[i].get_op()) {
            continue;
        }
        return *seq1[i].get_op() < *seq2[i].get_op();
    }
    return false;
}
bool operator==(const SchedOpSeq &ops1, const SchedOpSeq &ops2)
{
    auto &seq1 = ops1.get_sched_ops();
    auto &seq2 = ops2.get_sched_ops();
    if (seq1.size() != seq2.size()) {
        return false;
    }
    for (size_t i = 0; i < seq1.size(); ++i) {
        if (seq1[i].get_cfg() != seq2[i].get_cfg()) {
            return false;
        } else if ((seq1[i].get_op() == nullptr) &&
                   (seq2[i].get_op() == nullptr)) {
            continue;
        } else if ((seq1[i].get_op() == nullptr) ||
                   (seq2[i].get_op() == nullptr)) {
            return false;
        } else if (*seq1[i].get_op() == *seq2[i].get_op()) {
            continue;
        }
        return false;
    }
    return true;
}

////////////////////////////////////////////////////////////////////////////////

// void to_json(nlohmann::json &j, const SchedOpSeq &opseq)
// {
//     // j = nlohmann::json{
//     //     {"id", opseq.get_id()},
//     //     {"seq", opseq.get_sched_ops()},
//     //     {"seq_fdims", opseq.get_fdims()},
//     //     {"num_warps", opseq.get_num_warps()},
//     //     {"smem_bytes", opseq.get_smem_bytes()},
//     //     {"tdims", opseq.get_tdims()},
//     // };
// }
// void from_json(const nlohmann::json &j, SchedOpSeq &opseq)
// {
// }

} // namespace ark