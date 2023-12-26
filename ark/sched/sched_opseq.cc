// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "sched/sched_opseq.h"

#include "env.h"
#include "logging.h"
#include "math_utils.h"

using namespace std;

namespace ark {

SchedOpSeq::SchedOpSeq(int id_) : id{id_} {
    LOG(DEBUG, "create SchedOpSeq", id);
}

SchedOpSeq::SchedOpSeq(int id_, const Op *op, const OpConfig *cfg) : id{id_} {
    this->append(op, cfg);
}

bool SchedOpSeq::is_send() const {
    for (auto &sop : this->seq) {
        if (sop.is_virtual()) {
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

bool SchedOpSeq::is_send_done() const {
    for (auto &sop : this->seq) {
        if (sop.is_virtual()) {
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

bool SchedOpSeq::is_recv() const {
    for (auto &sop : this->seq) {
        if (sop.is_virtual()) {
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

bool SchedOpSeq::is_sync() const {
    for (auto &sop : this->seq) {
        if (sop.is_virtual()) {
            continue;
        }
        const OpType &ot = sop.get_op()->type;
        if (ot == OP_DEVICE_SYNC) {
            continue;
        }
        return false;
    }
    return true;
}

bool SchedOpSeq::is_comm() const {
    return this->is_send() || this->is_send_done() || this->is_recv() ||
           this->is_sync();
}

bool SchedOpSeq::append(const Op *op, const OpConfig *cfg) {
    assert(op != nullptr);
    int dx = 0;
    int dy = 0;
    int dz = 0;
    int wn;
    int sb;
    if (cfg != nullptr) {
        wn = cfg->num_warps;
        sb = cfg->smem_bytes;
    } else {
        wn = 0;
        sb = 0;
    }
    if ((op->outputs.size() > 0) && (wn > 0)) {
        const Dims &s = op->outputs[0]->shape;
        OpTile tile = cfg->output_tiles[0];
        if (tile.x < 0) tile.x = s.dims4()[2];
        if (tile.y < 0) tile.y = s.dims4()[3];

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
    }
    if (dz == 0) {
    } else if ((this->seq.size() == 0) || (this->tdims[0] == 0)) {
        this->num_warps = wn;
        this->smem_bytes = sb;
        this->tdims[0] = dz;
        this->tdims[1] = dy;
        this->tdims[2] = dx;
    } else {
        // Merge condition
        if (wn != this->num_warps || this->tdims[0] != dz ||
            this->tdims[1] != dy || this->tdims[2] != dx) {
            return false;
        }
        this->smem_bytes = max(this->smem_bytes, sb);
    }
    this->seq.emplace_back(op, cfg, "");
    return true;
}
bool operator<(const SchedOpSeq &ops1, const SchedOpSeq &ops2) {
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
bool operator==(const SchedOpSeq &ops1, const SchedOpSeq &ops2) {
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

}  // namespace ark
