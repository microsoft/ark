// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "sched_stream.h"

#include <algorithm>
#include <list>
#include <map>
#include <vector>

#include "logging.h"
#include "math.h"
#include "sched_branch.h"

namespace ark {

class SchedStream::Impl {
   private:
    int sm_id_begin;
    int sm_id_end;
    int num_warps_per_sm;
    int smem_bytes_per_sm;

    struct BranchInfo {
        SchedBranch sched_branch;
        std::map<int, int> sm_id_to_smem_per_warp;
    };

    std::vector<std::unique_ptr<BranchInfo>> branch_infos;

   public:
    Impl(int sm_id_begin, int sm_id_end, int num_warps_per_sm,
         int smem_bytes_per_sm);
    ~Impl();

   protected:
    void add_items(const std::vector<SchedItem> &items);
    void sync();
    void clear();
    std::vector<Stream> get_streams();
    int get_num_sm() const;

    friend class SchedStream;
};

SchedStream::Impl::Impl(int sm_id_begin_, int sm_id_end_, int num_warps_per_sm_,
                        int smem_bytes_per_sm_)
    : sm_id_begin{sm_id_begin_},
      sm_id_end{sm_id_end_},
      num_warps_per_sm{num_warps_per_sm_},
      smem_bytes_per_sm{smem_bytes_per_sm_} {
    this->branch_infos.emplace_back(std::make_unique<BranchInfo>());
}

SchedStream::Impl::~Impl() {}

void SchedStream::Impl::add_items(const std::vector<SchedItem> &items) {
    for (auto &item : items) {
        if (item.num_warps_per_uop > this->num_warps_per_sm) {
            LOG(ERROR, "uop requires more warps (", item.num_warps_per_uop,
                ") than available on SM (", this->num_warps_per_sm, ")");
        }
        if (item.smem_bytes_per_uop > this->smem_bytes_per_sm) {
            LOG(ERROR, "uop requires more shared memory (",
                item.smem_bytes_per_uop, ") than available on SM (",
                this->smem_bytes_per_sm, ")");
        }
    }

    BranchInfo *branch_info = this->branch_infos.back().get();

    // Sort items in decreasing order of smem_bytes_per_uop / num_warps_per_uop.
    std::vector<SchedItem> sorted_items(items);
    std::sort(sorted_items.begin(), sorted_items.end(),
              [](const SchedItem &a, const SchedItem &b) {
                  int smem_per_warp_a =
                      math::div_up(a.smem_bytes_per_uop, a.num_warps_per_uop);
                  int smem_per_warp_b =
                      math::div_up(b.smem_bytes_per_uop, b.num_warps_per_uop);
                  if (smem_per_warp_a != smem_per_warp_b) {
                      return smem_per_warp_a > smem_per_warp_b;
                  } else {
                      return a.opseq_id < b.opseq_id;
                  }
              });

    int num_sms = this->sm_id_end - this->sm_id_begin;
    int warps_to_schedule = 0;
    for (auto &item : sorted_items) {
        warps_to_schedule += item.num_warps_per_uop * item.num_uops;
    }
    int target_warps_per_sm = warps_to_schedule / num_sms;
    if (target_warps_per_sm > this->num_warps_per_sm) {
        target_warps_per_sm = this->num_warps_per_sm;
    }

    // opseq_id -> SchedItem
    std::map<int, SchedItem> remaining_items;
    for (auto &item : sorted_items) {
        remaining_items[item.opseq_id] = item;
    }

    // Cache the last scheduled uop_idx for each opseq_id.
    // opseq_id -> uop_idx
    std::map<int, int> cache_scheduled_uop_idx;

    while (!remaining_items.empty()) {
        std::vector<int> n_remaining_warps(num_sms, this->num_warps_per_sm);
        std::vector<int> done_items;
        bool no_progress = true;

        for (auto &p : remaining_items) {
            auto &item = p.second;
            int current_sm_idx = this->sm_id_begin;
            int current_warp_idx = 0;
            int uop_idx = cache_scheduled_uop_idx[item.opseq_id];

            while (uop_idx < item.num_uops &&
                   current_sm_idx < this->sm_id_end) {
                int rem_warp =
                    n_remaining_warps[current_sm_idx - this->sm_id_begin];
                if (rem_warp < item.num_warps_per_uop) {
                    // No room on this SM for this uop, move to next SM
                    current_sm_idx++;
                    current_warp_idx = 0;
                    continue;
                }

                int target_rem_warp =
                    this->num_warps_per_sm -
                    std::max(item.num_warps_per_uop, target_warps_per_sm);
                if (rem_warp <= target_rem_warp) {
                    // This SM has too many warps scheduled, move to next SM
                    current_sm_idx++;
                    current_warp_idx = 0;
                    continue;
                }

                int smem_per_warp =
                    branch_info->sm_id_to_smem_per_warp[current_sm_idx];
                int item_smem_per_warp = math::div_up(item.smem_bytes_per_uop,
                                                      item.num_warps_per_uop);
                int max_smem_per_warp =
                    std::max(smem_per_warp, item_smem_per_warp);
                if (max_smem_per_warp > 0) {
                    int max_warps_per_sm =
                        std::min(this->num_warps_per_sm,
                                 this->smem_bytes_per_sm / max_smem_per_warp);
                    if (current_warp_idx + item.num_warps_per_uop >
                        max_warps_per_sm) {
                        // If we schedule this uop, we will exceed the
                        // shared memory bytes per SM. Move to next SM.
                        current_sm_idx++;
                        current_warp_idx = 0;
                        continue;
                    }
                }
                branch_info->sm_id_to_smem_per_warp[current_sm_idx] =
                    max_smem_per_warp;

                // Schedule this uop on this SM
                branch_info->sched_branch.add(
                    item.opseq_id, uop_idx, current_sm_idx, current_warp_idx,
                    current_warp_idx + item.num_warps_per_uop);
                if (current_warp_idx + item.num_warps_per_uop >
                    this->num_warps_per_sm) {
                    LOG(ERROR, "unexpected error");
                }
                n_remaining_warps[current_sm_idx - this->sm_id_begin] -=
                    item.num_warps_per_uop;
                current_warp_idx += item.num_warps_per_uop;
                uop_idx++;
                no_progress = false;
            }
            if (uop_idx == item.num_uops) {
                done_items.push_back(item.opseq_id);
            }
            cache_scheduled_uop_idx[item.opseq_id] = uop_idx;
        }

        if (no_progress) {
            // No progress can be made, which means that the current
            // smem_per_warp is too small. Create a new SchedBranch and
            // try again.
            // TODO: upon this case, we need to force __syncthreads() on
            // all SMs of this stream for safety.
            this->sync();
            branch_info = this->branch_infos.back().get();
            continue;
        }

        // Remove items that are done
        for (int done_opseq_id : done_items) {
            remaining_items.erase(done_opseq_id);
        }
    }

    if (sorted_items.size() > 0) {
        this->sync();
    }
}

void SchedStream::Impl::sync() {
    this->branch_infos.emplace_back(std::make_unique<BranchInfo>());
}

void SchedStream::Impl::clear() { this->branch_infos.clear(); }

std::vector<Stream> SchedStream::Impl::get_streams() {
    std::vector<Stream> streams;
    for (auto &branch_info : this->branch_infos) {
        Stream stream;
        stream.branches = branch_info->sched_branch.get_branches(
            branch_info->sm_id_to_smem_per_warp);
        streams.emplace_back(std::move(stream));
    }
    return streams;
}

int SchedStream::Impl::get_num_sm() const {
    return this->sm_id_end - this->sm_id_begin;
}

SchedStream::SchedStream(int sm_id_begin, int sm_id_end, int num_warps_per_sm,
                         int smem_bytes_per_sm) {
    this->impl = std::make_unique<Impl>(sm_id_begin, sm_id_end,
                                        num_warps_per_sm, smem_bytes_per_sm);
}

SchedStream::~SchedStream() {}

void SchedStream::add_items(const std::vector<SchedItem> &items) {
    this->impl->add_items(items);
}

void SchedStream::sync() { this->impl->sync(); }

void SchedStream::clear() { this->impl->clear(); }

std::vector<Stream> SchedStream::get_streams() {
    return this->impl->get_streams();
}

int SchedStream::get_num_sm() const { return this->impl->get_num_sm(); }

}  // namespace ark
