// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "sched_stream.h"
#include "logging.h"
#include "math.h"
#include "sched_branch.h"
#include <list>
#include <map>
#include <vector>

namespace ark {

class SchedStream::Impl
{
  private:
    int sm_id_begin;
    int sm_id_end;
    int num_warps_per_sm;
    int smem_bytes_per_sm;
    std::vector<std::unique_ptr<SchedBranch>> branches;

  public:
    Impl(int sm_id_begin, int sm_id_end, int num_warps_per_sm,
         int smem_bytes_per_sm);
    ~Impl();

  protected:
    void add_items(const std::vector<SchedItem> &items);
    void sync();
    void clear();
    std::vector<std::vector<Branch>> get_branches();
    int get_num_sm() const;

    friend class SchedStream;
};

SchedStream::Impl::Impl(int sm_id_begin_, int sm_id_end_, int num_warps_per_sm_,
                        int smem_bytes_per_sm_)
    : sm_id_begin{sm_id_begin_}, sm_id_end{sm_id_end_},
      num_warps_per_sm{num_warps_per_sm_}, smem_bytes_per_sm{smem_bytes_per_sm_}
{
    this->branches.emplace_back(std::make_unique<SchedBranch>());
}

SchedStream::Impl::~Impl()
{
}

void SchedStream::Impl::add_items(const std::vector<SchedItem> &items)
{
    for (auto &item : items) {
        if (item.num_warps_per_uop > this->num_warps_per_sm) {
            LOG(ERROR, "uop requires more warps than available on SM");
        }
        if (item.smem_bytes_per_uop > this->smem_bytes_per_sm) {
            LOG(ERROR, "uop requires more shared memory than available on SM");
        }
    }

    int num_sms = this->sm_id_end - this->sm_id_begin;
    int warps_to_schedule = 0;
    for (auto &item : items) {
        warps_to_schedule += item.num_warps_per_uop * item.num_uops;
    }
    int target_warps_per_sm = warps_to_schedule / num_sms;
    if (target_warps_per_sm > this->num_warps_per_sm) {
        target_warps_per_sm = this->num_warps_per_sm;
    }

    // opseq_id -> SchedItem
    std::map<int, SchedItem> remaining_items;
    for (auto &item : items) {
        remaining_items[item.opseq_id] = item;
    }

    // opseq_id -> uop_idx
    std::map<int, int> uop_idx_cache;

    while (!remaining_items.empty()) {
        std::vector<int> remaining_smem_bytes(num_sms, this->smem_bytes_per_sm);
        std::vector<int> remaining_warps(num_sms, this->num_warps_per_sm);
        std::vector<int> done_items;
        bool no_progress = true;

        for (auto &p : remaining_items) {
            auto &item = p.second;
            int current_sm_idx = this->sm_id_begin;
            int current_warp_idx = 0;
            int uop_idx = uop_idx_cache[item.opseq_id];

            while (uop_idx < item.num_uops &&
                   current_sm_idx < this->sm_id_end) {
                int rem_smem =
                    remaining_smem_bytes[current_sm_idx - this->sm_id_begin];
                int rem_warp =
                    remaining_warps[current_sm_idx - this->sm_id_begin];
                if (rem_smem < item.smem_bytes_per_uop ||
                    rem_warp < item.num_warps_per_uop) {
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
                // Schedule this uop on this SM
                this->branches.back()->add(
                    item.opseq_id, uop_idx, current_sm_idx, current_warp_idx,
                    current_warp_idx + item.num_warps_per_uop);
                remaining_smem_bytes[current_sm_idx - this->sm_id_begin] -=
                    item.smem_bytes_per_uop;
                remaining_warps[current_sm_idx - this->sm_id_begin] -=
                    item.num_warps_per_uop;
                current_warp_idx += item.num_warps_per_uop;
                uop_idx++;
                no_progress = false;
            }
            if (uop_idx == item.num_uops) {
                done_items.push_back(item.opseq_id);
            }
            uop_idx_cache[item.opseq_id] = uop_idx;
        }

        if (no_progress) {
            auto &item = remaining_items.begin()->second;
            LOG(ERROR,
                "Unable to schedule any items (given resources: "
                "num_warps_per_sm=",
                this->num_warps_per_sm,
                " smem_bytes_per_sm=", this->smem_bytes_per_sm,
                "). For example: opseq_id=", item.opseq_id,
                " num_uops=", item.num_uops,
                " num_warps_per_uop=", item.num_warps_per_uop,
                " smem_bytes_per_uop=", item.smem_bytes_per_uop);
        }

        // Remove items that are done
        for (int done_opseq_id : done_items) {
            remaining_items.erase(done_opseq_id);
        }
    }
}

void SchedStream::Impl::sync()
{
    this->branches.emplace_back(std::make_unique<SchedBranch>());
}

void SchedStream::Impl::clear()
{
    branches.clear();
}

std::vector<std::vector<Branch>> SchedStream::Impl::get_branches()
{
    std::vector<std::vector<Branch>> branches;
    for (auto &branch : this->branches) {
        branches.emplace_back(std::move(branch->get_branches()));
    }
    return branches;
}

int SchedStream::Impl::get_num_sm() const
{
    return this->sm_id_end - this->sm_id_begin;
}

SchedStream::SchedStream(int sm_id_begin, int sm_id_end, int num_warps_per_sm,
                         int smem_bytes_per_sm)
{
    this->impl = std::make_unique<Impl>(sm_id_begin, sm_id_end,
                                        num_warps_per_sm, smem_bytes_per_sm);
}

SchedStream::~SchedStream()
{
}

void SchedStream::add_items(const std::vector<SchedItem> &items)
{
    this->impl->add_items(items);
}

void SchedStream::sync()
{
    this->impl->sync();
}

void SchedStream::clear()
{
    this->impl->clear();
}

std::vector<std::vector<Branch>> SchedStream::get_branches()
{
    return this->impl->get_branches();
}

int SchedStream::get_num_sm() const
{
    return this->impl->get_num_sm();
}

} // namespace ark
