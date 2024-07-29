// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "codegen.hpp"

#include <list>

#include "ark/data_type.hpp"
#include "env.h"
#include "file_io.h"
#include "logging.h"
#include "model/model_buffer.hpp"
#include "model/model_data_type.hpp"
#include "model/model_op.hpp"
#include "model/model_tensor.hpp"
#include "range.hpp"
#include "utils/utils_math.hpp"

static std::string replace(
    const std::string &template_str,
    const std::map<std::string, std::string> &replacements) {
    std::string result = template_str;
    for (const auto &kv : replacements) {
        size_t pos = 0;
        while ((pos = result.find(kv.first, pos)) != std::string::npos) {
            result.replace(pos, kv.first.length(), kv.second);
            pos += kv.second.length();
        }
    }
    return result;
}

namespace ark {

class SyncStateInfo {
   public:
    SyncStateInfo(size_t id) : id(id) {}

    size_t id;
};

class CodeGenerator::Impl {
   public:
    Impl(const PlanJson &plan,
         const std::map<size_t, void*> &buffer_id_to_addr,
         const std::string &name);
    ~Impl() = default;

   private:
    std::string def_op(const Json &op_json, size_t task_id, size_t op_idx);

    std::string def_task(const Json &task_json);

    std::string def_channels(int world_size);

    std::string task_seq(size_t proc_b, size_t proc_e, size_t proc_s,
                         size_t proc_cur, size_t task_b, size_t task_e,
                         size_t task_s, size_t task_gran, size_t num_slots,
                         size_t slot_num_warps, size_t slot_sram_bytes,
                         size_t task_id);

    std::string resource_group(const Json &rg_json, const Json &task_infos,
                               const Range<size_t> &proc_range);

    std::string sync_process_range(const Range<size_t> &ranges, int state_id);

   protected:
    friend class CodeGenerator;

    std::map<size_t, void*> buffer_id_to_addr;
    std::string name_;
    int rank_;
    int world_size_;
    size_t num_procs_;
    size_t num_warps_per_proc_;
    std::string code_;
};

CodeGenerator::Impl::Impl(const PlanJson &plan,
                          const std::map<size_t, void*> &buffer_id_to_addr,
                          const std::string &name)
    : buffer_id_to_addr(buffer_id_to_addr), name_(name) {
    rank_ = plan.at("Rank");
    world_size_ = plan.at("WorldSize");
    num_procs_ = plan.at("NumProcessors");
    num_warps_per_proc_ = plan.at("NumWarpsPerProcessor");

    std::stringstream definitions_ss;
    for (auto &task_json : plan.at("TaskInfos")) {
        definitions_ss << this->def_task(task_json);
    }

    if (world_size_ > 1) {
        definitions_ss << this->def_channels(world_size_);
    }

    std::map<Range<size_t>, SyncStateInfo> sync_state_info;

    auto get_state_id = [&sync_state_info](const Range<size_t> &range) {
        auto it = sync_state_info.find(range);
        if (it == sync_state_info.end()) {
            size_t id = sync_state_info.size();
            sync_state_info.emplace(range, SyncStateInfo(id));
            return id;
        }
        return it->second.id;
    };

    std::list<Range<size_t>> unsynced;
    std::stringstream body_ss;
    size_t pg_idx = 0;
    for (auto &pg : plan.at("ProcessorGroups")) {
        Range<size_t> proc_range(pg["ProcessorRange"][0],
                                 pg["ProcessorRange"][1]);
        size_t begin = *proc_range.begin();
        size_t end = *proc_range.end();
        if (end == begin) continue;

        if (pg_idx > 0) {
            bool need_sync = false;
            auto it = unsynced.begin();
            while (it != unsynced.end()) {
                auto &range = *it;
                auto intersec = proc_range.intersection(range);
                if (intersec.empty()) {
                    it++;
                    continue;
                }
                if (intersec.size() < range.size()) {
                    // range is not a part of proc_range, so we need to
                    // sync range here.
                    size_t state_id = get_state_id(range);
                    body_ss << sync_process_range(range, state_id);
                    if (intersec.size() < proc_range.size()) {
                        // proc_range is not a part of range, so we need to
                        // sync proc_range later.
                        need_sync = true;
                    }
                } else {
                    // intersec.size() == range.size(), which means that
                    // range is a part of proc_range. In this case, we don't
                    // need to sync range here, because we will sync
                    // proc_range later by setting `need_sync` to true.
                    need_sync = true;
                }
                it = unsynced.erase(it);
            }
            if (need_sync) {
                size_t state_id = get_state_id(proc_range);
                body_ss << sync_process_range(proc_range, state_id);
            }
        }
        for (auto &rg : pg["ResourceGroups"]) {
            body_ss << resource_group(rg, plan.at("TaskInfos"), proc_range);
        }
        unsynced.push_back(proc_range);
        pg_idx++;
    }

    for (auto &kv : sync_state_info) {
        definitions_ss << "__device__ sync::State ARK_LOOP_SYNC_STATE_"
                       << kv.second.id << ";\n";
    }

    const std::string &ark_root = get_env().path_root_dir;
    const std::string &template_path =
        ark_root + "/include/kernels/kernel_template.in";
    if (!is_file(template_path)) {
        ERR(SchedulerError, "kernel template file not found: ", template_path);
    }
    std::string template_code = read_file(template_path);
    std::map<std::string, std::string> replacements = {
        {"@NUM_BLOCKS@", std::to_string(num_procs_)},
        {"@NUM_WARPS_PER_BLOCK@", std::to_string(num_warps_per_proc_)},
        {"@DEFINITIONS@", definitions_ss.str()},
        {"@BODY@", body_ss.str()},
        {"@NAME@", (name_.empty() ? "" : "_" + name_)},
    };
    code_ = replace(template_code, replacements);
}

std::string CodeGenerator::Impl::def_op(const Json &op_json, size_t task_id,
                                        size_t op_idx) {
    auto op = ModelOp::deserialize(op_json);
    auto impl_name = op->impl_name(op_json["Config"]);
    auto impl_args = op->impl_args(op_json["Config"]);
    std::stringstream ss;
    ss << "__forceinline__ __device__ void t" << task_id << "_o" << op_idx
       << "(";
    size_t arg_idx = 0;
    for (auto &arg : impl_args) {
        if (arg.type_name() == "TENSOR") {
            auto tns = arg.value<ModelTensorRef>();
            ss << tns->data_type()->type_str() << "*";
        } else if (arg.type_name() == "OFFSET") {
            ss << "uint64_t";
        } else {
            ss << arg.type_str();
        }
        ss << " _" << arg_idx++ << ", ";
    }
    ss << "int _idx, int _spw) {\n  " << impl_name << "(";
    for (size_t i = 0; i < impl_args.size(); ++i) {
        ss << "_" << i << ", ";
    }
    ss << "_idx, _spw);\n}\n";
    return ss.str();
}

std::string CodeGenerator::Impl::def_task(const Json &task_json) {
    std::stringstream ss;
    size_t op_idx = 0;
    for (auto &op_json : task_json["Ops"]) {
        ss << this->def_op(op_json, task_json["Id"], op_idx++);
    }
    ss << "__device__ void t" << task_json["Id"]
       << "(int _idx, int _spw) {\n";
    op_idx = 0;
    for (auto &op_json : task_json["Ops"]) {
        auto op = ModelOp::deserialize(op_json);
        auto impl_args = op->impl_args(op_json["Config"]);
        ss << "  t" << task_json["Id"] << "_o" << op_idx++ << "(";
        for (size_t i = 0; i < impl_args.size(); ++i) {
            auto &arg = impl_args[i];
            if (arg.type_name() == "TENSOR") {
                auto tns = arg.value<ModelTensorRef>();
                void* buffer_addr = buffer_id_to_addr.at(tns->buffer()->id());
                size_t offset = ModelOffset(tns).value();
                ss << "(" << tns->data_type()->type_str() << "*)((char*)"
                    << buffer_addr << " + " << offset << ")";
                
            } else if (arg.type_name() == "OFFSET") {
                auto moff = arg.value<ModelOffset>();
                void* buffer_addr = buffer_id_to_addr.at(moff.buffer_id());
                size_t offset = moff.value();
                ss << "(uint64_t)((char*)" << buffer_addr << " + " << offset
                   << ")";
            } else {
                ss << arg.serialize().begin().value();
            }
            ss << ", ";
        }
        ss << "_idx, _spw);\n";
    }
    ss << "}\n";
    return ss.str();
}

std::string CodeGenerator::Impl::def_channels(int world_size) {
    std::stringstream ss;
    ss << "__constant__ mscclpp::SimpleProxyChannelDeviceHandle ";
    ss << "ARK_PROXY_CHANS[" << world_size << "];\n";
    ss << "__constant__ mscclpp::SimpleProxyChannelDeviceHandle ";
    ss << "ARK_PROXY_SECONDARY_CHANS[" << world_size << "];\n";
    ss << "__constant__ mscclpp::SmChannelDeviceHandle ";
    ss << "ARK_SM_CHANS[" << world_size << "];\n";
    return ss.str();
}

std::string CodeGenerator::Impl::task_seq(
    size_t proc_b, size_t proc_e, size_t proc_s, size_t proc_cur, size_t task_b,
    size_t task_e, size_t task_s, size_t task_gran, size_t num_slots,
    size_t slot_num_warps, size_t slot_sram_bytes, size_t task_id) {
    std::stringstream ss;
    ss << "task_seq<" << proc_b << ", " << proc_e << ", " << proc_s << ", "
       << proc_cur << ", " << task_b << ", " << task_e << ", " << task_s << ", "
       << task_gran << ", " << num_slots << ", " << slot_num_warps << ", "
       << slot_sram_bytes << ", t" << task_id << ">();\n";
    return ss.str();
}

std::string CodeGenerator::Impl::resource_group(
    const Json &rg_json, const Json &task_infos,
    const Range<size_t> &proc_range) {
    Range<size_t> rg_proc_range(rg_json["ProcessorRange"][0],
                                rg_json["ProcessorRange"][1]);
    if (*rg_proc_range.begin() < *proc_range.begin() ||
        *rg_proc_range.end() > *proc_range.end()) {
        ERR(SchedulerError, "invalid processor range of resource group");
    }
    Range<size_t> rg_warp_range(rg_json["WarpRange"][0],
                                rg_json["WarpRange"][1]);
    Range<size_t> rg_sram_range(rg_json["SramRange"][0],
                                rg_json["SramRange"][1]);
    size_t total_warps = rg_warp_range.size();
    size_t total_sram = rg_sram_range.size();
    size_t proc_cur = *rg_proc_range.begin();
    size_t proc_b = *rg_proc_range.begin();
    size_t proc_e = *rg_proc_range.end();
    size_t proc_s = rg_proc_range.step();
    std::map<size_t, Json> task_infos_map;
    for (auto &task_info : task_infos) {
        task_infos_map[task_info.at("Id").get<size_t>()] = task_info;
    }
    std::stringstream ss;
    for (auto &tg : rg_json["TaskGroups"]) {
        size_t task_id = tg["TaskId"];
        auto &task_info = task_infos_map.at(task_id);
        Range<size_t> task_range(tg["TaskRange"][0], tg["TaskRange"][1]);
        size_t task_gran = tg["Granularity"];
        size_t num_warps_per_task = task_info["NumWarps"];
        size_t sram_bytes_per_task = task_info["SramBytes"];
        // number of concurrent tasks per processor
        size_t n_slots;
        if (sram_bytes_per_task > 0) {
            n_slots = std::min(total_warps / num_warps_per_task,
                               total_sram / sram_bytes_per_task);
        } else {
            n_slots = total_warps / num_warps_per_task;
        }
        if (n_slots == 0) {
            ERR(SchedulerError, "not enough resources for task group: ",
                tg.dump());
        }

        size_t task_b = *task_range.begin();
        size_t task_e = *task_range.end();
        size_t task_s = task_range.step();

        size_t slot_n_warps = num_warps_per_task;
        size_t slot_n_sram = total_sram / n_slots;

        //
        // Distribute tasks to processors.
        //
        // A sequence [b, e, s] means the range starts from `b`, ends at
        // `e - 1`, and the step size is `s`.
        //
        // Processor ID sequence: [proc_b, proc_e, proc_s], total `n_procs`
        // Task ID sequence: [task_b, task_e, task_s], total `n_tasks`
        //
        // The distribution starts from the processor ID `proc_cur` and wraps
        // around (`proc_cur - proc_b` is always a multiple of `proc_s`).
        // If `task_gran` is 1, the distribution is round-robin; otherwise,
        // the distribution assigns `task_gran` consequent tasks to each
        // processor, as long as there are enough tasks.
        // We distribute tasks from smaller task IDs to larger task IDs.
        // Therefore, the `t`-th assigned task ID of the processor ID
        // `(proc_cur + proc_s*p)%n_procs` is (p in range [0, n_procs-1]):
        //
        // ```
        // task_b + task_s*(
        //     p*task_gran +
        //     t/task_gran*task_gran*n_procs +
        //     t%task_gran
        // )
        // ```
        //
        // where the division is integer division.
        //
        // Within a single processor, `n_slots` consequent tasks are
        // distributed to warps and SRAMs. Specifically, say that
        // "k-th slot" refers to the set of warps `k * slot_n_warps` ~
        // `(k+1) * slot_n_warps - 1` and SRAMs `k * slot_n_sram` ~
        // `(k+1) * slot_n_sram - 1`, then the `t`-th task is assigned to
        // the `t%n_slots`-th slot.
        //
        // Therefore, the `i`-th assigned task ID of the processor ID
        // `(proc_cur + p)%n_procs` and the `k`-th slot is (p in range
        // [0, n_procs-1], k in range [0, n_slots-1]) the same as the above
        // formula with `t` replaced by `k + i*n_slots`:
        //
        // ```
        // task_b + task_s*(
        //     p*task_gran +
        //     (k + i*n_slots)/task_gran*task_gran*n_procs +
        //     (k + i*n_slots)%task_gran
        // )
        // ```
        //
        // The corresponding CUDA code is generated as follows, saying that
        // `blockIdx.x` is the processor ID:
        //
        // ```
        // if ((blockIdx.x >= proc_b) &&
        //     (blockIdx.x < proc_e) &&
        //     ((blockIdx.x - proc_b) % proc_s == 0)) {
        //   size_t p = ((blockIdx.x + gridDim.x - proc_cur) % gridDim.x) /
        //              proc_s;
        //   size_t k = threadIdx.x / warp_size / slot_n_warps;
        //   size_t task_id_base = task_b + task_s*p*task_gran;
        //   for (size_t t = k; ; t += n_slots) {
        //     size_t task_id = task_id_base + task_s*(
        //       t/task_gran*task_gran*n_procs + t%task_gran
        //     );
        //     if (task_id >= task_e) break;
        //     task_func(_buf, task_id, sram_per_warp);
        //   }
        // }
        // ```
        ss << "  ";
        ss << this->task_seq(proc_b, proc_e, proc_s, proc_cur, task_b, task_e,
                             task_s, task_gran, n_slots, slot_n_warps,
                             slot_n_sram, task_id);

        // Update `proc_cur` to the next of the last scheduled one
        size_t n_procs = rg_proc_range.size();
        size_t n_tasks = task_range.size();
        size_t proc_cur_idx = (proc_cur - proc_b) / proc_s;
        proc_cur_idx += math::div_up(n_tasks, task_gran);
        proc_cur_idx = proc_cur_idx % n_procs;
        proc_cur = proc_b + proc_cur_idx * proc_s;
    }
    return ss.str();
}

std::string CodeGenerator::Impl::sync_process_range(const Range<size_t> &range,
                                                    int state_id) {
    std::stringstream cond;
    if (range.size() == 1) {
        cond << "blockIdx.x == " << *range.begin();
    } else {
        if (*range.begin() == 0) {
            cond << "blockIdx.x < " << *range.end();
        } else {
            cond << "blockIdx.x >= " << *range.begin() << " && blockIdx.x < "
                 << *range.end();
        }
        if (range.step() > 1) {
            cond << " && ";
            if (*range.begin() == 0) {
                cond << "blockIdx.x % " << range.step() << " == 0";
            } else {
                cond << "(blockIdx.x - " << *range.begin() << ") % "
                     << range.step() << " == 0";
            }
        }
    }
    std::stringstream ret;
    ret << "  if (" << cond.str() << ") { ";
    ret << "sync_gpu<" << range.size() << ">(ARK_LOOP_SYNC_STATE_" << state_id
        << "); }\n";
    return ret.str();
}

CodeGenerator::CodeGenerator(
    const PlanJson &plan, const std::map<size_t, void*> &buffer_id_to_addr,
    const std::string &name)
    : impl_(std::make_shared<Impl>(plan, buffer_id_to_addr, name)) {}

std::string CodeGenerator::code() const { return impl_->code_; }

size_t CodeGenerator::num_procs() const { return impl_->num_procs_; }

size_t CodeGenerator::num_warps_per_proc() const {
    return impl_->num_warps_per_proc_;
}

}  // namespace ark
