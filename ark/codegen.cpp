// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "codegen.hpp"

#include <list>
#include <utility>

#include "ark/data_type.hpp"
#include "buffer_registry.hpp"
#include "env.h"
#include "file_io.h"
#include "logging.hpp"
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
            if ((kv.first == "@GLOBAL_ARGS@" || kv.first == "@FUNCTION_ARGS@" ||
                 kv.first == "@ARG_TYPES@") &&
                kv.second.empty()) {
                size_t comma_pos = pos;
                if (comma_pos >= 2 && result.substr(comma_pos - 2, 2) == ", ") {
                    result.erase(comma_pos - 2, 2);
                    pos -= 2;
                }

            } else {
                pos += kv.second.length();
            }
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
         const std::map<size_t, size_t> &buffer_id_to_offset,
         const std::set<size_t> &extra_buffer_ids, const std::string &name);
    ~Impl() = default;

   private:
    std::pair<std::string, size_t> def_op(const Json &op_json);

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

    std::set<size_t> op_hashes_;
    std::set<size_t> task_hashes_;
    std::map<size_t, size_t> buffer_id_to_offset_;
    std::set<size_t> extra_buffer_ids_;
    std::string name_;
    int rank_;
    int world_size_;
    size_t num_procs_;
    size_t num_warps_per_proc_;
    std::string code_;
};

CodeGenerator::Impl::Impl(const PlanJson &plan,
                          const std::map<size_t, size_t> &buffer_id_to_offset,
                          const std::set<size_t> &extra_buffer_ids,
                          const std::string &name)
    : buffer_id_to_offset_(buffer_id_to_offset),
      extra_buffer_ids_(extra_buffer_ids),
      name_(name) {
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
        ERR(InvalidUsageError,
            "kernel template file not found: ", template_path,
            ". Please make sure the ARK_ROOT environment variable is set "
            "correctly.");
    }

    // Generate the global arguments
    std::stringstream global_args_ss, function_args_ss, arg_types_ss;
    for (auto buf_id : extra_buffer_ids_) {
        std::string arg_name = "_ext_buf_" + std::to_string(buf_id);
        global_args_ss << "void *" << arg_name << ", ";
        function_args_ss << arg_name << ", ";
        arg_types_ss << "void *, ";
    }
    std::string global_args = global_args_ss.str();
    std::string function_args = function_args_ss.str();
    std::string arg_types = arg_types_ss.str();
    if (!global_args.empty()) {
        global_args.pop_back();
        global_args.pop_back();
    }
    if (!function_args.empty()) {
        function_args.pop_back();
        function_args.pop_back();
    }
    if (!arg_types.empty()) {
        arg_types.pop_back();
        arg_types.pop_back();
    }

    std::string template_code = read_file(template_path);
    std::map<std::string, std::string> replacements = {
        {"@NUM_BLOCKS@", std::to_string(num_procs_)},
        {"@NUM_WARPS_PER_BLOCK@", std::to_string(num_warps_per_proc_)},
        {"@DEFINITIONS@", definitions_ss.str()},
        {"@BODY@", body_ss.str()},
        {"@NAME@", (!name_.empty() ? "" : name_)},
        {"@GLOBAL_ARGS@", global_args},
        {"@FUNCTION_ARGS@", function_args},
        {"@ARG_TYPES@", arg_types},
    };
    code_ = replace(template_code, replacements);
}

std::pair<std::string, size_t> CodeGenerator::Impl::def_op(
    const Json &op_json) {
    auto op = ModelOp::deserialize(op_json);
    auto impl_name = op->impl_name(op_json["Config"]);
    auto impl_args = op->impl_args(op_json["Config"]);
    std::stringstream ss_desc;
    size_t arg_idx = 0;
    for (auto &arg : impl_args) {
        if (arg.type_name() == "TENSOR") {
            auto tns = arg.value<ModelTensorRef>();
            ss_desc << tns->data_type()->type_str() << "*";
        } else if (arg.type_name() == "OFFSET") {
            ss_desc << "uint64_t";
        } else {
            ss_desc << arg.type_str();
        }
        ss_desc << " _" << arg_idx++ << ", ";
    }
    ss_desc << "int _idx, int _spw) {\n  " << impl_name << "(";
    for (size_t i = 0; i < impl_args.size(); ++i) {
        ss_desc << "_" << i << ", ";
    }
    ss_desc << "_idx, _spw);\n}\n";
    auto desc_str = ss_desc.str();
    size_t op_hash = std::hash<std::string>{}(desc_str);
    std::stringstream ss;
    ss << "__forceinline__ __device__ void __op_" << std::hex << op_hash
       << std::dec << "(";
    ss << desc_str;
    return {ss.str(), op_hash};
}

std::string CodeGenerator::Impl::def_task(const Json &task_json) {
    std::stringstream ss;
    std::stringstream ss_hash_concat;
    std::vector<size_t> op_hash_list;
    for (auto &op_json : task_json["Ops"]) {
        auto [def_str, hash] = this->def_op(op_json);
        if (op_hashes_.find(hash) == op_hashes_.end()) {
            ss << def_str;
            op_hashes_.insert(hash);
        }
        ss_hash_concat << std::hex << hash;
        op_hash_list.push_back(hash);
    }
    size_t task_hash = std::hash<std::string>{}(ss_hash_concat.str());
    std::stringstream ss_desc;
    auto &buf_reg = BufferRegistry::get_instance();
    size_t op_idx = 0;
    std::map<std::string, size_t> ptr_str_to_index;
    std::vector<std::string> ptr_str_list;
    for (auto &op_json : task_json["Ops"]) {
        auto op = ModelOp::deserialize(op_json);
        auto impl_args = op->impl_args(op_json["Config"]);
        ss_desc << "  __op_" << std::hex << op_hash_list[op_idx++] << std::dec
                << "(";
        for (auto &arg : impl_args) {
            if (arg.type_name() == "TENSOR") {
                auto tns = arg.value<ModelTensorRef>();
                size_t buffer_id = tns->buffer()->id();
                auto it = buffer_id_to_offset_.find(buffer_id);
                auto buf_info = buf_reg.get(buffer_id);
                std::string ptr_str;
                if ((buf_info && buf_info->is_external) ||
                    (it == buffer_id_to_offset_.end())) {
                    ptr_str = "_ext_buf_" + std::to_string(buffer_id);
                } else {
                    size_t buffer_offset;
                    buffer_offset = it->second;
                    size_t offset = buffer_offset + ModelOffset(tns).value();
                    ptr_str = "&_buf[" + std::to_string(offset) + "]";
                }
                size_t ptr_idx;
                if (ptr_str_to_index.find(ptr_str) == ptr_str_to_index.end()) {
                    ptr_idx = ptr_str_to_index.size();
                    ptr_str_to_index[ptr_str] = ptr_idx;
                    ptr_str_list.push_back(ptr_str);
                } else {
                    ptr_idx = ptr_str_to_index[ptr_str];
                }
                ss_desc << "(" << tns->data_type()->type_str() << "*)_"
                        << ptr_idx;
            } else if (arg.type_name() == "OFFSET") {
                auto moff = arg.value<ModelOffset>();
                size_t buffer_id = moff.buffer_id();
                auto buf_info = buf_reg.get(buffer_id);
                if (buf_info && buf_info->is_external) {
                    ERR(InternalError, "cannot offset external buffer");
                }
                size_t buffer_offset;
                auto it = buffer_id_to_offset_.find(buffer_id);
                if (it == buffer_id_to_offset_.end()) {
                    ERR(InternalError, "buffer ID not found: ", buffer_id);
                }
                buffer_offset = it->second;
                size_t offset = buffer_offset + moff.value();
                ss_desc << offset;
            } else {
                ss_desc << arg.serialize().begin().value();
            }
            ss_desc << ", ";
        }
        ss_desc << "_idx, _spw);\n";
    }
    if (task_hashes_.find(task_hash) == task_hashes_.end()) {
        ss << "__device__ void __task_" << std::hex << task_hash << std::dec
           << "(";
        for (size_t i = 0; i < ptr_str_list.size(); ++i) {
            ss << "void *_" << i << ", ";
        }
        ss << "int _idx, int _spw) {\n" << ss_desc.str() << "}\n";
        task_hashes_.insert(task_hash);
    }
    ss << "__forceinline__ __device__ void __t" << task_json["Id"]
       << "(char *_buf, int _idx, int _spw, @GLOBAL_ARGS@) {\n";
    ss << "  __task_" << std::hex << task_hash << std::dec << "(";
    for (auto &ptr_str : ptr_str_list) {
        ss << ptr_str << ", ";
    }
    ss << "_idx, _spw);\n}\n";
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
       << slot_sram_bytes << ", __t" << task_id
       << ">(_buf, @FUNCTION_ARGS@);\n";
    return ss.str();
}

std::string CodeGenerator::Impl::resource_group(
    const Json &rg_json, const Json &task_infos,
    const Range<size_t> &proc_range) {
    Range<size_t> rg_proc_range(rg_json["ProcessorRange"][0],
                                rg_json["ProcessorRange"][1]);
    if (*rg_proc_range.begin() < *proc_range.begin() ||
        *rg_proc_range.end() > *proc_range.end()) {
        ERR(PlanError, "invalid processor range of resource group");
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
            ERR(PlanError, "not enough resources for task group: ", tg.dump());
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
    const PlanJson &plan, const std::map<size_t, size_t> &buffer_id_to_offset,
    const std::set<size_t> &extra_buffer_ids, const std::string &name)
    : impl_(std::make_shared<Impl>(plan, buffer_id_to_offset, extra_buffer_ids,
                                   name)) {}

std::string CodeGenerator::code() const { return impl_->code_; }

size_t CodeGenerator::num_procs() const { return impl_->num_procs_; }

size_t CodeGenerator::num_warps_per_proc() const {
    return impl_->num_warps_per_proc_;
}

}  // namespace ark
