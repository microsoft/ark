// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "codegen.hpp"

#include <set>

#include "ark/data_type.hpp"
#include "env.h"
#include "file_io.h"
#include "logging.h"
#include "model/model_data_type.hpp"
#include "model/model_op.hpp"
#include "model/model_tensor.hpp"
#include "nlohmann/json.hpp"
#include "range.hpp"

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

class BufferInfo {
   public:
    BufferInfo(size_t id) : id(id), bytes(0), is_input(true), is_output(true) {}

    // ID of this buffer
    const size_t id;

    // Total bytes of this buffer
    size_t bytes;

    // True if none of tensors in this buffer is a result tensor or a write
    // tensor of a non-virtual Op, i.e., this buffer is an input buffer
    bool is_input;

    // True if none of tensors in this buffer is a read tensor of a non-virtual
    // Op, i.e., this buffer is an output buffer
    bool is_output;

    // IDs of tensors in this buffer
    std::set<size_t> tensor_ids;

    // IDs of tasks that read/write from/to this buffer
    std::set<size_t> task_ids;
};

class SyncStateInfo {
   public:
    SyncStateInfo() {
        static size_t next_id = 0;
        id = next_id++;
    }

    size_t id;
};

class CodeGenerator::Impl {
   public:
    Impl(const std::string &plan, const std::string &name);
    ~Impl() = default;

   private:
    void plan_memory(const nlohmann::json &plan);

    std::string def_op(const nlohmann::json &op_json, size_t task_id,
                       size_t op_idx);

    std::string def_task(const nlohmann::json &task_json);

    std::string task_seq(size_t proc_b, size_t proc_e, size_t proc_s,
                         size_t proc_cur, size_t task_b, size_t task_e,
                         size_t task_s, size_t task_gran, size_t num_slots,
                         size_t slot_num_warps, size_t slot_sram_bytes,
                         size_t task_id);

    std::string resource_group(const nlohmann::json &rg_json,
                               const nlohmann::json &task_infos,
                               const Range<size_t> &proc_range);

   protected:
    friend class CodeGenerator;

    std::string name_;
    size_t num_procs_;
    size_t num_warps_per_proc_;
    std::map<size_t, std::shared_ptr<BufferInfo>> buffer_id_to_info_;
    std::map<size_t, size_t> buffer_id_to_offset_;
    std::map<size_t, CodeGenerator::TensorInfo> tensor_id_to_info_;
    size_t total_bytes_;
    std::string code_;
};

CodeGenerator::Impl::Impl(const std::string &plan, const std::string &name)
    : name_(name) {
    auto j = nlohmann::json::parse(plan);
    this->plan_memory(j);

    num_procs_ = j["NumProcessors"];
    num_warps_per_proc_ = j["NumWarpsPerProcessor"];

    std::stringstream definitions_ss;
    for (auto &task_json : j["TaskInfos"]) {
        definitions_ss << this->def_task(task_json);
    }

    std::map<Range<size_t>, SyncStateInfo> sync_state_info;

    std::stringstream body_ss;
    size_t pg_idx = 0;
    for (auto &pg : j["ProcessorGroups"]) {
        Range<size_t> proc_range(pg["ProcessorRange"][0],
                                 pg["ProcessorRange"][1]);
        size_t begin = *proc_range.begin();
        size_t end = *proc_range.end();
        if (end == begin) continue;

        if (pg_idx > 0) {
            // sync pg
            if (begin == 0) {
                body_ss << "  if (blockIdx.x < " << end << ") {";
            } else if (begin + 1 == end) {
                body_ss << "  if (blockIdx.x == " << begin << ") {";
            } else {
                body_ss << "  if (blockIdx.x >= " << begin
                        << " && blockIdx.x < " << end << ") {";
            }
            size_t state_id = sync_state_info[proc_range].id;
            body_ss << " sync_gpu<" << end - begin << ">(ARK_LOOP_SYNC_STATE_"
                    << state_id << "); }\n";
        }
        for (auto &rg : pg["ResourceGroups"]) {
            body_ss << this->resource_group(rg, j["TaskInfos"], proc_range);
        }
        pg_idx++;
    }

    for (auto &kv : sync_state_info) {
        definitions_ss << "__device__ sync::State ARK_LOOP_SYNC_STATE_"
                       << kv.second.id << ";\n";
    }

    const std::string &ark_root = get_env().path_root_dir;
    const std::string &template_path =
        ark_root + "/include/kernels/kernel_template.in";
    std::string template_code = read_file(template_path);
    std::map<std::string, std::string> replacements = {
        {"@NUM_BLOCKS@", std::to_string(num_procs_)},
        {"@NUM_WARPS_PER_BLOCK@", std::to_string(num_warps_per_proc_)},
        {"@DEFINITIONS@", definitions_ss.str()},
        {"@BODY@", body_ss.str()},
        {"@NAME@", name_},
    };
    code_ = replace(template_code, replacements);
}

void CodeGenerator::Impl::plan_memory(const nlohmann::json &plan) {
    auto get_or_create_buffer_info = [&](size_t buffer_id) {
        if (buffer_id_to_info_.find(buffer_id) == buffer_id_to_info_.end()) {
            auto buf_info = std::make_shared<BufferInfo>(buffer_id);
            buffer_id_to_info_[buffer_id] = buf_info;
            return buf_info;
        }
        return buffer_id_to_info_[buffer_id];
    };

    auto tensor_stride_bytes = [](const nlohmann::json &tns) {
        Dims strides(tns["Strides"].get<std::vector<DimType>>());
        size_t nelems = strides.nelems();
        return nelems * DataType::from_name(tns["DataType"]).bytes();
    };

    auto retrieve_tensor_and_buffer_info = [&](const nlohmann::json &tensor,
                                               size_t task_id, bool is_input,
                                               bool is_output) {
        size_t tensor_id = tensor["Id"].get<size_t>();
        auto buf_info = get_or_create_buffer_info(tensor["BufferId"]);
        buf_info->bytes =
            std::max(buf_info->bytes, tensor_stride_bytes(tensor));
        buf_info->is_input = is_input;
        buf_info->is_output = is_output;
        buf_info->tensor_ids.insert(tensor_id);
        buf_info->task_ids.insert(task_id);
        if (tensor_id_to_info_.find(tensor_id) == tensor_id_to_info_.end()) {
            CodeGenerator::TensorInfo info;
            info.id = tensor_id;
            info.bytes = tensor_stride_bytes(tensor);
            // offset is undetermined yet
            tensor_id_to_info_.emplace(tensor_id, info);
        }
    };

    for (auto &task_info : plan["TaskInfos"]) {
        for (auto &op : task_info["Ops"]) {
            size_t task_id = task_info["Id"].get<size_t>();
            for (auto &tns : op["ReadTensors"]) {
                retrieve_tensor_and_buffer_info(tns, task_id, true, false);
            }
            for (auto &tns : op["WriteTensors"]) {
                retrieve_tensor_and_buffer_info(tns, task_id, false, true);
            }
            for (auto &tns : op["ResultTensors"]) {
                retrieve_tensor_and_buffer_info(tns, task_id, false, true);
            }
        }
    }

    // TODO: improve memory planning
    size_t offset = 0;
    for (auto &kv : buffer_id_to_info_) {
        buffer_id_to_offset_[kv.first] = offset;
        for (auto &tns_id : kv.second->tensor_ids) {
            tensor_id_to_info_.at(tns_id).offset = offset;
        }
        offset += kv.second->bytes;
    }
    total_bytes_ = offset;
}

std::string CodeGenerator::Impl::def_op(const nlohmann::json &op_json,
                                        size_t task_id, size_t op_idx) {
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

std::string CodeGenerator::Impl::def_task(const nlohmann::json &task_json) {
    std::stringstream ss;
    size_t op_idx = 0;
    for (auto &op_json : task_json["Ops"]) {
        ss << this->def_op(op_json, task_json["Id"], op_idx++);
    }
    ss << "__noinline__ __device__ void t" << task_json["Id"]
       << "(char* _buf, int _idx, int _spw) {\n";
    op_idx = 0;
    for (auto &op_json : task_json["Ops"]) {
        auto op = ModelOp::deserialize(op_json);
        auto impl_args = op->impl_args(op_json["Config"]);
        ss << "  t" << task_json["Id"] << "_o" << op_idx++ << "(";
        for (size_t i = 0; i < impl_args.size(); ++i) {
            auto &arg = impl_args[i];
            if (arg.type_name() == "TENSOR") {
                auto tns = arg.value<ModelTensorRef>();
                auto st = tns->strides();
                auto of = tns->offsets();
                int ndims = st.ndims();
                auto info = tensor_id_to_info_.at(tns->id());
                size_t offset = info.offset;
                for (int idx = ndims - 1; idx >= 0; --idx) {
                    size_t inc = of[idx];
                    for (int j = idx + 1; j < ndims; ++j) {
                        inc *= st[j];
                    }
                    offset += inc * tns->data_type()->bytes();
                }
                ss << "(" << tns->data_type()->type_str() << "*)&_buf["
                   << offset << "]";
            } else {
                ss << arg.serialize()[1];
            }
            ss << ", ";
        }
        ss << "_idx, _spw);\n";
    }
    ss << "}\n";
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
       << slot_sram_bytes << ", t" << task_id << ">(_buf);\n";
    return ss.str();
}

std::string CodeGenerator::Impl::resource_group(
    const nlohmann::json &rg_json, const nlohmann::json &task_infos,
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
    std::stringstream ss;
    for (auto &tg : rg_json["TaskGroups"]) {
        size_t task_id = tg["TaskId"];
        auto &task_info = task_infos[task_id];
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
            ERR(SchedulerError, "not enough resources for task group");
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
        //   proc_s; size_t k = threadIdx.x / warp_size / slot_n_warps; size_t
        //   task_id_base = task_b + task_s*p*task_gran; for (size_t t = k; ; t
        //   += n_slots) {
        //     size_t task_id = task_id_base + task_s*(
        //       t/task_gran*task_gran*n_procs + t%task_gran
        //     );
        //     if (task_id >= task_e) break;
        //     task_func(_buf, task_id, sram_per_warp);
        //   }
        //   __syncthreads();
        // }
        // ```
        ss << "  ";
        ss << this->task_seq(proc_b, proc_e, proc_s, proc_cur, task_b, task_e,
                             task_s, task_gran, n_slots, slot_n_warps,
                             slot_n_sram, task_id);
    }
    return ss.str();
}

CodeGenerator::CodeGenerator(const std::string &plan, const std::string &name)
    : impl_(std::make_shared<Impl>(plan, name)) {}

std::string CodeGenerator::code() const { return impl_->code_; }

size_t CodeGenerator::num_procs() const { return impl_->num_procs_; }

size_t CodeGenerator::num_warps_per_proc() const {
    return impl_->num_warps_per_proc_;
}

size_t CodeGenerator::total_memory_bytes() const { return impl_->total_bytes_; }

const CodeGenerator::TensorInfo &CodeGenerator::tensor_info(
    size_t tensor_id) const {
    auto it = impl_->tensor_id_to_info_.find(tensor_id);
    if (it == impl_->tensor_id_to_info_.end()) {
        ERR(NotFoundError, "tensor ID not found: ", tensor_id);
    }
    return it->second;
}

}  // namespace ark
