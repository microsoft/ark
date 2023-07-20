// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_SCHED_OP_H_
#define ARK_SCHED_OP_H_

#include "gpu/gpu_mgr.h"
#include "include/ark.h"
#include "ops/ops_config.h"

namespace ark {

const OpConfig *sched_op_config(const Op *op, const GpuInfo &gpu_info);

class SchedOp
{
  public:
    SchedOp(const Op *op_, const OpConfig *cfg_, const std::string name);
    const Op *get_op() const
    {
        return op;
    }
    int get_num_warps() const
    {
        return const_cast<OpConfig *>(cfg)->num_warps;
    }
    const Dims &get_tnums() const
    {
        return tnums;
    }
    const std::string &get_name() const
    {
        return name;
    }
    const OpConfig *get_cfg() const
    {
        return cfg;
    }
    const std::string func_string() const;
    const std::string func_string_matmul() const;
    const std::string func_string_send() const;
    const std::string func_string_recv() const;
    const std::string func_string_send_done() const;
    const std::string func_string_send_mm() const;
    const std::string func_string_recv_mm() const;
    const std::string func_string_reduce(const std::string &type) const;
    const std::string func_string_layernorm() const;
    const std::string func_string_softmax() const;
    const std::string func_string_scale() const;
    const std::string func_string_gelu() const;
    const std::string func_string_add() const;
    const std::string func_string_mul() const;
    const std::string func_string_im2col() const;
    const std::string func_string_transpose() const;

  private:
    const Op *op;
    const OpConfig *cfg;
    std::string name;
    // The number of tiles along each axis of the operator.
    Dims tnums;
};

void to_json(nlohmann::json &j, const SchedOp &sop);
void from_json(const nlohmann::json &j, SchedOp &sop);

} // namespace ark

#endif // ARK_SCHED_OP_H_
