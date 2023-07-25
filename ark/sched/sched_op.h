// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_SCHED_OP_H_
#define ARK_SCHED_OP_H_

#include "gpu/gpu_mgr.h"
#include "include/ark.h"
#include "ops/ops_common.h"
#include "json.h"

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
    bool is_virtual() const;

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
