// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>

#include "gpu/gpu_buf.h"
#include "logging.h"
#include "model.h"

namespace ark {
extern const OpConfigMap MsllDeviceSyncConfigMap;

MsllDeviceSyncOp::MsllDeviceSyncOp(const std::string &prec_type, Tensor *input,
                                   Tensor *output, int nranks,
                                   const std::string &name)
    : Op{OP_DEVICE_SYNC_MSLL,
         prec_type,
         {input},
         {output},
         {{nranks}},
         name,
         &MsllDeviceSyncConfigMap,
         -1,
         true} {}

std::string MsllDeviceSyncOp::function_name(const OpConfig &) const {
    int nranks;
    this->args.get(&nranks, 0);
    return Op::function_name("ark::comm::device_sync_msll", {{nranks}});
}

OpArgs MsllDeviceSyncOp::function_call_args(const OpConfig &) const {
    return {};
}

const OpConfigMap MsllDeviceSyncConfigMap = {
    {{OP_ARCH_CUDA_ANY, "none"},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 0, {{-1, -1}, {-1, -1}}, {{-1, -1}}, false, true},
     }},
};

}  // namespace ark
