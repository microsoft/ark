// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>

#include "gpu/gpu_buf.h"
#include "logging.h"
#include "model.h"

namespace ark {
extern const OpConfigMap MscclppDeviceSyncConfigMap;

MscclppDeviceSyncOp::MscclppDeviceSyncOp(const std::string &prec_type,
                                         Tensor *input, Tensor *output,
                                         int nranks, const std::string &name)
    : Op{OP_DEVICE_SYNC_MSCCLPP,
         prec_type,
         {input},
         {output},
         {{nranks}},
         name,
         &MscclppDeviceSyncConfigMap,
         -1,
         true} {}

std::string MscclppDeviceSyncOp::function_name(const OpConfig &) const {
    int nranks;
    this->args.get(&nranks, 0);
    return Op::function_name("ark::comm::device_sync_msll", {{nranks}});
}

OpArgs MscclppDeviceSyncOp::function_call_args(const OpConfig &) const {
    return {};
}

Tensor *Model::device_sync_mscclpp(Tensor *input, int nranks,
                                   const std::string &name) {
    MscclppDeviceSyncOp op{"none", input, input, nranks, name};
    return this->impl->add_op(op)[0];
}

const OpConfigMap MscclppDeviceSyncConfigMap = {
    {{OP_ARCH_CUDA_ANY, "none"},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 0, {{-1, -1}, {-1, -1}}, {{-1, -1}}, false, true},
     }},
};

}  // namespace ark
