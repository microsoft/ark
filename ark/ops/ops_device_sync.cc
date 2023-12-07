// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>

#include "gpu/gpu_buf.h"
#include "logging.h"
#include "model.h"

namespace ark {
extern const OpConfigMap DeviceSyncConfigMap;

DeviceSyncOp::DeviceSyncOp(const std::string &prec_type, Tensor *input,
                           Tensor *output, int nranks, const std::string &name)
    : Op{OP_DEVICE_SYNC,       prec_type, {input}, {output}, {{nranks}}, name,
         &DeviceSyncConfigMap, -1,        true} {}

std::string DeviceSyncOp::function_name(const OpConfig &) const {
    int nranks;
    this->args.get(&nranks, 0);
    return Op::function_name("ark::comm::device_sync", {{nranks}});
}

OpArgs DeviceSyncOp::function_call_args(const OpConfig &) const { return {}; }

Tensor *Model::device_sync(Tensor *input, int nranks, const std::string &name) {
    DeviceSyncOp op{"none", input, input, nranks, name};
    return this->impl->add_op(op)[0];
}

const OpConfigMap DeviceSyncConfigMap = {
    {{OP_ARCH_CUDA_ANY, "none"},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 0, {{-1, -1}}, {{-1, -1}}, false, true},
     }},
};

}  // namespace ark
