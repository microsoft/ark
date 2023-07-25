// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "math.h"
#include "model.h"
#include "ops_common.h"

using namespace std;

namespace ark {

Tensor *Model::all_reduce(Tensor *input, int gpu_id, int gpu_num,
                          Tensor *output, const string &name)
{
    assert(input != nullptr);
    if (output != nullptr) {
        LOGERR("all_reduce output is not supported");
    }
    LOG(DEBUG, "all_reduce ", input->shape, " ", gpu_id, " ", gpu_num);
    if (input->ndims() > 1) {
        LOGERR("supports only 1D input");
    }
    if (math::pad(input->shape[0], input->pads[0]) < (size_t)input->ldims[0]) {
        LOGERR("all_reduce of a split tensor is not supported");
    }

    int base = this->impl->next_eid;
    // all to all allreduce
    vector<Tensor *> recv_dep_tensors;
    for (int gpu_dst = 0; gpu_dst < gpu_num; gpu_dst++) {
        if (gpu_dst == gpu_id)
            continue;
        Tensor *send_tensor =
            this->send(input, base + gpu_id * gpu_num + gpu_dst, gpu_dst);
        Tensor *send_done_tensor =
            this->send_done(input, base + gpu_id * gpu_num + gpu_dst, gpu_dst);
        recv_dep_tensors.push_back(send_tensor);
        recv_dep_tensors.push_back(send_done_tensor);
    }
    Tensor *add_tensor = input;
    for (int gpu_src = 0; gpu_src < gpu_num; gpu_src++) {
        Tensor *recv_buf = this->tensor(input->shape, input->type);
        if (gpu_src == gpu_id)
            continue;
        if (add_tensor != nullptr) {
            recv_dep_tensors.push_back(add_tensor);
        }
        Tensor *recv = this->recv(this->identity(recv_buf, recv_dep_tensors),
                                  base + gpu_src * gpu_num + gpu_id, gpu_src);
        add_tensor = this->add(add_tensor, this->identity(recv_buf, {recv}));
    }

    this->impl->next_eid += gpu_num * gpu_num;
    return add_tensor;
}

} // namespace ark
