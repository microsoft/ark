// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "math.h"
#include "model.h"
#include "ops_common.h"
#include <cassert>

namespace ark {

Tensor *Model::all_reduce(Tensor *input, int gpu_id, int gpu_num,
                          Tensor *output, const std::string &)
{
    assert(input != nullptr);
    if (output != nullptr) {
        LOG(ERROR, "all_reduce output is not supported");
    }
    if (input->ndims() > 1) {
        LOG(ERROR, "supports only 1D input");
    }
    if (math::pad(input->shape[0], input->pads[0]) < (size_t)input->ldims[0]) {
        LOG(ERROR, "all_reduce of a split tensor is not supported");
    }

    int base = this->impl->next_eid;
    Tensor *prev_recv = nullptr;
    Tensor *cumulate = input;
    for (int i = 1; i < gpu_num; i++) {
        int gpu_dst = (gpu_id + i) % gpu_num;
        int gpu_src = (gpu_id + gpu_num - i) % gpu_num;
        Tensor *send_data;
        if (prev_recv != nullptr) {
            send_data = this->identity(input, {prev_recv});
        } else {
            send_data = input;
        }
        Tensor *send_tensor =
            this->send(send_data, base + gpu_id * gpu_num + gpu_dst, gpu_dst);
        Tensor *send_done_tensor =
            this->send_done(this->identity(input, {send_tensor}),
                            base + gpu_id * gpu_num + gpu_dst, gpu_dst);
        Tensor *recv_buf = this->tensor(input->shape, input->type);
        Tensor *recv = this->recv(this->identity(recv_buf, {send_done_tensor}),
                                  base + gpu_src * gpu_num + gpu_id, gpu_src);
        prev_recv = recv;
        cumulate = this->add(cumulate, this->identity(recv_buf, {recv}));
    }
    this->impl->next_eid += gpu_num * gpu_num;
    return cumulate;
}

} // namespace ark
