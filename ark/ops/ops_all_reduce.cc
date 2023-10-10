// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>

#include "logging.h"
#include "math.h"
#include "model.h"
#include "ops_common.h"

namespace ark {

Tensor *Model::all_reduce(Tensor *input, int gpu_id, int gpu_num,
                          Tensor *output, const std::string &) {
    assert(input != nullptr);
    if (output != nullptr) {
        LOG(ERROR, "all_reduce output is not supported");
    }
    if (input->ndims() > 1) {
        LOG(ERROR, "supports only 1D input");
    }
    if (!input->is_sequential()) {
        LOG(WARN,
            "all_reduce may not work correctly if the input tensor is "
            "not contiguous");
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
        send_data = this->send(send_data, base + gpu_id, gpu_dst);
        Tensor *send_done_tensor =
            this->send_done(send_data, base + gpu_id, gpu_dst);
        Tensor *recv_buf = this->tensor(input->shape, input->type);
        recv_buf = this->identity(recv_buf, {send_done_tensor});
        Tensor *recv = this->recv(base + gpu_src, gpu_src, 0, recv_buf);
        prev_recv = recv;
        cumulate = this->add(cumulate, recv);
    }
    this->impl->next_eid += gpu_num;
    return cumulate;
}

Tensor *Model::local_all_reduce(Tensor *input, int gpu_id, int gpu_num,
                                Tensor *output, const std::string &)
{
    assert(input != nullptr);
    if (output != nullptr) {
        LOG(ERROR, "all_reduce output is not supported");
    }
    if (input->ndims() > 1) {
        LOG(ERROR, "supports only 1D input");
    }
    if (!input->is_sequential()) {
        LOG(WARN, "all_reduce may not work correctly if the input tensor is "
                  "not contiguous");
    }
    int sid = this->impl->next_eid;
    Tensor* out = this->local_reduce_scatter_mscclpp(input, gpu_id, sid, gpu_num);
    return this->local_all_gather_mscclpp(out, gpu_id, sid, gpu_num);
}

} // namespace ark
