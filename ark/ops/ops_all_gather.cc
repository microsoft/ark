// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "math.h"
#include "model.h"
#include "ops_common.h"
#include <cassert>

namespace ark {

std::vector<Tensor *> Model::all_gather(Tensor *input, int gpu_id, int gpu_num,
                                        const std::vector<Tensor *> &output,
                                        const std::string &)
{
    assert(input != nullptr);
    if (input->ndims() > 1) {
        LOG(INFO,
            "warning: if the send tensor if not contiguous, the all_gather "
            "may not work correctly");
    }
    if (!output.empty() && output.size() != (size_t)gpu_num) {
        LOG(ERROR, "all_gather output size should be 0 or gpu_num");
    }
    CHECK(gpu_num > 1);

    std::vector<Tensor *> result(gpu_num);

    int base = this->impl->next_eid;
    Tensor *prev_recv = nullptr;
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
            this->send(send_data, base + gpu_id, gpu_dst);
        Tensor *send_done_tensor =
            this->send_done(this->identity(input, {send_tensor}),
                            base + gpu_id, gpu_dst);
        Tensor *recv_buf;
        if (!output.empty()) {
            CHECK(output.size() > (size_t)gpu_src);
            CHECK(output[gpu_src]->shape == input->shape);
            CHECK(output[gpu_src]->type == input->type);
            recv_buf = output[gpu_src];
        } else {
            recv_buf = this->tensor(input->shape, input->type);
        }
        Tensor *recv = this->recv(this->identity(recv_buf, {send_done_tensor}),
                                  base + gpu_src, gpu_src);
        prev_recv = recv;
        result[gpu_src] = this->identity(recv_buf, {recv});
    }
    result[gpu_id] = this->identity(input, {prev_recv});
    this->impl->next_eid += gpu_num;
    return result;
}

} // namespace ark
