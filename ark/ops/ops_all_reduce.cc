// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/logging.h"
#include "ark/math.h"
#include "ark/model_io.h"

using namespace std;

namespace ark {

Tensor *Model::all_reduce(Tensor *input, int gpu_id, int gpu_num,
                          Tensor *output, const string &name)
{
    assert(input != nullptr);
    LOG(DEBUG, "all_reduce ", input->shape, " ", gpu_id, " ", gpu_num);
    if (input->ndims() > 1) {
        LOGERR("supports only 1D input");
    }
    if ((math::pad(input->shape[0], input->pads[0]) <
         (size_t)input->ldims[0]) ||
        (math::pad(input->shape[1], input->pads[1]) <
         (size_t)input->ldims[1]) ||
        (math::pad(input->shape[2], input->pads[2]) <
         (size_t)input->ldims[2]) ||
        (math::pad(input->shape[3], input->pads[3]) <
         (size_t)input->ldims[3])) {
        LOGERR("all_reduce of a split tensor is not supported");
    }

    int base = this->next_eid;
    // all to all allreduce
    vector<Tensor *> send_tensors;
    for (int gpu_dst = 0; gpu_dst < gpu_num; gpu_dst++) {
        if (gpu_dst == gpu_id)
            continue;
        Tensor *send_t =
            this->send(input, base + gpu_id * gpu_num + gpu_dst, gpu_dst);
        this->send_done(input, base + gpu_id * gpu_num + gpu_dst);
        send_tensors.push_back(send_t);
    }
    Tensor *recv_buf = this->tensor(input->shape, input->type);
    Tensor *add = nullptr;
    for (int gpu_src = 0; gpu_src < gpu_num; gpu_src++) {
        if (gpu_src == gpu_id)
            continue;
        if (add != nullptr) {
            send_tensors.push_back(add);
        }
        Tensor *recv = this->recv(this->identity(recv_buf, send_tensors),
                                  base + gpu_src * gpu_num + gpu_id, gpu_src);
        add = this->add(this->identity(input, {recv}), recv_buf,
                        this->identity(input));
    }

    this->next_eid += gpu_num * gpu_num;
    return input;
}

} // namespace ark
