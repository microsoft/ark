// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/logging.h"
#include "ark/math.h"
#include "ark/model_io.h"

using namespace std;

namespace ark {

std::vector<Tensor *> *Model::all_gather(Tensor *input, int gpu_id, int gpu_num,
                                         const string &name)
{
    assert(input != nullptr);
    LOG(DEBUG, "all_gather ", input->shape, " ", gpu_id, " ", gpu_num);
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
        LOGERR("all_gather of a split tensor is not supported");
    }

    int base = this->next_eid;
    vector<Tensor *> send_tensors;
    for (int gpu_dst = 0; gpu_dst < gpu_num; gpu_dst++) {
        if (gpu_dst == gpu_id)
            continue;
        Tensor *send_t =
            this->send(input, base + gpu_id * gpu_num + gpu_dst, gpu_dst);
        this->send_done(input, base + gpu_id * gpu_num + gpu_dst);
        send_tensors.push_back(send_t);
    }
    Tensor *add_tensor = input;
    vector<Tensor *> *recv_tensors = new vector<Tensor *>();
    for (int gpu_src = 0; gpu_src < gpu_num; gpu_src++) {
        Tensor *recv_buf = this->tensor(input->shape, input->type);
        if (gpu_src == gpu_id)
            continue;
        if (add_tensor != nullptr) {
            send_tensors.push_back(add_tensor);
        }
        Tensor *recv = this->recv(this->identity(recv_buf, send_tensors),
                                  base + gpu_src * gpu_num + gpu_id, gpu_src);
        recv_tensors->push_back(recv);
    }

    this->next_eid += gpu_num * gpu_num;
    return recv_tensors;
}

} // namespace ark
