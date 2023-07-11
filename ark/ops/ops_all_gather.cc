// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/logging.h"
#include "ark/math.h"
#include "ark/model_io.h"

using namespace std;

namespace ark {

std::vector<Tensor *> Model::all_gather(Tensor *input, int gpu_id, int gpu_num,
                                        std::vector<Tensor *> output,
                                        const string &name)
{
    LOG(DEBUG, "all_gather ", input);
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
    Tensor *recv_buf;
    for (int gpu_src = 0; gpu_src < gpu_num; gpu_src++) {
        recv_buf = nullptr;
        // if gpu_src == gpu_id, the tensor is local
        if (gpu_src == gpu_id) {
            output.push_back(input);
            continue;
        }
        if (output.size() > gpu_src) {
            recv_buf = output[gpu_src];
        }
        if (recv_buf == nullptr) {
            recv_buf = this->tensor(input->shape, input->type);
            output.push_back(recv_buf);
        }
        if (add_tensor != nullptr) {
            send_tensors.push_back(add_tensor);
        }
        Tensor *recv = this->recv(this->identity(recv_buf, send_tensors),
                                  base + gpu_src * gpu_num + gpu_id, gpu_src);
    }

    this->next_eid += gpu_num * gpu_num;
    return output;
}

} // namespace ark
