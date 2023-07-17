// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "math.h"
#include "model_io.h"

using namespace std;

namespace ark {

std::vector<Tensor *> Model::all_gather(Tensor *input, int gpu_id, int gpu_num,
                                        std::vector<Tensor *> output,
                                        const string &name)
{
    assert(input != nullptr);
    LOG(DEBUG, "all_gather ", input->shape, " ", gpu_id, " ", gpu_num);
    if (input->ndims() > 1) {
        LOG(INFO,
            "warning: if the send tensor if not contiguous, the all_gather "
            "may not work correctly");
    }
    LOG(DEBUG, "all gather output size: ", output.size());

    int base = this->next_eid;
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
    recv_dep_tensors.push_back(input);

    Tensor *recv_buf;
    for (int gpu_src = 0; gpu_src < gpu_num; gpu_src++) {
        recv_buf = nullptr;
        // if gpu_src == gpu_id, the tensor is local
        if (gpu_src == gpu_id) {
            output.push_back(input);
            continue;
        }
        if (output.size() > (size_t)gpu_src) {
            recv_buf = output[gpu_src];
        }
        if (recv_buf == nullptr) {
            recv_buf = this->tensor(input->shape, input->type);
            output.push_back(recv_buf);
        }
        Tensor *recv = this->recv(this->identity(recv_buf, recv_dep_tensors),
                                  base + gpu_src * gpu_num + gpu_id, gpu_src);
    }

    this->next_eid += gpu_num * gpu_num;
    return output;
}

} // namespace ark
