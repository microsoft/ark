// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "math_utils.h"
#include "ops_common.hpp"

namespace ark {

ModelTensorRef Model::all_reduce(ModelTensorRef input, int gpu_id, int gpu_num,
                                 [[maybe_unused]] ModelTensorRef output,
                                 const std::string &) {
    if (!input->is_sequential()) {
        LOG(WARN,
            "all_reduce may not work correctly if the input tensor is "
            "not contiguous");
    }
    ModelTensorRef prev_recv;
    ModelTensorRef cumulate = input;
    for (int i = 1; i < gpu_num; i++) {
        int gpu_dst = (gpu_id + i) % gpu_num;
        int gpu_src = (gpu_id + gpu_num - i) % gpu_num;
        ModelTensorRef send_data;
        if (prev_recv) {
            send_data = this->identity(input, {prev_recv});
        } else {
            send_data = input;
        }
        send_data = this->send(send_data, gpu_id, gpu_dst);
        ModelTensorRef send_done_tensor =
            this->send_done(send_data, gpu_id, gpu_dst);
        ModelTensorRef recv_buf =
            this->tensor(input->shape(), input->data_type());
        recv_buf = this->identity(recv_buf, {send_done_tensor});
        ModelTensorRef recv = this->recv(gpu_src, gpu_src, 0, recv_buf);
        prev_recv = recv;
        cumulate = this->add(cumulate, recv);
    }
    return cumulate;
}

}  // namespace ark
