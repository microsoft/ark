// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_common.hpp"

namespace ark {

Tensor Model::all_reduce(Tensor input, int gpu_id, int gpu_num,
                         [[maybe_unused]] Tensor output, const std::string &) {
    Tensor prev_recv;
    Tensor cumulate = input;
    for (int i = 1; i < gpu_num; i++) {
        int gpu_dst = (gpu_id + i) % gpu_num;
        int gpu_src = (gpu_id + gpu_num - i) % gpu_num;
        Tensor send_data;
        if (prev_recv.is_none()) {
            send_data = input;
        } else {
            send_data = this->identity(input, {prev_recv});
        }
        send_data = this->send(send_data, gpu_id, gpu_dst);
        Tensor send_done_tensor = this->send_done(send_data, gpu_id, gpu_dst);
        Tensor recv_buf = this->tensor(input.shape(), input.data_type());
        recv_buf = this->identity(recv_buf, {send_done_tensor});
        Tensor recv = this->recv(gpu_src, gpu_src, 0, recv_buf);
        prev_recv = recv;
        cumulate = this->add(cumulate, recv);
    }
    return cumulate;
}

}  // namespace ark