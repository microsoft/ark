// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_common.hpp"

namespace ark {

Tensor Model::all_reduce(Tensor input, int gpu_id, int gpu_num, Tensor output,
                         const std::string &) {
    std::vector<int> tags(gpu_num);
    for (int i = 0; i < gpu_num; i++) {
        tags[i] = this->unique_tag();
    }
    if (output.is_null()) {
        output = this->tensor(input.shape(), input.data_type(), input.strides(),
                              input.offsets(), input.padded_shape());
    }
    if (output != input) {
        output = this->copy(input, output);
    }
    Tensor prev_recv = NullTensor;
    Tensor cumulate = output;
    for (int i = 1; i < gpu_num; i++) {
        int gpu_dst = (gpu_id + i) % gpu_num;
        int gpu_src = (gpu_id + gpu_num - i) % gpu_num;
        Tensor send_data;
        if (prev_recv.is_null()) {
            send_data = output;
        } else {
            send_data = this->identity(output, {prev_recv});
        }
        send_data = this->send(send_data, gpu_dst, tags[gpu_id]);
        Tensor send_done_tensor = this->send_done(send_data);
        Tensor recv_buf = this->tensor(output.shape(), output.data_type());
        Tensor recv = this->identity(recv_buf, {send_done_tensor});
        recv = this->recv(recv_buf, gpu_src, tags[gpu_src]);
        prev_recv = recv;
        cumulate = this->add(cumulate, recv, cumulate);
    }
    return cumulate;
}

}  // namespace ark
