// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/logging.h"
#include "ark/math.h"
#include "ark/model_io.h"

using namespace std;

namespace ark {

// Tensor *Model::all_reduce(Tensor *input, int gpu_id, int gpu_num,
//                           Tensor *output, const string &name)
// {
//     assert(input != nullptr);
//     LOG(DEBUG, "all_reduce ", input->shape, " ", gpu_id, " ", gpu_num);
//     if (input->ndims() > 1) {
//         LOGERR("supports only 1D input");
//     }
//     if ((math::pad(input->shape[0], input->pads[0]) <
//          (size_t)input->ldims[0]) ||
//         (math::pad(input->shape[1], input->pads[1]) <
//          (size_t)input->ldims[1]) ||
//         (math::pad(input->shape[2], input->pads[2]) <
//          (size_t)input->ldims[2]) ||
//         (math::pad(input->shape[3], input->pads[3]) <
//          (size_t)input->ldims[3])) {
//         LOGERR("all_reduce of a split tensor is not supported");
//     }
//     DimType total_num = input->ldims.size();
//     DimType num_per_shard = math::div_up(total_num, gpu_num);
//     vector<Tensor *> shards =
//         this->sharding(input, 0, num_per_shard, name + "/sharding");

//     int gpu_dst = (gpu_id + 1) % gpu_num;
//     int gpu_src = (gpu_id + gpu_num - 1) % gpu_num;
//     int base = this->next_eid;
//     Tensor *reduce_scatter_dep;
//     // reduce-scatter
//     Tensor *add;
//     for (int i = 1; i < gpu_num; ++i) {
//         int send_shard_id = (gpu_id + gpu_num - i + 1) % gpu_num;

//         Tensor *send =
//             this->send(shards[send_shard_id], base + send_shard_id, gpu_dst);

//         int recv_shard_id = (gpu_id + gpu_num - i) % gpu_num;
//         Tensor *recv_shard = shards[recv_shard_id];
//         Tensor *recv_buf = this->tensor(recv_shard->shape, recv_shard->type);
//         Tensor *recv = this->recv(this->identity(recv_buf, {send}),
//                                   base + recv_shard_id, gpu_src);
//         add = this->add(this->identity(recv_shard, {recv}), recv_buf,
//                         this->identity(recv_shard));
//         reduce_scatter_dep = add;
//     }
//     base += gpu_num;
//     // all-gather
//     for (int i = 1; i < gpu_num; ++i) {
//         int send_shard_id = (gpu_id + gpu_num - i) % gpu_num;
//         Tensor *send_shard = this->identity(add, {reduce_scatter_dep});
//         // send_shard->exported = true;
//         Tensor *send_t = this->send(send_shard, base + send_shard_id,
//         gpu_dst);

//         int recv_shard_id = (gpu_id + gpu_num - i - 1) % gpu_num;
//         LOG(DEBUG, "GPU ", gpu_id, " send shard ", send_shard_id,
//             " recv shard ", recv_shard_id);
//         Tensor *recv_shard = shards[recv_shard_id];
//         Tensor *recv_buf = this->tensor(recv_shard->shape, recv_shard->type);
//         Tensor *recv = this->recv(this->identity(recv_buf, {send_t}),
//                                   base + recv_shard_id, gpu_src);
//         Tensor *dummy = this->tensor(recv_shard->shape, recv_shard->type);
//         this->add(this->identity(recv_buf, {recv}), dummy,
//                   this->identity(recv_shard));
//     }

//     this->next_eid += 2 * gpu_num;
//     return input;
// }

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
    Tensor *send_t;
    for (int gpu_dst = 0; gpu_dst < gpu_num; gpu_dst++) {
        if (gpu_dst == gpu_id)
            continue;
        send_t = this->send(input, base + gpu_dst, gpu_dst);
    }
    Tensor *recv_buf = this->tensor(input->shape, input->type);
    for (int gpu_src = 0; gpu_src < gpu_num; gpu_src++) {
        if (gpu_src == gpu_id)
            continue;
        Tensor *recv = this->recv(this->identity(recv_buf, {send_t}),
                                  base + gpu_id, gpu_src);
        Tensor *add = this->add(this->identity(input, {recv}), recv_buf,
                                this->identity(input));
    }

    this->next_eid += 1;
    return input;
}

} // namespace ark
