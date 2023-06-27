// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/logging.h"
#include "ark/math.h"
#include "ark/model_io.h"

using namespace std;

namespace ark {

//
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
    DimType total_num = input->ldims.size();
    DimType num_per_shard = math::div_up(total_num, gpu_num);
    vector<Tensor *> shards =
        this->sharding(input, 0, num_per_shard, name + "/sharding");
    vector<Tensor *> bufs;
    bufs.resize(gpu_num);

    int gpu_dst = (gpu_id + 1) % gpu_num;
    int gpu_src = (gpu_id + gpu_num - 1) % gpu_num;
    int base = this->next_eid;

    int icnt = 0;
    int rcnt = 0;
    int scnt = 0;
    // reduce-scatter
    for (int i = 1; i < gpu_num; ++i) {
        int shard_send_id = (gpu_id + gpu_num - i + 1) % gpu_num;

        Tensor *send =
            this->send(shards[shard_send_id], base + shard_send_id, gpu_dst, 0,
                       nullptr, name + "/send_" + to_string(scnt++));
        bufs[shard_send_id] = shards[shard_send_id];

        int shard_recv_id = (gpu_id + gpu_num - i) % gpu_num;
        Tensor *recv_shard = shards[shard_recv_id];
        Tensor *recv_buf = this->tensor(recv_shard->shape, recv_shard->type);
        Tensor *recv = this->recv(this->identity(recv_buf, {send}),
                                  base + shard_recv_id, gpu_src, 0, nullptr,
                                  name + "/recv_" + to_string(rcnt++));
        Tensor *add = this->add(this->identity(recv_shard, {recv}), recv_buf,
                                this->identity(recv_shard),
                                name + "/add_" + to_string(i - 1));
    }

    // all-gather
    for (int i = 1; i < gpu_num; ++i) {
    }

    this->next_eid += gpu_num;
    return input;
}

} // namespace ark
