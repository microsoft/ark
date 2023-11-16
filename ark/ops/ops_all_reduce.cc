// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>

#include "logging.h"
#include "math.h"
#include "model.h"
#include "ops_common.h"

#ifdef ARK_USE_MSCCLPP
#include <mscclpp/packet.hpp>
constexpr int MSCCLPP_PACKET_SIZE = sizeof(mscclpp::LLPacket);
#else
constexpr int MSCCLPP_PACKET_SIZE = 16;
#endif

namespace ark {

Tensor *Model::all_reduce(Tensor *input, int gpu_id, int gpu_num,
                          Tensor *output, const std::string &) {
    assert(input != nullptr);
    if (output != nullptr) {
        ERR(InvalidUsageError, "all_reduce output is not supported");
    }
    if (!input->is_sequential()) {
        LOG(WARN,
            "all_reduce may not work correctly if the input tensor is "
            "not contiguous");
    }
    int base = this->impl->next_eid;
    Tensor *prev_recv = nullptr;
    Tensor *cumulate = input;
    for (int i = 1; i < gpu_num; i++) {
        int gpu_dst = (gpu_id + i) % gpu_num;
        int gpu_src = (gpu_id + gpu_num - i) % gpu_num;
        Tensor *send_data;
        if (prev_recv != nullptr) {
            send_data = this->identity(input, {prev_recv});
        } else {
            send_data = input;
        }
        send_data = this->send(send_data, base + gpu_id, gpu_dst);
        Tensor *send_done_tensor =
            this->send_done(send_data, base + gpu_id, gpu_dst);
        Tensor *recv_buf = this->tensor(input->shape, input->type);
        recv_buf = this->identity(recv_buf, {send_done_tensor});
        Tensor *recv = this->recv(base + gpu_src, gpu_src, 0, recv_buf);
        prev_recv = recv;
        cumulate = this->add(cumulate, recv);
    }
    this->impl->next_eid += gpu_num;
    return cumulate;
}

Tensor *Model::local_all_reduce_mscclpp(Tensor *input, int gpu_id, int gpu_num,
                                        const std::string &) {
    assert(input != nullptr);
    if (!input->is_sequential()) {
        LOG(WARN,
            "all_reduce may not work correctly if the input tensor is "
            "not contiguous");
    }
    ark::Dims ori_shape = input->shape;
    Tensor *input_reshaped = this->reshape(input, {input->shape.size()});
    Tensor *out =
        this->local_reduce_scatter_mscclpp(input_reshaped, gpu_id, gpu_num);
    Tensor *res = this->local_all_gather_mscclpp(out, gpu_id, gpu_num);
    return this->reshape(res, ori_shape);
}

Tensor *Model::local_all_reduce_packet_mscclpp(Tensor *input, int gpu_id,
                                               int gpu_num,
                                               const std::string &) {
    assert(input != nullptr);
    // We only support out-of-place all_reduce
    if (input->ndims() > 1) {
        ERR(InvalidUsageError, "supports only 1D input");
    }
    if (!input->is_sequential()) {
        LOG(WARN,
            "all_reduce may not work correctly if the input tensor is "
            "not contiguous");
    }
    Tensor *out = this->tensor(input->shape, input->type);
    // only half of the packets are used to store data
    const int num_packets = input->shape_bytes() / (MSCCLPP_PACKET_SIZE / 2);
    const int scratch_nelems = num_packets *
                               2 /*oringinal data & reduced result*/ *
                               2 /*double buffer*/;
    Dims scratch_shape = {
        static_cast<ark::DimType>(scratch_nelems * MSCCLPP_PACKET_SIZE)};
    Tensor *scratch = this->tensor(scratch_shape, UINT8);
    int npeer = gpu_num - 1;
    std::vector<Tensor *> outputs;
    std::vector<Tensor *> remote_scratches;
    size_t nelems_per_rank =
        input->shape_bytes() / input->type_bytes() / gpu_num;
    size_t npackets_per_rank = num_packets / gpu_num;
    int flag = this->impl->reduce_packet_flag;
    size_t scratch_base_offset =
        (flag & 1) ? 0 : num_packets * MSCCLPP_PACKET_SIZE;
    size_t scratch_result_offset = (flag & 1)
                                       ? 2 * num_packets * MSCCLPP_PACKET_SIZE
                                       : 3 * num_packets * MSCCLPP_PACKET_SIZE;
    int id = this->impl->next_eid;
    std::vector<Tensor *> sharded_inputs =
        this->sharding(input, 0, nelems_per_rank);
    std::vector<Tensor *> sharded_outputs =
        this->sharding(out, 0, nelems_per_rank);
    for (int i = 0; i < npeer; ++i) {
        int remote_rank = i < gpu_id ? i : i + 1;
        Tensor *remote_scratch = this->tensor(scratch_shape, UINT8);
        remote_scratches.push_back(remote_scratch);
        Tensor *out = this->put_packet_mscclpp(
            sharded_inputs[remote_rank], scratch, remote_scratch, id, gpu_id,
            remote_rank,
            scratch_base_offset +
                npackets_per_rank * gpu_id * MSCCLPP_PACKET_SIZE,
            flag);
        outputs.push_back(out);
    }
    Tensor *input_sharded = this->identity(sharded_inputs[gpu_id], outputs);
    // This op should reduce from the scratch buffer and write to the remote.
    Tensor *out_stage2 = this->reduce_and_write_packet_mscclpp(
        input_sharded, scratch, sharded_outputs[gpu_id], remote_scratches, id,
        gpu_id, npeer, nelems_per_rank, scratch_base_offset,
        scratch_result_offset, flag);
    // Get the result from the scratch buffer.
    Tensor *scratch_stage3 = this->identity(scratch, {out_stage2});
    outputs.clear();
    for (int i = 0; i < npeer; ++i) {
        int remote_rank = i < gpu_id ? i : i + 1;
        size_t dst_offset = nelems_per_rank * remote_rank * input->type_bytes();
        size_t src_offset = scratch_result_offset + npackets_per_rank *
                                                        remote_rank *
                                                        MSCCLPP_PACKET_SIZE;
        Tensor *res =
            this->get_packet_mscclpp(scratch_stage3, out, src_offset,
                                     dst_offset, npackets_per_rank, flag);
        outputs.push_back(res);
    }
    this->impl->next_eid += 1;
    this->impl->reduce_packet_flag += 1;
    return this->identity(out, outputs);
}

}  // namespace ark
