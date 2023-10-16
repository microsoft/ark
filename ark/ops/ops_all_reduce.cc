// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>

#include "logging.h"
#include "math.h"
#include "model.h"
#include "ops_common.h"

#ifdef ARK_USE_MSCCLPP
#include <mscclpp/packet.hpp>
#endif

namespace ark {

Tensor *Model::all_reduce(Tensor *input, int gpu_id, int gpu_num,
                          Tensor *output, const std::string &) {
    assert(input != nullptr);
    if (output != nullptr) {
        LOG(ERROR, "all_reduce output is not supported");
    }
    if (input->ndims() > 1) {
        LOG(ERROR, "supports only 1D input");
    }
    if (!input->is_sequential()) {
        LOG(WARN,
            "all_reduce may not work correctly if the input tensor is "
            "not contiguous");
    }
    if (math::pad(input->shape[0], input->pads[0]) < (size_t)input->ldims[0]) {
        LOG(ERROR, "all_reduce of a split tensor is not supported");
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

Tensor *Model::local_all_reduce(Tensor *input, int gpu_id, int gpu_num,
                                const std::string &) {
    assert(input != nullptr);
    if (input->ndims() > 1) {
        LOG(ERROR, "supports only 1D input");
    }
    if (!input->is_sequential()) {
        LOG(WARN, "all_reduce may not work correctly if the input tensor is "
                  "not contiguous");
    }
    int sid = this->impl->next_eid;
    Tensor* out = this->local_reduce_scatter_mscclpp(input, gpu_id, sid, gpu_num);
    Tensor *res = this->local_all_gather_mscclpp(out, gpu_id, sid, gpu_num);
    this->impl->next_eid += gpu_num;
    return res;
}

Tensor *Model::local_all_reduce_packet(Tensor *input, int gpu_id, int gpu_num,
                                       const std::string &) {
    assert(input != nullptr);
    // We only support out-of-place all_reduce
    if (input->ndims() > 1) {
        LOG(ERROR, "supports only 1D input");
    }
    if (!input->is_sequential()) {
        LOG(WARN,
            "all_reduce may not work correctly if the input tensor is "
            "not contiguous");
    }
    Tensor *out = this->tensor(input->shape, input->type);
    // only half of the packets are used to store data
    const int num_packets = input->shape_bytes() / (sizeof(mscclpp::LLPacket) / 2);
    const int scratch_nelems = num_packets *
                              2 /*oringinal data & reduced result*/ *
                              2 /*double buffer*/;
    Dims scratch_shape = {
        static_cast<ark::DimType>(scratch_nelems * sizeof(mscclpp::LLPacket))};
    Tensor *scratch = this->tensor(scratch_shape, UINT8);
    scratch->exported = true;
    int npeer = gpu_num - 1;
    std::vector<Tensor*> outputs;
    std::vector<Tensor*> remote_scratchs;
    size_t nelems_per_rank = input->shape_bytes() / input->type_bytes() / gpu_num;
    size_t npackets_per_rank = num_packets / gpu_num;
    int flag = this->impl->reduce_packet_flag;
    size_t scratch_base_offset = (flag & 1) ? 0 : num_packets * sizeof(mscclpp::LLPacket);
    size_t scratch_result_offset =
        (flag & 1) ? 2 * num_packets * sizeof(mscclpp::LLPacket)
                   : 3 * num_packets * sizeof(mscclpp::LLPacket);
    int id = this->impl->next_eid;
    for (int i = 0; i < npeer; ++i) {
        int remote_rank = i < gpu_id ? i : i + 1;
        Tensor *remote_scratch = this->tensor(scratch_shape, UINT8);
        remote_scratchs.push_back(remote_scratch);
        Tensor *out = this->put_packet_mscclpp(
            input, scratch, remote_scratch, id, gpu_id, remote_rank,
            nelems_per_rank * remote_rank * input->type_bytes(),
            scratch_base_offset +
                npackets_per_rank * gpu_id * sizeof(mscclpp::LLPacket),
            nelems_per_rank * input->type_bytes(), flag);
        outputs.push_back(out);
    }
    Tensor *scratch_stage2 = this->identity(scratch, outputs);
    // // This op should reduce from the scratch buffer and write to the remote. The input is all peers scratch buffset
    Tensor *out_stage2 = this->reduce_and_write_packet_mscclpp(
        scratch_stage2, out, remote_scratchs, id, gpu_id, npeer,
        nelems_per_rank, scratch_base_offset, scratch_result_offset, flag);
    // Tensor* out = this->impl->add_op()[0];
    this->impl->next_eid += 1;
    this->impl->reduce_packet_flag += 1;
    return this->identity(out, {out_stage2});
}

} // namespace ark
