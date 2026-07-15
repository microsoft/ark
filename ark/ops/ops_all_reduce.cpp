// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "buffer_registry.hpp"
#include "ops_common.hpp"
#include "ops_communication.hpp"

namespace ark {

Tensor Model::all_reduce(Tensor input, int gpu_id, int gpu_num, Tensor output,
                         const std::string &) {
    std::vector<int> tags(gpu_num);
    for (int i = 0; i < gpu_num; i++) {
        tags[i] = this->unique_tag();
    }
    if (output.is_null()) {
        output = this->copy(input);
    } else if (output.ref()->buffer()->id() == input.ref()->buffer()->id()) {
        // In-place: copy input so the ring loop does not mutate send data.
        // TODO: This catches the common case (output IS input). Sub-buffer
        // offset aliasing or aliasing through different buffer objects backed
        // by the same allocation is not currently detected.
        input = this->copy(input);
    }
    Tensor prev_recv = NullTensor;
    Tensor cumulate = output;
    for (int i = 1; i < gpu_num; i++) {
        int gpu_dst = (gpu_id + i) % gpu_num;
        int gpu_src = (gpu_id + gpu_num - i) % gpu_num;
        Tensor send_data;
        if (prev_recv.is_null()) {
            send_data = input;
        } else {
            send_data = this->identity(input, {prev_recv});
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

Tensor Model::all_reduce_packet(Tensor input, int rank, int rank_num,
                                Tensor output, const std::string &) {
    int n_peers = rank_num - 1;
    if (n_peers < 1) {
        ERR(ModelError, "all_reduce_packet requires rank_num >= 2");
    }
    if (input.shape().ndims() < 2) {
        ERR(ModelError, "all_reduce_packet requires a 2-D input");
    }
    DimType cols = input.shape()[-1];
    size_t elems_per_pkt = 8 / input.data_type().bytes();
    if (cols % elems_per_pkt != 0) {
        ERR(ModelError, "all_reduce_packet: cols (", cols,
            ") must be divisible by ", elems_per_pkt);
    }

    // Copy external input into an internal (mscclpp-registered) buffer.
    auto input_info =
        BufferRegistry::get_instance().get(input.ref()->buffer()->id());
    if (input.is_external() || (input_info && input_info->is_external)) {
        input = this->copy(input);
    }

    if (output.is_null()) {
        output = this->tensor(input.shape(), input.data_type());
    }

    std::vector<ModelTensorRef> peer_output_refs;
    for (int i = 0; i < rank_num; ++i) {
        if (i == rank) continue;
        peer_output_refs.push_back(std::make_shared<ModelTensor>(
            output.data_type().ref(), std::make_shared<ModelBuffer>(i),
            output.shape()));
    }

    return impl_
        ->create_op<ModelOpAllReducePacket>(
            "all_reduce_packet", input.ref(), output.ref(), rank, rank_num,
            peer_output_refs)
        ->result_tensors()[0];
}

Tensor Model::all_reduce_rsag(Tensor input, int rank, int rank_num,
                              Tensor output, const std::string &) {
    int n_peers = rank_num - 1;
    if (n_peers < 1) {
        ERR(ModelError, "all_reduce_rsag requires rank_num >= 2");
    }
    DimType nelems = input.shape().nelems();
    size_t elems_per_int4 = 16 / input.data_type().bytes();
    if (nelems % (static_cast<DimType>(rank_num) *
                  static_cast<DimType>(elems_per_int4)) != 0) {
        ERR(ModelError, "all_reduce_rsag: nelems (", nelems,
            ") must be divisible by rank_num*", elems_per_int4);
    }

    // Copy external input into an internal (mscclpp-registered) buffer.
    auto input_info =
        BufferRegistry::get_instance().get(input.ref()->buffer()->id());
    if (input.is_external() || (input_info && input_info->is_external)) {
        input = this->copy(input);
    }

    if (output.is_null()) {
        output = this->tensor(input.shape(), input.data_type());
    }

    std::vector<ModelTensorRef> peer_output_refs;
    for (int i = 0; i < rank_num; ++i) {
        if (i == rank) continue;
        peer_output_refs.push_back(std::make_shared<ModelTensor>(
            output.data_type().ref(), std::make_shared<ModelBuffer>(i),
            output.shape()));
    }

    return impl_
        ->create_op<ModelOpAllReduceRsag>("all_reduce_rsag", input.ref(),
                                          output.ref(), rank, rank_num,
                                          peer_output_refs)
        ->result_tensors()[0];
}

Tensor Model::all_reduce_allpair_packet(Tensor input, int rank, int rank_num,
                                        Tensor output, const std::string &) {
    int n_peers = rank_num - 1;
    if (n_peers < 1) {
        ERR(ModelError, "all_reduce_allpair_packet requires rank_num >= 2");
    }
    DimType nelems = input.shape().nelems();
    // LL8Packet packs 2 bf16/fp16 per packet; the whole-array packet count must
    // be exact (grid-strided kernel, no remainder handling).
    size_t elems_per_pkt = 4 / input.data_type().bytes();  // 2 for fp16/bf16
    if (nelems % static_cast<DimType>(elems_per_pkt) != 0) {
        ERR(ModelError, "all_reduce_allpair_packet: nelems (", nelems,
            ") must be divisible by ", elems_per_pkt);
    }

    // Copy external input into an internal (mscclpp-registered) buffer.
    auto input_info =
        BufferRegistry::get_instance().get(input.ref()->buffer()->id());
    if (input.is_external() || (input_info && input_info->is_external)) {
        input = this->copy(input);
    }

    if (output.is_null()) {
        output = this->tensor(input.shape(), input.data_type());
    }

    std::vector<ModelTensorRef> peer_output_refs;
    for (int i = 0; i < rank_num; ++i) {
        if (i == rank) continue;
        peer_output_refs.push_back(std::make_shared<ModelTensor>(
            output.data_type().ref(), std::make_shared<ModelBuffer>(i),
            output.shape()));
    }

    return impl_
        ->create_op<ModelOpAllReduceAllpairPacket>(
            "all_reduce_allpair_packet", input.ref(), output.ref(), rank,
            rank_num, peer_output_refs)
        ->result_tensors()[0];
}


}  // namespace ark
