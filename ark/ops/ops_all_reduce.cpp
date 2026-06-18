// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cstdint>

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

    size_t nelems = input.shape().nelems();
    size_t elems_per_uint32 = sizeof(uint32_t) / input.data_type().bytes();
    if (nelems % (elems_per_uint32 * 2 * rank_num) != 0) {
        ERR(ModelError, "all_reduce_packet: nelems (", nelems,
            ") must be divisible by ", elems_per_uint32 * 2 * rank_num);
    }

    auto &buf_reg = BufferRegistry::get_instance();
    size_t input_id = input.ref()->buffer()->id();
    auto input_info = buf_reg.get(input_id);
    bool input_external =
        input.is_external() || (input_info && input_info->is_external);
    if (input_external && !input.offsets().is_no_dim() &&
        !input.offsets().is_zeros()) {
        ERR(ModelError,
            "all_reduce_packet does not support external input offsets");
    }

    if (input_external && (input.padded_shape() != input.shape() ||
                           input.strides() != input.padded_shape())) {
        ERR(ModelError, "all_reduce_packet supports only dense external input");
    }

    if (!output.is_null()) {
        size_t output_id = output.ref()->buffer()->id();
        auto output_info = buf_reg.get(output_id);
        bool output_external =
            output.is_external() || (output_info && output_info->is_external);
        if (output_external && !output.offsets().is_no_dim() &&
            !output.offsets().is_zeros()) {
            ERR(ModelError,
                "all_reduce_packet does not support external output offsets");
        }
        if (output_external && (output.padded_shape() != output.shape() ||
                                output.strides() != output.padded_shape())) {
            ERR(ModelError,
                "all_reduce_packet supports only dense external output");
        }
        if (input_external && output_external) {
            bool same_buffer = input_id == output_id;
            bool ranges_overlap = false;
            if (input_info && output_info && input_info->data &&
                output_info->data) {
                auto input_begin =
                    reinterpret_cast<std::uintptr_t>(input_info->data);
                auto output_begin =
                    reinterpret_cast<std::uintptr_t>(output_info->data);
                auto input_end = input_begin + input.ref()->shape_bytes();
                auto output_end = output_begin + output.ref()->shape_bytes();
                ranges_overlap =
                    input_begin < output_end && output_begin < input_end;
            }
            if (same_buffer || ranges_overlap) {
                ERR(ModelError,
                    "all_reduce_packet does not support aliased external "
                    "input/output");
            }
        }
    } else {
        output = this->tensor(input.shape(), input.data_type());
    }

    // Scratch layout: [input_section | result_section]
    // Each section holds NPkts = nelems_int32 / 2 packets of 16 bytes each.
    // Total: 2 × NPkts × 16 = nelems_int32 × 16 = nelems_fp16 × 8 bytes.
    size_t nelems_int32 = nelems / elems_per_uint32;
    size_t n_pkts = nelems_int32 / 2;  // each packet carries uint2 = 2×u32
    size_t packet_size = 16;           // sizeof(mscclpp::LL16Packet)
    size_t scratch_bytes = 2 * n_pkts * packet_size;
    Dims scratch_shape(static_cast<DimType>(scratch_bytes));
    Tensor scratch = this->tensor(scratch_shape, UINT8);

    // Peer scratch refs — remote buffers at the same offset (symmetric alloc).
    std::vector<ModelTensorRef> peer_scratch_refs;
    for (int i = 0; i < rank_num; ++i) {
        if (i == rank) continue;
        peer_scratch_refs.push_back(std::make_shared<ModelTensor>(
            UINT8.ref(), std::make_shared<ModelBuffer>(i), scratch_shape));
    }

    uint32_t flag = 1;  // Hardcoded; per-call rotation deferred.
    return impl_
        ->create_op<ModelOpAllReducePacketFused>(
            "all_reduce_packet", input.ref(), output.ref(), rank, rank_num,
            flag, scratch.ref(), peer_scratch_refs)
        ->result_tensors()[0];
}

}  // namespace ark
