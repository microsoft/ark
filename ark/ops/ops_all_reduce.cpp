// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "buffer_registry.hpp"
#include "ops_common.hpp"
#include "ops_communication.hpp"

namespace {

constexpr ark::DimType kQwen3HiddenSize = 4096;
constexpr ark::DimType kQwen3PrefillTokens = 2048;
constexpr ark::DimType kQwen3DecodeElements = kQwen3HiddenSize;
constexpr ark::DimType kQwen3PrefillElements =
    kQwen3PrefillTokens * kQwen3HiddenSize;
constexpr int kMaxQwen3Tp = 8;

bool is_supported_qwen3_route(ark::Tensor input, int rank_num,
                              ark::DimType nelems) {
    constexpr int kFp16ElemsPerPacket = 4;
    return input.data_type() == ark::FP16 && input.shape().nelems() == nelems &&
           rank_num >= 2 && rank_num <= kMaxQwen3Tp &&
           nelems % (kFp16ElemsPerPacket * rank_num) == 0;
}

ark::Tensor all_reduce_ring(ark::Model &model, ark::Tensor input, int gpu_id,
                            int gpu_num, ark::Tensor output) {
    std::vector<int> tags(gpu_num);
    for (int i = 0; i < gpu_num; i++) {
        tags[i] = model.unique_tag();
    }
    if (output.is_null()) {
        output = model.copy(input);
    } else if (output.ref()->buffer()->id() == input.ref()->buffer()->id()) {
        // In-place: copy input so the ring loop does not mutate send data.
        // TODO: This catches the common case (output IS input). Sub-buffer
        // offset aliasing or aliasing through different buffer objects backed
        // by the same allocation is not currently detected.
        input = model.copy(input);
    }
    ark::Tensor prev_recv = ark::NullTensor;
    ark::Tensor cumulate = output;
    for (int i = 1; i < gpu_num; i++) {
        int gpu_dst = (gpu_id + i) % gpu_num;
        int gpu_src = (gpu_id + gpu_num - i) % gpu_num;
        ark::Tensor send_data;
        if (prev_recv.is_null()) {
            send_data = input;
        } else {
            send_data = model.identity(input, {prev_recv});
        }
        send_data = model.send(send_data, gpu_dst, tags[gpu_id]);
        ark::Tensor send_done_tensor = model.send_done(send_data);
        ark::Tensor recv_buf = model.tensor(output.shape(), output.data_type());
        ark::Tensor recv = model.identity(recv_buf, {send_done_tensor});
        recv = model.recv(recv_buf, gpu_src, tags[gpu_src]);
        prev_recv = recv;
        cumulate = model.add(cumulate, recv, cumulate);
    }
    return cumulate;
}

}  // namespace

namespace ark {

Tensor Model::all_reduce(Tensor input, int gpu_id, int gpu_num, Tensor output,
                         const std::string &) {
    // Route only the Qwen3 shapes that this prerequisite owns. A rank/model
    // mismatch falls back to the legacy ring path so the graph keeps one rank
    // authority instead of mixing Model::rank() with an explicit rank.
    if (gpu_id == this->rank() &&
        is_supported_qwen3_route(input, gpu_num, kQwen3PrefillElements)) {
        return this->all_reduce_prefill(input, gpu_id, gpu_num, output,
                                        "all_reduce_prefill");
    }
    if (gpu_id == this->rank() &&
        is_supported_qwen3_route(input, gpu_num, kQwen3DecodeElements)) {
        return this->all_reduce_packet(input, gpu_id, gpu_num, output,
                                       "all_reduce_packet");
    }
    return all_reduce_ring(*this, input, gpu_id, gpu_num, output);
}

Tensor Model::all_reduce_packet_impl(Tensor input, int rank, int rank_num,
                                     Tensor output,
                                     const std::string &op_name) {
    int n_peers = rank_num - 1;
    if (n_peers < 1) {
        ERR(ModelError, op_name, " requires rank_num >= 2");
    }
    if (rank < 0 || rank >= rank_num) {
        ERR(ModelError, op_name, ": rank ", rank,
            " must be in [0, rank_num)");
    }

    size_t nelems = input.shape().nelems();
    size_t elems_per_uint32 = sizeof(uint32_t) / input.data_type().bytes();
    if (elems_per_uint32 == 0) {
        ERR(ModelError, op_name, ": unsupported data type ",
            input.data_type().name());
    }
    if (nelems % (elems_per_uint32 * 2 * rank_num) != 0) {
        ERR(ModelError, op_name, ": nelems (", nelems,
            ") must be divisible by ", elems_per_uint32 * 2 * rank_num);
    }

    auto input_info =
        BufferRegistry::get_instance().get(input.ref()->buffer()->id());
    std::shared_ptr<BufferRegistry::Info> output_info;
    if (!output.is_null()) {
        output_info =
            BufferRegistry::get_instance().get(output.ref()->buffer()->id());
    }
    bool output_aliases_input =
        !output.is_null() &&
        (output.ref()->buffer()->id() == input.ref()->buffer()->id() ||
         (input_info && output_info && input_info->data != nullptr &&
          input_info->data == output_info->data));

    // Copy external input into an internal buffer so it resides in mscclpp
    // registered memory. Internal ARK tensors are already registered. In-place
    // calls also need a copy so the collective does not overwrite input shards
    // before all peers read them.
    if (input.is_external() || (input_info && input_info->is_external) ||
        output_aliases_input) {
        input = this->copy(input);
    }

    if (output.is_null()) {
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
            op_name, input.ref(), output.ref(), rank, rank_num, flag,
            scratch.ref(), peer_scratch_refs)
        ->result_tensors()[0];
}

Tensor Model::all_reduce_packet(Tensor input, int rank, int rank_num,
                                Tensor output, const std::string &name) {
    return this->all_reduce_packet_impl(
        input, rank, rank_num, output,
        name.empty() ? std::string("all_reduce_packet") : name);
}

Tensor Model::all_reduce_prefill(Tensor input, int rank, int rank_num,
                                 Tensor output, const std::string &name) {
    if (!is_supported_qwen3_route(input, rank_num, kQwen3PrefillElements)) {
        ERR(ModelError,
            "all_reduce_prefill supports only FP16 Qwen3 prefill tensors with ",
            kQwen3PrefillElements,
            " elements, rank_num in [2, 8], and packet-aligned shards; got dtype=",
            input.data_type().name(), ", nelems=", input.shape().nelems(),
            ", rank_num=", rank_num);
    }
    if (rank != this->rank()) {
        ERR(ModelError, "all_reduce_prefill rank ", rank,
            " must match model rank ", this->rank());
    }
    return this->all_reduce_packet_impl(
        input, rank, rank_num, output,
        name.empty() ? std::string("all_reduce_prefill") : name);
}

}  // namespace ark
