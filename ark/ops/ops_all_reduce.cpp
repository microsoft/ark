// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "buffer_registry.hpp"
#include "ops_common.hpp"
#include "ops_communication.hpp"

namespace {

constexpr size_t kAutoPacketMaxBytes = 32 << 10;

void require_valid_all_reduce_rank(int rank, int rank_num,
                                   const char *op_name) {
    if (rank_num < 2) {
        ERR(ark::ModelError, op_name, " requires rank_num >= 2");
    }
    if (rank < 0 || rank >= rank_num) {
        ERR(ark::ModelError, op_name,
            " requires rank to satisfy 0 <= rank < rank_num");
    }
}

bool all_reduce_packet_supported(ark::Tensor input, int rank_num) {
    if (rank_num < 2) {
        return false;
    }
    size_t dtype_bytes = input.data_type().bytes();
    if (dtype_bytes == 0 || sizeof(uint32_t) % dtype_bytes != 0) {
        return false;
    }
    size_t elems_per_uint32 = sizeof(uint32_t) / dtype_bytes;
    size_t divisor = elems_per_uint32 * 2 * rank_num;
    return input.shape().nelems() % divisor == 0;
}

void require_all_reduce_packet_supported(ark::Tensor input, int rank,
                                         int rank_num) {
    require_valid_all_reduce_rank(rank, rank_num, "all_reduce_packet");
    size_t dtype_bytes = input.data_type().bytes();
    if (dtype_bytes == 0 || sizeof(uint32_t) % dtype_bytes != 0) {
        ERR(ark::ModelError, "all_reduce_packet: unsupported data type ",
            input.data_type().name());
    }
    size_t nelems = input.shape().nelems();
    size_t elems_per_uint32 = sizeof(uint32_t) / dtype_bytes;
    if (nelems % (elems_per_uint32 * 2 * rank_num) != 0) {
        ERR(ark::ModelError, "all_reduce_packet: nelems (", nelems,
            ") must be divisible by ", elems_per_uint32 * 2 * rank_num);
    }
}

}  // namespace

namespace ark {

Tensor Model::all_reduce(Tensor input, int gpu_id, int gpu_num, Tensor output,
                         const std::string &) {
    std::vector<int> tags(gpu_num);
    for (int i = 0; i < gpu_num; i++) {
        tags[i] = this->unique_tag();
    }
    auto input_info =
        BufferRegistry::get_instance().get(input.ref()->buffer()->id());
    if (input.is_external() || (input_info && input_info->is_external)) {
        input = this->copy(input);
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

std::string Model::all_reduce_route(Tensor input, int rank, int rank_num,
                                    const std::string &route) {
    require_valid_all_reduce_rank(rank, rank_num, "all_reduce_route");

    std::string requested = route.empty() ? "auto" : route;
    if (requested == "auto") {
        size_t input_bytes = input.shape().nelems() * input.data_type().bytes();
        if (input_bytes <= kAutoPacketMaxBytes &&
            all_reduce_packet_supported(input, rank_num)) {
            return "packet";
        }
        return "ring";
    }
    if (requested == "packet") {
        require_all_reduce_packet_supported(input, rank, rank_num);
        return "packet";
    }
    if (requested == "ring") {
        return "ring";
    }
    ERR(ModelError, "unknown all_reduce route: ", requested);
    return "";
}

Tensor Model::all_reduce_routed(Tensor input, int rank, int rank_num,
                                Tensor output, const std::string &route,
                                const std::string &name) {
    std::string selected = this->all_reduce_route(input, rank, rank_num, route);
    if (selected == "packet") {
        return this->all_reduce_packet(input, rank, rank_num, output, name);
    }
    if (selected == "ring") {
        return this->all_reduce(input, rank, rank_num, output, name);
    }
    ERR(ModelError, "unknown selected all_reduce route: ", selected);
    return NullTensor;
}

Tensor Model::all_reduce_packet(Tensor input, int rank, int rank_num,
                                Tensor output, const std::string &) {
    require_all_reduce_packet_supported(input, rank, rank_num);

    size_t nelems = input.shape().nelems();
    size_t elems_per_uint32 = sizeof(uint32_t) / input.data_type().bytes();

    // Copy external input into an internal buffer so it resides in mscclpp
    // registered memory. Internal ARK tensors are already registered.
    auto input_info =
        BufferRegistry::get_instance().get(input.ref()->buffer()->id());
    if (input.is_external() || (input_info && input_info->is_external)) {
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
            "all_reduce_packet", input.ref(), output.ref(), rank, rank_num,
            flag, scratch.ref(), peer_scratch_refs)
        ->result_tensors()[0];
}

}  // namespace ark
