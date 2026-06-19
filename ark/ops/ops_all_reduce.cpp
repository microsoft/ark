// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "buffer_registry.hpp"
#include "ops_common.hpp"
#include "ops_communication.hpp"

namespace ark {

Tensor Model::all_reduce(Tensor input, int gpu_id, int gpu_num, Tensor output,
                         const std::string &) {
    auto is_registered_external = [](const Tensor &tensor) {
        if (tensor.is_null()) {
            return false;
        }
        auto info = BufferRegistry::get_instance().get(
            tensor.ref()->buffer()->id());
        return tensor.is_external() || (info && info->is_external);
    };

    auto ring_all_reduce = [&]() {
        if (is_registered_external(input)) {
            input = this->copy(input);
        }
        std::vector<int> tags(gpu_num);
        for (int i = 0; i < gpu_num; i++) {
            tags[i] = this->unique_tag();
        }
        if (output.is_null()) {
            output = this->copy(input);
        } else if (output.ref()->buffer()->id() == input.ref()->buffer()->id()) {
            // In-place: copy input so the ring loop does not mutate send data.
            // TODO: This catches the common case (output IS input). Sub-buffer
            // offset aliasing or aliasing through different buffer objects
            // backed by the same allocation is not currently detected.
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
    };

    if (gpu_num < 2) {
        return ring_all_reduce();
    }

    auto has_flattenable_layout = [](const Tensor &tensor) {
        Dims strides = tensor.strides();
        Dims padded_shape = tensor.padded_shape();
        Dims offsets = tensor.offsets();
        if (strides != padded_shape) {
            return false;
        }
        for (auto offset : offsets.vector()) {
            if (offset != 0) {
                return false;
            }
        }
        return true;
    };

    size_t nelems = input.shape().nelems();
    size_t elems_per_uint32 = sizeof(uint32_t) / input.data_type().bytes();
    size_t packet_alignment = elems_per_uint32 * 2 * gpu_num;
    constexpr size_t kLargeMessageThresholdBytes = 153600;
    bool packet_supported = packet_alignment != 0 &&
                            (nelems % packet_alignment) == 0 &&
                            !is_registered_external(output);
    if (input.ref()->shape_bytes() <= kLargeMessageThresholdBytes) {
        if (packet_supported) {
            if (!output.is_null() && output.ref()->buffer()->id() ==
                                         input.ref()->buffer()->id()) {
                input = this->copy(input);
            }
            return this->all_reduce_packet(input, gpu_id, gpu_num, output);
        }
        return ring_all_reduce();
    }

    // Large route size: shard evenly across ranks.
    if (nelems % gpu_num != 0) {
        return ring_all_reduce();
    }

    constexpr size_t kPrefillSmTileElems = 64 * 8 * 8;
    size_t nelems_per_rank = nelems / gpu_num;
    // Large route tile: each rank shard must fill SM reduce tiles.
    if ((nelems_per_rank % kPrefillSmTileElems) != 0) {
        return ring_all_reduce();
    }
    // Large route peers: RecvReduceSend kernels support at most 7 peers.
    if (gpu_num - 1 > kMaxRecvReduceSendPeers) {
        return ring_all_reduce();
    }

    if (is_registered_external(input)) {
        input = this->copy(input);
    }

    // Large path: scatter shards to peer scratch, synchronize, reduce this
    // rank's shard, write the reduced shard to peers, then synchronize before
    // exposing the gathered output.
    Tensor final_output = output;
    Tensor collective_output = output;
    if (collective_output.is_null() ||
        is_registered_external(collective_output)) {
        collective_output = this->tensor(input.shape(), input.data_type());
    }

    // Large route layout: tensors reshaped for sharding must be contiguous.
    if (!has_flattenable_layout(input) ||
        !has_flattenable_layout(collective_output)) {
        return ring_all_reduce();
    }

    Tensor reshaped_input = this->reshape(input, {static_cast<DimType>(nelems)});
    Tensor reshaped_output =
        this->reshape(collective_output, {static_cast<DimType>(nelems)});
    DimType nelems_per_rank_dim = static_cast<DimType>(nelems_per_rank);
    std::vector<Tensor> sharded_inputs =
        this->sharding(reshaped_input, 0, nelems_per_rank_dim);
    std::vector<Tensor> sharded_outputs =
        this->sharding(reshaped_output, 0, nelems_per_rank_dim);

    int send_tag = this->unique_tag();
    int output_tag = this->unique_tag();

    std::vector<int> remote_ranks;
    for (int i = 0; i < gpu_num; ++i) {
        if (i != gpu_id) {
            remote_ranks.push_back(i);
        }
    }

    int n_peers = gpu_num - 1;
    Tensor scratch = this->tensor({nelems_per_rank_dim * n_peers},
                                  reshaped_input.data_type());
    std::vector<Tensor> send_deps;
    ark::Dims scratch_strides = {nelems_per_rank_dim * n_peers};
    ark::Dims scratch_padded = {nelems_per_rank_dim};
    for (int dst = 0; dst < gpu_num; ++dst) {
        if (dst == gpu_id) continue;
        int remote_slot = dst < gpu_id ? gpu_id - 1 : gpu_id;
        Tensor remote_scratch = this->tensor(
            {nelems_per_rank_dim}, reshaped_input.data_type(), scratch_strides,
            ark::Dims(nelems_per_rank_dim * remote_slot), scratch_padded, dst);
        Tensor send = impl_
                          ->create_op<ModelOpAllReducePrefillSendSm>(
                              "all_reduce_prefill_send",
                              sharded_inputs[dst].ref(), dst, send_tag,
                              remote_scratch.ref())
                          ->result_tensors()[0];
        send_deps.push_back(send);
    }

    Tensor sends_done = this->identity(reshaped_input, send_deps);
    // Use the default Proxy DeviceSync as an inter-rank barrier independent of
    // the SM data movement in the large-message path.
    Tensor scatter_sync = this->device_sync(sends_done, gpu_id, gpu_num);
    Tensor local_input = this->identity(sharded_inputs[gpu_id], {scatter_sync});

    std::vector<ModelTensorRef> peer_output_refs;
    ark::Dims output_strides = {static_cast<DimType>(nelems)};
    ark::Dims output_padded = {nelems_per_rank_dim};
    for (int peer : remote_ranks) {
        Tensor peer_output = this->tensor(
            {nelems_per_rank_dim}, reshaped_output.data_type(), output_strides,
            ark::Dims(nelems_per_rank_dim * gpu_id), output_padded, peer);
        peer_output_refs.push_back(peer_output.ref());
    }

    Tensor local_reduced =
        impl_
            ->create_op<ModelOpAllReducePrefillRecvReduceSendSm>(
                "all_reduce_prefill_reduce_scatter", local_input.ref(),
                sharded_outputs[gpu_id].ref(), gpu_id, remote_ranks, send_tag,
                output_tag, peer_output_refs, scratch.ref())
            ->result_tensors()[0];

    std::vector<Tensor> recv_deps;
    recv_deps.push_back(local_reduced);
    for (int peer : remote_ranks) {
        Tensor recv = impl_
                          ->create_op<ModelOpAllReducePrefillRecvNoWait>(
                              "all_reduce_prefill_recv",
                              sharded_outputs[peer].ref(), peer, output_tag)
                          ->result_tensors()[0];
        recv_deps.push_back(recv);
    }

    Tensor gather_done = this->identity(local_reduced, recv_deps);
    Tensor gather_sync = this->device_sync(gather_done, gpu_id, gpu_num);
    Tensor result = this->identity(collective_output, {gather_sync});
    if (!final_output.is_null() && final_output.ref()->buffer()->id() !=
                                      collective_output.ref()->buffer()->id()) {
        result = this->copy(result, final_output);
    }
    return result;
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
