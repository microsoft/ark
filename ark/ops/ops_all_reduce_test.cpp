// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/executor.hpp"
#include "gpu/gpu.hpp"
#include "model/model_buffer.hpp"
#include "model/model_node.hpp"
#include "model/model_op.hpp"
#include "model/model_tensor.hpp"
#include "ops_test_common.hpp"

template <typename T, int NumGpus>
void baseline_all_reduce(std::vector<void *> &outputs,
                         const std::vector<ark::Dims> &output_shapes,
                         const std::vector<void *> &,
                         const std::vector<ark::Dims> &, int) {
    // Calculate sum from 1 to NumGpus.
    T expected = 0;
    for (int i = 1; i <= NumGpus; ++i) {
        expected += T(i);
    }

    T *out = static_cast<T *>(outputs[0]);
    for (ark::DimType i = 0; i < output_shapes[0].nelems(); ++i) {
        out[i] = expected;
    }
}

template <int NumGpus>
void test_all_reduce_internal(ark::DimType nelem) {
    for (int gpu_id = 0; gpu_id < NumGpus; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id, nelem]() {
            UNITTEST_SKIP(ark::unittest::get_gpu_count() < NumGpus);
            // Each GPU's data is equal to its GPU ID + 1.
            ark::Model m(gpu_id, NumGpus);
            ark::Tensor ones = m.tensor({nelem}, ark::FP16);
            ark::Tensor data = m.mul(ones, float(gpu_id + 1));
            ark::Tensor output = m.all_reduce(data, gpu_id, NumGpus);

            std::vector<ark::half_t> ones_vec(ones.shape().nelems(),
                                              ark::half_t(1.0f));
            auto result = ark::op_test(
                "all_reduce", m, {ones}, {output},
                baseline_all_reduce<ark::half_t, NumGpus>, {ones_vec.data()});
            UNITTEST_LOG(result);
            UNITTEST_EQ(result.max_diff[0], 0.0f);
            return ark::unittest::SUCCESS;
        });
    }
    ark::unittest::wait_all_processes();
}

ark::Tensor all_reduce_packet(ark::Model &m, ark::Tensor input, int rank,
                              int rank_num, int flag, ark::Tensor output) {
    int tag_send_reduce = m.unique_tag();
    int tag_output = m.unique_tag();
    if (output.is_null()) {
        output = m.tensor(input.shape(), input.data_type(), input.strides(),
                          input.offsets(), input.padded_shape());
    }
    std::vector<int> remote_ranks;
    for (int i = 0; i < rank_num; i++) {
        if (i != rank) {
            remote_ranks.push_back(i);
        }
    }
    // need to make sure input is contiguous, and we flatten the input tensor
    ark::Tensor reshaped_input = m.reshape(input, {input.shape().nelems()});
    ark::Tensor reshaped_output = m.reshape(output, {output.shape().nelems()});
    int nelems_per_rank = reshaped_input.shape().nelems() / rank_num;
    uint32_t nbytes_per_rank =
        nelems_per_rank * reshaped_input.data_type().bytes();
    std::vector<ark::Tensor> sharded_inputs =
        m.sharding(reshaped_input, 0, nelems_per_rank);
    std::vector<ark::Tensor> sharded_outputs =
        m.sharding(reshaped_output, 0, nelems_per_rank);
    int npeer = rank_num - 1;
    size_t scratch_off = flag % 2 == 0 ? 0 : nbytes_per_rank * npeer * 2;
    ark::Dims scratch_strides = {nbytes_per_rank * 2 * npeer * 2};
    for (int i = 0; i < rank_num; i++) {
        if (i != rank) {
            int off_index = i < rank ? rank - 1 : rank;
            ark::Tensor scratch_tensor = m.tensor(
                nbytes_per_rank * 2, ark::UINT8, scratch_strides,
                ark::Dims(scratch_off + nbytes_per_rank * off_index * 2),
                ark::Dims(nbytes_per_rank * 2), i);
            m.send_packet(sharded_inputs[i], i, tag_send_reduce, flag,
                          scratch_tensor);
        }
    }
    std::vector<ark::Tensor> deps;
    ark::Tensor scratch =
        m.tensor(nbytes_per_rank * 2 * npeer, ark::UINT8, scratch_strides,
                 scratch_off, nbytes_per_rank * 2 * npeer);
    std::vector<ark::Tensor> outputs;
    size_t out_off = flag % 2 == 0 ? 0 : nbytes_per_rank * 2;
    ark::Dims out_shape = {nbytes_per_rank * 2};
    ark::Dims out_strides = {nbytes_per_rank * 2 *
                             2};  // packet + double buffer
    for (int i = 0; i < rank_num; i++) {
        if (i != rank) {
            outputs.push_back(m.tensor(out_shape, ark::UINT8, out_strides,
                                       out_off, out_shape, i));
        }
    }
    deps.push_back(m.recv_reduce_send_packet(
        sharded_inputs[rank], remote_ranks, tag_send_reduce, tag_output, flag,
        sharded_outputs[rank], outputs, scratch));
    for (int i = 0; i < rank_num; i++) {
        if (i != rank) {
            ark::Tensor scratch_tensor =
                m.tensor(out_shape, ark::UINT8, out_strides, ark::Dims(out_off),
                         out_shape);
            deps.push_back(m.recv_packet(sharded_outputs[i], i, tag_output,
                                         flag, scratch_tensor));
        }
    }
    return m.identity(output, deps);
}

template <int NumGpus>
void test_all_reduce_packet_internal(ark::DimType nelem) {
    for (int gpu_id = 0; gpu_id < NumGpus; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id, nelem]() {
            UNITTEST_SKIP(ark::unittest::get_gpu_count() < NumGpus);
            // Each GPU's data is equal to its GPU ID + 1.
            ark::Model m(gpu_id, NumGpus);
            ark::Tensor ones = m.tensor({nelem}, ark::FP16);
            ark::Tensor data = m.mul(ones, float(gpu_id + 1));
            ark::Tensor output =
                all_reduce_packet(m, data, gpu_id, NumGpus, 1, data);

            std::vector<ark::half_t> ones_vec(ones.shape().nelems(),
                                              ark::half_t(1.0f));
            auto result = ark::op_test(
                "all_reduce_packet", m, {ones}, {output},
                baseline_all_reduce<ark::half_t, NumGpus>, {ones_vec.data()});
            UNITTEST_LOG(result);
            UNITTEST_EQ(result.max_diff[0], 0.0f);
            return ark::unittest::SUCCESS;
        });
    }
    ark::unittest::wait_all_processes();
}

ark::Tensor all_reduce_sm(ark::Model &m, ark::Tensor input, int rank,
                          int rank_num, ark::Tensor output) {
    int send_tag = m.unique_tag();
    int recv_tag = m.unique_tag();
    if (output.is_null()) {
        output = m.tensor(input.shape(), input.data_type(), input.strides(),
                          input.offsets(), input.padded_shape());
    }
    std::vector<int> remote_ranks;
    for (int i = 0; i < rank_num; i++) {
        if (i != rank) {
            remote_ranks.push_back(i);
        }
    }
    ark::Tensor reshaped_input = m.reshape(input, {input.shape().nelems()});
    ark::Tensor reshaped_output = m.reshape(output, {output.shape().nelems()});
    int nelems_per_rank = reshaped_input.shape().nelems() / rank_num;
    int npeer = rank_num - 1;
    ark::Tensor scratch_tensor =
        m.tensor(nelems_per_rank * npeer, reshaped_input.data_type());
    std::vector<ark::Tensor> sharded_inputs =
        m.sharding(reshaped_input, 0, nelems_per_rank);
    std::vector<ark::Tensor> sharded_scratch =
        m.sharding(scratch_tensor, 0, nelems_per_rank);
    std::vector<ark::Tensor> shared_outputs =
        m.sharding(reshaped_output, 0, nelems_per_rank);
    for (int i = 0; i < rank_num; i++) {
        if (i != rank) {
            int remote_off = i < rank ? rank - 1 : rank;
            ark::Tensor scratch =
                m.tensor(nelems_per_rank, reshaped_input.data_type(),
                         {nelems_per_rank * npeer},
                         ark::Dims(nelems_per_rank * remote_off),
                         ark::Dims(nelems_per_rank), i);
            m.send(sharded_inputs[i], i, send_tag, scratch);
        }
    }
    m.device_sync(reshaped_input, rank, rank_num);
    m.recv_reduce_send(sharded_inputs[rank], remote_ranks, send_tag, recv_tag,
                       sharded_inputs[rank]);
    for (int i = 0; i < rank_num; i++) {
        if (i != rank) {
            int peer_id = i < rank ? i : i - 1;
            m.recv(sharded_inputs[peer_id], i, recv_tag);
        }
    }
    ark::Tensor res = m.device_sync(input, rank, rank_num);
    return res;
}

template <int NumGpus>
void test_all_reduce_sm_internal(ark::DimType nelem) {
    auto config_rule = [nelem](const std::string op_str, const std::string) {
        const int tile_y = 64 /*nthreads per wrap*/ * 8 /*nelems per thread*/ *
                           8 /*num wraps*/;
        const int num_tasks = nelem / tile_y / NumGpus;
        auto op = nlohmann::json::parse(op_str);
        nlohmann::json config;
        if (op.at("Type") == "Send") {
            config["ChannelType"] = "Sm";
            config["Signal"] = false;
            config["Tile"] = {1, tile_y};
            config["NumTasks"] = num_tasks;
            config["NumWarps"] = 8;
            config["SramBytes"] = 0;
        } else if (op.at("Type") == "DeviceSync") {
            config["ChannelType"] = "Sm";
            config["NumTasks"] = 1;
            config["NumWarps"] = 1;
            config["SramBytes"] = 0;
        } else if (op.at("Type") == "Recv") {
            config["ChannelType"] = "Sm";
            config["NumTasks"] = 1;
            config["NumWarps"] = 1;
            config["SramBytes"] = 0;
            config["Wait"] = false;
        } else if (op.at("Type") == "RecvReduceSend") {
            config["NumTasks"] = num_tasks;
            config["NumWarps"] = 8;
            config["SramBytes"] = 0;
            config["Tile"] = {1, tile_y};
        }
        return config.dump();
    };
    for (int gpu_id = 0; gpu_id < NumGpus; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id, nelem, config_rule]() {
            UNITTEST_SKIP(ark::unittest::get_gpu_count() < NumGpus);
            // Each GPU's data is equal to its GPU ID + 1.
            ark::Model m(gpu_id, NumGpus);
            ark::Tensor ones = m.tensor({nelem}, ark::FP16);
            ark::Tensor data = m.mul(ones, float(gpu_id + 1));
            ark::Tensor output = all_reduce_sm(m, data, gpu_id, NumGpus, data);

            std::vector<ark::half_t> ones_vec(ones.shape().nelems(),
                                              ark::half_t(1.0f));
            auto result =
                ark::op_test("all_reduce_sm", m, {ones}, {output},
                             baseline_all_reduce<ark::half_t, NumGpus>,
                             {ones_vec.data()}, {config_rule});
            UNITTEST_LOG(result);
            UNITTEST_EQ(result.max_diff[0], 0.0f);
            return ark::unittest::SUCCESS;
        });
    }
    ark::unittest::wait_all_processes();
}

template <int NumGpus>
void test_all_reduce_inplace_internal(ark::DimType nelem) {
    for (int gpu_id = 0; gpu_id < NumGpus; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id, nelem]() {
            UNITTEST_SKIP(ark::unittest::get_gpu_count() < NumGpus);
            // Each GPU's data is equal to its GPU ID + 1.
            ark::Model m(gpu_id, NumGpus);
            ark::Tensor ones = m.tensor({nelem}, ark::FP16);
            ark::Tensor data = m.mul(ones, float(gpu_id + 1));
            // In-place: pass the same tensor as both input and output.
            ark::Tensor output = m.all_reduce(data, gpu_id, NumGpus, data);

            // Verify the output is truly in-place (same buffer as input).
            UNITTEST_EQ(output.ref()->buffer()->id(),
                        data.ref()->buffer()->id());

            std::vector<ark::half_t> ones_vec(ones.shape().nelems(),
                                              ark::half_t(1.0f));
            auto result = ark::op_test(
                "all_reduce_inplace", m, {ones}, {output},
                baseline_all_reduce<ark::half_t, NumGpus>, {ones_vec.data()});
            UNITTEST_LOG(result);
            UNITTEST_EQ(result.max_diff[0], 0.0f);
            return ark::unittest::SUCCESS;
        });
    }
    ark::unittest::wait_all_processes();
}

// The corruption was most visible with gpu_num >= 3 (multiple ring
// iterations); this 2-GPU case validates the in-place copy path still
// works for the simpler ring.
ark::unittest::State test_all_reduce_inplace_2gpus() {
    test_all_reduce_inplace_internal<2>(64);
    test_all_reduce_inplace_internal<2>(8192);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_all_reduce_inplace_3gpus() {
    test_all_reduce_inplace_internal<3>(64);
    test_all_reduce_inplace_internal<3>(8192);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_all_reduce_inplace_4gpus() {
    test_all_reduce_inplace_internal<4>(64);
    test_all_reduce_inplace_internal<4>(8192);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_all_reduce_4gpus() {
    test_all_reduce_internal<4>(64);
    test_all_reduce_internal<4>(8192);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_all_reduce_8gpus() {
    test_all_reduce_internal<8>(64);
    test_all_reduce_internal<8>(8192);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_all_reduce_packet_4gpus() {
    test_all_reduce_packet_internal<4>(2048);
    test_all_reduce_packet_internal<4>(8192);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_all_reduce_packet_8gpus() {
    test_all_reduce_packet_internal<8>(2048);
    test_all_reduce_packet_internal<8>(8192);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_all_reduce_sm_4gpus() {
    test_all_reduce_sm_internal<4>(2048 * 1024);
    test_all_reduce_sm_internal<4>(8192 * 1024);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_all_reduce_sm_8gpus() {
    test_all_reduce_sm_internal<8>(2048 * 1024);
    test_all_reduce_sm_internal<8>(8192 * 1024);
    return ark::unittest::SUCCESS;
}

template <int NumGpus>
void test_all_reduce_packet_fused_internal(ark::DimType nelem) {
    for (int gpu_id = 0; gpu_id < NumGpus; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id, nelem]() {
            UNITTEST_SKIP(ark::unittest::get_gpu_count() < NumGpus);
            // Each GPU's data is equal to its GPU ID + 1.
            ark::Model m(gpu_id, NumGpus);
            ark::Tensor ones = m.tensor({nelem}, ark::FP16);
            ark::Tensor data = m.mul(ones, float(gpu_id + 1));
            ark::Tensor output = m.all_reduce_packet(data, gpu_id, NumGpus);

            std::vector<ark::half_t> ones_vec(ones.shape().nelems(),
                                              ark::half_t(1.0f));
            auto result = ark::op_test(
                "all_reduce_packet_fused", m, {ones}, {output},
                baseline_all_reduce<ark::half_t, NumGpus>, {ones_vec.data()});
            UNITTEST_LOG(result);
            UNITTEST_EQ(result.max_diff[0], 0.0f);
            return ark::unittest::SUCCESS;
        });
    }
    ark::unittest::wait_all_processes();
}

// Variant with external-buffer (placeholder) input — exercises staging the
// external input into registered memory before the fused packet collective.
// Cannot use op_test() because placeholders require pre-allocated GPU memory;
// drive the executor manually instead.
template <int NumGpus>
void test_all_reduce_packet_fused_ext_internal(ark::DimType nelem) {
    for (int gpu_id = 0; gpu_id < NumGpus; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id, nelem]() {
            UNITTEST_SKIP(ark::unittest::get_gpu_count() < NumGpus);

            UNITTEST_EQ(ark::gpuSetDevice(gpu_id), ark::gpuSuccess);

            // Allocate GPU memory and fill with (gpu_id + 1).
            ark::half_t *d_input = nullptr;
            size_t nbytes = nelem * sizeof(ark::half_t);
            UNITTEST_EQ(ark::gpuMalloc(&d_input, nbytes), ark::gpuSuccess);
            std::vector<ark::half_t> h_input(nelem,
                                             ark::half_t(float(gpu_id + 1)));
            UNITTEST_EQ(ark::gpuMemcpy(d_input, h_input.data(), nbytes,
                                       ark::gpuMemcpyHostToDevice),
                        ark::gpuSuccess);

            ark::Model m(gpu_id, NumGpus);
            ark::Tensor input =
                m.placeholder({nelem}, ark::FP16, {}, {}, {}, -1, d_input);
            ark::Tensor output = m.all_reduce_packet(input, gpu_id, NumGpus);

            ark::DefaultExecutor exe(m, gpu_id);
            exe.launch();
            exe.run(1);
            exe.stop();

            std::vector<ark::half_t> h_output(nelem);
            exe.tensor_read(output, h_output);

            float expected = float(NumGpus * (NumGpus + 1)) / 2.0f;
            for (ark::DimType i = 0; i < nelem; ++i) {
                UNITTEST_EQ(float(h_output[i]), expected);
            }

            UNITTEST_EQ(ark::gpuFree(d_input), ark::gpuSuccess);
            return ark::unittest::SUCCESS;
        });
    }
    ark::unittest::wait_all_processes();
}

ark::unittest::State test_all_reduce_packet_fused_ext_2gpus() {
    test_all_reduce_packet_fused_ext_internal<2>(4096);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_all_reduce_packet_fused_2gpus() {
    test_all_reduce_packet_fused_internal<2>(4096);
    return ark::unittest::SUCCESS;
}

struct PrefillRouteCounts {
    int sm_send_count = 0;
    int sm_reduce_count = 0;
};

PrefillRouteCounts count_prefill_routes(ark::Model &model) {
    PrefillRouteCounts counts;
    for (auto &node : model.nodes()) {
        auto &op = node->op;
        if (op->is_virtual()) continue;
        auto cfg = op->default_config(ark::ARCH_CUDA_80);
        if (op->type() == ark::ModelOpT::from_name("Send") &&
            cfg.contains("ChannelType") && cfg.at("ChannelType") == "Sm") {
            ++counts.sm_send_count;
        }
        if (op->type() == ark::ModelOpT::from_name("RecvReduceSend") &&
            cfg.contains("NumWarps") && cfg.at("NumWarps").get<int>() == 8) {
            ++counts.sm_reduce_count;
        }
    }
    return counts;
}

ark::unittest::State test_all_reduce_size_dispatch_model() {
    {
        ark::Model model(0, 2);
        ark::Tensor tns = model.tensor({4096}, ark::FP16);
        model.all_reduce(tns, 0, 2);

        bool found_packet = false;
        for (auto &node : model.nodes()) {
            auto &op = node->op;
            if (op->is_virtual()) continue;
            found_packet |=
                op->type() == ark::ModelOpT::from_name("AllReducePacketFused");
        }
        UNITTEST_TRUE(found_packet);
    }
    {
        ark::Model model(0, 2);
        ark::Tensor tns = model.tensor({131072}, ark::FP16);
        model.all_reduce(tns, 0, 2);

        int packet_count = 0;
        int sm_send_count = 0;
        int sm_reduce_count = 0;
        int recv_nowait_count = 0;
        int proxy_device_sync_count = 0;
        for (auto &node : model.nodes()) {
            auto &op = node->op;
            if (op->is_virtual()) continue;
            if (op->type() ==
                ark::ModelOpT::from_name("AllReducePacketFused")) {
                ++packet_count;
            }
            auto cfg = op->default_config(ark::ARCH_CUDA_80);
            if (op->type() == ark::ModelOpT::from_name("Send") &&
                cfg.contains("ChannelType") && cfg.at("ChannelType") == "Sm") {
                ++sm_send_count;
            }
            if (op->type() == ark::ModelOpT::from_name("RecvReduceSend") &&
                cfg.at("NumWarps").get<int>() == 8) {
                ++sm_reduce_count;
            }
            if (op->type() == ark::ModelOpT::from_name("Recv") &&
                cfg.at("Wait") == false) {
                ++recv_nowait_count;
            }
            if (op->type() == ark::ModelOpT::from_name("DeviceSync") &&
                cfg.at("ChannelType") == "Proxy") {
                ++proxy_device_sync_count;
            }
        }
        UNITTEST_EQ(packet_count, 0);
        UNITTEST_EQ(sm_send_count, 1);
        UNITTEST_EQ(sm_reduce_count, 1);
        UNITTEST_EQ(recv_nowait_count, 1);
        UNITTEST_EQ(proxy_device_sync_count, 2);
    }
    {
        ark::Model model(0, 2);
        ark::Tensor tns = model.tensor({77824}, ark::FP16);
        model.all_reduce(tns, 0, 2);

        int packet_count = 0;
        int sm_send_count = 0;
        int sm_reduce_count = 0;
        for (auto &node : model.nodes()) {
            auto &op = node->op;
            if (op->is_virtual()) continue;
            if (op->type() ==
                ark::ModelOpT::from_name("AllReducePacketFused")) {
                ++packet_count;
            }
            auto cfg = op->default_config(ark::ARCH_CUDA_80);
            if (op->type() == ark::ModelOpT::from_name("Send") &&
                cfg.contains("ChannelType") && cfg.at("ChannelType") == "Sm") {
                ++sm_send_count;
            }
            if (op->type() == ark::ModelOpT::from_name("RecvReduceSend") &&
                cfg.contains("NumWarps") &&
                cfg.at("NumWarps").get<int>() == 8) {
                ++sm_reduce_count;
            }
        }
        UNITTEST_EQ(packet_count, 0);
        UNITTEST_EQ(sm_send_count, 0);
        UNITTEST_EQ(sm_reduce_count, 0);
    }
    {
        ark::Model model(0, 9);
        ark::Tensor tns = model.tensor({110592}, ark::FP16);
        model.all_reduce(tns, 0, 9);

        PrefillRouteCounts counts = count_prefill_routes(model);
        UNITTEST_EQ(counts.sm_send_count, 0);
        UNITTEST_EQ(counts.sm_reduce_count, 0);
    }
    {
        ark::Model model(0, 2);
        ark::Tensor base = model.tensor({129, 1024}, ark::FP16);
        ark::Tensor tns = model.refer(base, {128, 1024}, {129, 1024},
                                      {0, 0}, {128, 1024});
        model.all_reduce(tns, 0, 2);

        PrefillRouteCounts counts = count_prefill_routes(model);
        UNITTEST_EQ(counts.sm_send_count, 0);
        UNITTEST_EQ(counts.sm_reduce_count, 0);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_all_reduce_large_dispatch_2gpus() {
    test_all_reduce_internal<2>(131072);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_all_reduce_large_dispatch_8gpus() {
    test_all_reduce_internal<8>(131072);
    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_all_reduce_4gpus);
    UNITTEST(test_all_reduce_8gpus);
    UNITTEST(test_all_reduce_packet_4gpus);
    UNITTEST(test_all_reduce_packet_8gpus);
    UNITTEST(test_all_reduce_packet_fused_ext_2gpus);
    UNITTEST(test_all_reduce_packet_fused_2gpus);
    UNITTEST(test_all_reduce_size_dispatch_model);
    UNITTEST(test_all_reduce_large_dispatch_2gpus);
    UNITTEST(test_all_reduce_large_dispatch_8gpus);
    UNITTEST(test_all_reduce_sm_4gpus);
    UNITTEST(test_all_reduce_sm_8gpus);
    UNITTEST(test_all_reduce_inplace_2gpus);
    UNITTEST(test_all_reduce_inplace_3gpus);
    UNITTEST(test_all_reduce_inplace_4gpus);
    return 0;
}
