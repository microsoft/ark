// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model/model_buffer.hpp"
#include "model/model_node.hpp"
#include "model/model_op.hpp"
#include "ops_test_common.hpp"

ark::unittest::State test_all_reduce_model() {
    // OpNode graph (parentheses indicate a OpNode):
    //
    //               +--> (S,SD,R,) --+--> (S,SD,R,) --+
    //               |                |                |
    //   (S,SD,R,) --+--> (Add,)      +--> (Add,)      +--> (Add,)
    //                      |               ^  |              ^
    //                      |               |  |              |
    //                      +---------------+  +--------------+

    ark::Model model;
    ark::Tensor input = model.tensor({1}, ark::FP32);
    ark::Tensor output = model.all_reduce(input, 0, 4);

    UNITTEST_TRUE(model.verify());

    auto compressed = model.compress();
    auto nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 6);

    auto nodes_iter = nodes.begin();
    auto node = *(nodes_iter++);
    // UNITTEST_EQ(node->get_name(), "send;send_done;recv;");
    UNITTEST_EQ(node->producers.size(), 0);
    UNITTEST_EQ(node->consumers.size(), 2);

    // UNITTEST_EQ(node->consumers[0]->get_name(), "add;");
    UNITTEST_EQ(node->consumers[0]->consumers.size(), 1);
    // UNITTEST_EQ((*(node->consumers[0]->consumers.begin()))->get_name(),
    // "add_1;");

    // UNITTEST_EQ(node->consumers[1]->get_name(),
    // "send_1;send_done_1;recv_1;");
    UNITTEST_EQ(node->consumers[1]->producers.size(), 1);
    UNITTEST_EQ(node->consumers[1]->consumers.size(), 2);

    node = node->consumers[1];

    // UNITTEST_EQ(node->consumers[0]->get_name(), "add_1;");
    UNITTEST_EQ(node->consumers[0]->producers.size(), 2);
    UNITTEST_EQ(node->consumers[0]->consumers.size(), 1);
    // UNITTEST_EQ((*(node->consumers[0]->consumers.begin()))->get_name(),
    // "add_2;");

    // UNITTEST_EQ(node->consumers[1]->get_name(),
    // "send_2;send_done_2;recv_2;");
    UNITTEST_EQ(node->consumers[1]->producers.size(), 1);
    UNITTEST_EQ(node->consumers[1]->consumers.size(), 1);
    // UNITTEST_EQ((*(node->consumers[1]->consumers.begin()))->get_name(),
    // "add_2;");
    UNITTEST_EQ(
        (*(node->consumers[1]->consumers.begin()))->ops[0]->result_tensors()[0],
        output.ref());

    return ark::unittest::SUCCESS;
}

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
            // Each GPU's data is equal to its GPU ID + 1.
            ark::Model m(gpu_id, NumGpus);
            ark::Tensor ones = m.tensor({nelem}, ark::FP16);
            ark::Tensor data = m.mul(ones, float(gpu_id + 1));
            ark::Tensor output = m.all_reduce(data, gpu_id, NumGpus);

            std::vector<ark::half_t> ones_vec(ones.shape().nelems(),
                                              ark::half_t(1.0f));
            auto result =
                ark::op_test("all_reduce", m, {ones}, {output},
                             baseline_all_reduce<ark::half_t, NumGpus>,
                             {ones_vec.data()}, false, gpu_id, NumGpus);
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
                std::make_shared<ark::ModelBuffer>(i), nbytes_per_rank * 2,
                ark::UINT8, scratch_strides,
                ark::Dims(scratch_off + nbytes_per_rank * off_index * 2),
                ark::Dims(nbytes_per_rank * 2));
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
    ark::Dims out_strides = {nbytes_per_rank * 2 * 2}; // packet + double buffer
    for (int i = 0; i < rank_num; i++) {
        if (i != rank) {
            outputs.push_back(m.tensor(std::make_shared<ark::ModelBuffer>(i),
                                       out_shape, ark::UINT8, out_strides,
                                       out_off, out_shape));
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
            // Each GPU's data is equal to its GPU ID + 1.
            ark::Model m(gpu_id, NumGpus);
            ark::Tensor ones = m.tensor({nelem}, ark::FP16);
            ark::Tensor data = m.mul(ones, float(gpu_id + 1));
            ark::Tensor output = all_reduce_packet(m, data, gpu_id, NumGpus, 1, data);

            std::vector<ark::half_t> ones_vec(ones.shape().nelems(),
                                              ark::half_t(1.0f));
            auto result =
                ark::op_test("all_reduce_packet", m, {ones}, {output},
                             baseline_all_reduce<ark::half_t, NumGpus>,
                             {ones_vec.data()}, false, gpu_id, NumGpus);
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
                m.tensor(std::make_shared<ark::ModelBuffer>(i), nelems_per_rank,
                         reshaped_input.data_type(), {nelems_per_rank * npeer},
                         ark::Dims(nelems_per_rank * remote_off),
                         ark::Dims(nelems_per_rank));
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
            // Each GPU's data is equal to its GPU ID + 1.
            ark::Model m(gpu_id, NumGpus);
            ark::Tensor ones = m.tensor({nelem}, ark::FP16);
            ark::Tensor data = m.mul(ones, float(gpu_id + 1));
            ark::Tensor output = all_reduce_sm(m, data, gpu_id, NumGpus, data);

            std::vector<ark::half_t> ones_vec(ones.shape().nelems(),
                                              ark::half_t(1.0f));
            auto result = ark::op_test(
                "all_reduce_sm", m, {ones}, {output},
                baseline_all_reduce<ark::half_t, NumGpus>, {ones_vec.data()},
                false, gpu_id, NumGpus, config_rule);
            UNITTEST_LOG(result);
            UNITTEST_EQ(result.max_diff[0], 0.0f);
            return ark::unittest::SUCCESS;
        });
    }
    ark::unittest::wait_all_processes();
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

int main() {
    // UNITTEST(test_all_reduce_model);
    UNITTEST(test_all_reduce_4gpus);
    UNITTEST(test_all_reduce_8gpus);
    UNITTEST(test_all_reduce_packet_4gpus);
    UNITTEST(test_all_reduce_packet_8gpus);
    UNITTEST(test_all_reduce_sm_4gpus);
    UNITTEST(test_all_reduce_sm_8gpus);
    return 0;
}
