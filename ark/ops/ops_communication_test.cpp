// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nlohmann/json.hpp>
#include <numeric>

#include "ark/executor.hpp"
#include "ark/planner.hpp"
#include "half.h"
#include "model/model_buffer.hpp"
#include "model/model_node.hpp"
#include "model/model_op.hpp"
#include "model/model_tensor.hpp"
#include "ops_test_common.hpp"

ark::unittest::State test_communication_host_ops() {
    // Host-only test: construct communication ops via Model API and exercise
    // default_config / impl_name / impl_args without a GPU.
    {
        // Send + SendDone + Recv
        ark::Model model(0, 2);
        ark::Tensor tns = model.tensor({1024}, ark::FP16);
        ark::Tensor out = model.send(tns, 1, 0);
        model.send_done(out);
        model.recv(model.tensor({1024}, ark::FP16), 1, 0);

        auto nodes = model.nodes();
        // Walk every node's op and call the three coverage-critical methods.
        for (auto &node : nodes) {
            auto &op = node->op;
            if (op->is_virtual()) continue;
            auto cfg = op->default_config(ark::ARCH_CUDA_80);
            UNITTEST_FALSE(cfg.empty());
            auto name = op->impl_name(cfg);
            UNITTEST_FALSE(name.empty());
            // impl_args may legitimately return an empty vector.
            (void)op->impl_args(cfg);
        }
    }
    {
        // SendPacket + RecvPacket
        ark::Model model(0, 2);
        ark::Tensor tns = model.tensor({1024}, ark::FP16);
        model.send_packet(tns, 1, 0, 1);
        model.recv_packet(model.tensor({1024}, ark::FP16), 1, 0, 1);

        auto nodes = model.nodes();
        for (auto &node : nodes) {
            auto &op = node->op;
            if (op->is_virtual()) continue;
            auto cfg = op->default_config(ark::ARCH_CUDA_80);
            UNITTEST_FALSE(cfg.empty());
            auto name = op->impl_name(cfg);
            UNITTEST_FALSE(name.empty());
            (void)op->impl_args(cfg);
        }
    }
    {
        // RecvReduceSendPacket
        ark::Model model(0, 2);
        ark::Tensor tns = model.tensor({1024}, ark::FP16);
        std::vector<ark::Tensor> shards = model.sharding(tns, 0, 512);
        model.send_packet(shards[1], 1, 0, 1);
        model.recv_reduce_send_packet(shards[0], {1}, 0, 1, 1, shards[0]);

        auto nodes = model.nodes();
        for (auto &node : nodes) {
            auto &op = node->op;
            if (op->is_virtual()) continue;
            auto cfg = op->default_config(ark::ARCH_CUDA_80);
            UNITTEST_FALSE(cfg.empty());
            auto name = op->impl_name(cfg);
            UNITTEST_FALSE(name.empty());
            (void)op->impl_args(cfg);
        }
    }
    {
        // DeviceSync
        ark::Model model(0, 2);
        ark::Tensor tns = model.tensor({1024}, ark::FP16);
        model.device_sync(tns, 0, 2);

        auto nodes = model.nodes();
        for (auto &node : nodes) {
            auto &op = node->op;
            if (op->is_virtual()) continue;
            auto cfg = op->default_config(ark::ARCH_CUDA_80);
            UNITTEST_FALSE(cfg.empty());
            auto name = op->impl_name(cfg);
            UNITTEST_FALSE(name.empty());
            (void)op->impl_args(cfg);
        }
    }
    {
        // RecvReduceSend
        ark::Model model(0, 2);
        ark::Tensor tns = model.tensor({1024}, ark::FP16);
        std::vector<ark::Tensor> shards = model.sharding(tns, 0, 512);
        ark::Tensor remote_scratch =
            model.tensor({512}, ark::FP16, {}, {}, {}, 1);
        ark::Tensor sent = model.send(shards[1], 1, 0, remote_scratch);
        ark::Tensor synced = model.device_sync(sent, 0, 2);
        ark::Tensor reduced = model.identity(shards[0], {synced});
        model.recv_reduce_send(reduced, {1}, 0, 1, reduced);

        auto nodes = model.nodes();
        for (auto &node : nodes) {
            auto &op = node->op;
            if (op->is_virtual()) continue;
            auto cfg = op->default_config(ark::ARCH_CUDA_80);
            UNITTEST_FALSE(cfg.empty());
            auto name = op->impl_name(cfg);
            UNITTEST_FALSE(name.empty());
            (void)op->impl_args(cfg);
        }
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_communication_send_recv_unidir() {
    // send from gpu 0 to gpu 1
    for (int gpu_id = 0; gpu_id < 2; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id]() {
            ark::Model model(gpu_id, 2);
            ark::Tensor tns = model.tensor({1024}, ark::FP16);
            if (gpu_id == 0) {
                ark::Tensor out_tns = model.send(tns, 1, 0);
                model.send_done(out_tns);
            }
            if (gpu_id == 1) {
                tns = model.recv(tns, 0, 0);
            }

            ark::DefaultExecutor exe(model, gpu_id);

            if (gpu_id == 0) {
                std::vector<ark::half_t> data(1024);
                std::iota(data.begin(), data.end(), 1.0f);
                exe.tensor_write(tns, data);
            }

            exe.barrier();

            exe.launch();
            exe.run(1);
            exe.stop();

            exe.barrier();

            if (gpu_id == 1) {
                std::vector<ark::half_t> data(1024);
                exe.tensor_read(tns, data);
                for (int i = 0; i < 1024; ++i) {
                    UNITTEST_EQ(data[i], ark::half_t(i + 1));
                }
            }
            return ark::unittest::SUCCESS;
        });
    }

    ark::unittest::wait_all_processes();

    // send from gpu 1 to gpu 0
    for (int gpu_id = 0; gpu_id < 2; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id]() {
            ark::Model model(gpu_id, 2);
            ark::Tensor tns = model.tensor({1024}, ark::FP16);
            if (gpu_id == 1) {
                auto out_tns = model.send(tns, 0, 0);
                model.send_done(out_tns);
            }
            if (gpu_id == 0) {
                tns = model.recv(tns, 1, 0);
            }

            ark::DefaultExecutor exe(model, gpu_id);

            if (gpu_id == 1) {
                std::vector<ark::half_t> data(1024);
                std::iota(data.begin(), data.end(), 1.0f);
                exe.tensor_write(tns, data);
            }

            exe.barrier();

            exe.launch();
            exe.run(1);
            exe.stop();

            exe.barrier();

            if (gpu_id == 0) {
                std::vector<ark::half_t> data(1024);
                exe.tensor_read(tns, data);
                for (int i = 0; i < 1024; ++i) {
                    UNITTEST_EQ(data[i], ark::half_t(i + 1));
                }
            }
            return ark::unittest::SUCCESS;
        });
    }

    ark::unittest::wait_all_processes();

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_communication_send_recv_bidir() {
    for (int gpu_id = 0; gpu_id < 2; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id]() {
            int remote_gpu_id = (gpu_id + 1) % 2;
            int tag = 0;

            ark::Model model(gpu_id, 2);
            ark::Tensor tns_data = model.tensor({1024}, ark::FP16);
            ark::Tensor tns = model.send(tns_data, remote_gpu_id, tag);
            tns = model.send_done(tns);

            ark::Tensor tns2_data = model.tensor({1024}, ark::FP16);
            // build dependency (send_done --> recv)
            ark::Tensor tns2 = model.identity(tns2_data, {tns});
            tns2 = model.recv(tns2_data, remote_gpu_id, tag);

            ark::DefaultExecutor exe(model, gpu_id);

            std::vector<ark::half_t> data(1024);
            std::iota(data.begin(), data.end(), ark::half_t(gpu_id + 1));
            exe.tensor_write(tns_data, data);

            exe.barrier();

            exe.launch();
            exe.run(1);
            exe.stop();

            exe.barrier();

            data.clear();
            data.resize(1024);
            exe.tensor_read(tns2_data, data);
            for (int i = 0; i < 1024; ++i) {
                UNITTEST_EQ(data[i], ark::half_t(remote_gpu_id + i + 1));
            }
            return ark::unittest::SUCCESS;
        });
    }

    ark::unittest::wait_all_processes();

    for (int gpu_id = 0; gpu_id < 2; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id]() {
            int remote_gpu_id = (gpu_id + 1) % 2;
            int tag = 0;

            ark::Model model(gpu_id, 2);
            ark::Tensor tns_data = model.tensor({1024}, ark::FP16);
            ark::Tensor tns = model.send(tns_data, remote_gpu_id, tag);
            tns = model.send_done(tns);

            ark::Tensor tns2_data = model.tensor({1024}, ark::FP16);
            // build dependency (send_done --> recv)
            ark::Tensor tns2 = model.identity(tns2_data, {tns});
            tns2 = model.recv(tns2_data, remote_gpu_id, tag);

            ark::Tensor sum = model.add(tns2, tns_data);

            ark::DefaultExecutor exe(model, gpu_id);

            std::vector<ark::half_t> data(1024);
            std::iota(data.begin(), data.end(), ark::half_t(gpu_id + 1));
            exe.tensor_write(tns_data, data);

            exe.barrier();

            exe.launch();
            exe.run(1);
            exe.stop();

            exe.barrier();

            data.clear();
            data.resize(1024);
            exe.tensor_read(sum, data);
            for (int i = 0; i < 1024; ++i) {
                UNITTEST_EQ(data[i],
                            ark::half_t(gpu_id + remote_gpu_id + 2 * i + 2));
            }
            return ark::unittest::SUCCESS;
        });
    }

    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_communication_send_recv_bidir_sm() {
    auto config_rule = [](const std::string op_str, const std::string) {
        auto op = nlohmann::json::parse(op_str);
        nlohmann::json config;
        if (op.at("Type") == "Send") {
            config["ChannelType"] = "Sm";
            config["Signal"] = true;
            config["Tile"] = {1, 256};
            config["NumTasks"] = 4;
            config["NumWarps"] = 4;
            config["SramBytes"] = 0;
        } else if (op.at("Type") == "SendDone") {
            config["ChannelType"] = "Sm";
            config["NumTasks"] = 1;
            config["NumWarps"] = 1;
            config["SramBytes"] = 0;
        } else if (op.at("Type") == "Recv") {
            config["ChannelType"] = "Sm";
            config["NumTasks"] = 1;
            config["NumWarps"] = 1;
            config["SramBytes"] = 0;
            config["Wait"] = true;
        }
        return config.dump();
    };

    for (int gpu_id = 0; gpu_id < 2; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id, config_rule]() {
            int remote_gpu_id = (gpu_id + 1) % 2;
            int tag = 0;

            ark::Model model(gpu_id, 2);
            ark::Tensor tns_data = model.tensor({1024}, ark::FP16);
            ark::Tensor tns = model.send(tns_data, remote_gpu_id, tag);
            tns = model.send_done(tns);

            ark::Tensor tns2_data = model.tensor({1024}, ark::FP16);
            // build dependency (send_done --> recv)
            ark::Tensor tns2 = model.identity(tns2_data, {tns});
            tns2 = model.recv(tns2_data, remote_gpu_id, tag);

            ark::DefaultExecutor exe(model, gpu_id, nullptr, {config_rule});

            std::vector<ark::half_t> data(1024);
            std::iota(data.begin(), data.end(), ark::half_t(gpu_id + 1));
            exe.tensor_write(tns_data, data);

            exe.barrier();

            exe.launch();
            exe.run(1);
            exe.stop();

            exe.barrier();

            data.clear();
            data.resize(1024);
            exe.tensor_read(tns2_data, data);
            for (int i = 0; i < 1024; ++i) {
                UNITTEST_EQ(data[i], ark::half_t(remote_gpu_id + i + 1));
            }
            return ark::unittest::SUCCESS;
        });
    }

    ark::unittest::wait_all_processes();

    for (int gpu_id = 0; gpu_id < 2; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id, config_rule]() {
            int remote_gpu_id = (gpu_id + 1) % 2;
            int tag = 0;

            ark::Model model(gpu_id, 2);
            ark::Tensor tns_data = model.tensor({1024}, ark::FP16);
            ark::Tensor tns = model.send(tns_data, remote_gpu_id, tag);
            tns = model.send_done(tns);

            ark::Tensor tns2_data = model.tensor({1024}, ark::FP16);
            // build dependency (send_done --> recv)
            ark::Tensor tns2 = model.identity(tns2_data, {tns});
            tns2 = model.recv(tns2_data, remote_gpu_id, tag);

            ark::Tensor sum = model.add(tns2, tns_data);

            ark::DefaultExecutor exe(model, gpu_id, nullptr, {config_rule});

            std::vector<ark::half_t> data(1024);
            std::iota(data.begin(), data.end(), ark::half_t(gpu_id + 1));
            exe.tensor_write(tns_data, data);

            exe.barrier();

            exe.launch();
            exe.run(1);
            exe.stop();

            exe.barrier();

            data.clear();
            data.resize(1024);
            exe.tensor_read(sum, data);
            for (int i = 0; i < 1024; ++i) {
                UNITTEST_EQ(data[i],
                            ark::half_t(gpu_id + remote_gpu_id + 2 * i + 2));
            }
            return ark::unittest::SUCCESS;
        });
    }

    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_communication_send_packet() {
    // send from gpu 0 to gpu 1
    for (int gpu_id = 0; gpu_id < 2; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id]() {
            ark::Model model(gpu_id, 2);
            ark::Tensor tns_data = model.tensor({1024}, ark::FP16);
            if (gpu_id == 0) {
                model.send_packet(tns_data, 1, 0, 1);
            }
            if (gpu_id == 1) {
                tns_data = model.recv_packet(tns_data, 0, 0, 1);
            }

            ark::DefaultExecutor exe(model, gpu_id);

            if (gpu_id == 0) {
                std::vector<ark::half_t> data(1024);
                std::iota(data.begin(), data.end(), 1.0f);
                exe.tensor_write(tns_data, data);
            }

            exe.barrier();
            exe.launch();
            exe.run(1);
            exe.stop();
            exe.barrier();

            if (gpu_id == 1) {
                std::vector<ark::half_t> data(1024);
                exe.tensor_read(tns_data, data);
                for (int i = 0; i < 1024; ++i) {
                    UNITTEST_EQ(data[i], ark::half_t(i + 1));
                }
            }
            return ark::unittest::SUCCESS;
        });
    }

    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_communication_send_recv_reduce_packet() {
    for (int gpu_id = 0; gpu_id < 2; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id]() {
            ark::Model model(gpu_id, 2);
            ark::Tensor tns_data = model.tensor({1024}, ark::FP16);
            std::vector<ark::Tensor> shard_tensors =
                model.sharding(tns_data, 0, 512);

            int peer_gpu_id = (gpu_id + 1) % 2;
            model.send_packet(shard_tensors[peer_gpu_id], peer_gpu_id, 0, 1);
            model.recv_reduce_send_packet(shard_tensors[gpu_id], {peer_gpu_id},
                                          0, 1, 1, shard_tensors[gpu_id]);
            model.recv_packet(shard_tensors[peer_gpu_id], peer_gpu_id, 1, 1);

            ark::DefaultExecutor exe(model, gpu_id);

            std::vector<ark::half_t> data(1024);
            std::iota(data.begin(), data.end(), 1.0f);
            exe.tensor_write(tns_data, data);

            exe.barrier();
            exe.launch();
            exe.run(1);
            exe.stop();
            exe.barrier();

            exe.tensor_read(tns_data, data);
            for (int i = 0; i < 1024; ++i) {
                UNITTEST_EQ(data[i], ark::half_t((i + 1) * 2));
            }
            return ark::unittest::SUCCESS;
        });
    }

    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_communication_send_recv_reduce() {
    auto config_rule = [](const std::string op_str, const std::string) {
        auto op = nlohmann::json::parse(op_str);
        nlohmann::json config;
        if (op.at("Type") == "Send") {
            constexpr int tile_y = 256;
            const auto &shape = op.at("WriteTensors")[0].at("PaddedShape");
            size_t num_tasks = 1;
            for (const auto &dim : shape) {
                num_tasks *= dim.get<size_t>();
            }
            num_tasks = (num_tasks + tile_y - 1) / tile_y;
            config["ChannelType"] = "Sm";
            config["Signal"] = false;
            config["Tile"] = {1, tile_y};
            config["NumTasks"] = num_tasks;
            config["NumWarps"] = 4;
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
        }
        return config.dump();
    };
    for (int gpu_id = 0; gpu_id < 2; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id, config_rule]() {
            ark::Model model(gpu_id, 2);
            ark::Tensor tns_data = model.tensor({1024}, ark::FP16);
            std::vector<ark::Tensor> shard_tensors =
                model.sharding(tns_data, 0, 512);

            int peer_gpu_id = (gpu_id + 1) % 2;
            ark::Tensor remote_scratch =
                model.tensor({512}, ark::FP16, {}, {}, {}, peer_gpu_id);
            ark::Tensor out = model.send(shard_tensors[peer_gpu_id],
                                         peer_gpu_id, 0, remote_scratch);
            out = model.device_sync(out, gpu_id, 2);
            ark::Tensor reduced = model.identity(shard_tensors[gpu_id], {out});
            reduced =
                model.recv_reduce_send(reduced, {peer_gpu_id}, 0, 1, reduced);
            model.recv(shard_tensors[peer_gpu_id], peer_gpu_id, 1);
            model.device_sync(reduced, gpu_id, 2);

            ark::Planner planner(model, gpu_id);
            planner.install_config_rule(config_rule);
            ark::Executor exe;
            exe.compile(planner.plan(), gpu_id);

            std::vector<ark::half_t> data(1024);
            std::iota(data.begin(), data.end(), 1.0f);
            exe.tensor_write(tns_data, data);

            exe.barrier();
            exe.launch();
            exe.run(1);
            exe.stop();
            exe.barrier();

            exe.tensor_read(tns_data, data);
            if (gpu_id == 1) {
                for (int i = 0; i < 1024; ++i) {
                    UNITTEST_EQ(data[i], ark::half_t((i + 1) * 2));
                }
            }
            return ark::unittest::SUCCESS;
        });
    }

    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_communication_allreduce_packet_fused_model() {
    auto count_ops = [](ark::Model &model, const std::string &type) {
        int count = 0;
        for (auto &node : model.nodes()) {
            auto &op = node->op;
            if (!op->is_virtual() &&
                op->type() == ark::ModelOpT::from_name(type)) {
                ++count;
            }
        }
        return count;
    };

    // Route selection is observable without running the graph.
    {
        ark::Model model(0, 8);
        ark::Tensor tns = model.tensor({4096}, ark::FP16);
        UNITTEST_EQ(model.all_reduce_route(tns, 0, 8), "packet");
        ark::Tensor result = model.all_reduce_routed(tns, 0, 8);
        UNITTEST_EQ(count_ops(model, "AllReducePacketFused"), 1);
    }
    {
        ark::Model model(0, 8);
        ark::Tensor tns = model.tensor({4096}, ark::FP16);
        ark::Tensor output = model.tensor(tns.shape(), tns.data_type());
        UNITTEST_EQ(model.all_reduce_route(tns, 0, 8), "packet");
        ark::Tensor result = model.all_reduce_routed(tns, 0, 8, output);
        // ARK ops return a versioned tensor ref. The explicit output tensor is
        // correct when the returned version writes the same backing buffer.
        UNITTEST_EQ(result.ref()->buffer()->id(), output.ref()->buffer()->id());
        UNITTEST_EQ(result.shape(), output.shape());
        UNITTEST_EQ(count_ops(model, "AllReducePacketFused"), 1);
    }
    {
        ark::Model model(0, 8);
        ark::Tensor tns = model.tensor({2048 * 4096}, ark::FP16);
        UNITTEST_EQ(model.all_reduce_route(tns, 0, 8), "ring");
        UNITTEST_EQ(model.all_reduce_route(tns, 0, 8, "ring"), "ring");
        ark::Tensor result = model.all_reduce_routed(tns, 0, 8);
        UNITTEST_EQ(count_ops(model, "AllReducePacketFused"), 0);
        UNITTEST_TRUE(count_ops(model, "Send") > 0);
    }
    {
        ark::Model model(0, 8);
        ark::Tensor tns = model.tensor({2048 * 4096}, ark::FP16);
        ark::Tensor output = model.tensor(tns.shape(), tns.data_type());
        ark::Tensor result = model.all_reduce_routed(tns, 0, 8, output);
        UNITTEST_EQ(result.ref()->buffer()->id(), output.ref()->buffer()->id());
        UNITTEST_EQ(result.shape(), output.shape());
        UNITTEST_EQ(count_ops(model, "AllReducePacketFused"), 0);
        UNITTEST_TRUE(count_ops(model, "Send") > 0);
        UNITTEST_EQ(count_ops(model, "Copy"), 1);
    }
    {
        ark::Model model(0, 2);
        ark::Tensor tns = model.tensor({1023}, ark::FP16);
        UNITTEST_EQ(model.all_reduce_route(tns, 0, 2), "ring");
        UNITTEST_THROW(model.all_reduce_route(tns, 0, 2, "packet"),
                       ark::ModelError);
        UNITTEST_THROW(model.all_reduce_route(tns, 0, 2, "unknown"),
                       ark::ModelError);
    }
    {
        ark::Model model(0, 2);
        ark::Tensor base = model.tensor({65, 64}, ark::FP16);
        ark::Tensor tns =
            model.refer(base, {64, 64}, {65, 64}, {0, 0}, {64, 64});
        UNITTEST_EQ(model.all_reduce_route(tns, 0, 2), "ring");
        UNITTEST_THROW(model.all_reduce_route(tns, 0, 2, "packet"),
                       ark::ModelError);
    }
    {
        ark::Model model(0, 2);
        ark::Tensor base = model.tensor({64, 64}, ark::FP16);
        ark::Tensor tns =
            model.refer(base, {63, 64}, {64, 64}, {1, 0}, {63, 64});
        UNITTEST_EQ(model.all_reduce_route(tns, 0, 2), "ring");
        UNITTEST_THROW(model.all_reduce_route(tns, 0, 2, "packet"),
                       ark::ModelError);
    }
    {
        ark::Model model(0, 2);
        ark::Tensor tns = model.tensor({1024}, ark::FP16);
        UNITTEST_THROW(model.all_reduce_route(tns, -1, 2), ark::ModelError);
        UNITTEST_THROW(model.all_reduce_route(tns, 2, 2), ark::ModelError);
        UNITTEST_THROW(model.all_reduce_routed(tns, -1, 2), ark::ModelError);
        UNITTEST_THROW(model.all_reduce_routed(tns, 2, 2), ark::ModelError);
        UNITTEST_THROW(model.all_reduce_packet(tns, -1, 2), ark::ModelError);
        UNITTEST_THROW(model.all_reduce_packet(tns, 2, 2), ark::ModelError);
    }

    // Single-GPU model-level test: construct the fused allreduce op and
    // verify impl_name / impl_args / default_config produce valid output.
    {
        ark::Model model(0, 2);
        ark::Tensor tns = model.tensor({1024}, ark::FP16);
        ark::Tensor result = model.all_reduce_packet(tns, 0, 2);

        auto nodes = model.nodes();
        bool found = false;
        for (auto &node : nodes) {
            auto &op = node->op;
            if (op->is_virtual()) continue;
            if (op->type() != ark::ModelOpT::from_name("AllReducePacketFused"))
                continue;
            found = true;
            auto cfg = op->default_config(ark::ARCH_CUDA_80);
            UNITTEST_FALSE(cfg.empty());
            auto name = op->impl_name(cfg);
            UNITTEST_FALSE(name.empty());
            // Verify the kernel name appears in the impl_name string.
            UNITTEST_TRUE(name.find("allreduce_packet_fused") !=
                          std::string::npos);
            auto args = op->impl_args(cfg);
            // (output, input, scratch_ptr, scratch_offset_remote, input_offset)
            UNITTEST_EQ(args.size(), 5);
        }
        UNITTEST_TRUE(found);
    }
    // Medium-size tensor (40000 bytes, 32KB <= x <= 153KB):
    // exercises the blocks_per_peer=8, num_warps=16 config path.
    {
        // 20000 FP16 = 40000 bytes => 32KB < 40KB <= 153KB.
        ark::Model model(0, 2);
        ark::Tensor tns = model.tensor({20000}, ark::FP16);
        ark::Tensor result = model.all_reduce_packet(tns, 0, 2);

        auto nodes = model.nodes();
        for (auto &node : nodes) {
            auto &op = node->op;
            if (op->is_virtual()) continue;
            if (op->type() != ark::ModelOpT::from_name("AllReducePacketFused"))
                continue;
            auto cfg = op->default_config(ark::ARCH_CUDA_80);
            UNITTEST_EQ(cfg.at("NumWarps").get<int>(), 16);
            UNITTEST_EQ(cfg.at("NumTasks").get<int>(), 8);
        }
    }
    // Large tensor (> 153600 bytes):
    // exercises blocks_per_peer=8, num_warps=32 config path.
    {
        // 80000 FP16 = 160000 bytes > 153600.
        ark::Model model(0, 2);
        ark::Tensor tns = model.tensor({80000}, ark::FP16);
        ark::Tensor result = model.all_reduce_packet(tns, 0, 2);

        auto nodes = model.nodes();
        for (auto &node : nodes) {
            auto &op = node->op;
            if (op->is_virtual()) continue;
            if (op->type() != ark::ModelOpT::from_name("AllReducePacketFused"))
                continue;
            auto cfg = op->default_config(ark::ARCH_CUDA_80);
            UNITTEST_EQ(cfg.at("NumWarps").get<int>(), 32);
            UNITTEST_EQ(cfg.at("NumTasks").get<int>(), 8);
        }
    }
    // Test Planner::plan() with AllReducePacketFused to exercise the
    // NumProcs override branch in planner.cpp.
    {
        ark::Model model(0, 2);
        ark::Tensor tns = model.tensor({1024}, ark::FP16);
        ark::Tensor result = model.all_reduce_packet(tns, 0, 2);

        ark::Planner planner(model, 0);
        auto plan = ark::Json::parse(planner.plan(false));
        // The plan should contain at least one TaskInfo for the fused op.
        UNITTEST_TRUE(plan.contains("TaskInfos"));
        bool found_fused = false;
        for (auto &ti : plan["TaskInfos"]) {
            for (auto &op : ti["Ops"]) {
                if (op.at("Type").get<std::string>() ==
                    "AllReducePacketFused") {
                    found_fused = true;
                    // NumProcs should have been set by the planner.
                    UNITTEST_TRUE(op["Config"].contains("NumProcs"));
                }
            }
        }
        UNITTEST_TRUE(found_fused);
    }
    // Verify rank_num < 2 is rejected.
    {
        ark::Model model(0, 1);
        ark::Tensor tns = model.tensor({1024}, ark::FP16);
        UNITTEST_THROW(model.all_reduce_packet(tns, 0, 1), ark::ModelError);
    }
    // Verify non-divisible tensor size is rejected.
    {
        ark::Model model(0, 2);
        ark::Tensor tns = model.tensor({1023}, ark::FP16);
        UNITTEST_THROW(model.all_reduce_packet(tns, 0, 2), ark::ModelError);
    }
    // Multi-peer model test: rank_num=4, rank=2 exercises peer-index mapping.
    {
        ark::Model model(2, 4);
        ark::Tensor tns = model.tensor({1024}, ark::FP16);
        ark::Tensor result = model.all_reduce_packet(tns, 2, 4);

        auto nodes = model.nodes();
        bool found = false;
        for (auto &node : nodes) {
            auto &op = node->op;
            if (op->is_virtual()) continue;
            if (op->type() != ark::ModelOpT::from_name("AllReducePacketFused"))
                continue;
            found = true;
            auto cfg = op->default_config(ark::ARCH_CUDA_80);
            // 1024 FP16 = 2048 bytes < 32KB → blocks_per_peer=4, n_peers=3
            UNITTEST_EQ(cfg.at("NumTasks").get<int>(), 12);
            UNITTEST_EQ(cfg.at("NumWarps").get<int>(), 32);
            auto name = op->impl_name(cfg);
            UNITTEST_TRUE(name.find("allreduce_packet_fused") !=
                          std::string::npos);
        }
        UNITTEST_TRUE(found);
    }
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_communication_host_ops);
    UNITTEST(test_communication_send_recv_unidir);
    UNITTEST(test_communication_send_recv_bidir);
    UNITTEST(test_communication_send_recv_bidir_sm);
    UNITTEST(test_communication_send_packet);
    UNITTEST(test_communication_send_recv_reduce_packet);
    UNITTEST(test_communication_send_recv_reduce);
    UNITTEST(test_communication_allreduce_packet_fused_model);
    return ark::unittest::SUCCESS;
}
