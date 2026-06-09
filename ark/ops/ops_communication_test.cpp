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
            // default_config does not include NumProcs; stamp it manually
            // (the planner does this at plan time).
            cfg["NumProcs"] = cfg["NumTasks"].get<int>();
            auto name = op->impl_name(cfg);
            UNITTEST_FALSE(name.empty());
            // Verify the kernel name appears in the impl_name string.
            UNITTEST_TRUE(name.find("allreduce_packet_fused") !=
                          std::string::npos);
            auto args = op->impl_args(cfg);
            // (output, input, scratch_ptr, scratch_offset_remote)
            UNITTEST_EQ(args.size(), 4);
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
