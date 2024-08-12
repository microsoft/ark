// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nlohmann/json.hpp>
#include <numeric>

#include "ark/executor.hpp"
#include "ark/planner.hpp"
#include "half.h"
#include "model/model_buffer.hpp"
#include "ops_test_common.hpp"

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
                tns = model.send(tns, 0, 0);
                model.send_done(tns);
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
            std::vector<ark::Tensor> shard_tensors = model.sharding(tns_data, 0, 512);

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
            config["ChannelType"] = "Sm";
            config["Signal"] = false;
            config["Tile"] = {1, 256};
            config["NumTasks"] = 4;
            config["NumWarps"] = 4;
            config["SramBytes"] = 0;
        }
        else if (op.at("Type") == "DeviceSync") {
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
            exe.compile(gpu_id, planner.plan());

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

int main() {
    ark::init();
    UNITTEST(test_communication_send_recv_unidir);
    UNITTEST(test_communication_send_recv_bidir);
    UNITTEST(test_communication_send_recv_bidir_sm);
    UNITTEST(test_communication_send_packet);
    UNITTEST(test_communication_send_recv_reduce_packet);
    UNITTEST(test_communication_send_recv_reduce);
    return ark::unittest::SUCCESS;
}
