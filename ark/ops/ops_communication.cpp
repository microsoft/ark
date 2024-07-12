// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_communication.hpp"

#include "ops_common.hpp"

namespace {
static const std::map<std::string, size_t> packet_payload_size_map = {
    {"mscclpp::LL8Packet", 4},
    {"mscclpp::LL16Packet", 8},
};
static const int MAX_NUM_PEERS = 7;
}  // namespace

namespace ark {

ModelOpSend::ModelOpSend(ModelTensorRef input, int remote_rank, int tag,
                         ModelTensorRef output)
    : ModelOp("Send") {
    check_null(input);
    if (output) {
        if (output->buffer()->rank() != remote_rank) {
            ERR(ModelError, "invalid buffer rank: ", output->buffer()->rank(),
                ", expected: ", remote_rank);
        }
    } else {
        output = std::make_shared<ModelTensor>(
            input->data_type(), std::make_shared<ModelBuffer>(remote_rank),
            input->shape(), input->strides(), input->offsets(),
            input->padded_shape());
    }
    input->buffer()->tag_send(remote_rank, tag);
    output->buffer()->tag_recv(-1, tag);
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);

    read_tensors_ = {input};
    write_tensors_ = {output};
    result_tensors_ = {result};
    verify();
}

std::string ModelOpSend::impl_name(const Json &config) const {
    check_fields_config(config,
                        {"ChannelType", "NumTasks", "NumWarps", "SramBytes"});
    auto &input = read_tensors_[0];
    auto &output = write_tensors_[0];
    int remote_rank = output->buffer()->rank();
    bool signal = config["Signal"];
    std::string channel_type = config["ChannelType"];
    if (channel_type != "Proxy" && channel_type != "SecondaryProxy" &&
        channel_type != "Sm") {
        ERR(ModelError, "invalid channel type: ", channel_type);
    }
    Dims unit_out_dims;
    if (config.find("Tile") != config.end()) {
        auto &tile_shape = config.at("Tile");
        unit_out_dims = {1, 1, tile_shape[0], tile_shape[1]};
    } else {
        unit_out_dims = output->strides().dims4();
    }
    return function_name_string(
        "put",
        {"comm::ChannelType::" + channel_type, std::to_string(signal),
         std::to_string(remote_rank), vec_string(input->strides().dims4()),
         vec_string(input->shape().dims4()),
         vec_string(output->strides().dims4()),
         vec_string(output->shape().dims4()), vec_string(unit_out_dims),
         std::to_string(1), std::to_string(0),
         output->data_type()->type_str()});
}

std::vector<ModelOpArg> ModelOpSend::impl_args([
    [maybe_unused]] const Json &config) const {
    return {ModelOffset(write_tensors_[0]), ModelOffset(read_tensors_[0])};
}

Json ModelOpSend::default_config([[maybe_unused]] const ArchRef arch) const {
    return {{"ChannelType", "Proxy"},
            {"Signal", true},
            {"NumTasks", 1},
            {"NumWarps", 1},
            {"SramBytes", 0}};
}

ModelOpSendDone::ModelOpSendDone(ModelTensorRef input) : ModelOp("SendDone") {
    check_null(input);
    ModelTensorRef result = std::make_shared<ModelTensor>(*input);
    read_tensors_ = {input};
    write_tensors_ = {};
    result_tensors_ = {result};
    verify();
}

std::string ModelOpSendDone::impl_name(const Json &config) const {
    check_fields_config(config,
                        {"ChannelType", "NumTasks", "NumWarps", "SramBytes"});
    std::string channel_type = config["ChannelType"];
    if (channel_type != "Proxy" && channel_type != "SecondaryProxy" &&
        channel_type != "Sm") {
        ERR(ModelError, "invalid channel type: ", channel_type);
    }
    auto &input = read_tensors_[0];
    int remote_rank = input->buffer()->rank();
    return function_name_string("flush", {"comm::ChannelType::" + channel_type,
                                          std::to_string(remote_rank)});
}

std::vector<ModelOpArg> ModelOpSendDone::impl_args([
    [maybe_unused]] const Json &config) const {
    return {};
}

Json ModelOpSendDone::default_config([
    [maybe_unused]] const ArchRef arch) const {
    return {{"ChannelType", "Proxy"},
            {"NumTasks", 1},
            {"NumWarps", 1},
            {"SramBytes", 0}};
}

ModelOpRecv::ModelOpRecv(ModelTensorRef output, int remote_rank, int tag)
    : ModelOp("Recv") {
    check_null(output);
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);
    ModelTensorRef input = std::make_shared<ModelTensor>(
        output->data_type(), std::make_shared<ModelBuffer>(remote_rank),
        output->shape(), output->strides(), output->offsets(),
        output->padded_shape());
    input->buffer()->tag_send(-1, tag);
    output->buffer()->tag_recv(remote_rank, tag);

    read_tensors_ = {input};
    write_tensors_ = {output};
    result_tensors_ = {result};
    verify();
}

std::string ModelOpRecv::impl_name(const Json &config) const {
    check_fields_config(config,
                        {"ChannelType", "NumTasks", "NumWarps", "SramBytes", "Wait"});
    std::string channel_type = config["ChannelType"];
    bool wait = config["Wait"];
    if (channel_type != "Proxy" && channel_type != "SecondaryProxy" &&
        channel_type != "Sm") {
        ERR(ModelError, "invalid channel type: ", channel_type);
    }
    auto &input = read_tensors_[0];
    int remote_rank = input->buffer()->rank();
    int max_spin_cnt = -1;
    return function_name_string(
        "wait",
        {"comm::ChannelType::" + channel_type, std::to_string(remote_rank),
         std::to_string(max_spin_cnt), std::to_string(wait)});
}

std::vector<ModelOpArg> ModelOpRecv::impl_args([
    [maybe_unused]] const Json &config) const {
    return {};
}

Json ModelOpRecv::default_config([[maybe_unused]] const ArchRef arch) const {
    return {{"ChannelType", "Proxy"},
            {"NumTasks", 1},
            {"NumWarps", 1},
            {"SramBytes", 0},
            {"Wait", true}};
}

ModelOpSendPacket::ModelOpSendPacket(ModelTensorRef input, int remote_rank,
                                     int tag, uint32_t flag,
                                     ModelTensorRef output)
    : ModelOp("SendPacket") {
    check_null(input);
    if (output) {
        // TODO: verify output shape and strides
        if (output->buffer()->rank() != remote_rank) {
            ERR(ModelError, "invalid buffer rank: ", output->buffer()->rank(),
                ", expected: ", remote_rank);
        }
    } else {
        // For packet output, expand the last dimension to 2x
        Dims output_shape(input->shape_bytes() * 2);
        output = std::make_shared<ModelTensor>(
            UINT8.ref(), std::make_shared<ModelBuffer>(remote_rank),
            output_shape);
    }
    output->buffer()->tag_recv(-1, tag);
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);

    read_tensors_ = {input};
    write_tensors_ = {output};
    result_tensors_ = {result};
    args_ = {{"Flag", ModelOpArg(flag)}};
    verify();
}

std::string ModelOpSendPacket::impl_name(const Json &config) const {
    check_fields_config(
        config, {"NumTasks", "NumWarps", "Tile", "SramBytes", "PacketType"});
    auto &input = read_tensors_[0];
    auto &output = write_tensors_[0];
    uint32_t flag = args_.at("Flag").value<uint32_t>();
    int remote_rank = output->buffer()->rank();
    int num_warps = config.at("NumWarps");
    auto &tile_shape = config.at("Tile");
    std::string packet_type = config.at("PacketType");
    Dims unit_out_dims = {1, 1, tile_shape[0], tile_shape[1]};
    const size_t packet_payload_size = packet_payload_size_map.at(packet_type);
    const size_t scale_factor =
        packet_payload_size / input->data_type()->bytes();
    if (scale_factor == 0) {
        ERR(ModelError,
            "unsupported data type: ", input->data_type()->type_str());
    }
    Dims in_dims[] = {input->strides().dims4(), input->shape().dims4()};
    for (auto &dim : in_dims) {
        dim[3] /= scale_factor;
    }
    Dims out_dims[] = {output->strides().dims4(), output->shape().dims4(),
                       unit_out_dims};
    for (auto &dim : out_dims) {
        dim[3] = dim[3] / packet_payload_size / 2;
    }
    return function_name_string(
        "write_packet", {std::to_string(remote_rank), vec_string(in_dims[0]),
                         vec_string(in_dims[1]), vec_string(out_dims[0]),
                         vec_string(out_dims[1]), vec_string(out_dims[2]),
                         std::to_string(num_warps), std::to_string(0),
                         packet_type, std::to_string(flag)});
}

std::vector<ModelOpArg> ModelOpSendPacket::impl_args([
    [maybe_unused]] const Json &config) const {
    return {ModelOffset(write_tensors_[0]), ModelOffset(read_tensors_[0])};
}

Json ModelOpSendPacket::default_config([
    [maybe_unused]] const ArchRef arch) const {
    Json config;
    if (arch->belongs_to(ARCH_ROCM)) {
        config["PacketType"] = "mscclpp::LL8Packet";
    } else {
        config["PacketType"] = "mscclpp::LL16Packet";
    }
    config["NumWarps"] = 1;
    config["SramBytes"] = 0;
    const auto &shape = result_tensors_[0]->shape().dims4();
    size_t tile_x = 1;
    size_t tile_y = 512;
    config["Tile"] = {tile_x, tile_y};
    size_t num_tasks = shape[0] * shape[1];
    num_tasks *= (shape[2] + tile_x - 1) / tile_x;
    num_tasks *= (shape[3] + tile_y - 1) / tile_y;
    config["NumTasks"] = num_tasks;
    return config;
}

ModelOpRecvPacket::ModelOpRecvPacket(ModelTensorRef output, int remote_rank,
                                     int tag, uint32_t flag,
                                     ModelTensorRef scratch)
    : ModelOp("RecvPacket") {
    check_null(output);
    int local_rank = output->buffer()->rank();
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);
    if (scratch) {
        if (scratch->buffer()->rank() != local_rank) {
            ERR(ModelError, "invalid buffer rank: ", scratch->buffer()->rank(),
                ", expected: ", local_rank);
        }
    } else {
        // For packet output, expand the last dimension to 2x
        Dims scratch_shape(output->shape_bytes() * 2);
        scratch = std::make_shared<ModelTensor>(
            UINT8.ref(), std::make_shared<ModelBuffer>(local_rank),
            scratch_shape);
    }
    ModelTensorRef input = std::make_shared<ModelTensor>(
        output->data_type(), std::make_shared<ModelBuffer>(remote_rank),
        output->shape());
    scratch->buffer()->tag_recv(remote_rank, tag);

    read_tensors_ = {input, scratch};
    write_tensors_ = {output};
    result_tensors_ = {result};
    args_ = {{"Flag", ModelOpArg(flag)}};
    verify();
}

std::string ModelOpRecvPacket::impl_name(const Json &config) const {
    check_fields_config(
        config, {"NumTasks", "NumWarps", "Tile", "SramBytes", "PacketType"});
    auto &input = read_tensors_[1];
    auto &peer_tensor = read_tensors_[0];
    auto &output = write_tensors_[0];
    uint32_t flag = args_.at("Flag").value<uint32_t>();
    int num_warps = config.at("NumWarps");
    auto &tile_shape = config.at("Tile");
    std::string packet_type = config.at("PacketType");
    int remote_rank = peer_tensor->buffer()->rank();
    Dims unit_out_dims = {1, 1, tile_shape[0], tile_shape[1]};
    const size_t packet_payload_size = packet_payload_size_map.at(packet_type);
    const size_t scale_factor =
        packet_payload_size / output->data_type()->bytes();
    if (scale_factor == 0) {
        ERR(ModelError,
            "unsupported data type: ", input->data_type()->type_str());
    }
    Dims in_dims[] = {input->strides().dims4(), input->shape().dims4()};
    for (auto &dim : in_dims) {
        dim[3] = dim[3] / packet_payload_size / 2;
    }
    Dims out_dims[] = {output->strides().dims4(), output->shape().dims4(),
                       unit_out_dims};
    for (auto &dim : out_dims) {
        dim[3] = dim[3] / scale_factor;
    }
    return function_name_string(
        "read_packet", {std::to_string(remote_rank), vec_string(in_dims[0]),
                        vec_string(in_dims[1]), vec_string(out_dims[0]),
                        vec_string(out_dims[1]), vec_string(out_dims[2]),
                        std::to_string(num_warps), std::to_string(0),
                        packet_type, std::to_string(flag)});
}

std::vector<ModelOpArg> ModelOpRecvPacket::impl_args([
    [maybe_unused]] const Json &config) const {
    return {ModelOffset(write_tensors_[0]), ModelOffset(read_tensors_[1])};
}

Json ModelOpRecvPacket::default_config([
    [maybe_unused]] const ArchRef arch) const {
    Json config;
    if (arch->belongs_to(ARCH_ROCM)) {
        config["PacketType"] = "mscclpp::LL8Packet";
    } else {
        config["PacketType"] = "mscclpp::LL16Packet";
    }
    config["NumWarps"] = 1;
    config["SramBytes"] = 0;
    const auto &shape = result_tensors_[0]->shape().dims4();
    size_t tile_x = 1;
    size_t tile_y = 128;
    config["Tile"] = {tile_x, tile_y};
    size_t num_tasks = shape[0] * shape[1];
    num_tasks *= (shape[2] + tile_x - 1) / tile_x;
    num_tasks *= (shape[3] + tile_y - 1) / tile_y;
    config["NumTasks"] = num_tasks;
    return config;
}

ModelOpRecvReduceSendPacket::ModelOpRecvReduceSendPacket(
    ModelTensorRef input, ModelTensorRef output, int rank,
    const std::vector<int> &remote_ranks, int recv_tag, int output_tag,
    uint32_t flag, std::vector<ModelTensorRef> &peer_output_refs,
    ModelTensorRef scratch)
    : ModelOp("RecvReduceSendPacket") {
    check_null(input);
    uint32_t n_remote_ranks = remote_ranks.size();
    // Need to check the scratch buffers are contiguous
    if (scratch) {
        if (scratch->buffer()->rank() != rank &&
            scratch->buffer()->rank() != -1) {
            ERR(ModelError, "invalid buffer rank: ", scratch->buffer()->rank(),
                ", expected: ", rank);
        }
    } else {
        Dims scratch_shape(input->shape_bytes() * 2 * n_remote_ranks);
        scratch = std::make_shared<ModelTensor>(
            UINT8.ref(), std::make_shared<ModelBuffer>(rank), scratch_shape);
    }
    if (!output) {
        output = std::make_shared<ModelTensor>(
            input->data_type(), std::make_shared<ModelBuffer>(rank),
            input->shape(), input->strides(), input->offsets(),
            input->padded_shape());
    }
    for (uint32_t i = 0; i < n_remote_ranks; ++i) {
        scratch->buffer()->tag_recv(remote_ranks[i], recv_tag);
        peer_output_refs[i]->buffer()->tag_recv(-1, output_tag);
    }
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);
    read_tensors_ = {input, scratch};
    write_tensors_ = {output};
    write_tensors_.insert(write_tensors_.end(), peer_output_refs.begin(),
                          peer_output_refs.end());
    result_tensors_ = {result};
    args_ = {
        {"Flag", ModelOpArg(flag)},
        {"NPeers", ModelOpArg(n_remote_ranks)},
        {"Rank", ModelOpArg(rank)},
    };
    verify();
}

std::string ModelOpRecvReduceSendPacket::impl_name(const Json &config) const {
    check_fields_config(
        config, {"NumTasks", "NumWarps", "Tile", "SramBytes", "PacketType"});
    auto &input = read_tensors_[0];
    auto &output = write_tensors_[0];
    uint32_t flag = args_.at("Flag").value<uint32_t>();
    uint32_t n_peers = args_.at("NPeers").value<uint32_t>();
    int rank = args_.at("Rank").value<int>();
    int num_warps = config.at("NumWarps");
    auto &tile_shape = config.at("Tile");
    std::string packet_type = config.at("PacketType");
    Dims unit_out_dims = {1, 1, tile_shape[0], tile_shape[1]};
    Dims in_dims[] = {input->strides().dims4(), input->shape().dims4()};
    Dims out_dims[] = {output->strides().dims4(), output->shape().dims4(),
                       unit_out_dims};
    return function_name_string(
        "read_reduce_and_write",
        {vec_string(in_dims[0]), vec_string(in_dims[1]),
         vec_string(out_dims[0]), vec_string(out_dims[1]),
         vec_string(out_dims[2]), std::to_string(num_warps), std::to_string(0),
         std::to_string(n_peers), std::to_string(rank), packet_type,
         input->data_type()->type_str(), std::to_string(flag)});
}

std::vector<ModelOpArg> ModelOpRecvReduceSendPacket::impl_args([
    [maybe_unused]] const Json &config) const {
    std::vector<ModelOpArg> args = {write_tensors_[0], read_tensors_[0],
                                    read_tensors_[1]};
    for (size_t i = 1; i < write_tensors_.size(); ++i) {
        args.push_back(ModelOffset(write_tensors_[i]));
    }
    for (int i = write_tensors_.size() - 1; i < MAX_NUM_PEERS; ++i) {
        args.push_back(0L);
    }
    return args;
}

Json ModelOpRecvReduceSendPacket::default_config([
    [maybe_unused]] const ArchRef arch) const {
    Json config;
    if (arch->belongs_to(ARCH_ROCM)) {
        config["PacketType"] = "mscclpp::LL8Packet";
    } else {
        config["PacketType"] = "mscclpp::LL16Packet";
    }
    config["NumWarps"] = 1;
    config["SramBytes"] = 0;
    const auto &shape = result_tensors_[0]->shape().dims4();
    size_t tile_x = 1;
    size_t tile_y = 128;
    config["Tile"] = {tile_x, tile_y};
    size_t num_tasks = shape[0] * shape[1];
    num_tasks *= (shape[2] + tile_x - 1) / tile_x;
    num_tasks *= (shape[3] + tile_y - 1) / tile_y;
    config["NumTasks"] = num_tasks;
    return config;
}

ModelOpRecvReduceSend::ModelOpRecvReduceSend(ModelTensorRef input,
                                             ModelTensorRef output, int rank,
                                             const std::vector<int> &remote_ranks,
                                             int recv_tag, int output_tag,
                                             std::vector<ModelTensorRef> &peer_output_refs,
                                             ModelTensorRef scratch)
    : ModelOp("RecvReduceSend") {
    check_null(input);
    uint32_t n_remote_ranks = remote_ranks.size();
    // Need to check the scratch buffers are contiguous
    if (scratch) {
        if (scratch->buffer()->rank() != rank &&
            scratch->buffer()->rank() != -1) {
            ERR(ModelError, "invalid buffer rank: ", scratch->buffer()->rank(),
                ", expected: ", rank);
        }
    } else {
        Dims scratch_shape(input->shape_bytes() * n_remote_ranks /
                           input->data_type()->bytes());
        scratch = std::make_shared<ModelTensor>(
            input->data_type(), std::make_shared<ModelBuffer>(rank),
            scratch_shape);
    }
    if (!output) {
        output = std::make_shared<ModelTensor>(
            input->data_type(), std::make_shared<ModelBuffer>(rank),
            input->shape(), input->strides(), input->offsets(),
            input->padded_shape());
    }
    for (uint32_t i = 0; i < n_remote_ranks; ++i) {
        scratch->buffer()->tag_recv(remote_ranks[i], recv_tag);
        peer_output_refs[i]->buffer()->tag_recv(-1, output_tag);
    }
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);
    read_tensors_ = {input, scratch};
    write_tensors_ = {output};
    write_tensors_.insert(write_tensors_.end(), peer_output_refs.begin(),
                          peer_output_refs.end());
    result_tensors_ = {result};
    args_ = {
        {"NPeers", ModelOpArg(n_remote_ranks)},
        {"Rank", ModelOpArg(rank)},
    };
    verify();
}

std::string ModelOpRecvReduceSend::impl_name(const Json &config) const {
    check_fields_config(config, {"NumTasks", "NumWarps", "Tile", "SramBytes"});
    auto &input = read_tensors_[0];
    auto &output = write_tensors_[0];
    uint32_t n_peers = args_.at("NPeers").value<uint32_t>();
    int rank = args_.at("Rank").value<int>();
    int num_warps = config.at("NumWarps");
    auto &tile_shape = config.at("Tile");
    Dims unit_out_dims = {1, 1, tile_shape[0], tile_shape[1]};
    Dims in_dims[] = {input->strides().dims4(), input->shape().dims4()};
    Dims out_dims[] = {output->strides().dims4(), output->shape().dims4(),
                       unit_out_dims};
    return function_name_string(
        "read_reduce_and_write",
        {vec_string(in_dims[0]), vec_string(in_dims[1]),
         vec_string(out_dims[0]), vec_string(out_dims[1]),
         vec_string(out_dims[2]), std::to_string(num_warps), std::to_string(0),
         std::to_string(n_peers), std::to_string(rank),
         input->data_type()->type_str(), input->data_type()->type_str()});
}

std::vector<ModelOpArg> ModelOpRecvReduceSend::impl_args([
    [maybe_unused]] const Json &config) const {
    std::vector<ModelOpArg> args = {write_tensors_[0], read_tensors_[0],
                                    read_tensors_[1]};
    for (size_t i = 1; i < write_tensors_.size(); ++i) {
        args.push_back(ModelOffset(write_tensors_[i]));
    }
    for (int i = write_tensors_.size() - 1; i < MAX_NUM_PEERS; ++i) {
        args.push_back(0L);
    }
    return args;
}

Json ModelOpRecvReduceSend::default_config([
    [maybe_unused]] const ArchRef arch) const {
    Json config;
    config["NumWarps"] = 1;
    config["SramBytes"] = 0;
    const auto &shape = result_tensors_[0]->shape().dims4();
    size_t tile_x = 1;
    size_t tile_y = 128;
    config["Tile"] = {tile_x, tile_y};
    size_t num_tasks = shape[0] * shape[1];
    num_tasks *= (shape[2] + tile_x - 1) / tile_x;
    num_tasks *= (shape[3] + tile_y - 1) / tile_y;
    config["NumTasks"] = num_tasks;
    return config;
}

ModelOpDeviceSync::ModelOpDeviceSync(ModelTensorRef input, int rank,
                                     int rank_num, ModelTensorRef output)
    : ModelOp("DeviceSync") {
    check_null(input);
    check_null(output);
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);
    read_tensors_ = {input};
    write_tensors_ = {output};
    result_tensors_ = {result};
    args_ = {{"Rank", rank}, {"PeerNum", rank_num - 1}};
    verify();
}

std::string ModelOpDeviceSync::impl_name(const Json &config) const {
    check_fields_config(config,
                        {"ChannelType", "NumTasks", "NumWarps", "SramBytes"});
    std::string channel_type = config["ChannelType"];
    if (channel_type != "Proxy" && channel_type != "SecondaryProxy" &&
        channel_type != "Sm") {
        ERR(ModelError, "invalid channel type: ", channel_type);
    }
    int rank = args_.at("Rank").value<int>();
    int peer_num = args_.at("PeerNum").value<int>();
    return function_name_string(
        "device_sync", {"comm::ChannelType::" + channel_type,
                        std::to_string(peer_num), std::to_string(rank)});
}

std::vector<ModelOpArg> ModelOpDeviceSync::impl_args([
    [maybe_unused]] const Json &config) const {
    return {};
}

Json ModelOpDeviceSync::default_config([[maybe_unused]] const ArchRef arch) const {
    return {{"ChannelType", "Proxy"},
            {"NumTasks", 1},
            {"NumWarps", 1},
            {"SramBytes", 0}};
}

Tensor Model::send(Tensor input, int remote_rank, int tag, Tensor output,
                   const std::string &name) {
    tags_.insert(tag);
    return impl_
        ->create_op<ModelOpSend>(name, input.ref(), remote_rank, tag,
                                 output.ref())
        ->result_tensors()[0];
}

Tensor Model::send_done(Tensor input, const std::string &name) {
    return impl_->create_op<ModelOpSendDone>(name, input.ref())
        ->result_tensors()[0];
}

Tensor Model::recv(Tensor output, int remote_rank, int tag,
                   const std::string &name) {
    tags_.insert(tag);
    return impl_->create_op<ModelOpRecv>(name, output.ref(), remote_rank, tag)
        ->result_tensors()[0];
}

Tensor Model::send_packet(Tensor input, int remote_rank, int tag, int flag,
                          Tensor output, const std::string &name) {
    tags_.insert(tag);
    return impl_
        ->create_op<ModelOpSendPacket>(name, input.ref(), remote_rank, tag,
                                       flag, output.ref())
        ->result_tensors()[0];
}

Tensor Model::recv_packet(Tensor output, int remote_rank, int tag, int flag,
                          Tensor scratch, const std::string &name) {
    tags_.insert(tag);
    return impl_
        ->create_op<ModelOpRecvPacket>(name, output.ref(), remote_rank, tag,
                                       flag, scratch.ref())
        ->result_tensors()[0];
}

Tensor Model::recv_reduce_send_packet(Tensor input,
                                      const std::vector<int> &remote_ranks,
                                      int recv_tag, int output_tag,
                                      unsigned int flag, Tensor output,
                                      std::vector<Tensor> peer_outputs,
                                      Tensor scratch, const std::string &name) {
    tags_.insert(recv_tag);
    tags_.insert(output_tag);
    std::vector<Tensor> result_tensors;
    std::vector<ModelTensorRef> scratch_refs;
    std::vector<ModelTensorRef> outputs_refs;
    int local_rank = this->rank();
    if (peer_outputs.empty()) {
        size_t shape_bytes = input.ref()->shape_bytes();
        Dims output_shape(shape_bytes * 2);  // For packet
        std::transform(remote_ranks.begin(), remote_ranks.end(),
                       std::back_inserter(peer_outputs), [&](int remote_rank) {
                           return std::make_shared<ModelTensor>(
                               UINT8.ref(),
                               std::make_shared<ModelBuffer>(remote_rank),
                               output_shape);
                       });
    }
    std::transform(peer_outputs.begin(), peer_outputs.end(),
                   std::back_inserter(outputs_refs),
                   [](const Tensor &t) { return t.ref(); });
    return impl_
        ->create_op<ModelOpRecvReduceSendPacket>(
            name, input.ref(), output.ref(), local_rank, remote_ranks, recv_tag,
            output_tag, flag, outputs_refs, scratch.ref())
        ->result_tensors()[0];
}

Tensor Model::recv_reduce_send(Tensor input,
                               const std::vector<int> &remote_ranks,
                               int recv_tag, int output_tag, Tensor output,
                               std::vector<Tensor> peer_outputs, Tensor scratch,
                               const std::string &name) {
    tags_.insert(recv_tag);
    tags_.insert(output_tag);
    std::vector<Tensor> result_tensors;
    std::vector<ModelTensorRef> scratch_refs;
    std::vector<ModelTensorRef> outputs_refs;
    int local_rank = this->rank();
    if (peer_outputs.empty()) {
        std::transform(remote_ranks.begin(), remote_ranks.end(),
                       std::back_inserter(peer_outputs), [&](int remote_rank) {
                           return std::make_shared<ModelTensor>(
                               input.ref()->data_type(),
                               std::make_shared<ModelBuffer>(remote_rank),
                               input.shape(), input.strides(), input.offsets(),
                               input.padded_shape());
                       });
    }
    std::transform(peer_outputs.begin(), peer_outputs.end(),
                   std::back_inserter(outputs_refs),
                   [](const Tensor &t) { return t.ref(); });
    return impl_
        ->create_op<ModelOpRecvReduceSend>(
            name, input.ref(), output.ref(), local_rank, remote_ranks, recv_tag,
            output_tag, outputs_refs, scratch.ref())
        ->result_tensors()[0];
}

Tensor Model::device_sync(Tensor input, int rank, int rank_num,
                          const std::string &name) {
    Tensor output = this->identity(input);
    return impl_
        ->create_op<ModelOpDeviceSync>(name, input.ref(), rank, rank_num,
                                       output.ref())
        ->result_tensors()[0];
}

}  // namespace ark
