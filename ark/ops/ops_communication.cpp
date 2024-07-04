// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_communication.hpp"

#include "ops_common.hpp"

namespace {
static const std::map<std::string, size_t> packet_type_map = {
    {"mscclpp::LL8Packet", 4},
    {"mscclpp::LL16Packet", 8},
};
} // namespace

namespace ark {

ModelOpSend::ModelOpSend(ModelTensorRef input, int remote_rank, int tag,
                         ModelTensorRef output)
    : ModelOp("Send") {
    check_null(input);
    if (output) {
        // TODO: verify output shape and strides
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
        {"comm::ChannelType::" + channel_type, std::to_string(true),
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
                        {"ChannelType", "NumTasks", "NumWarps", "SramBytes"});
    std::string channel_type = config["ChannelType"];
    if (channel_type != "Proxy" && channel_type != "SecondaryProxy" &&
        channel_type != "Sm") {
        ERR(ModelError, "invalid channel type: ", channel_type);
    }
    auto &input = read_tensors_[0];
    int remote_rank = input->buffer()->rank();
    int max_spin_cnt = -1;
    return function_name_string(
        "wait", {"comm::ChannelType::" + channel_type,
                 std::to_string(remote_rank), std::to_string(max_spin_cnt)});
}

std::vector<ModelOpArg> ModelOpRecv::impl_args([
    [maybe_unused]] const Json &config) const {
    return {};
}

Json ModelOpRecv::default_config([[maybe_unused]] const ArchRef arch) const {
    return {{"ChannelType", "Proxy"},
            {"NumTasks", 1},
            {"NumWarps", 1},
            {"SramBytes", 0}};
}

ModelOpSendPacket::ModelOpSendPacket(ModelTensorRef input, int remote_rank,
                                     int tag, int flag, ModelTensorRef output)
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
        Dims output_shape = input->shape();
        int n_dims = output_shape.ndims();
        output_shape[n_dims - 1] =
            output_shape[n_dims - 1] * 2 * input->data_type()->bytes();
        output = std::make_shared<ModelTensor>(
            UINT8.ref(), std::make_shared<ModelBuffer>(remote_rank),
            output_shape);
    }
    input->buffer()->tag_send(remote_rank, tag);
    output->buffer()->tag_recv(-1, tag);
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);

    read_tensors_ = {input};
    write_tensors_ = {output};
    result_tensors_ = {result};
    flag_ = flag;
    verify();
}

std::string ModelOpSendPacket::impl_name(const Json &config) const {
    check_fields_config(
        config, {"NumTasks", "NumWarps", "Tile", "SramBytes", "PacketType"});
    auto &input = read_tensors_[0];
    auto &output = write_tensors_[0];
    int remote_rank = output->buffer()->rank();
    Dims unit_out_dims;
    int num_warps = config.at("NumWarps");
    auto &tile_shape = config.at("Tile");
    std::string packet_type = config.at("PacketType");
    unit_out_dims = {1, 1, tile_shape[0], tile_shape[1]};
    size_t scale_factor =
        packet_type_map.at(packet_type) / input->data_type()->bytes();
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
        dim[3] = dim[3] / packet_type_map.at(packet_type) / 2;
    }
    return function_name_string(
        "write_packet", {std::to_string(remote_rank), vec_string(in_dims[0]),
                         vec_string(in_dims[1]), vec_string(out_dims[0]),
                         vec_string(out_dims[1]), vec_string(out_dims[2]),
                         std::to_string(num_warps), std::to_string(0),
                         packet_type, std::to_string(flag_)});
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
                                     int tag, int flag, ModelTensorRef scratch)
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
        Dims scratch_shape = output->shape();
        int n_dims = scratch_shape.ndims();
        scratch_shape[n_dims - 1] =
            scratch_shape[n_dims - 1] * 2 * output->data_type()->bytes();
        scratch = std::make_shared<ModelTensor>(
            UINT8.ref(), std::make_shared<ModelBuffer>(local_rank),
            scratch_shape);
    }
    ModelTensorRef input = std::make_shared<ModelTensor>(
        output->data_type(), std::make_shared<ModelBuffer>(remote_rank),
        output->shape());
    scratch->buffer()->tag_recv(remote_rank, tag);
    input->buffer()->tag_send(-1, tag);

    read_tensors_ = {input, scratch};
    write_tensors_ = {output};
    result_tensors_ = {result};
    flag_ = flag;
    verify();
}

std::string ModelOpRecvPacket::impl_name(const Json &config) const {
    check_fields_config(
        config, {"NumTasks", "NumWarps", "Tile", "SramBytes", "PacketType"});
    auto &input = read_tensors_[1];
    auto &output = write_tensors_[0];
    Dims unit_out_dims;
    int num_warps = config.at("NumWarps");
    auto &tile_shape = config.at("Tile");
    std::string packet_type = config.at("PacketType");
    unit_out_dims = {1, 1, tile_shape[0], tile_shape[1]};
    size_t scale_factor =
        packet_type_map.at(packet_type) / input->data_type()->bytes();
    if (scale_factor == 0) {
        ERR(ModelError,
            "unsupported data type: ", input->data_type()->type_str());
    }
    Dims in_dims[] = {input->strides().dims4(), input->shape().dims4()};
    for (auto &dim : in_dims) {
        dim[3] /= scale_factor / 2;
    }
    Dims out_dims[] = {output->strides().dims4(), output->shape().dims4(),
                       unit_out_dims};
    for (auto &dim : out_dims) {
        dim[3] = dim[3] / packet_type_map.at(packet_type);
    }
    return function_name_string(
        "read_packet", {vec_string(in_dims[0]), vec_string(in_dims[1]),
                        vec_string(out_dims[0]), vec_string(out_dims[1]),
                        vec_string(out_dims[2]), std::to_string(num_warps),
                        std::to_string(0), packet_type, std::to_string(flag_)});
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
    size_t tile_y = 512;
    config["Tile"] = {tile_x, tile_y};
    size_t num_tasks = shape[0] * shape[1];
    num_tasks *= (shape[2] + tile_x - 1) / tile_x;
    num_tasks *= (shape[3] + tile_y - 1) / tile_y;
    config["NumTasks"] = num_tasks;
    return config;
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

}  // namespace ark
