// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_communication.hpp"

#include "ops_common.hpp"

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
            input->shape(), input->strides(), input->offsets(), input->pads());
    }
    input->buffer()->tag_send(remote_rank, tag);
    output->buffer()->tag_recv(-1, tag);
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);

    read_tensors_ = {input};
    write_tensors_ = {output};
    result_tensors_ = {result};
    verify();
}

std::string ModelOpSend::impl_name([
    [maybe_unused]] const nlohmann::json &config) const {
    auto &input = read_tensors_[0];
    auto &output = write_tensors_[0];
    int channel_id = output->buffer()->rank();
    return function_name_string(
        "send",
        {std::to_string(channel_id), vec_string(input->strides().dims4()),
         vec_string(input->shape().dims4()),
         vec_string(output->strides().dims4()),
         vec_string(output->shape().dims4()),
         vec_string(output->strides().dims4()), std::to_string(1),
         std::to_string(0), output->data_type()->type_str()});
}

std::vector<ModelOpArg> ModelOpSend::impl_args([
    [maybe_unused]] const nlohmann::json &config) const {
    return {ModelOffset(write_tensors_[0]), ModelOffset(read_tensors_[0])};
}

nlohmann::ordered_json ModelOpSend::default_config() const {
    return {{"NumTasks", 1}, {"NumWarps", 1}, {"SramBytes", 0}};
}

ModelOpSendDone::ModelOpSendDone(ModelTensorRef input) : ModelOp("SendDone") {
    check_null(input);
    ModelTensorRef result = std::make_shared<ModelTensor>(*input);
    read_tensors_ = {input};
    write_tensors_ = {};
    result_tensors_ = {result};
    verify();
}

std::string ModelOpSendDone::impl_name([
    [maybe_unused]] const nlohmann::json &config) const {
    auto &input = read_tensors_[0];
    int channel_id = input->buffer()->rank();
    return function_name_string("send_done", {std::to_string(channel_id)});
}

std::vector<ModelOpArg> ModelOpSendDone::impl_args([
    [maybe_unused]] const nlohmann::json &config) const {
    return {};
}

nlohmann::ordered_json ModelOpSendDone::default_config() const {
    return {{"NumTasks", 1}, {"NumWarps", 1}, {"SramBytes", 0}};
}

ModelOpRecv::ModelOpRecv(ModelTensorRef output, int remote_rank, int tag)
    : ModelOp("Recv") {
    check_null(output);
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);
    ModelTensorRef input = std::make_shared<ModelTensor>(
        output->data_type(), std::make_shared<ModelBuffer>(remote_rank),
        output->shape(), output->strides(), output->offsets(), output->pads());
    input->buffer()->tag_send(-1, tag);
    output->buffer()->tag_recv(remote_rank, tag);

    read_tensors_ = {input};
    write_tensors_ = {output};
    result_tensors_ = {result};
    verify();
}

std::string ModelOpRecv::impl_name([
    [maybe_unused]] const nlohmann::json &config) const {
    auto &input = read_tensors_[0];
    int channel_id = input->buffer()->rank();
    return function_name_string("recv", {std::to_string(channel_id)});
}

std::vector<ModelOpArg> ModelOpRecv::impl_args([
    [maybe_unused]] const nlohmann::json &config) const {
    return {};
}

nlohmann::ordered_json ModelOpRecv::default_config() const {
    return {{"NumTasks", 1}, {"NumWarps", 1}, {"SramBytes", 0}};
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

}  // namespace ark
