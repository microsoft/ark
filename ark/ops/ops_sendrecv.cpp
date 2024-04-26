// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_sendrecv.hpp"

#include "ops_common.hpp"

namespace ark {

ModelOpSend::ModelOpSend(ModelTensorRef input, int sid, int rank, int dst_rank,
                         DimType bytes)
    : ModelOp("Send") {
    DimType max_bytes = input->strides().size() * input->data_type()->bytes();
    if (max_bytes < bytes) {
        LOG(ERROR, "invalid bytes: ", bytes, ", max: ", max_bytes);
    }
    if (bytes == 0) {
        bytes = max_bytes;
    }
    input->set_exported();

    ModelTensorRef recvbuf = std::make_shared<ModelTensor>(
        input->data_type(), std::make_shared<ModelBuffer>(), input->shape());
    recvbuf->set_imported_rank(dst_rank);

    ModelTensorRef result = std::make_shared<ModelTensor>(*recvbuf);

    read_tensors_ = {input};
    write_tensors_ = {recvbuf};
    result_tensors_ = {result};
    args_["Rank"] = rank;
    args_["DstRank"] = dst_rank;
    args_["Bytes"] = bytes;
    args_["Sid"] = sid;

    verify();
}

ModelTensorRef Model::send(ModelTensorRef input, int sid, int dst_rank,
                           DimType bytes, const std::string &name) {
    return impl_
        ->create_op<ModelOpSend>(name, input, sid, rank_, dst_rank, bytes)
        ->result_tensors()[0];
}

ModelOpSendDone::ModelOpSendDone(ModelTensorRef input, int rank, int dst_rank)
    : ModelOp("SendDone") {
    ModelTensorRef result = std::make_shared<ModelTensor>(*input);
    read_tensors_ = {};
    write_tensors_ = {input};
    result_tensors_ = {result};
    args_["Rank"] = rank;
    args_["DstRank"] = dst_rank;

    verify();
}

ModelTensorRef Model::send_done(ModelTensorRef input, int, int dst_rank,
                                const std::string &name) {
    return impl_->create_op<ModelOpSendDone>(name, input, rank_, dst_rank)
        ->result_tensors()[0];
}

ModelOpRecv::ModelOpRecv(ModelTensorRef output, int, int rank, int src_rank,
                         DimType bytes)
    : ModelOp("Recv") {
    if (output == nullptr) {
        if (bytes == 0) {
            LOG(ERROR, "receive bytes cannot be 0");
        }
        output = std::make_shared<ModelTensor>(
            BYTE, std::make_shared<ModelBuffer>(), Dims{bytes});
    }
    output->set_exported();
    DimType max_bytes = output->shape().size() * output->data_type()->bytes();
    if (max_bytes < bytes) {
        LOG(ERROR, "invalid bytes: ", bytes, ", max: ", max_bytes);
    }
    if (bytes == 0) {
        bytes = max_bytes;
    }
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);

    read_tensors_ = {};
    write_tensors_ = {output};
    result_tensors_ = {result};
    args_["Rank"] = rank;
    args_["SrcRank"] = src_rank;

    verify();
}

ModelTensorRef Model::recv(int sid, int src_rank, DimType bytes,
                           ModelTensorRef output, const std::string &name) {
    return impl_
        ->create_op<ModelOpRecv>(name, output, sid, rank_, src_rank, bytes)
        ->result_tensors()[0];
}

}  // namespace ark
