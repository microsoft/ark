// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_SENDRECV_HPP_
#define ARK_OPS_SENDRECV_HPP_

#include "ark/dims.hpp"
#include "ark/model.hpp"
#include "model/model_op.hpp"

namespace ark {

class ModelOpSend : public ModelOp {
   public:
    ModelOpSend() = default;
    ModelOpSend(ModelTensorRef input, int sid, int rank, int dst_rank,
                DimType bytes);
};

class ModelOpSendDone : public ModelOp {
   public:
    ModelOpSendDone() = default;
    ModelOpSendDone(ModelTensorRef input, int rank, int dst_rank);
};

class ModelOpRecv : public ModelOp {
   public:
    ModelOpRecv() = default;
    ModelOpRecv(ModelTensorRef output, int, int rank, int src_rank,
                DimType bytes);
};

}  // namespace ark

#endif  // ARK_OPS_SENDRECV_HPP_
