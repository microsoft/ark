// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"

namespace ark {

TensorOp::TensorOp(const std::vector<Tensor *> &deps, Tensor *output,
                   const std::string &name)
    : Op{OP_TENSOR, "none", deps, {output}, {}, name, nullptr, -1} {}

Tensor *Model::tensor(const Dims &shape, const TensorType &ttype,
                      TensorBuf *buf, const Dims &ldims, const Dims &offs,
                      const Dims &pads, const std::vector<Tensor *> &deps,
                      bool exported, int imported_rank,
                      const std::string &name) {
    if (buf == nullptr) {
        buf = this->impl->create_tensor_buf();
    }
    int tensor_id = (int)this->impl->tns_storage.size();
    this->impl->tns_storage.emplace_back(
        std::make_unique<Tensor>(shape, ttype, buf, ldims, offs, pads, exported,
                                 imported_rank, tensor_id, name));
    std::set<Tensor *> dep_set;
    for (auto &dep : deps) {
        dep_set.emplace(dep);
    }
    std::vector<Tensor *> dep_vec;
    for (auto &dep : dep_set) {
        dep_vec.emplace_back(dep);
    }
    TensorOp op{dep_vec, this->impl->tns_storage.back().get(), name};
    return this->impl->add_op(op)[0];
}

}  // namespace ark
