// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"
#include <cassert>

namespace ark {

TensorOp::TensorOp(const std::vector<Tensor *> &deps, Tensor *output,
                   const std::string &name)
    : Op{OP_TENSOR, OP_PREC_NONE, deps, {output}, {}, name, nullptr, -1}
{
}

Tensor *Model::tensor(const Dims &shape, TensorType type, TensorBuf *buf,
                      const Dims &ldims, const Dims &offs, const Dims &pads,
                      const std::vector<Tensor *> &deps, bool exported,
                      int imported_rank, const std::string &name)
{
    if (buf == nullptr) {
        buf = this->impl->create_tensor_buf();
    }
    Tensor *ret =
        new Tensor{shape,    type,          buf,
                   ldims,    offs,          pads,
                   exported, imported_rank, (int)this->impl->tns_storage.size(),
                   name};
    assert(ret != nullptr);
    this->impl->tns_storage.emplace_back(ret);
    std::set<Tensor *> dep_set;
    for (auto &dep : deps) {
        dep_set.emplace(dep);
    }
    std::vector<Tensor *> dep_vec;
    for (auto &dep : dep_set) {
        dep_vec.emplace_back(dep);
    }
    TensorOp op{dep_vec, ret, name};
    return this->impl->add_op(op)[0];
}

} // namespace ark
