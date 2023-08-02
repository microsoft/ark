// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"
#include "tensor.h"

using namespace std;

namespace ark {

TensorOp::TensorOp(const vector<Tensor *> &deps, Tensor *output,
                   const string &name)
    : Op{OP_TENSOR, OP_PREC_NONE, deps, {output}, {}, name, nullptr, -1}
{
}

Tensor *Model::tensor(const Dims &shape, TensorType type, TensorBuf *buf,
                      const Dims &ldims, const Dims &offs, const Dims &pads,
                      const vector<Tensor *> &deps, bool exported,
                      int imported_rank, const std::string &name)
{
    LOG(DEBUG, "tensor ", name, " ", shape, " ", type, " ", ldims, " ", offs,
        " ", pads);
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
    set<Tensor *> dep_set;
    for (auto &dep : deps) {
        dep_set.emplace(dep);
    }
    vector<Tensor *> dep_vec;
    for (auto &dep : dep_set) {
        dep_vec.emplace_back(dep);
    }
    TensorOp op{dep_vec, ret, name};
    return this->impl->add_op(op)[0];
}

} // namespace ark
