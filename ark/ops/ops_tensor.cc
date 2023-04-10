// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/logging.h"
#include "ark/model_io.h"

using namespace std;

namespace ark {

Tensor *Model::tensor(const Dims &shape, TensorType type, TensorBuf *buf,
                      const Dims &ldims, const Dims &offs, const Dims &pads,
                      const vector<Tensor *> &deps, bool exported,
                      bool imported, const std::string &name)
{
    LOG(DEBUG, "tensor ", name, " ", shape, " ", type, " ", ldims, " ", offs,
        " ", pads);
    if (buf == nullptr) {
        buf = this->create_tensor_buf();
    }
    Tensor *ret = new Tensor{shape,    type,     buf,
                             ldims,    offs,     pads,
                             exported, imported, (int)this->tns_storage.size(),
                             name};
    assert(ret != nullptr);
    this->tns_storage.emplace_back(ret);
    set<Tensor *> dep_set;
    for (auto &dep : deps) {
        dep_set.emplace(dep);
    }
    vector<Tensor *> dep_vec;
    for (auto &dep : dep_set) {
        dep_vec.emplace_back(dep);
    }
    this->create_op(OP_TENSOR, OP_PREC_NONE, dep_vec, {ret}, {}, name);
    return ret;
}

} // namespace ark
