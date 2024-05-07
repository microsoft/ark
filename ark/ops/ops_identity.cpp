// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_identity.hpp"

#include <set>

#include "ops_common.hpp"

namespace ark {

ModelOpIdentity::ModelOpIdentity(ModelTensorRef input,
                                 const std::vector<ModelTensorRef> &deps)
    : ModelOpTensor(input->buffer(), input->shape(), input->data_type(),
                    input->strides(), input->offsets(), input->pads()) {
    std::set<ModelTensorRef> dep_set;
    dep_set.emplace(input);
    read_tensors_.emplace_back(input);
    for (auto &dep : deps) {
        if (dep_set.emplace(dep).second) {
            read_tensors_.emplace_back(dep);
        }
    }

    verify();
}

Tensor Model::identity(Tensor input, const std::vector<Tensor> &deps,
                       const std::string &name) {
    std::vector<ModelTensorRef> deps_ref;
    for (auto &dep : deps) {
        deps_ref.emplace_back(dep.ref_);
    }
    return impl_->create_op<ModelOpIdentity>(name, input.ref_, deps_ref)
        ->result_tensors()[0];
}

}  // namespace ark
