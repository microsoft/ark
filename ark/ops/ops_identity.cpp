// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_identity.hpp"

#include <set>

#include "ops_common.hpp"

namespace ark {

ModelOpIdentity::ModelOpIdentity(ModelTensorRef input,
                                 const std::vector<ModelTensorRef> &deps)
    : ModelOpTensor(input->buffer(), input->shape(), input->data_type(),
                    input->strides(), input->offsets(), input->pads(),
                    input->exported(), input->imported_rank()) {
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

ModelTensorRef Model::identity(ModelTensorRef input,
                               const std::vector<ModelTensorRef> &deps,
                               const std::string &name) {
    return impl_->create_op<ModelOpIdentity>(name, input, deps)
        ->result_tensors()[0];
}

}  // namespace ark
