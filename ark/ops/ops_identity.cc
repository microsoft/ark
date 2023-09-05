// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"
#include <cassert>

namespace ark {

// Returns an identical tensor of `input` with execution dependencies `deps`.
Tensor *Model::identity(Tensor *input, const std::vector<Tensor *> &deps,
                        const std::string &name)
{
    assert(input != nullptr);
    std::set<Tensor *> dep_set;
    dep_set.emplace(input);
    for (auto &dep : deps) {
        dep_set.emplace(dep);
    }
    std::vector<Tensor *> dep_vec;
    for (auto &dep : dep_set) {
        dep_vec.emplace_back(dep);
    }
    return this->tensor(input->shape, input->type, input->buf, input->ldims,
                        input->offs, input->pads, dep_vec, input->exported,
                        input->imported_rank, name + "/identity");
}

} // namespace ark
