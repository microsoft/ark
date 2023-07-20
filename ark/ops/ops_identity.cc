// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"

using namespace std;

namespace ark {

// Returns an identical tensor of `input` with execution dependencies `deps`.
Tensor *Model::identity(Tensor *input, const vector<Tensor *> &deps,
                        Tensor *output, const string &name)
{
    assert(input != nullptr);
    LOG(DEBUG, "identity ", input->shape);
    set<Tensor *> dep_set;
    dep_set.emplace(input);
    for (auto &dep : deps) {
        dep_set.emplace(dep);
    }
    vector<Tensor *> dep_vec;
    for (auto &dep : dep_set) {
        dep_vec.emplace_back(dep);
    }
    if (output != nullptr) {
        // TODO: copy data.
        LOGERR("identity: output is not supported yet");
    }
    return this->tensor(input->shape, input->type, input->buf, input->ldims,
                        input->offs, input->pads, dep_vec, input->exported,
                        input->imported, name + "/identity");
}

} // namespace ark
