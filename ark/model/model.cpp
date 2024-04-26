// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/model.hpp"

namespace ark {

Model Model::compress() const {
    Model model(*this);
    model.compress_nodes();
    return model;
}

}  // namespace ark
