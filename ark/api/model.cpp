// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/model.hpp"

#include <limits>

#include "logging.hpp"

namespace ark {

Model::Model(int rank, int world_size) : ModelGraph(rank, world_size) {
    static size_t next_id = 0;
    id_ = next_id++;
}

Model::Model(const Model &other) : ModelGraph(other), id_(other.id()) {}

size_t Model::id() const { return id_; }

Model Model::compress() const {
    Model model(*this);
    model.compress_nodes();
    return model;
}

int Model::unique_tag() {
    size_t num_ints = size_t(std::numeric_limits<int>::max()) * 2 + 2;
    if (tags_.size() == num_ints) {
        ERR(ModelError, "no more unique tags");
    }
    int next_val;
    if (tags_.empty()) {
        next_val = std::numeric_limits<int>::min();
    } else if (*tags_.rbegin() < std::numeric_limits<int>::max()) {
        next_val = *tags_.rbegin() + 1;
    } else {
        next_val = std::numeric_limits<int>::min();
        for (int tag : tags_) {
            if (tag == next_val) {
                next_val++;
            } else {
                break;
            }
        }
    }
    tags_.insert(next_val);
    return next_val;
}

}  // namespace ark
