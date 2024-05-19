// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_common.hpp"

namespace ark {

// Shard `input` along `axis` into `dim_per_shard`-dimensional shards.
std::vector<Tensor> Model::sharding(Tensor input, DimType axis,
                                    DimType dim_per_shard,
                                    const std::string &name) {
    if (axis >= DIMS_LEN) {
        ERR(InvalidUsageError, "invlaid axis value: ", axis);
    }
    if ((input.shape()[axis] % dim_per_shard) != 0) {
        ERR(InvalidUsageError, "dimension length of axis ", axis, " (",
            input.shape()[axis],
            ") is not divided by the dimension per shard (", dim_per_shard,
            ").");
    }
    std::vector<Tensor> shards;
    DimType num_shard = input.shape()[axis] / dim_per_shard;
    Dims shard_shape = input.shape();
    Dims shard_offs = input.offsets();
    for (DimType i = 0; i < num_shard; ++i) {
        shard_shape[axis] = dim_per_shard;
        Tensor shard =
            this->refer(input, shard_shape, input.strides(), shard_offs, {},
                        name + "/shard_" + std::to_string(i));
        shards.emplace_back(shard);
        shard_offs[axis] += dim_per_shard;
    }
    return shards;
}

}  // namespace ark
