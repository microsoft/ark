// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_common.hpp"
#include "utils/utils_math.hpp"

namespace ark {

// Shard `input` along `axis` into `dim_per_shard`-dimensional shards.
std::vector<Tensor> Model::sharding(Tensor input, DimType axis,
                                    DimType dim_per_shard,
                                    const std::string &name) {
    if (axis >= DIMS_LEN) {
        ERR(InvalidUsageError, "invlaid axis value: ", axis);
    }
    if ((input.shape()[axis] % dim_per_shard) != 0) {
        // If the total dimension is not divided by the per-shard size,
        // we need to check whether we can put a padding here.
        // If the padded dimension of the input tensor is smaller than
        // the leading dimension size, it means that the input tensor refers to
        // a part of a buffer -- in this case, we cannot put a padding because
        // the tensor has adjacent data.
        DimType pdim = math::pad(input.shape()[axis], input.pads()[axis]);
        if (pdim < input.strides()[axis]) {
            ERR(InvalidUsageError, "the dimension of axis ", axis, " (",
                input.shape()[axis],
                ") is not divided by the dimension per shard (", dim_per_shard,
                ") and this tensor cannot be padded.");
        }
    }
    std::vector<Tensor> shards;
    DimType num_shard = math::div_up(input.shape()[axis], dim_per_shard);
    Dims shard_shape = input.shape();
    Dims shard_offs = input.offsets();
    Dims shard_pads = input.pads();
    for (DimType i = 0; i < num_shard; ++i) {
        DimType dim;
        if (i == (num_shard - 1)) {
            dim = input.shape()[axis] - (i * dim_per_shard);
            shard_pads[axis] = input.pads()[axis];
        } else {
            dim = dim_per_shard;
            shard_pads[axis] = 1;
        }
        shard_shape[axis] = dim;
        Tensor shard =
            this->refer(input, shard_shape, input.strides(), shard_offs,
                        shard_pads, name + "/shard_" + std::to_string(i));
        shards.emplace_back(shard);
        shard_offs[axis] += dim;
    }
    return shards;
}

}  // namespace ark
