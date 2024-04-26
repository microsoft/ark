// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_common.hpp"

#include <algorithm>
#include <cassert>
#include <ostream>
#include <string>

#include "logging.h"

namespace ark {

void check_match_data_type(ModelTensorRef a, ModelTensorRef b) {
    if (a->data_type() != b->data_type()) {
        ERR(InvalidUsageError,
            "data types mismatch: ", a->data_type()->type_name(), ", ",
            b->data_type()->type_name());
    }
}

void check_match_shape(ModelTensorRef a, ModelTensorRef b) {
    if (a->shape() != b->shape()) {
        ERR(InvalidUsageError, "shapes mismatch: ", a->shape(), ", ",
            b->shape());
    }
}

void check_shape(ModelTensorRef tensor, const Dims &shape) {
    if (tensor->shape() != shape) {
        ERR(InvalidUsageError, "shape mismatch: ", tensor->shape(), " and ",
            shape);
    }
}

Dims broadcast_shape(const Dims &dims1, const Dims &dims2) {
    std::vector<DimType> output_dims_reversed;
    int ndims = std::max(dims1.ndims(), dims2.ndims());
    for (int i = 1; i < ndims + 1; ++i) {
        int d1 = (i - 1 < dims1.ndims()) ? dims1[-i] : 1;
        int d2 = (i - 1 < dims2.ndims()) ? dims2[-i] : 1;
        if (d1 == d2) {
            output_dims_reversed.push_back(d1);
        } else if (d1 == 1) {
            output_dims_reversed.push_back(d2);
        } else if (d2 == 1) {
            output_dims_reversed.push_back(d1);
        } else {
            ERR(InvalidUsageError,
                "input and other cannot be broadcasted: ", dims1, ", ", dims2);
        }
    }
    std::reverse(output_dims_reversed.begin(), output_dims_reversed.end());
    return Dims{output_dims_reversed};
}

std::string tolower(const std::string &str) {
    std::string ret = str;
    std::transform(ret.begin(), ret.end(), ret.begin(), ::tolower);
    return ret;
}

}  // namespace ark
