// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/logging.h"
#include "ark/model_io.h"

using namespace std;

namespace ark {

Tensor *Model::linear(Tensor *input, DimType out_features, bool bias,
                      Tensor *output, DimType splitk, bool is_relu,
                      const string &name, int gran_lev)
{
    assert(input != nullptr);
    if ((input->type != FP16) && (input->type != FP32)) {
        LOGERR("unsupported input data type: ", type_str(input->type));
    }
    if ((output != nullptr) && (input->type != output->type)) {
        LOGERR("invalid output data type: ", type_str(output->type));
    }
    const Dims &is = input->shape;
    assert(is[2] == 1);
    DimType in_dim = is[3];
    Tensor *weight = this->tensor({1, out_features, 1, in_dim}, input->type);
    Tensor *ret = this->matmul(input, weight, output, splitk, false, true,
                               is_relu, name + "/matmul", gran_lev);
    return ret;
}

} // namespace ark
