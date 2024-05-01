// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_COMMON_HPP_
#define ARK_OPS_COMMON_HPP_

#include <memory>

#include "ark/model.hpp"
#include "logging.h"
#include "model/model_data_type.hpp"
#include "model/model_graph_impl.hpp"
#include "model/model_op.hpp"
#include "model/model_tensor.hpp"

namespace ark {

void check_none(Tensor tensor);

void check_match_data_type(Tensor t, ModelDataType dt);

void check_match_data_type(Tensor a, Tensor b);

void check_match_shape(Tensor a, Tensor b);

void check_match_shape(Tensor tensor, const Dims &shape);

/// Return the output shape of broadcasting between two shapes.
/// Follow NumPy rules.
/// https://numpy.org/doc/stable/user/basics.broadcasting.html
/// @param dims1 The first shape.
/// @param dims2 The second shape.
Dims broadcast_shape(const Dims &dims1, const Dims &dims2);

void check_broadcast_shape(Tensor from, Tensor to);

std::string tolower(const std::string &str);

}  // namespace ark

#endif  // ARK_OPS_COMMON_HPP_
