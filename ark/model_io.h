// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_IO_H_
#define ARK_MODEL_IO_H_

#include "ark/include/ark.h"
#include <iostream>

namespace ark {

std::ostream &operator<<(std::ostream &os, const Model &og);
std::istream &operator>>(std::istream &is, Model &og);

const std::string type_str(const TensorType &type);
const std::string op_str(const Op &op);

std::ostream &operator<<(std::ostream &os, const OpType &s);

} // namespace ark

#endif // ARK_MODEL_IO_H_