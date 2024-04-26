// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_NAMED_TYPE_HPP_
#define ARK_NAMED_TYPE_HPP_

#include <string>

namespace ark {

class NamedT {
   public:
    NamedT(const std::string &type_name) : type_name_(type_name) {}
    NamedT &operator=(const NamedT &) = default;

    const std::string &type_name() const { return type_name_; }

   private:
    std::string type_name_;
};

bool operator==(const NamedT &lhs, const NamedT &rhs);

}  // namespace ark

#endif  // ARK_NAMED_TYPE_HPP_
