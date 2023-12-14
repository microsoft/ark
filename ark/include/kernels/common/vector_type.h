// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_VECTOR_TYPE_H_
#define ARK_KERNELS_VECTOR_TYPE_H_

#include <type_traits>

namespace ark {
namespace type {

template <typename DataType, int Size>
struct Vtype {};

template <typename Vtype, typename = void>
struct VtypeExists : std::false_type {};

template <typename Vtype>
struct VtypeExists<Vtype, std::void_t<typename Vtype::type>> : std::true_type {
};

}  // namespace type
}  // namespace ark

#endif  // ARK_KERNELS_VECTOR_TYPE_H_
