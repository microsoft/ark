// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_VECTOR_TYPE_H_
#define ARK_KERNELS_VECTOR_TYPE_H_

#include <type_traits>

namespace ark {
namespace type {

template <typename DataType, int Size>
struct Vtype {};

template <typename DataType, int Size, typename = void>
struct VtypeExists : std::false_type {};

template <typename DataType, int Size>
struct VtypeExists<DataType, Size, std::void_t<typename Vtype<DataType, Size>::type>> : std::true_type {
};

template <typename T, typename = void>
struct IsBuiltinVector2 : std::false_type {};

template <typename T, typename = void>
struct IsBuiltinVector4 : std::false_type {};

template <typename T>
struct IsBuiltinVector2<
    T, std::void_t<decltype(std::declval<T>().x, std::declval<T>().y)>>
    : std::true_type {};

template <typename T>
struct IsBuiltinVector4<
    T, std::void_t<decltype(std::declval<T>().x, std::declval<T>().y,
                            std::declval<T>().z, std::declval<T>().w)>>
    : std::true_type {};

}  // namespace type
}  // namespace ark

#endif  // ARK_KERNELS_VECTOR_TYPE_H_
