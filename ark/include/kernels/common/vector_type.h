// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_VECTOR_TYPE_H_
#define ARK_KERNELS_VECTOR_TYPE_H_

#include <type_traits>

namespace ark {
namespace type {

template <typename DataType, int Size>
struct Vtype {
    using type = void;
};

template <typename DataType>
struct Vtype<DataType, 1> {
    using type = DataType;
};

template <typename DataType, int Size>
struct VtypeExists {
    enum {
        value =
            std::is_same_v<typename Vtype<DataType, Size>::type, void> ? 0 : 1
    };
};

template <typename DataType, int CurrentVal = 256>
struct VtypeMaxSize {
    enum {
        value = VtypeExists<DataType, CurrentVal>::value
                    ? CurrentVal
                    : VtypeMaxSize<DataType, CurrentVal / 2>::value
    };
};

template <typename DataType>
struct VtypeMaxSize<DataType, 0> {
    enum { value = 0 };
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
