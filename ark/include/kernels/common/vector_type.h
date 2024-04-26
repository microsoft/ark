// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_VECTOR_TYPE_H_
#define ARK_KERNELS_VECTOR_TYPE_H_

#include <type_traits>

#include "static_math.h"

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

template <typename DataType>
struct Constant {
    static DEVICE DataType zero() { return DataType(0); }
    static DEVICE DataType lowest() { return DataType(0); }
};

}  // namespace type

template <typename IntrinsicType, typename InputVtype>
struct IntrinsicCompute1Exists {
    template <typename U>
    static auto test(const InputVtype &)
        -> decltype(&U::compute, std::true_type{});

    template <typename>
    static auto test(...) -> std::false_type;

    static constexpr bool value = decltype(test<IntrinsicType>(
        type::Constant<InputVtype>::zero()))::value;
};

template <typename IntrinsicType, typename InputVtype>
struct IntrinsicCompute2Exists {
    template <typename U>
    static auto test(const InputVtype &, const InputVtype &)
        -> decltype(&U::compute, std::true_type{});

    template <typename>
    static auto test(...) -> std::false_type;

    static constexpr bool value = decltype(test<IntrinsicType>(
        type::Constant<InputVtype>::zero(),
        type::Constant<InputVtype>::zero()))::value;
};

template <typename IntrinsicType, typename InputType, int CurrentVal = 256>
struct IntrinsicCompute1VtypeMaxSize {
    constexpr static int VtypeExists =
        type::VtypeExists<InputType, CurrentVal>::value;
    constexpr static int ICExists =
        VtypeExists
            ? IntrinsicCompute1Exists<
                  IntrinsicType,
                  typename type::Vtype<InputType, CurrentVal>::type>::value
            : 0;
    enum {
        value = ICExists
                    ? CurrentVal
                    : IntrinsicCompute1VtypeMaxSize<IntrinsicType, InputType,
                                                    CurrentVal / 2>::value
    };
};

template <typename IntrinsicType, typename InputType>
struct IntrinsicCompute1VtypeMaxSize<IntrinsicType, InputType, 1> {
    enum { value = 1 };
};

template <typename IntrinsicType, typename InputType, int CurrentVal = 256>
struct IntrinsicCompute2VtypeMaxSize {
    constexpr static int VtypeExists =
        type::VtypeExists<InputType, CurrentVal>::value;
    constexpr static int ICExists =
        VtypeExists
            ? IntrinsicCompute2Exists<
                  IntrinsicType,
                  typename type::Vtype<InputType, CurrentVal>::type>::value
            : 0;
    enum {
        value = ICExists
                    ? CurrentVal
                    : IntrinsicCompute2VtypeMaxSize<IntrinsicType, InputType,
                                                    CurrentVal / 2>::value
    };
};

template <typename IntrinsicType, typename InputType>
struct IntrinsicCompute2VtypeMaxSize<IntrinsicType, InputType, 1> {
    enum { value = 1 };
};

template <int NumElem, typename ElemIntrinsic>
struct VectorCompute {
    template <typename OutputType, typename InputType>
    static DEVICE void compute(OutputType *out, const InputType *in) {
        constexpr int TmpMin =
            math::min<type::VtypeMaxSize<OutputType>::value, NumElem>::value;
        constexpr int VtypeSize =
            math::min<TmpMin,
                      IntrinsicCompute1VtypeMaxSize<ElemIntrinsic, InputType,
                                                    TmpMin>::value>::value;
        using OutputVtype = typename type::Vtype<OutputType, VtypeSize>::type;
        using InputVtype = typename type::Vtype<InputType, VtypeSize>::type;
        constexpr int NumVtype = NumElem / VtypeSize;

        static_assert(NumElem % VtypeSize == 0,
                      "NumElem must be divisible by VtypeSize");
        OutputVtype *out_vtype = reinterpret_cast<OutputVtype *>(out);
        const InputVtype *in_vtype = reinterpret_cast<const InputVtype *>(in);
#pragma unroll
        for (int i = 0; i < NumVtype; ++i) {
            out_vtype[i] = ElemIntrinsic::compute(in_vtype[i]);
        }
    }

    template <typename OutputType, typename InputType>
    static DEVICE void compute(OutputType *out, const InputType *in0,
                               const InputType *in1) {
        constexpr int TmpMin =
            math::min<type::VtypeMaxSize<OutputType>::value, NumElem>::value;
        constexpr int VtypeSize =
            math::min<TmpMin,
                      IntrinsicCompute2VtypeMaxSize<ElemIntrinsic, InputType,
                                                    TmpMin>::value>::value;
        using OutputVtype = typename type::Vtype<OutputType, VtypeSize>::type;
        using InputVtype = typename type::Vtype<InputType, VtypeSize>::type;
        constexpr int NumVtype = NumElem / VtypeSize;

        static_assert(NumElem % VtypeSize == 0,
                      "NumElem must be divisible by VtypeSize");
        OutputVtype *out_vtype = reinterpret_cast<OutputVtype *>(out);
        const InputVtype *in0_vtype = reinterpret_cast<const InputVtype *>(in0);
        const InputVtype *in1_vtype = reinterpret_cast<const InputVtype *>(in1);
#pragma unroll
        for (int i = 0; i < NumVtype; ++i) {
            out_vtype[i] = ElemIntrinsic::compute(in0_vtype[i], in1_vtype[i]);
        }
    }
};

template <typename OutDims, typename OutDataType, typename UnitOutDims>
struct DefaultNelemPerThread {
    static constexpr int ConsecutiveDimLen =
        (OutDims::W == 1 && UnitOutDims::W == 1)
            ? math::min<OutDims::H, UnitOutDims::H>::value
            : math::min<OutDims::W, UnitOutDims::W>::value;

    static const int value =
        (sizeof(OutDataType) <= 2 && ConsecutiveDimLen % 8 == 0) ? 8
        : (ConsecutiveDimLen % 4 == 0)                           ? 4
        : (ConsecutiveDimLen % 2 == 0)                           ? 2
                                                                 : 1;
};

}  // namespace ark

#endif  // ARK_KERNELS_VECTOR_TYPE_H_
