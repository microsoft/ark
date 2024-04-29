// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_CAST_H_
#define ARK_KERNELS_CAST_H_

#include "common/broadcast.h"
#include "common/type_intrinsics.h"
#include "common/vector_type.h"

namespace ark {

template <typename _InShape, typename _FromType, typename _ToType,
          int _NelemPerThread>
struct Cast;

template <typename _InShape, typename _FromType, typename _ToType>
struct Cast<_InShape, _FromType, _ToType, 2> {
    using InputType = _FromType;
    using OutputType = _ToType;
    static const int NelemPerThread = 2;

    static DEVICE void compute(_ToType *output, const _FromType *input) {
        if constexpr (_InShape::W == 1) {
            *output = type::Cast::compute<_ToType>(*input);
        } else if constexpr (type::VtypeExists<_FromType, 2>::value &&
                             type::VtypeExists<_ToType, 2>::value) {
            using ToType2 = typename type::Vtype<_ToType, 2>::type;
            using FromType2 = typename type::Vtype<_FromType, 2>::type;
            ToType2 *pout = reinterpret_cast<ToType2 *>(output);
            const FromType2 *pin = reinterpret_cast<const FromType2 *>(input);
            *pout = type::Cast::compute<ToType2>(*pin);
        } else {
            output[0] = type::Cast::compute<_ToType>(input[0]);
            output[1] = type::Cast::compute<_ToType>(input[1]);
        }
    }
};

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename FromType, typename ToType>
DEVICE void cast(ToType *out, FromType *in, int uop_idx, int) {
    Broadcast1<InDims, InShape, OutDims, OutShape, UnitOutDims, NumWarps,
               SmemBytes, Cast<InShape, FromType, ToType, 2>>::run(out, in,
                                                                   uop_idx);
}

}  // namespace ark

#endif  // ARK_KERNELS_CAST_H_
