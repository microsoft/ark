// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_UNIT_OP_H_
#define ARK_KERNELS_UNIT_OP_H_

#include "common.h"
#include "smem.h"
#include "static_math.h"

namespace ark {

// Helper for defining unit operators. Follows the NCHW tensor layout.
//
// OutDims(ark::Vec):
//      Shape of the actual output data layout.
//
// OutShape(ark::Vec):
//      Shape of the output data that the entire operator is operating on.
//      This is usually the same as OutDims, but can be smaller for operators
//      that operate on a subset of the data.
//
// UnitOutShape(ark::Vec):
//      Shape of the output data that a unit operator is operating on.
//      Each dimension should be equal to or smaller than the corresponding
//      dimension in OutShape.
//      Even if each dimension in OutShape is not divided by the corresponding
//      dimension in UnitOutShape, UnitOp provides no mechanism for informing
//      unit operators whether they are operating on the boundary of the data
//      or not. Instead, it should be managed by the underlying implementation.
//
// ThreadsNum(int): Number of threads that a unit operator is using.
//
// SmemBytes(int):  Bytes of shared memory that a unit operator is using.
template <typename OutDims, typename OutShape, typename UnitOutShape,
          int ThreadsNum, int SmemBytes>
struct UnitOp
{
    static_assert(OutDims::N >= OutShape::N,
                  "Dimension N is smaller than tensor shape");
    static_assert(OutDims::C >= OutShape::C,
                  "Dimension C is smaller than tensor shape");
    static_assert(OutDims::H >= OutShape::H,
                  "Dimension H is smaller than tensor shape");
    static_assert(OutDims::W >= OutShape::W,
                  "Dimension W is smaller than tensor shape");

    static_assert(OutDims::N >= UnitOutShape::N,
                  "Shape is not divisible by unit shape in dimension N");
    static_assert(OutDims::C >= UnitOutShape::C,
                  "Shape is not divisible by unit shape in dimension C");
    static_assert(OutDims::H >= UnitOutShape::H,
                  "Shape is not divisible by unit shape in dimension H");
    static_assert(OutDims::W >= UnitOutShape::W,
                  "Shape is not divisible by unit shape in dimension W");

    static_assert(UnitOutShape::N > 0,
                  "Unit shape is not positive in dimension N");
    static_assert(UnitOutShape::C > 0,
                  "Unit shape is not positive in dimension C");
    static_assert(UnitOutShape::H > 0,
                  "Unit shape is not positive in dimension H");
    static_assert(UnitOutShape::W > 0,
                  "Unit shape is not positive in dimension W");

    static_assert(ThreadsNum > 0, "# of threads is not positive");
    static_assert(SmemBytes >= 0, "Bytes of shared memory is negative");

    static const int ThreadsNum = ThreadsNum;
    static const int SmemBytes = SmemBytes;

    // Do not use `threadIdx` and use this function instead.
    static DEVICE int thread_id()
    {
        return math::mod<ThreadsNum>(threadIdx.x);
    }

    // Return a shared memory pointer.
    template <typename T> static DEVICE T *shared_memory()
    {
        static_assert(sizeof(T) <= SmemBytes,
                      "Shared memory is not large enough");
        return SharedMemory<T, ThreadsNum>();
    }
};

} // namespace ark

#endif // ARK_KERNELS_UNIT_OP_H_
