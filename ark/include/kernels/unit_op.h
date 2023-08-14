// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_UNIT_OP_H_
#define ARK_KERNELS_UNIT_OP_H_

#include "device.h"
#include "smem.h"
#include "static_math.h"
#include "sync.h"
#include "vec.h"

namespace ark {

/// Helper for defining unit operators. Follows the NCHW tensor layout.
///
/// @tparam _OutDims Shape of the actual output data layout. Each dimension
/// should be divided by the corresponding dimension in @ref _UnitOutDims.
/// @tparam _OutShape Shape of the output data that the entire operator is
/// operating on. This is usually the same as @ref _OutDims, but can be smaller
/// for operators that operate on a subset of the data.
/// @tparam _UnitOutDims Shape of the output data that a unit operator is
/// operating on. Shape of the output data that a unit operator is operating on.
/// Each dimension should be equal to or smaller than the corresponding
/// dimension in @ref _OutShape. Even if each dimension in @ref _OutShape is not
/// divided by the corresponding dimension in @ref _UnitOutDims, @ref UnitOp
/// provides no mechanism for informing unit operators whether they are
/// operating on the boundary of the data or not. Instead, it should be managed
/// by the underlying implementation.
/// @tparam _NumThreads Number of threads that a unit operator is using.
/// @tparam _SmemBytes Bytes of shared memory that a unit operator is using.
///
template <typename _OutDims, typename _OutShape, typename _UnitOutDims,
          int _NumThreads, int _SmemBytes>
struct UnitOp
{
    static_assert(_OutDims::N >= _OutShape::N,
                  "Dimension N is smaller than tensor shape");
    static_assert(_OutDims::C >= _OutShape::C,
                  "Dimension C is smaller than tensor shape");
    static_assert(_OutDims::H >= _OutShape::H,
                  "Dimension H is smaller than tensor shape");
    static_assert(_OutDims::W >= _OutShape::W,
                  "Dimension W is smaller than tensor shape");

    static_assert(_UnitOutDims::N > 0,
                  "Unit dimension is not positive in dimension N");
    static_assert(_UnitOutDims::C > 0,
                  "Unit dimension is not positive in dimension C");
    static_assert(_UnitOutDims::H > 0,
                  "Unit dimension is not positive in dimension H");
    static_assert(_UnitOutDims::W > 0,
                  "Unit dimension is not positive in dimension W");

    static_assert(_OutDims::N % _UnitOutDims::N == 0,
                  "Dimension N is not divisible by the unit dimension");
    static_assert(_OutDims::C % _UnitOutDims::C == 0,
                  "Dimension C is not divisible by the unit dimension");
    static_assert(_OutDims::H % _UnitOutDims::H == 0,
                  "Dimension H is not divisible by the unit dimension");
    static_assert(_OutDims::W % _UnitOutDims::W == 0,
                  "Dimension W is not divisible by the unit dimension");

    static_assert(_NumThreads > 0, "# of threads is not positive");
    static_assert(_SmemBytes >= 0, "Bytes of shared memory is negative");

    // Number of unit operators in each dimension.
    using UnitOpDims =
        Vec<_OutDims::N / _UnitOutDims::N, _OutDims::C / _UnitOutDims::C,
            _OutDims::H / _UnitOutDims::H, _OutDims::W / _UnitOutDims::W>;

    static const int NumThreads = _NumThreads;
    static const int SmemBytes = _SmemBytes;

    /// Do not use `threadIdx` and use this function instead.
    static DEVICE int thread_id()
    {
        return math::mod<NumThreads>(threadIdx.x);
    }

    /// Convert a unit operator ID to the corresponding index along the N
    /// dimension.
    /// @param uop_id Unit operator ID.
    static DEVICE int uop_idx_n(int uop_id)
    {
        return uop_id / UnitOpDims::CHW;
    }

    /// Convert a unit operator ID to the corresponding index along the C
    /// dimension.
    /// @param uop_id Unit operator ID.
    static DEVICE int uop_idx_c(int uop_id)
    {
        return (uop_id / UnitOpDims::HW) % UnitOpDims::C;
    }

    /// Convert a unit operator ID to the corresponding index along the H
    /// dimension.
    /// @param uop_id Unit operator ID.
    static DEVICE int uop_idx_h(int uop_id)
    {
        return (uop_id / UnitOpDims::W) % UnitOpDims::H;
    }

    /// Convert a unit operator ID to the corresponding index along the W
    /// dimension.
    /// @param uop_id Unit operator ID.
    static DEVICE int uop_idx_w(int uop_id)
    {
        return uop_id % UnitOpDims::W;
    }

    /// Return a shared memory pointer.
    /// @tparam T Type of the underlying data.
    /// @param smem_per_warp Bytes of shared memory per warp.
    template <typename T> static DEVICE T *shared_memory(int smem_per_warp)
    {
        static_assert(sizeof(T) <= SmemBytes,
                      "Shared memory is not large enough");
        return SharedMemory<T, NumThreads>::get(smem_per_warp);
    }

    /// Do not use `__syncthreads()` and use this function instead.
    static DEVICE void sync_threads()
    {
        sync_warps<NumThreads>();
    }
};

} // namespace ark

#endif // ARK_KERNELS_UNIT_OP_H_
