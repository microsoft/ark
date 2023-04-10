#ifndef ARK_KERNELS_BASE_OP_H_
#define ARK_KERNELS_BASE_OP_H_

#include "static_math.h"

namespace ark {

template <int LeadingDim, int ThreadsNum, int SmemBytes, int TDimM, int TDimN,
          int TDimK>
struct BaseOp
{
    static DEVICE int thread_id()
    {
        return math::mod<ThreadsNum>(threadIdx.x);
    }
    static DEVICE int tile_offset(int tx, int ty)
    {
        constexpr int NumM = LeadingDim * TDimN;
        return NumM * tx + TDimM * ty;
    }
    static DEVICE int elem_offset(int thread_id)
    {
        return math::div<TDimM>(thread_id) * LeadingDim +
               math::mod<TDimM>(thread_id);
    }
    static DEVICE int midx(int ty, int thread_id)
    {
        return TDimM * ty + math::mod<TDimM>(thread_id);
    }
    static DEVICE int nidx(int tx, int thread_id)
    {
        return TDimN * tx + math::div<TDimM>(thread_id);
    }
    static DEVICE int offset(int tx, int ty, int thread_id)
    {
        // Same as LeadingDim * nidx(tx, thread_id) + midx(ty, thread_id).
        return tile_offset(tx, ty) + elem_offset(thread_id);
    }
};

} // namespace ark

#endif // ARK_KERNELS_BASE_OP_H_
