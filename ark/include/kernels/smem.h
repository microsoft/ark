#ifndef ARK_KERNELS_SMEM_H_
#define ARK_KERNELS_SMEM_H_

#include "arch.h"
#include "static_math.h"

namespace ark {

#if defined(ARK_THREADS_PER_BLOCK)
template <typename T, int ThreadsNum> struct SharedMemory
{
    static const int SmemOffset = ThreadsNum * ark::Arch::MaxSmemBytesPerBlock /
                                  (ARK_THREADS_PER_BLOCK * sizeof(int));
    DEVICE operator T *()
    {
        return (
            T *)&_ARK_SMEM[(threadIdx.x >> math::log2_up<ThreadsNum>::value) *
                           SmemOffset];
    }
    DEVICE operator const T *() const
    {
        return (
            T *)&_ARK_SMEM[(threadIdx.x >> math::log2_up<ThreadsNum>::value) *
                           SmemOffset];
    }
};
#endif // defined(ARK_THREADS_PER_BLOCK)

} // namespace ark

#endif // ARK_KERNELS_SMEM_H_