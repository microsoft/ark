// Simple mixed-precision matrix multiplication (C = A x B).
// Assume `A` is in column-major and `B` is in row-major.
// CAUTION: `m` and `n` should be even numbers!
#include "ark_kernels.h"
extern "C" __global__ void simple_matmul_nt(ark::half *C, ark::half *A,
                                            ark::half *B, unsigned int m,
                                            unsigned int n, unsigned int k,
                                            bool is_relu)
{
    unsigned int coldiv2 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int rowdiv2 = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int mdiv2 = m >> 1;
    unsigned int ndiv2 = n >> 1;
    unsigned int col = coldiv2 << 1;
    unsigned int row = rowdiv2 << 1;
    if (col < n && row < m) {
        __half2 *pA = (__half2 *)A;
        __half2 *pB = (__half2 *)B;
        __half2 *pC = (__half2 *)C;
#if 0
        float2 s2_0;
        float2 s2_1;
        s2_0.x = 0;
        s2_0.y = 0;
        s2_1.x = 0;
        s2_1.y = 0;
        for (unsigned int i = 0; i < k; ++i) {
            // (2x1) x (1x2) = (2x2)
            float2 a2 = __half22float2(pA[i * mdiv2 + rowdiv2]);
            float2 b2 = __half22float2(pB[i * ndiv2 + coldiv2]);
            s2_0.x += a2.x * b2.x;
            s2_0.y += a2.y * b2.x;
            s2_1.x += a2.x * b2.y;
            s2_1.y += a2.y * b2.y;
        }
        pC[col * mdiv2 + rowdiv2] = __float22half2_rn(s2_0);
        pC[col * mdiv2 + mdiv2 + rowdiv2] = __float22half2_rn(s2_1);
#else
        __half2 s2_0 = __half2half2((__half)0x0);
        __half2 s2_1 = __half2half2((__half)0x0);
        for (unsigned int i = 0; i < k; ++i) {
            // __half2 lb = __low2half2(pB[i * ndiv2 + coldiv2]);
            // __half2 hb = __high2half2(pB[i * ndiv2 + coldiv2]);
            // s2_0 += pA[i * mdiv2 + rowdiv2] * lb;
            // s2_1 += pA[i * mdiv2 + rowdiv2] * hb;
            __half a2x = __low2half(pA[i * mdiv2 + rowdiv2]);
            __half a2y = __high2half(pA[i * mdiv2 + rowdiv2]);
            __half b2x = __low2half(pB[i * ndiv2 + coldiv2]);
            __half b2y = __high2half(pB[i * ndiv2 + coldiv2]);
            s2_0 = __hadd2(s2_0, __halves2half2(a2x * b2x, a2y * b2x));
            s2_1 = __hadd2(s2_1, __halves2half2(a2x * b2y, a2y * b2y));
        }
        if (is_relu) {
            pC[col * mdiv2 + rowdiv2] = s2_0;
            pC[col * mdiv2 + mdiv2 + rowdiv2] = s2_1;
        } else {
            float2 fs2_0 = __half22float2(s2_0);
            float2 fs2_1 = __half22float2(s2_1);
            pC[col * mdiv2 + rowdiv2] =
                __floats2half2_rn(fmaxf(fs2_0.x, 0.0f), fmaxf(fs2_0.y, 0.0f));
            pC[col * mdiv2 + mdiv2 + rowdiv2] =
                __floats2half2_rn(fmaxf(fs2_1.x, 0.0f), fmaxf(fs2_1.y, 0.0f));
        }
#endif
    }
}
