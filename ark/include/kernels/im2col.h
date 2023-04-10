#ifndef ARK_KERNELS_IM2COL_H_
#define ARK_KERNELS_IM2COL_H_

#include "sync.h"
#include "transform.h"

namespace ark {

template <typename InShape, typename InLDims, unsigned int KernelHeight,
          unsigned int KernelWidth, unsigned int StrideHeight,
          unsigned int StrideWidth, unsigned int PadHeight,
          unsigned int PadWidth, unsigned int DilationHeight,
          unsigned int DilationWidth, int TN, int SB, int TDM, int TDN, int TDK>
struct TransformIm2Col
{
    static const unsigned int InN = InLDims::H * InLDims::W;

    static const unsigned int Height = InShape::H;
    static const unsigned int Width = InShape::W;

    static const unsigned int PatchNumHeight =
        (Height - KernelHeight + 2 * PadHeight) / StrideHeight + 1;
    static const unsigned int PatchNumWidth =
        (Width - KernelWidth + 2 * PadWidth) / StrideWidth + 1;
    static const unsigned int PatchNum =
        math::mul<unsigned int, PatchNumHeight, PatchNumWidth>::value;
    static const unsigned int OutHeight = math::pad<PatchNum, TDM>::value;

    static const unsigned int KHW = KernelHeight * KernelWidth;

    static const unsigned int MaxMIdx = PatchNum;
    static const unsigned int MaxNIdx = InShape::N * InShape::C * KHW;

    // Index of the input element is derived as follows:
    //   channel_idx = nidx / (KernelHeight*KernelWidth);
    //   per_channel_patch_idx = midx;
    //   per_channel_patch_pos_width
    //      = (per_channel_patch_idx % PatchNumWidth) * StrideWidth;
    //   per_channel_patch_pos_height
    //      = (per_channel_patch_idx / PatchNumWidth) * StrideHeight;
    //   per_patch_elem_idx = nidx % (KernelHeight*KernelWidth);
    //   per_patch_elem_pos_width = per_patch_elem_idx % KernelWidth;
    //   per_patch_elem_pos_height = per_patch_elem_idx / KernelWidth;
    //   elem_width =
    //      per_channel_patch_pos_width
    //      + per_patch_elem_pos_width - PadWidth;
    //   elem_height =
    //      per_channel_patch_pos_height
    //      + per_patch_elem_pos_height - PadHeight;
    //   elem_idx = elem_width + elem_height * Width + channel_idx * InN;
    //
    // with exception:
    //   if (elem_width < 0) or (elem_width > Width - 1) --> output is zero
    //   if (elem_height < 0) or (elem_height > Height - 1) --> output is zero
    //
    // This function reads a half value while avoiding 2-byte misaligned access.
    // Return the value as a float for efficiency.
    // CAUTION: This function assumes that `x` address is 4-byte aligned.
    static DEVICE float read_elem(__half *x, int midx, int nidx)
    {
        int elem_width = math::mod<PatchNumWidth>(midx) * StrideWidth +
                         math::mod<KernelWidth>(nidx) - PadWidth;
        int elem_height = math::div<PatchNumWidth>(midx) * StrideHeight +
                          math::div<KernelWidth>(math::mod<KHW>(nidx)) -
                          PadHeight;

        if (elem_height < 0 || elem_height >= Height || elem_width < 0 ||
            elem_width >= Width) {
            return 0.0f;
        }

        int idx = elem_width + elem_height * Width + math::div<KHW>(nidx) * InN;

        float2 fx = __half22float2(((__half2 *)x)[idx >> 1]);
        return ((float *)&fx)[idx & 1];
    }

    //
    static DEVICE __half2 compute(__half2 *x, int midx, int nidx)
    {
        float f1 = 0;
        float f2 = 0;
        if (nidx <= MaxNIdx) {
            if (midx <= MaxMIdx) {
                f1 = read_elem((__half *)x, midx, nidx);
            }
            if (midx + 1 <= MaxMIdx) {
                f2 = read_elem((__half *)x, midx + 1, nidx);
            }
        }
        __syncwarp();
        return __floats2half2_rn(f1, f2);
    }
};

// Half-precision image to column operation.
// TODO: support dilation.
template <typename InShape, typename InLDims, unsigned int KernelHeight,
          unsigned int KernelWidth, unsigned int StrideHeight,
          unsigned int StrideWidth, unsigned int PadHeight,
          unsigned int PadWidth, unsigned int DilationHeight,
          unsigned int DilationWidth, int TN, int SB, int TDM, int TDN, int TDK>
DEVICE void im2col(ark::half *y, ark::half *x, int tx, int ty, int tz)
{
    using TransformIm2Col =
        TransformIm2Col<InShape, InLDims, KernelHeight, KernelWidth,
                        StrideHeight, StrideWidth, PadHeight, PadWidth,
                        DilationHeight, DilationWidth, TN, SB, TDM, TDN, TDK>;
    Transform<TransformIm2Col, TransformIm2Col::OutHeight, -1, -1, TN, SB, TDM,
              TDN, TDK>::run(y, x, tx, ty, tz);
    sync_warps<TN>();
}

} // namespace ark

#endif // ARK_KERNELS_IM2COL_H_
