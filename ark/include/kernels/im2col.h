// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_IM2COL_H_
#define ARK_KERNELS_IM2COL_H_

#include "ewise.h"
#include "sync.h"

namespace ark {

template <typename _InShape, typename _InDims, typename _OutDims,
          typename _UnitOutDims, typename _DataType, int _NelemPerThread,
          int KernelHeight, int KernelWidth, int StrideHeight, int StrideWidth,
          int PadHeight, int PadWidth, int DilationHeight, int DilationWidth>
struct Im2Col;

template <typename _InShape, typename _InDims, typename _OutDims,
          typename _UnitOutDims, int KernelHeight, int KernelWidth,
          int StrideHeight, int StrideWidth, int PadHeight, int PadWidth,
          int DilationHeight, int DilationWidth>
struct Im2Col<_InShape, _InDims, _OutDims, _UnitOutDims, fp16, 2, KernelHeight,
              KernelWidth, StrideHeight, StrideWidth, PadHeight, PadWidth,
              DilationHeight, DilationWidth> {
    using InDims = _InDims;
    using OutDims = _OutDims;
    using DataType = fp16;
    static const int NelemPerThread = 2;

    static const int InN = InDims::HW;

    static const int Height = _InShape::H;
    static const int Width = _InShape::W;

    static const int PatchNumHeight =
        (Height - KernelHeight + 2 * PadHeight) / StrideHeight + 1;
    static const int PatchNumWidth =
        (Width - KernelWidth + 2 * PadWidth) / StrideWidth + 1;
    static const int PatchNum = math::mul<PatchNumHeight, PatchNumWidth>::value;
    static const int OutHeight = math::pad<PatchNum, _UnitOutDims::H>::value;

    static const int KHW = math::mul<KernelHeight, KernelWidth>::value;

    static const int MaxMIdx = PatchNum;
    static const int MaxNIdx = math::mul<_InShape::NC, KHW>::value;

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
    static DEVICE float read_elem(fp16 *x, int midx, int nidx) {
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

        float2 fx = __half22float2(((fp16x2 *)x)[idx >> 1]);
        return ((float *)&fx)[idx & 1];
    }

    static DEVICE void compute(fp16 *out, fp16 *in, int idx_n, int idx_c,
                               int idx_h, int idx_w) {
        out += idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W +
               idx_w;

        int midx = idx_w;
        int nidx = idx_h + idx_c * OutDims::H + idx_n * OutDims::CH;

        float f1 = 0;
        float f2 = 0;
        if (nidx <= MaxNIdx) {
            if (midx <= MaxMIdx) {
                f1 = read_elem(in, midx, nidx);
            }
            if (midx + 1 <= MaxMIdx) {
                f2 = read_elem(in, midx + 1, nidx);
            }
        }
        sync_warps<Arch::ThreadsPerWarp>();
        *(fp16x2 *)out = __floats2half2_rn(f1, f2);
    }
};

// Half-precision image to column operation.
// TODO: support dilation.
template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes, int KernelHeight, int KernelWidth, int StrideHeight,
          int StrideWidth, int PadHeight, int PadWidth, int DilationHeight,
          int DilationWidth>
DEVICE void im2col(fp16 *y, fp16 *x, int uop_idx, int) {
    Ewise1<OutDims, OutShape, UnitOutDims, NumThreads, SmemBytes,
           Im2Col<InShape, InDims, OutDims, UnitOutDims, fp16, 2, KernelHeight,
                  KernelWidth, StrideHeight, StrideWidth, PadHeight, PadWidth,
                  DilationHeight, DilationWidth>>::run(y, x, uop_idx);
    sync_warps<NumThreads>();
}

}  // namespace ark

#endif  // ARK_KERNELS_IM2COL_H_
