// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_TRANSPOSE_H_
#define ARK_KERNELS_TRANSPOSE_H_

#include "ewise.h"

namespace ark {

template <typename InDims, typename OutDims, typename DataType,
          int NelemPerThread>
struct Transpose0132
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        out += idx_n * OutDims::C * OutDims::H * OutDims::W +
               idx_c * OutDims::H * OutDims::W + idx_h * OutDims::W + idx_w;
        //
        in += idx_n * InDims::C * InDims::H * InDims::W +
              idx_c * InDims::H * InDims::W + idx_w * InDims::W + idx_h;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::W];
        }
    }
};

template <typename InDims, typename OutDims, typename DataType,
          int NelemPerThread>
struct Transpose0213
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        out += idx_n * OutDims::C * OutDims::H * OutDims::W +
               idx_c * OutDims::H * OutDims::W + idx_h * OutDims::W + idx_w;
        //
        in += idx_n * InDims::C * InDims::H * InDims::W +
              idx_h * InDims::H * InDims::W + idx_c * InDims::W + idx_w;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i];
        }
    }
};

template <typename InDims, typename OutDims, typename DataType,
          int NelemPerThread>
struct Transpose0231
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        out += idx_n * OutDims::C * OutDims::H * OutDims::W +
               idx_c * OutDims::H * OutDims::W + idx_h * OutDims::W + idx_w;
        //
        in += idx_n * InDims::C * InDims::H * InDims::W +
              idx_h * InDims::H * InDims::W + idx_w * InDims::W + idx_c;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::W];
        }
    }
};

template <typename InDims, typename OutDims, typename DataType,
          int NelemPerThread>
struct Transpose0312
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        out += idx_n * OutDims::C * OutDims::H * OutDims::W +
               idx_c * OutDims::H * OutDims::W + idx_h * OutDims::W + idx_w;
        //
        in += idx_n * InDims::C * InDims::H * InDims::W +
              idx_w * InDims::H * InDims::W + idx_c * InDims::W + idx_h;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::H * InDims::W];
        }
    }
};

template <typename InDims, typename OutDims, typename DataType,
          int NelemPerThread>
struct Transpose0321
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        out += idx_n * OutDims::C * OutDims::H * OutDims::W +
               idx_c * OutDims::H * OutDims::W + idx_h * OutDims::W + idx_w;
        //
        in += idx_n * InDims::C * InDims::H * InDims::W +
              idx_w * InDims::H * InDims::W + idx_h * InDims::W + idx_c;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::H * InDims::W];
        }
    }
};

template <typename InDims, typename OutDims, typename DataType,
          int NelemPerThread>
struct Transpose1023
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        out += idx_n * OutDims::C * OutDims::H * OutDims::W +
               idx_c * OutDims::H * OutDims::W + idx_h * OutDims::W + idx_w;
        //
        in += idx_c * InDims::C * InDims::H * InDims::W +
              idx_n * InDims::H * InDims::W + idx_h * InDims::W + idx_w;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i];
        }
    }
};

template <typename InDims, typename OutDims, typename DataType,
          int NelemPerThread>
struct Transpose1032
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        out += idx_n * OutDims::C * OutDims::H * OutDims::W +
               idx_c * OutDims::H * OutDims::W + idx_h * OutDims::W + idx_w;
        //
        in += idx_c * InDims::C * InDims::H * InDims::W +
              idx_n * InDims::H * InDims::W + idx_w * InDims::W + idx_h;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::W];
        }
    }
};

template <typename InDims, typename OutDims, typename DataType,
          int NelemPerThread>
struct Transpose1203
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        out += idx_n * OutDims::C * OutDims::H * OutDims::W +
               idx_c * OutDims::H * OutDims::W + idx_h * OutDims::W + idx_w;
        //
        in += idx_c * InDims::C * InDims::H * InDims::W +
              idx_h * InDims::H * InDims::W + idx_n * InDims::W + idx_w;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i];
        }
    }
};

template <typename InDims, typename OutDims, typename DataType,
          int NelemPerThread>
struct Transpose1230
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        out += idx_n * OutDims::C * OutDims::H * OutDims::W +
               idx_c * OutDims::H * OutDims::W + idx_h * OutDims::W + idx_w;
        //
        in += idx_c * InDims::C * InDims::H * InDims::W +
              idx_h * InDims::H * InDims::W + idx_w * InDims::W + idx_n;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::W];
        }
    }
};

template <typename InDims, typename OutDims, typename DataType,
          int NelemPerThread>
struct Transpose1302
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        out += idx_n * OutDims::C * OutDims::H * OutDims::W +
               idx_c * OutDims::H * OutDims::W + idx_h * OutDims::W + idx_w;
        //
        in += idx_c * InDims::C * InDims::H * InDims::W +
              idx_w * InDims::H * InDims::W + idx_n * InDims::W + idx_h;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::H * InDims::W];
        }
    }
};

template <typename InDims, typename OutDims, typename DataType,
          int NelemPerThread>
struct Transpose1320
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        out += idx_n * OutDims::C * OutDims::H * OutDims::W +
               idx_c * OutDims::H * OutDims::W + idx_h * OutDims::W + idx_w;
        //
        in += idx_c * InDims::C * InDims::H * InDims::W +
              idx_w * InDims::H * InDims::W + idx_h * InDims::W + idx_n;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::H * InDims::W];
        }
    }
};

template <typename InDims, typename OutDims, typename DataType,
          int NelemPerThread>
struct Transpose2013
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        out += idx_n * OutDims::C * OutDims::H * OutDims::W +
               idx_c * OutDims::H * OutDims::W + idx_h * OutDims::W + idx_w;
        //
        in += idx_h * InDims::C * InDims::H * InDims::W +
              idx_n * InDims::H * InDims::W + idx_c * InDims::W + idx_w;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i];
        }
    }
};

template <typename InDims, typename OutDims, typename DataType,
          int NelemPerThread>
struct Transpose2031
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        out += idx_n * OutDims::C * OutDims::H * OutDims::W +
               idx_c * OutDims::H * OutDims::W + idx_h * OutDims::W + idx_w;
        //
        in += idx_h * InDims::C * InDims::H * InDims::W +
              idx_n * InDims::H * InDims::W + idx_w * InDims::W + idx_c;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::W];
        }
    }
};

template <typename InDims, typename OutDims, typename DataType,
          int NelemPerThread>
struct Transpose2103
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        out += idx_n * OutDims::C * OutDims::H * OutDims::W +
               idx_c * OutDims::H * OutDims::W + idx_h * OutDims::W + idx_w;
        //
        in += idx_h * InDims::C * InDims::H * InDims::W +
              idx_c * InDims::H * InDims::W + idx_n * InDims::W + idx_w;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i];
        }
    }
};

template <typename InDims, typename OutDims, typename DataType,
          int NelemPerThread>
struct Transpose2130
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        out += idx_n * OutDims::C * OutDims::H * OutDims::W +
               idx_c * OutDims::H * OutDims::W + idx_h * OutDims::W + idx_w;
        //
        in += idx_h * InDims::C * InDims::H * InDims::W +
              idx_c * InDims::H * InDims::W + idx_w * InDims::W + idx_n;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::W];
        }
    }
};

template <typename InDims, typename OutDims, typename DataType,
          int NelemPerThread>
struct Transpose2301
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        out += idx_n * OutDims::C * OutDims::H * OutDims::W +
               idx_c * OutDims::H * OutDims::W + idx_h * OutDims::W + idx_w;
        //
        in += idx_h * InDims::C * InDims::H * InDims::W +
              idx_w * InDims::H * InDims::W + idx_n * InDims::W + idx_c;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::H * InDims::W];
        }
    }
};

template <typename InDims, typename OutDims, typename DataType,
          int NelemPerThread>
struct Transpose2310
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        out += idx_n * OutDims::C * OutDims::H * OutDims::W +
               idx_c * OutDims::H * OutDims::W + idx_h * OutDims::W + idx_w;
        //
        in += idx_h * InDims::C * InDims::H * InDims::W +
              idx_w * InDims::H * InDims::W + idx_c * InDims::W + idx_n;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::H * InDims::W];
        }
    }
};

template <typename InDims, typename OutDims, typename DataType,
          int NelemPerThread>
struct Transpose3012
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        out += idx_n * OutDims::C * OutDims::H * OutDims::W +
               idx_c * OutDims::H * OutDims::W + idx_h * OutDims::W + idx_w;
        //
        in += idx_w * InDims::C * InDims::H * InDims::W +
              idx_n * InDims::H * InDims::W + idx_c * InDims::W + idx_h;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::C * InDims::H * InDims::W];
        }
    }
};

template <typename InDims, typename OutDims, typename DataType,
          int NelemPerThread>
struct Transpose3021
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        out += idx_n * OutDims::C * OutDims::H * OutDims::W +
               idx_c * OutDims::H * OutDims::W + idx_h * OutDims::W + idx_w;
        //
        in += idx_w * InDims::C * InDims::H * InDims::W +
              idx_n * InDims::H * InDims::W + idx_h * InDims::W + idx_c;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::C * InDims::H * InDims::W];
        }
    }
};

template <typename InDims, typename OutDims, typename DataType,
          int NelemPerThread>
struct Transpose3102
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        out += idx_n * OutDims::C * OutDims::H * OutDims::W +
               idx_c * OutDims::H * OutDims::W + idx_h * OutDims::W + idx_w;
        //
        in += idx_w * InDims::C * InDims::H * InDims::W +
              idx_c * InDims::H * InDims::W + idx_n * InDims::W + idx_h;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::C * InDims::H * InDims::W];
        }
    }
};

template <typename InDims, typename OutDims, typename DataType,
          int NelemPerThread>
struct Transpose3120
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        out += idx_n * OutDims::C * OutDims::H * OutDims::W +
               idx_c * OutDims::H * OutDims::W + idx_h * OutDims::W + idx_w;
        //
        in += idx_w * InDims::C * InDims::H * InDims::W +
              idx_c * InDims::H * InDims::W + idx_h * InDims::W + idx_n;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::C * InDims::H * InDims::W];
        }
    }
};

template <typename InDims, typename OutDims, typename DataType,
          int NelemPerThread>
struct Transpose3201
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        out += idx_n * OutDims::C * OutDims::H * OutDims::W +
               idx_c * OutDims::H * OutDims::W + idx_h * OutDims::W + idx_w;
        //
        in += idx_w * InDims::C * InDims::H * InDims::W +
              idx_h * InDims::H * InDims::W + idx_n * InDims::W + idx_c;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::C * InDims::H * InDims::W];
        }
    }
};

template <typename InDims, typename OutDims, typename DataType,
          int NelemPerThread>
struct Transpose3210
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        out += idx_n * OutDims::C * OutDims::H * OutDims::W +
               idx_c * OutDims::H * OutDims::W + idx_h * OutDims::W + idx_w;
        //
        in += idx_w * InDims::C * InDims::H * InDims::W +
              idx_h * InDims::H * InDims::W + idx_c * InDims::W + idx_n;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::C * InDims::H * InDims::W];
        }
    }
};

////////////////////////////////////////////////////////////////////////////////

template <typename InDims, typename OutDims, typename OutShape,
          typename UnitOutShape, int ThreadsNum, int SmemBytes,
          typename Transpose>
DEVICE void _transpose(float *out, float *in, int tx, int ty, int tz)
{
    Ewise1<OutDims, OutShape, UnitOutShape, ThreadsNum, SmemBytes,
           Transpose>::run(out, in, tz / OutShape::C, tz % OutShape::C, ty, tx);
}

// TODO: we need to use NelemPerThread=2 for half in the future, if out is a
// __half pointer, this can cause a memory bug, because GPU DRAM access should
// be always 4-byte aligned
template <typename InDims, typename OutDims, typename OutShape,
          typename UnitOutShape, int ThreadsNum, int SmemBytes,
          typename Transpose>
DEVICE void _transpose(ark::half *out, ark::half *in, int tx, int ty, int tz)
{
    Ewise1<OutDims, OutShape, UnitOutShape, ThreadsNum, SmemBytes,
           Transpose>::run(out, in, tz / OutShape::C, tz % OutShape::C, ty, tx);
}

#define _DEC_TRANSPOSE(tp_type)                                                \
    template <typename InDims, typename OutDims, typename OutShape,            \
              typename UnitOutShape, int ThreadsNum, int SmemBytes,            \
              typename DataType>                                               \
    DEVICE void transpose##tp_type(DataType *out, DataType *in, int tx,        \
                                   int ty, int tz)                             \
    {                                                                          \
        _transpose<InDims, OutDims, OutShape, UnitOutShape, ThreadsNum,        \
                   SmemBytes,                                                  \
                   Transpose##tp_type<InDims, OutDims, DataType, 1>>(          \
            out, in, tx, ty, tz);                                              \
    }

_DEC_TRANSPOSE(0132)
_DEC_TRANSPOSE(0213)
_DEC_TRANSPOSE(0231)
_DEC_TRANSPOSE(0312)
_DEC_TRANSPOSE(0321)
_DEC_TRANSPOSE(1023)
_DEC_TRANSPOSE(1032)
_DEC_TRANSPOSE(1203)
_DEC_TRANSPOSE(1230)
_DEC_TRANSPOSE(1302)
_DEC_TRANSPOSE(1320)
_DEC_TRANSPOSE(2013)
_DEC_TRANSPOSE(2031)
_DEC_TRANSPOSE(2103)
_DEC_TRANSPOSE(2130)
_DEC_TRANSPOSE(2301)
_DEC_TRANSPOSE(2310)
_DEC_TRANSPOSE(3012)
_DEC_TRANSPOSE(3021)
_DEC_TRANSPOSE(3102)
_DEC_TRANSPOSE(3120)
_DEC_TRANSPOSE(3201)
_DEC_TRANSPOSE(3210)

} // namespace ark

#endif // ARK_KERNELS_TRANSPOSE_H_
