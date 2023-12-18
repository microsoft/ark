// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_TRANSPOSE_H_
#define ARK_KERNELS_TRANSPOSE_H_

#include "common/ewise.h"

namespace ark {

template <typename _InDims, typename _OutDims, typename _OutShape,
          typename _DataType, int _NelemPerThread>
struct Transpose0132 {
    using InDims = _InDims;
    using OutDims = _OutDims;
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        if (idx_w >= _OutShape::W || idx_h >= _OutShape::H ||
            idx_c >= _OutShape::C || idx_n >= _OutShape::N) {
            return;
        }
        out += idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W +
               idx_w;
        //
        in += idx_n * InDims::CHW + idx_c * InDims::HW + idx_w * InDims::W +
              idx_h;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::W];
        }
    }
};

template <typename _InDims, typename _OutDims, typename _OutShape,
          typename _DataType, int _NelemPerThread>
struct Transpose0213 {
    using InDims = _InDims;
    using OutDims = _OutDims;
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        if (idx_w >= _OutShape::W || idx_h >= _OutShape::H ||
            idx_c >= _OutShape::C || idx_n >= _OutShape::N) {
            return;
        }
        out += idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W +
               idx_w;
        //
        in += idx_n * InDims::CHW + idx_h * InDims::HW + idx_c * InDims::W +
              idx_w;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i];
        }
    }
};

template <typename _InDims, typename _OutDims, typename _OutShape,
          typename _DataType, int _NelemPerThread>
struct Transpose0231 {
    using InDims = _InDims;
    using OutDims = _OutDims;
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        if (idx_w >= _OutShape::W || idx_h >= _OutShape::H ||
            idx_c >= _OutShape::C || idx_n >= _OutShape::N) {
            return;
        }
        out += idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W +
               idx_w;
        //
        in += idx_n * InDims::CHW + idx_w * InDims::HW + idx_c * InDims::W +
              idx_h;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::HW];
        }
    }
};

template <typename _InDims, typename _OutDims, typename _OutShape,
          typename _DataType, int _NelemPerThread>
struct Transpose0312 {
    using InDims = _InDims;
    using OutDims = _OutDims;
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        if (idx_w >= _OutShape::W || idx_h >= _OutShape::H ||
            idx_c >= _OutShape::C || idx_n >= _OutShape::N) {
            return;
        }
        out += idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W +
               idx_w;
        //
        in += idx_n * InDims::CHW + idx_h * InDims::HW + idx_w * InDims::W +
              idx_c;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::W];
        }
    }
};

template <typename _InDims, typename _OutDims, typename _OutShape,
          typename _DataType, int _NelemPerThread>
struct Transpose0321 {
    using InDims = _InDims;
    using OutDims = _OutDims;
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        if (idx_w >= _OutShape::W || idx_h >= _OutShape::H ||
            idx_c >= _OutShape::C || idx_n >= _OutShape::N) {
            return;
        }
        out += idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W +
               idx_w;
        //
        in += idx_n * InDims::CHW + idx_w * InDims::HW + idx_h * InDims::W +
              idx_c;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::HW];
        }
    }
};

template <typename _InDims, typename _OutDims, typename _OutShape,
          typename _DataType, int _NelemPerThread>
struct Transpose1023 {
    using InDims = _InDims;
    using OutDims = _OutDims;
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        if (idx_w >= _OutShape::W || idx_h >= _OutShape::H ||
            idx_c >= _OutShape::C || idx_n >= _OutShape::N) {
            return;
        }
        out += idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W +
               idx_w;
        //
        in += idx_c * InDims::CHW + idx_n * InDims::HW + idx_h * InDims::W +
              idx_w;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i];
        }
    }
};

template <typename _InDims, typename _OutDims, typename _OutShape,
          typename _DataType, int _NelemPerThread>
struct Transpose1032 {
    using InDims = _InDims;
    using OutDims = _OutDims;
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        if (idx_w >= _OutShape::W || idx_h >= _OutShape::H ||
            idx_c >= _OutShape::C || idx_n >= _OutShape::N) {
            return;
        }
        out += idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W +
               idx_w;
        //
        in += idx_c * InDims::CHW + idx_n * InDims::HW + idx_w * InDims::W +
              idx_h;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::W];
        }
    }
};

template <typename _InDims, typename _OutDims, typename _OutShape,
          typename _DataType, int _NelemPerThread>
struct Transpose1203 {
    using InDims = _InDims;
    using OutDims = _OutDims;
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        if (idx_w >= _OutShape::W || idx_h >= _OutShape::H ||
            idx_c >= _OutShape::C || idx_n >= _OutShape::N) {
            return;
        }
        out += idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W +
               idx_w;
        //
        in += idx_h * InDims::CHW + idx_n * InDims::HW + idx_c * InDims::W +
              idx_w;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i];
        }
    }
};

template <typename _InDims, typename _OutDims, typename _OutShape,
          typename _DataType, int _NelemPerThread>
struct Transpose1230 {
    using InDims = _InDims;
    using OutDims = _OutDims;
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        if (idx_w >= _OutShape::W || idx_h >= _OutShape::H ||
            idx_c >= _OutShape::C || idx_n >= _OutShape::N) {
            return;
        }
        out += idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W +
               idx_w;
        //
        in += idx_w * InDims::CHW + idx_n * InDims::HW + idx_c * InDims::W +
              idx_h;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::CHW];
        }
    }
};

template <typename _InDims, typename _OutDims, typename _OutShape,
          typename _DataType, int _NelemPerThread>
struct Transpose1302 {
    using InDims = _InDims;
    using OutDims = _OutDims;
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        if (idx_w >= _OutShape::W || idx_h >= _OutShape::H ||
            idx_c >= _OutShape::C || idx_n >= _OutShape::N) {
            return;
        }
        out += idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W +
               idx_w;
        //
        in += idx_h * InDims::CHW + idx_n * InDims::HW + idx_w * InDims::W +
              idx_c;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::W];
        }
    }
};

template <typename _InDims, typename _OutDims, typename _OutShape,
          typename _DataType, int _NelemPerThread>
struct Transpose1320 {
    using InDims = _InDims;
    using OutDims = _OutDims;
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        if (idx_w >= _OutShape::W || idx_h >= _OutShape::H ||
            idx_c >= _OutShape::C || idx_n >= _OutShape::N) {
            return;
        }
        out += idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W +
               idx_w;
        //
        in += idx_w * InDims::CHW + idx_n * InDims::HW + idx_h * InDims::W +
              idx_c;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::CHW];
        }
    }
};

template <typename _InDims, typename _OutDims, typename _OutShape,
          typename _DataType, int _NelemPerThread>
struct Transpose2013 {
    using InDims = _InDims;
    using OutDims = _OutDims;
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        if (idx_w >= _OutShape::W || idx_h >= _OutShape::H ||
            idx_c >= _OutShape::C || idx_n >= _OutShape::N) {
            return;
        }
        out += idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W +
               idx_w;
        //
        in += idx_c * InDims::CHW + idx_h * InDims::HW + idx_n * InDims::W +
              idx_w;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i];
        }
    }
};

template <typename _InDims, typename _OutDims, typename _OutShape,
          typename _DataType, int _NelemPerThread>
struct Transpose2031 {
    using InDims = _InDims;
    using OutDims = _OutDims;
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        if (idx_w >= _OutShape::W || idx_h >= _OutShape::H ||
            idx_c >= _OutShape::C || idx_n >= _OutShape::N) {
            return;
        }
        out += idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W +
               idx_w;
        //
        in += idx_c * InDims::CHW + idx_w * InDims::HW + idx_n * InDims::W +
              idx_h;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::HW];
        }
    }
};

template <typename _InDims, typename _OutDims, typename _OutShape,
          typename _DataType, int _NelemPerThread>
struct Transpose2103 {
    using InDims = _InDims;
    using OutDims = _OutDims;
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        if (idx_w >= _OutShape::W || idx_h >= _OutShape::H ||
            idx_c >= _OutShape::C || idx_n >= _OutShape::N) {
            return;
        }
        out += idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W +
               idx_w;
        //
        in += idx_h * InDims::CHW + idx_c * InDims::HW + idx_n * InDims::W +
              idx_w;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i];
        }
    }
};

template <typename _InDims, typename _OutDims, typename _OutShape,
          typename _DataType, int _NelemPerThread>
struct Transpose2130 {
    using InDims = _InDims;
    using OutDims = _OutDims;
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        if (idx_w >= _OutShape::W || idx_h >= _OutShape::H ||
            idx_c >= _OutShape::C || idx_n >= _OutShape::N) {
            return;
        }
        out += idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W +
               idx_w;
        //
        in += idx_w * InDims::CHW + idx_c * InDims::HW + idx_n * InDims::W +
              idx_h;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::CHW];
        }
    }
};

template <typename _InDims, typename _OutDims, typename _OutShape,
          typename _DataType, int _NelemPerThread>
struct Transpose2301 {
    using InDims = _InDims;
    using OutDims = _OutDims;
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        if (idx_w >= _OutShape::W || idx_h >= _OutShape::H ||
            idx_c >= _OutShape::C || idx_n >= _OutShape::N) {
            return;
        }
        out += idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W +
               idx_w;
        //
        in += idx_h * InDims::CHW + idx_w * InDims::HW + idx_n * InDims::W +
              idx_c;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::HW];
        }
    }
};

template <typename _InDims, typename _OutDims, typename _OutShape,
          typename _DataType, int _NelemPerThread>
struct Transpose2310 {
    using InDims = _InDims;
    using OutDims = _OutDims;
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        if (idx_w >= _OutShape::W || idx_h >= _OutShape::H ||
            idx_c >= _OutShape::C || idx_n >= _OutShape::N) {
            return;
        }
        out += idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W +
               idx_w;
        //
        in += idx_w * InDims::CHW + idx_h * InDims::HW + idx_n * InDims::W +
              idx_c;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::CHW];
        }
    }
};

template <typename _InDims, typename _OutDims, typename _OutShape,
          typename _DataType, int _NelemPerThread>
struct Transpose3012 {
    using InDims = _InDims;
    using OutDims = _OutDims;
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        if (idx_w >= _OutShape::W || idx_h >= _OutShape::H ||
            idx_c >= _OutShape::C || idx_n >= _OutShape::N) {
            return;
        }
        out += idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W +
               idx_w;
        //
        in += idx_c * InDims::CHW + idx_h * InDims::HW + idx_w * InDims::W +
              idx_n;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::W];
        }
    }
};

template <typename _InDims, typename _OutDims, typename _OutShape,
          typename _DataType, int _NelemPerThread>
struct Transpose3021 {
    using InDims = _InDims;
    using OutDims = _OutDims;
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        if (idx_w >= _OutShape::W || idx_h >= _OutShape::H ||
            idx_c >= _OutShape::C || idx_n >= _OutShape::N) {
            return;
        }
        out += idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W +
               idx_w;
        //
        in += idx_c * InDims::CHW + idx_w * InDims::HW + idx_h * InDims::W +
              idx_n;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::HW];
        }
    }
};

template <typename _InDims, typename _OutDims, typename _OutShape,
          typename _DataType, int _NelemPerThread>
struct Transpose3102 {
    using InDims = _InDims;
    using OutDims = _OutDims;
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        if (idx_w >= _OutShape::W || idx_h >= _OutShape::H ||
            idx_c >= _OutShape::C || idx_n >= _OutShape::N) {
            return;
        }
        out += idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W +
               idx_w;
        //
        in += idx_h * InDims::CHW + idx_c * InDims::HW + idx_w * InDims::W +
              idx_n;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::W];
        }
    }
};

template <typename _InDims, typename _OutDims, typename _OutShape,
          typename _DataType, int _NelemPerThread>
struct Transpose3120 {
    using InDims = _InDims;
    using OutDims = _OutDims;
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        if (idx_w >= _OutShape::W || idx_h >= _OutShape::H ||
            idx_c >= _OutShape::C || idx_n >= _OutShape::N) {
            return;
        }
        out += idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W +
               idx_w;
        //
        in += idx_w * InDims::CHW + idx_c * InDims::HW + idx_h * InDims::W +
              idx_n;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::CHW];
        }
    }
};

template <typename _InDims, typename _OutDims, typename _OutShape,
          typename _DataType, int _NelemPerThread>
struct Transpose3201 {
    using InDims = _InDims;
    using OutDims = _OutDims;
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        if (idx_w >= _OutShape::W || idx_h >= _OutShape::H ||
            idx_c >= _OutShape::C || idx_n >= _OutShape::N) {
            return;
        }
        out += idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W +
               idx_w;
        //
        in += idx_h * InDims::CHW + idx_w * InDims::HW + idx_c * InDims::W +
              idx_n;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::HW];
        }
    }
};

template <typename _InDims, typename _OutDims, typename _OutShape,
          typename _DataType, int _NelemPerThread>
struct Transpose3210 {
    using InDims = _InDims;
    using OutDims = _OutDims;
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        if (idx_w >= _OutShape::W || idx_h >= _OutShape::H ||
            idx_c >= _OutShape::C || idx_n >= _OutShape::N) {
            return;
        }
        out += idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W +
               idx_w;
        //
        in += idx_w * InDims::CHW + idx_h * InDims::HW + idx_c * InDims::W +
              idx_n;
        *out = *in;
#pragma unroll
        for (int i = 1; i < NelemPerThread; ++i) {
            out[i] = in[i * InDims::CHW];
        }
    }
};

////////////////////////////////////////////////////////////////////////////////

// TODO: support NelemPerThread > 1
template <typename InDims, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumWarps, int SmemBytes, typename Transpose,
          typename DataType>
DEVICE void _transpose(DataType *out, DataType *in, int uop_idx) {
    Ewise1<OutDims, OutShape, UnitOutDims, NumWarps, SmemBytes, Transpose>::run(
        out, in, uop_idx);
}

#define _DEC_TRANSPOSE(tp_type)                                              \
    template <typename InDims, typename OutDims, typename OutShape,          \
              typename UnitOutDims, int NumWarps, int SmemBytes,             \
              typename DataType>                                             \
    DEVICE void transpose##tp_type(DataType *out, DataType *in, int uop_idx, \
                                   int) {                                    \
        _transpose<                                                          \
            InDims, OutDims, OutShape, UnitOutDims, NumWarps, SmemBytes,     \
            Transpose##tp_type<InDims, OutDims, OutShape, DataType, 1>>(     \
            out, in, uop_idx);                                               \
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

}  // namespace ark

#endif  // ARK_KERNELS_TRANSPOSE_H_
