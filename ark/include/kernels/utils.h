// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_UTILS_H_
#define ARK_KERNELS_UTILS_H_

namespace ark {

template <int _D0 = 1, int _D1 = 1, int _D2 = 1, int _D3 = 1> struct Vec
{
    static_assert(_D0 >= 0, "");
    static_assert(_D1 >= 0, "");
    static_assert(_D2 >= 0, "");
    static_assert(_D3 >= 0, "");

    // 4D representation.
    static const int D0 = _D0;
    static const int D1 = _D1;
    static const int D2 = _D2;
    static const int D3 = _D3;
    static const int N = _D0;
    static const int C = _D1;
    static const int H = _D2;
    static const int W = _D3;

    // 3D representation.
    static const int X = _D0;
    static const int Y = _D1;
    static const int Z = _D2;
};

} // namespace ark

#endif // ARK_KERNELS_UTILS_H_
