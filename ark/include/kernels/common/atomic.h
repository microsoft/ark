// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_ATOMIC_H_
#define ARK_KERNELS_ATOMIC_H_

#include <mscclpp/atomic_device.hpp>

#include "device.h"

namespace ark {

template <typename T>
DEVICE T atomicLoadRelaxed(T *ptr) {
    return mscclpp::atomicLoad(ptr, mscclpp::memoryOrderRelaxed);
}

template <typename T>
DEVICE void atomicStoreRelaxed(T *ptr, const T &val) {
    mscclpp::atomicStore(ptr, val, mscclpp::memoryOrderRelaxed);
}

}  // namespace ark

#endif  // ARK_KERNELS_ATOMIC_H_
