// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_COMMON_H_
#define ARK_GPU_COMMON_H_

#include <cstdint>
#include <cuda.h>

#define ARK_GPU_NAME_PREFIX "gpu."
#define ARK_GPU_DATA_NAME ARK_GPU_NAME_PREFIX "data."
#define ARK_GPU_SC_RC_NAME ARK_GPU_NAME_PREFIX "sc_rc."
#define ARK_GPU_INFO_NAME ARK_GPU_NAME_PREFIX "info."

namespace ark {

// Constants.
enum
{
    REQUEST_INVALID = -1,
    MAX_NUM_SID = 256
};

//
union alignas(8) Request {
    uint64_t value = REQUEST_INVALID;
    struct
    {
        uint64_t req : 2;  // Request type
        uint64_t dst : 8;  // Dst segment ID
        uint64_t src : 8;  // Src segment ID
        uint64_t rank : 7; // Rank
        uint64_t len : 34; // Length
        uint64_t rsv : 5;  // Unused (reserved)
    } fields;
};

//
typedef CUresult GpuState;

} // namespace ark

#endif // ARK_GPU_COMM_COMMON_H_
