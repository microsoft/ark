// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_COMM_H_
#define ARK_KERNELS_COMM_H_

#include "common.h"
#include "unit_op.h"

namespace ark {
namespace comm {

// Doorbell types.
struct ReqType
{
    static const unsigned int Send = 0;
    static const unsigned int Recv = 1;
};

// Constructs 64-bit doorbell to be sent to FPGA.
template <unsigned int RType, unsigned int DstSid, unsigned int SrcSid,
          unsigned int Cid, unsigned long long int Length>
struct Doorbell
{
    static const unsigned long long int value =
        ((Length & 0x3ffffffff) << 25) + ((Cid & 0x7f) << 18) +
        ((SrcSid & 0xff) << 10) + ((DstSid & 0xff) << 2) + (RType & 0x3);
};

struct State
{
    int flag = 0;
};

#if (ARK_COMM_SW != 0)
__device__ int _ARK_COMM_SW_SEND_LOCK = 0;
#endif // (ARK_COMM_SW != 0)

// Send a doorbell to FPGA to request transaction.
template <unsigned int Cid, unsigned int SrcSid, unsigned int DstSid,
          unsigned long long int Length>
DEVICE void send()
{
    using UnitOp = UnitOp<ark::Vec<>, ark::Vec<>, ark::Vec<>, 32, 0>;
    if (UnitOp::thread_id() != 0) {
        return;
    }
    volatile unsigned int *done = &(_ARK_SC[SrcSid]);
    while (!(*done)) {
    }
    *done = 0;
    constexpr unsigned long long int dbval =
        Doorbell<ReqType::Send, DstSid, SrcSid, Cid, Length>::value;
#if (ARK_COMM_SW != 0)
#if 1
    constexpr unsigned long long int invalid = (unsigned long long int)-1;
    while (atomicCAS(&_ARK_COMM_SW_SEND_LOCK, 0, 1) != 0) {
    }
    while (*_ARK_DOORBELL != invalid) {
    }
    *_ARK_DOORBELL = dbval;
    atomicExch(&_ARK_COMM_SW_SEND_LOCK, 0);
#else
    // This version is slower than the above one.
    constexpr unsigned long long int invalid = (unsigned long long int)-1;
    for (;;) {
        while (*_ARK_DOORBELL != invalid) {
        }
        if (atomicCAS((unsigned long long int *)_ARK_DOORBELL,
                      (unsigned long long int)invalid,
                      (unsigned long long int)dbval) == invalid) {
            break;
        }
    }
#endif // 1
#else
    *_ARK_DOORBELL = dbval;
#endif // (ARK_COMM_SW != 0)
}

// Poll SC and reset.
template <unsigned int SrcSid> DEVICE void send_done()
{
    using UnitOp = UnitOp<ark::Vec<>, ark::Vec<>, ark::Vec<>, 32, 0>;
    if (UnitOp::thread_id() != 0) {
        return;
    }
    volatile unsigned int *done = &(_ARK_SC[SrcSid]);
    while (!(*done)) {
    }
    *done = 0;
}

//
template <unsigned int DstSid> DEVICE void recv()
{
    using UnitOp = UnitOp<ark::Vec<>, ark::Vec<>, ark::Vec<>, 32, 0>;
    if (UnitOp::thread_id() != 0) {
        return;
    }
    volatile unsigned int *len = &(_ARK_RC[DstSid]);
    while (!(*len)) {
    }
    *len = 0;
}

} // namespace comm
} // namespace ark

#endif // ARK_KERNELS_COMM_H_
