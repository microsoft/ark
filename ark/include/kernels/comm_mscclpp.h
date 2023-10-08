// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_COMM_MSCCLPP_H_
#define ARK_KERNELS_COMM_MSCCLPP_H_

#include "common.h"
#include "ewise.h"
#include "unit_op.h"
#include <cstdlib>
#include <cuda_runtime.h>
#include <mscclpp/proxy_channel_device.hpp>
#include <mscclpp/sm_channel_device.hpp>

extern __constant__ mscclpp::SimpleProxyChannelDeviceHandle _ARK_PROXY_CHANS[];
extern __constant__ mscclpp::SmChannelDeviceHandle _ARK_SM_CHANS[];

namespace ark {
namespace comm {

template<typename OutDims, typename DataType>
struct MscclppEwiseSumCompType
{
};

union BytesPack {
    uint32_t u16[8];
    uint32_t u32[4];
    uint64_t u64[2];
};

template <typename OutDims> struct MscclppEwiseSumCompType<OutDims, half>
{
    using DataType = half;
    static const int NelemPerThread = 8;
    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        BytesPack vs;
        BytesPack vd;
        int idx = (idx_n * OutDims::CHW + idx_c * OutDims::HW +
                   idx_h * OutDims::W + idx_w) /
                  NelemPerThread;
        longlong2 *ps = reinterpret_cast<longlong2 *>(in) + idx;
        longlong2 *pd = reinterpret_cast<longlong2 *>(out) + idx;
        asm volatile("ld.volatile.global.v2.b64 {%0,%1}, [%2];" : "=l"(vs.u64[0]), "=l"(vs.u64[1]) : "l"(ps) : "memory");
        asm volatile("ld.volatile.global.v2.b64 {%0,%1}, [%2];" : "=l"(vd.u64[0]), "=l"(vd.u64[1]) : "l"(pd) : "memory");
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            __half2 d, s, t;
            d.x = vd.u16[i * 2];
            d.y = vd.u16[i * 2 + 1];
            s.x = vs.u16[i * 2];
            s.y = vs.u16[i * 2 + 1];
            t = __hadd2(d, s);
            vd.u16[i * 2] = t.x;
            vd.u16[i * 2 + 1] = t.y;
        }
        asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" : : "l"(pd), "l"(vd.u64[0]), "l"(vd.u64[1]) : "memory");
    }
};

// Send a trigger to proxy to request transaction.
template <unsigned int Rank, unsigned int DstRank,
          unsigned long long int Length>
DEVICE void send_mscclpp(size_t dst_offset, size_t src_offset, int, int)
{
    using UnitOp = UnitOp<ark::Vec<>, ark::Vec<>, ark::Vec<>, 32, 0>;
    if (UnitOp::thread_id() != 0) {
        return;
    }
    constexpr unsigned int cid = DstRank < Rank ? DstRank : DstRank - 1;
    mscclpp::SimpleProxyChannelDeviceHandle &proxy_chan = _ARK_PROXY_CHANS[cid];
    proxy_chan.putWithSignal(dst_offset, src_offset, Length);
}

// Poll SC and reset.
template <unsigned int Rank, unsigned int DstRank>
DEVICE void send_done_mscclpp(int, int)
{
    using UnitOp = UnitOp<ark::Vec<>, ark::Vec<>, ark::Vec<>, 32, 0>;
    if (UnitOp::thread_id() != 0) {
        return;
    }
    constexpr unsigned int cid = DstRank < Rank ? DstRank : DstRank - 1;
    mscclpp::SimpleProxyChannelDeviceHandle &proxy_chan = _ARK_PROXY_CHANS[cid];
    proxy_chan.flush();
}

//
template <unsigned int Rank, unsigned int DstRank>
DEVICE void recv_mscclpp(int, int)
{
    using UnitOp = UnitOp<ark::Vec<>, ark::Vec<>, ark::Vec<>, 32, 0>;
    if (UnitOp::thread_id() != 0) {
        return;
    }
    constexpr unsigned int cid = DstRank < Rank ? DstRank : DstRank - 1;
    mscclpp::SimpleProxyChannelDeviceHandle &proxy_chan = _ARK_PROXY_CHANS[cid];
    proxy_chan.wait();
}

template <unsigned int NRanks>
DEVICE void device_sync_mscclpp(int, int)
{
    using UnitOp = UnitOp<ark::Vec<>, ark::Vec<>, ark::Vec<>, 32, 0>;
    if (UnitOp::thread_id() != 0) {
        return;
    }
    for (int i = 0; i < NRanks - 1; ++i) {
        _ARK_SM_CHANS[i].signal();
    }
    for (int i = 0; i < NRanks - 1; ++i) {
        _ARK_SM_CHANS[i].wait();
    }
}

// Do reduce scatter in a single node
template <typename Dims, typename Shape, typename UnitOutDims, int NumThreads,
          unsigned int NPeers, unsigned int Rank, unsigned long long Offset>
DEVICE void read_and_reduce_mscclpp(size_t dst_offset, size_t src_offset_0,
                                    size_t src_offset_1, size_t src_offset_2,
                                    size_t src_offset_3, size_t src_offset_4,
                                    size_t src_offset_5, size_t src_offset_6,
                                    ark::half *, int uop_idx, int)
{
    // treat channel dst as src since we read from it, and reduce to local
    // memory

    // All channels have the same src_, so we can use any channel to get dst
    using UnitOp = UnitOp<Dims, Shape, UnitOutDims, NumThreads, 0>;
    half *dst = reinterpret_cast<half *>((uint8_t *)_ARK_SM_CHANS[0].src_ +
                                         dst_offset + Offset);
    size_t peer_offsets[] = {src_offset_0, src_offset_1, src_offset_2,
                             src_offset_3, src_offset_4, src_offset_5,
                             src_offset_6};
    for (int i = 0; i < NPeers; ++i) {
        int chan_idx = (Rank + i) % NPeers;
        half *src =
            reinterpret_cast<half *>((uint8_t *)_ARK_SM_CHANS[chan_idx].dst_ +
                                     peer_offsets[chan_idx] + Offset);
        Ewise1<Dims, Shape, UnitOutDims, NumThreads, 0,
               MscclppEwiseSumCompType<Dims, half>>::run(dst, src, uop_idx);
    }
}

} // namespace comm
} // namespace ark

#endif // ARK_KERNELS_COMM_H_
