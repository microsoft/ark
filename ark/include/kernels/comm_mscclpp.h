// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_COMM_MSCCLPP_H_
#define ARK_KERNELS_COMM_MSCCLPP_H_

#include "common.h"
#include "ewise.h"
#include "unit_op.h"
#include <cstdlib>
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

template <typename OutDims> struct MscclppEwiseSumCompType<OutDims, half>
{
    using DataType = half;
    static const int NelemPerThread = 2;
    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w)
    {
        __half2 *dst = reinterpret_cast<__half2 *>(out);
        __half2 *src = reinterpret_cast<__half2 *>(in);
        int idx = idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W + idx_w;
        if (threadIdx.x == 2) {
            printf("idx is %d\n", idx);
        }
        dst[idx/2] = __hadd2(dst[idx/2], src[idx/2]);
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
          unsigned int PeerRank, unsigned int Rank, unsigned long long Offset>
DEVICE void read_and_reduce_mscclpp(size_t dst_offset, size_t src_offset,
                                    ark::half *, int uop_idx, int)
{
    // treat channel dst as src since we read from it, and reduce to local
    // memory
    int channel_id = PeerRank < Rank ? PeerRank : PeerRank - 1;
    half *src = reinterpret_cast<half *>(
        (uint8_t *)_ARK_SM_CHANS[channel_id].dst_ + src_offset + Offset);
    half *dst = reinterpret_cast<half *>(
        (uint8_t *)_ARK_SM_CHANS[channel_id].src_ + dst_offset + Offset);
    using UnitOp = UnitOp<Dims, Shape, UnitOutDims, NumThreads, 0>;
    Ewise1<Dims, Shape, UnitOutDims, NumThreads, 0,
           MscclppEwiseSumCompType<Dims, half>>::run(dst, src, uop_idx);
}

} // namespace comm
} // namespace ark

#endif // ARK_KERNELS_COMM_H_
