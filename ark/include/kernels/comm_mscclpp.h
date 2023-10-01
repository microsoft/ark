// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_COMM_MSCCLPP_H_
#define ARK_KERNELS_COMM_MSCCLPP_H_

#include "common.h"
#include "reduce.h"
#include "unit_op.h"
#include <cstdlib>
#include <mscclpp/proxy_channel_device.hpp>
#include <mscclpp/sm_channel_device.hpp>

extern __constant__ mscclpp::SimpleProxyChannelDeviceHandle _ARK_PROXY_CHANS[];
extern __constant__ mscclpp::SmChannelDeviceHandle _ARK_SM_CHANS[];

namespace ark {
namespace comm {

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

template <unsigned int NPeers>
DEVICE void device_sync_mscclpp(int, int)
{
    using UnitOp = UnitOp<ark::Vec<>, ark::Vec<>, ark::Vec<>, 32, 0>;
    if (UnitOp::thread_id() != 0) {
        return;
    }
    for (int i = 0; i < NPeers; ++i) {
        _ARK_SM_CHANS[i].signal();
    }
    for (int i = 0; i < NPeers; ++i) {
        _ARK_SM_CHANS[i].wait();
    }
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          unsigned int Rank, unsigned int DstRank>
DEVICE void read_and_redcue_mscclpp(size_t dst_offset, size_t src_offset, int,
                                    int)
{
    // here we need to sync first
    using UnitOp = UnitOp<OutDims, OutShape, UnitOutDims, NumThreads, 0>;
    UnitOp::sync_threads();

    // run reduce_e_sum to reduce the value
}

} // namespace comm
} // namespace ark

#endif // ARK_KERNELS_COMM_H_
