// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_COMM_MSCCLPP_H_
#define ARK_KERNELS_COMM_MSCCLPP_H_

#include <cuda_runtime.h>

#include <mscclpp/proxy_channel_device.hpp>
#include <mscclpp/sm_channel_device.hpp>

#include "common.h"
#include "ewise.h"
#include "unit_op.h"

extern __constant__ mscclpp::SimpleProxyChannelDeviceHandle _ARK_PROXY_CHANS[];
extern __constant__ mscclpp::SmChannelDeviceHandle _ARK_SM_CHANS[];

namespace ark {
namespace comm {

template <typename OutDims, typename DataType>
struct MscclppEwiseSumCompType {};

template <int NBytes>
union BytesPack {};

template <>
union BytesPack<16> {
    uint16_t u16[8];
    uint32_t u32[4];
    uint64_t u64[2];
};

template <>
union BytesPack<8> {
    uint16_t u16[4];
    uint32_t u32[2];
};

DEVICE void load(BytesPack<16> &v, const longlong2 *p) {
    asm volatile("ld.volatile.global.v2.b64 {%0,%1}, [%2];"
                 : "=l"(v.u64[0]), "=l"(v.u64[1])
                 : "l"(p)
                 : "memory");
}

DEVICE void load(BytesPack<8> &v, const uint2 *p) {
    asm volatile("ld.volatile.global.v2.b32 {%0,%1}, [%2];"
                 : "=r"(v.u32[0]), "=r"(v.u32[1])
                 : "l"(p)
                 : "memory");
}

DEVICE void store(longlong2 *p, const BytesPack<16> &v) {
    asm volatile("st.volatile.global.v2.b64 [%0], {%1,%2};"
                 :
                 : "l"(p), "l"(v.u64[0]), "l"(v.u64[1])
                 : "memory");
}

DEVICE void store(uint2 *p, const BytesPack<8> &v) {
    asm volatile("st.volatile.global.v2.b32 [%0], {%1,%2};"
                 :
                 : "l"(p), "r"(v.u32[0]), "r"(v.u32[1])
                 : "memory");
}

DEVICE void add_half8(BytesPack<16> &dst, BytesPack<16> &src) {
    __half2 *pd = reinterpret_cast<__half2 *>(dst.u32);
    __half2 *ps = reinterpret_cast<__half2 *>(src.u32);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        union {
            __half2 h2; uint32_t u32;
        } d, s;
        d.h2 = pd[i];
        s.h2 = ps[i];
        pd[i] = __hadd2(d.h2, s.h2);
    }
}

DEVICE void add_half4(BytesPack<8> &dst, BytesPack<8> &src) {
    __half *pd = reinterpret_cast<__half *>(dst.u16);
    __half *ps = reinterpret_cast<__half *>(src.u16);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        __half2 d, s;
        d.x = pd[i * 2];
        d.y = pd[i * 2 + 1];
        s.x = ps[i * 2];
        s.y = ps[i * 2 + 1];
        d = __hadd2(d, s);
        pd[i * 2] = d.x;
        pd[i * 2 + 1] = d.y;
    }
}

// Send a trigger to proxy to request transaction.
template <unsigned int Rank, unsigned int DstRank,
          unsigned long long int Length>
DEVICE void send_mscclpp(size_t dst_offset, size_t src_offset, int, int) {
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
DEVICE void send_done_mscclpp(int, int) {
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
DEVICE void recv_mscclpp(int, int) {
    using UnitOp = UnitOp<ark::Vec<>, ark::Vec<>, ark::Vec<>, 32, 0>;
    if (UnitOp::thread_id() != 0) {
        return;
    }
    constexpr unsigned int cid = DstRank < Rank ? DstRank : DstRank - 1;
    mscclpp::SimpleProxyChannelDeviceHandle &proxy_chan = _ARK_PROXY_CHANS[cid];
    proxy_chan.wait();
}

template <unsigned int NRanks>
DEVICE void device_sync_mscclpp(int, int) {
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
          unsigned int NPeers, unsigned int Rank, unsigned long long Offset,
          unsigned long long Length>
DEVICE void read_and_reduce_mscclpp(size_t src_offset_0, size_t src_offset_1,
                                    size_t src_offset_2, size_t src_offset_3,
                                    size_t src_offset_4, size_t src_offset_5,
                                    size_t src_offset_6, ark::fp16 *src,
                                    int uop_idx, int) {
    // treat channel dst as src since we read from it, and reduce to local
    // memory
    using UnitOp = UnitOp<Dims, Shape, UnitOutDims, NumThreads, 0>;
    constexpr int total_tiles =
        math::div_up<Shape::NCHW, UnitOutDims::NCHW>::value;
    constexpr int total_threads = total_tiles * NumThreads;
    constexpr size_t nInt4 = Length / sizeof(int4);
    const int tid = uop_idx * NumThreads + UnitOp::thread_id();
    BytesPack<16> *dst = reinterpret_cast<BytesPack<16> *>(
        (uint8_t *)src + Offset);
    size_t peer_offsets[] = {src_offset_0, src_offset_1, src_offset_2,
                             src_offset_3, src_offset_4, src_offset_5,
                             src_offset_6};
    for (int i = 0; i < NPeers; ++i) {
        int chan_idx = (Rank + i) % NPeers;
        const size_t index_offset4 = (peer_offsets[chan_idx] + Offset) / sizeof(int4);
        union {
            BytesPack<16> data;
            int4 val;
        } ret;
        for (int idx = tid; idx < nInt4; idx += total_threads) {
            BytesPack<16> tmp = dst[idx];
            ret.val = _ARK_SM_CHANS[chan_idx].read<int4>(index_offset4 + idx);
            add_half8(tmp, ret.data);
            dst[idx] = tmp;
        }
    }
}

template <typename Dims, typename Shape, typename UnitOutDims, int NumThreads,
          unsigned int NPeers, unsigned int Rank, unsigned long long Stride>
DEVICE void gather_from_peers_mscclpp(
    size_t ori_offset, size_t target_offset_0, size_t target_offset_1,
    size_t target_offset_2, size_t target_offset_3, size_t target_offset_4,
    size_t target_offset_5, size_t target_offset_6, ark::fp16 *, int uop_idx,
    int) {
    using UnitOp = UnitOp<Dims, Shape, UnitOutDims, NumThreads, 0>;
    constexpr size_t shape_width = Shape::W * sizeof(ark::fp16);
    constexpr size_t output_width = UnitOutDims::W * sizeof(ark::fp16);
    const int tid = UnitOp::thread_id();
    const int tile_hid = UnitOp::uop_idx_h(uop_idx);
    const int tile_wid = UnitOp::uop_idx_w(uop_idx);
    const size_t offset_in_width =
        tile_wid * UnitOutDims::W * sizeof(ark::fp16);
    size_t bytes_per_width = UnitOutDims::W * sizeof(ark::fp16);
    if (offset_in_width + output_width > shape_width) {
        bytes_per_width = shape_width - offset_in_width;
    }
    // if (tid == 0 && uop_idx == 1) {
    //   printf("tile hid %d, tile wid %d offset in width %lld bytes per width "
    //          "%lld output width %lld shape width %lld\n",
    //          tile_hid, tile_wid, offset_in_width, bytes_per_width, output_width,
    //          shape_width);
    // }
    size_t peer_offsets[] = {target_offset_0, target_offset_1, target_offset_2,
                             target_offset_3, target_offset_4, target_offset_5,
                             target_offset_6};
#pragma unroll
    for (int i = 0; i < NPeers; ++i) {
        int chan_idx = (Rank + i) % NPeers;
        int remote_rank = chan_idx < Rank ? chan_idx : chan_idx + 1;
        for (int j = tile_hid * UnitOutDims::H;
             j < tile_hid * UnitOutDims::H + UnitOutDims::H; ++j) {
          size_t offset =
              shape_width * remote_rank + j * Stride + offset_in_width;
        //   if (tid == 0 && uop_idx == 1) {
        //     printf("chan idx %d, remote rank %d, offset %lld\n", chan_idx,
        //            remote_rank, offset);
        //   }
          _ARK_SM_CHANS[chan_idx].get(peer_offsets[chan_idx] + offset,
                                      ori_offset + offset, bytes_per_width, tid,
                                      NumThreads);
        }
    }
}


template <typename Dims, typename Shape, typename UnitOutDims, int NumThreads,
          unsigned int DstRank, unsigned int Rank, unsigned long long DstOffset,
          unsigned long long Length, int Flag>
DEVICE void put_packet_mscclpp(size_t dst_offset, size_t src_offset,
                               int uop_idx, int) {
    using UnitOp = UnitOp<Dims, Shape, UnitOutDims, NumThreads, 0>;
    constexpr int total_tiles =
        math::div_up<Shape::NCHW, UnitOutDims::NCHW>::value;
    constexpr int total_threads = total_tiles * NumThreads;
    constexpr int chan_idx = DstRank < Rank ? DstRank : DstRank - 1;
    const int tid = uop_idx * NumThreads + UnitOp::thread_id();
    _ARK_SM_CHANS[chan_idx].putPackets(dst_offset + DstOffset, src_offset,
                                       Length, tid, total_threads, Flag);
}

template <typename Dims, typename Shape, typename UnitOutDims, int NumThreads,
          unsigned int NPeers, unsigned int NElemsPerRank, unsigned int Rank,
          unsigned long long RemoteDstOffset, unsigned long long ScratchOffset,
          int Flag>
DEVICE void reduce_and_write_packet_mscclpp(
    ark::fp16 *dst, ark::fp16 *src, void *scratch, size_t peer_offset_0,
    size_t peer_offset_1, size_t peer_offset_2, size_t peer_offset_3,
    size_t peer_offset_4, size_t peer_offset_5, size_t peer_offset_6,
    int uop_idx, int) {
    // All channels have the same src_, so we can use any channel to get dst
    using UnitOp = UnitOp<Dims, Shape, UnitOutDims, NumThreads, 0>;
    constexpr int total_tiles =
        math::div_up<Shape::NCHW, UnitOutDims::NCHW>::value;
    constexpr int total_threads = total_tiles * NumThreads;
    constexpr int npackets_per_rank =
        NElemsPerRank * sizeof(ark::fp16) / (sizeof(mscclpp::LLPacket) / 2);
    uint8_t *scratch_base = (uint8_t *)scratch + ScratchOffset;
    const int tid = uop_idx * NumThreads + UnitOp::thread_id();
    size_t peer_offsets[] = {peer_offset_0, peer_offset_1, peer_offset_2,
                             peer_offset_3, peer_offset_4, peer_offset_5,
                             peer_offset_6};
    for (int idx = tid; idx < npackets_per_rank; idx += total_threads) {
        BytesPack<8> data;
        load(data, (uint2 *)src + idx);
        for (int index = 0; index < NPeers; index++) {
            const int remote_rank = index < Rank ? index : index + 1;
            mscclpp::LLPacket *pkt = (mscclpp::LLPacket *)(scratch_base) +
                                     remote_rank * npackets_per_rank;
            uint2 val = pkt[idx].read(Flag);
            BytesPack<8> packet;
            packet.u32[0] = val.x;
            packet.u32[1] = val.y;
            add_half4(data, packet);
        }
        store((uint2 *)dst + idx, data);
        for (int index = 0; index < NPeers; index++) {
            mscclpp::LLPacket *dst_pkt =
                (mscclpp::LLPacket *)((char *)_ARK_SM_CHANS[index].dst_ +
                                      peer_offsets[index] + RemoteDstOffset);
            dst_pkt[idx + Rank * npackets_per_rank].write(data.u32[0], data.u32[1], Flag);
        }
    }
}

template <typename Dims, typename Shape, typename UnitOutDims, int NumThreads,
          int NPacket, unsigned long long DstOffset,
          unsigned long long SrcOffset, int Flag>
DEVICE void get_from_packet_mscclpp(void *dst, void *src, int uop_idx, int) {
    using UnitOp = UnitOp<Dims, Shape, UnitOutDims, NumThreads, 0>;
    constexpr int total_tiles =
        math::div_up<Shape::NCHW, UnitOutDims::NCHW>::value;
    constexpr int total_threads = total_tiles * NumThreads;
    const int tid = uop_idx * NumThreads + UnitOp::thread_id();
    mscclpp::LLPacket *dst_pkt = (mscclpp::LLPacket *)((char *)src + SrcOffset);
    BytesPack<8> packet;
    uint2 *dst_pkt_base = (uint2 *)((char *)dst + DstOffset);
    for (int idx = tid; idx < NPacket; idx += total_threads) {
        uint2 data = dst_pkt[idx].read(Flag);
        packet.u32[0] = data.x;
        packet.u32[1] = data.y;
        store(dst_pkt_base + idx, packet);
    }
}

}  // namespace comm
}  // namespace ark

#endif  // ARK_KERNELS_COMM_H_
