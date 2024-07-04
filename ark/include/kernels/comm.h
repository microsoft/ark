// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_COMM_H_
#define ARK_KERNELS_COMM_H_

#include <mscclpp/packet_device.hpp>
#include <mscclpp/proxy_channel_device.hpp>
#include <mscclpp/sm_channel_device.hpp>

#include "common/atomic.h"
#include "common/broadcast.h"
#include "common/fp16.h"
#include "common/unit_op.h"

extern __constant__ mscclpp::SimpleProxyChannelDeviceHandle ARK_PROXY_CHANS[];
extern __constant__ mscclpp::SimpleProxyChannelDeviceHandle
    ARK_PROXY_SECONDARY_CHANS[];
extern __constant__ mscclpp::SmChannelDeviceHandle ARK_SM_CHANS[];

namespace ark {
namespace comm {

enum class ChannelType {
    Proxy,
    SecondaryProxy,
    Sm,
};

template <ChannelType ChanType>
DEVICE void signal(int ChanId) {
    if constexpr (ChanType == ChannelType::Proxy) {
        ARK_PROXY_CHANS[ChanId].signal();
    } else if constexpr (ChanType == ChannelType::SecondaryProxy) {
        ARK_PROXY_SECONDARY_CHANS[ChanId].signal();
    } else if constexpr (ChanType == ChannelType::Sm) {
        ARK_SM_CHANS[ChanId].signal();
    }
}

template <ChannelType ChanType, int64_t MaxSpinCount = -1>
DEVICE void wait(int ChanId) {
    if constexpr (ChanType == ChannelType::Proxy) {
        ARK_PROXY_CHANS[ChanId].wait(MaxSpinCount);
    } else if constexpr (ChanType == ChannelType::SecondaryProxy) {
        ARK_PROXY_SECONDARY_CHANS[ChanId].wait(MaxSpinCount);
    } else if constexpr (ChanType == ChannelType::Sm) {
        ARK_SM_CHANS[ChanId].wait(MaxSpinCount);
    }
}

template <ChannelType ChanType>
DEVICE void flush(int ChanId) {
    static_assert(ChanType == ChannelType::Proxy ||
                      ChanType == ChannelType::SecondaryProxy,
                  "Invalid channel type");
    if constexpr (ChanType == ChannelType::Proxy) {
        ARK_PROXY_CHANS[ChanId].flush();
    } else if constexpr (ChanType == ChannelType::SecondaryProxy) {
        ARK_PROXY_SECONDARY_CHANS[ChanId].flush();
    }
}

template <ChannelType ChanType>
DEVICE void put(int ChanId, size_t remote_offset, size_t local_offset,
                size_t bytes) {
    static_assert(ChanType == ChannelType::Proxy ||
                      ChanType == ChannelType::SecondaryProxy,
                  "Invalid channel type");
    if constexpr (ChanType == ChannelType::Proxy) {
        ARK_PROXY_CHANS[ChanId].put(remote_offset, local_offset, bytes);
    } else if constexpr (ChanType == ChannelType::SecondaryProxy) {
        ARK_PROXY_SECONDARY_CHANS[ChanId].put(remote_offset, local_offset,
                                              bytes);
    }
}

template <ChannelType ChanType>
DEVICE void putWithSignal(int ChanId, size_t remote_offset, size_t local_offset,
                          size_t bytes) {
    static_assert(ChanType == ChannelType::Proxy ||
                      ChanType == ChannelType::SecondaryProxy,
                  "Invalid channel type");
    if constexpr (ChanType == ChannelType::Proxy) {
        ARK_PROXY_CHANS[ChanId].putWithSignal(remote_offset, local_offset,
                                              bytes);
    } else if constexpr (ChanType == ChannelType::SecondaryProxy) {
        ARK_PROXY_SECONDARY_CHANS[ChanId].putWithSignal(remote_offset,
                                                        local_offset, bytes);
    }
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename DataType>
DEVICE void read(int ChanId, size_t remote_offset, size_t local_offset,
                 int uop_idx, [[maybe_unused]] int smem_per_warp) {
    const mscclpp::SmChannelDeviceHandle &chan = ARK_SM_CHANS[ChanId];
    char *local = reinterpret_cast<char *>(chan.src_) + local_offset;
    char *remote = reinterpret_cast<char *>(chan.dst_) + remote_offset;
    DataType *local_data = reinterpret_cast<DataType *>(local);
    DataType *remote_data = reinterpret_cast<DataType *>(remote);
    DefaultBroadcast1<InDims, InShape, DataType, OutDims, OutShape, DataType,
                      type::Identity, true, false, UnitOutDims, NumWarps,
                      SmemBytes>::run(local_data, remote_data, uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename DataType>
DEVICE void write(int ChanId, size_t remote_offset, size_t local_offset,
                  int uop_idx, [[maybe_unused]] int smem_per_warp) {
    const mscclpp::SmChannelDeviceHandle &chan = ARK_SM_CHANS[ChanId];
    char *local = reinterpret_cast<char *>(chan.src_) + local_offset;
    char *remote = reinterpret_cast<char *>(chan.dst_) + remote_offset;
    DataType *local_data = reinterpret_cast<DataType *>(local);
    DataType *remote_data = reinterpret_cast<DataType *>(remote);
    DefaultBroadcast1<InDims, InShape, DataType, OutDims, OutShape, DataType,
                      type::Identity, false, true, UnitOutDims, NumWarps,
                      SmemBytes>::run(remote_data, local_data, uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename PacketType, int Flag>
DEVICE void writePacket(int chan_id, size_t remote_offset, size_t local_offset,
                        int uop_idx, [[maybe_unused]] int smem_per_warp) {
    using Instruct = type::DataToPacket<Flag, PacketType>;
    using Payload = typename PacketType::Payload;
    const mscclpp::SmChannelDeviceHandle &chan = ARK_SM_CHANS[chan_id];
    char *local = reinterpret_cast<char *>(chan.src_) + local_offset;
    char *remote = reinterpret_cast<char *>(chan.dst_) + remote_offset;
    Payload *local_data = reinterpret_cast<Payload *>(local);
    PacketType *remote_data = reinterpret_cast<PacketType *>(remote);
    DefaultBroadcast1<InDims, InShape, Payload, OutDims, OutShape, PacketType,
                      Instruct, false, false, UnitOutDims, NumWarps,
                      SmemBytes>::run(remote_data, local_data, uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename PacketType, int Flag>
DEVICE void readPacket(size_t output_offset, size_t scratch_offset, int uop_idx,
                       [[maybe_unused]] int smem_per_warp) {
    using Instruct = type::PacketToData<Flag, PacketType>;
    using Payload = typename PacketType::Payload;
    char *base_addr = reinterpret_cast<char *>(ARK_SM_CHANS[0].src_);
    char *scratch = base_addr + local_offset;
    char *output = base_addr + output_offset;
    PacketType *scratch_data = reinterpret_cast<PacketType *>(scratch);
    Payload *output_data = reinterpret_cast<Payload *>(output);
    DefaultBroadcast1<InDims, InShape, PacketType, OutDims, OutShape, Payload,
                      Instruct, false, false, UnitOutDims, NumWarps,
                      SmemBytes>::run(output_data, scratch_data, uop_idx);
}

template <int NBytes>
union BytesPack {};

template <>
union BytesPack<16> {
    uint16_t u16[8];
    uint32_t u32[4];
    uint64_t u64[2];
    ulonglong2 u128;
};

template <>
union BytesPack<8> {
    uint16_t u16[4];
    uint32_t u32[2];
    uint64_t u64;
};

DEVICE void store(ulonglong2 *p, const BytesPack<16> &v) {
#if defined(ARK_TARGET_CUDA_ARCH)
    asm volatile("st.volatile.global.v2.b64 [%0], {%1,%2};"
                 :
                 : "l"(p), "l"(v.u64[0]), "l"(v.u64[1])
                 : "memory");
#else   // !defined(ARK_TARGET_CUDA_ARCH)
    atomicStoreRelaxed(reinterpret_cast<uint64_t *>(&(p->x)), v.u64[0]);
    atomicStoreRelaxed(reinterpret_cast<uint64_t *>(&(p->y)), v.u64[1]);
#endif  // !defined(ARK_TARGET_CUDA_ARCH)
}

DEVICE void store(uint64_t *p, const BytesPack<8> &v) {
    atomicStoreRelaxed(p, v.u64);
}

DEVICE void add_half8(BytesPack<16> &dst, BytesPack<16> &src) {
    __half2 *pd = reinterpret_cast<__half2 *>(dst.u32);
    __half2 *ps = reinterpret_cast<__half2 *>(src.u32);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        union {
            __half2 h2;
            uint32_t u32;
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

template <unsigned int NRanks>
DEVICE void device_sync(int, int) {
    using UnitOp = UnitOp<ark::Vec<>, ark::Vec<>, ark::Vec<>, 1, 0>;
    if (UnitOp::thread_id() != 0) {
        return;
    }
    for (int i = 0; i < NRanks - 1; ++i) {
        ARK_SM_CHANS[i].signal();
    }
    for (int i = 0; i < NRanks - 1; ++i) {
        ARK_SM_CHANS[i].wait(-1);
    }
}

// Do reduce scatter in a single node
template <typename Dims, typename Shape, typename UnitOutDims, int NumWarps,
          unsigned int NPeers, unsigned int Rank, unsigned long long Offset,
          unsigned long long Length>
DEVICE void ring_read_and_reduce(size_t src_offset_0, size_t src_offset_1,
                                 size_t src_offset_2, size_t src_offset_3,
                                 size_t src_offset_4, size_t src_offset_5,
                                 size_t src_offset_6, ark::fp16 *src,
                                 int uop_idx, int) {
    // treat channel dst as src since we read from it, and reduce to local
    // memory
    using UnitOp = UnitOp<Dims, Shape, UnitOutDims, NumWarps, 0>;
    constexpr int total_tiles =
        math::div_up<Shape::NCHW, UnitOutDims::NCHW>::value;
    constexpr int total_threads = total_tiles * UnitOp::NumThreads;
    constexpr size_t nInt4 = Length / sizeof(int4);
    const int tid = uop_idx * UnitOp::NumThreads + UnitOp::thread_id();
    BytesPack<16> *dst =
        reinterpret_cast<BytesPack<16> *>((uint8_t *)src + Offset);
    size_t peer_offsets[] = {src_offset_0, src_offset_1, src_offset_2,
                             src_offset_3, src_offset_4, src_offset_5,
                             src_offset_6};
    for (int i = 0; i < NPeers; ++i) {
        int chan_idx = (Rank + i) % NPeers;
        const size_t index_offset4 =
            (peer_offsets[chan_idx] + Offset) / sizeof(int4);
        union {
            BytesPack<16> data;
            int4 val;
        } ret;
        for (int idx = tid; idx < nInt4; idx += total_threads) {
            BytesPack<16> tmp = dst[idx];
            ret.val = ARK_SM_CHANS[chan_idx].read<int4>(index_offset4 + idx);
            add_half8(tmp, ret.data);
            store((ulonglong2 *)&dst[idx], tmp);
        }
    }
}

// Do reduce scatter in a single node with AMD
template <typename Dims, typename Shape, typename UnitOutDims, int NumWarps,
          unsigned int NPeers, unsigned int Rank, unsigned long long Offset,
          unsigned long long Length>
DEVICE void parallel_read_and_reduce(size_t src_offset_0, size_t src_offset_1,
                                     size_t src_offset_2, size_t src_offset_3,
                                     size_t src_offset_4, size_t src_offset_5,
                                     size_t src_offset_6, ark::fp16 *src,
                                     int uop_idx, int) {
    // treat channel dst as src since we read from it, and reduce to local
    // memory
    using UnitOp = UnitOp<Dims, Shape, UnitOutDims, NumWarps, 0>;
    constexpr int total_tiles =
        math::div_up<Shape::NCHW, UnitOutDims::NCHW>::value;
    constexpr int total_threads = total_tiles * UnitOp::NumThreads;
    constexpr size_t nInt4 = Length / sizeof(int4);
    const int tid = uop_idx * UnitOp::NumThreads + UnitOp::thread_id();
    BytesPack<16> *dst =
        reinterpret_cast<BytesPack<16> *>((uint8_t *)src + Offset);
    size_t peer_offsets[] = {src_offset_0, src_offset_1, src_offset_2,
                             src_offset_3, src_offset_4, src_offset_5,
                             src_offset_6};
    for (int idx = tid; idx < nInt4; idx += total_threads) {
        BytesPack<16> tmp = dst[idx];
        for (int i = 0; i < NPeers; ++i) {
            int chan_idx = (Rank + i) % NPeers;
            const size_t index_offset4 =
                (peer_offsets[chan_idx] + Offset) / sizeof(int4);
            union {
                BytesPack<16> data;
                int4 val;
            } ret;
            ret.val = ARK_SM_CHANS[chan_idx].read<int4>(index_offset4 + idx);
            add_half8(tmp, ret.data);
        }
        store((ulonglong2 *)&dst[idx], tmp);
    }
}

template <typename Dims, typename Shape, typename UnitOutDims, int NumWarps,
          unsigned int NPeers, unsigned int Rank, unsigned long long Offset,
          unsigned long long Length>
DEVICE void read_and_reduce(size_t src_offset_0, size_t src_offset_1,
                            size_t src_offset_2, size_t src_offset_3,
                            size_t src_offset_4, size_t src_offset_5,
                            size_t src_offset_6, ark::fp16 *src, int uop_idx,
                            int) {
    // TODO: support length not multiple of 16
    static_assert(Length % sizeof(int4) == 0, "Length must be multiple of 16");
#if defined(ARK_TARGET_CUDA_ARCH)
    return ring_read_and_reduce<Dims, Shape, UnitOutDims, NumWarps, NPeers,
                                Rank, Offset, Length>(
        src_offset_0, src_offset_1, src_offset_2, src_offset_3, src_offset_4,
        src_offset_5, src_offset_6, src, uop_idx, 0);
#else   // !defined(ARK_TARGET_CUDA_ARCH)
    return parallel_read_and_reduce<Dims, Shape, UnitOutDims, NumWarps, NPeers,
                                    Rank, Offset, Length>(
        src_offset_0, src_offset_1, src_offset_2, src_offset_3, src_offset_4,
        src_offset_5, src_offset_6, src, uop_idx, 0);
#endif  // !defined(ARK_TARGET_CUDA_ARCH)
}

template <typename Dims, typename Shape, typename UnitOutDims, int NumWarps,
          unsigned int NPeers, unsigned int Rank, unsigned long long Stride>
DEVICE void ring_gather_from_peers(
    size_t ori_offset, size_t target_offset_0, size_t target_offset_1,
    size_t target_offset_2, size_t target_offset_3, size_t target_offset_4,
    size_t target_offset_5, size_t target_offset_6, ark::fp16 *, int uop_idx,
    int) {
    using UnitOp = UnitOp<Dims, Shape, UnitOutDims, NumWarps, 0>;
    constexpr size_t shape_width = Shape::W * sizeof(ark::fp16);
    constexpr size_t output_width = UnitOutDims::W * sizeof(ark::fp16);
    constexpr size_t stride = Dims::W * sizeof(ark::fp16);
    const int tid = UnitOp::thread_id();
    const int tile_hid = UnitOp::uop_idx_h(uop_idx);
    const int tile_wid = UnitOp::uop_idx_w(uop_idx);
    const size_t offset_in_width =
        tile_wid * UnitOutDims::W * sizeof(ark::fp16);
    size_t bytes_per_width = UnitOutDims::W * sizeof(ark::fp16);
    if (offset_in_width + output_width > shape_width) {
        bytes_per_width = shape_width - offset_in_width;
    }
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
                shape_width * remote_rank + j * stride + offset_in_width;
            ARK_SM_CHANS[chan_idx].get(peer_offsets[chan_idx] + offset,
                                       ori_offset + offset, bytes_per_width,
                                       tid, UnitOp::NumThreads);
        }
    }
}

template <typename Dims, typename Shape, typename UnitOutDims, int NumWarps,
          unsigned int NPeers, unsigned int Rank, unsigned long long Stride>
DEVICE void parallel_gather_from_peers(
    size_t ori_offset, size_t target_offset_0, size_t target_offset_1,
    size_t target_offset_2, size_t target_offset_3, size_t target_offset_4,
    size_t target_offset_5, size_t target_offset_6, ark::fp16 *, int uop_idx,
    int) {
    using UnitOp = UnitOp<Dims, Shape, UnitOutDims, NumWarps, 0>;
    constexpr size_t shape_width = Shape::W * sizeof(ark::fp16);
    constexpr size_t output_width = UnitOutDims::W * sizeof(ark::fp16);
    constexpr size_t stride = Dims::W * sizeof(ark::fp16);
    const int tid = UnitOp::thread_id();
    const int tile_hid = UnitOp::uop_idx_h(uop_idx);
    const int tile_wid = UnitOp::uop_idx_w(uop_idx);
    const size_t offset_in_width =
        tile_wid * UnitOutDims::W * sizeof(ark::fp16);
    size_t bytes_per_width = UnitOutDims::W * sizeof(ark::fp16);
    if (offset_in_width + output_width > shape_width) {
        bytes_per_width = shape_width - offset_in_width;
    }
    size_t peer_offsets[] = {target_offset_0, target_offset_1, target_offset_2,
                             target_offset_3, target_offset_4, target_offset_5,
                             target_offset_6};
    const size_t unit_size = bytes_per_width >= (16 * UnitOp::NumThreads)
                                 ? 16 * UnitOp::NumThreads
                                 : bytes_per_width;
    for (int i = tile_hid * UnitOutDims::H;
         i < tile_hid * UnitOutDims::H + UnitOutDims::H; ++i) {
        int base = 0;
        for (; base < bytes_per_width; base += unit_size) {
#pragma unroll
            for (int j = 0; j < NPeers; ++j) {
                int chan_idx = (Rank + j) % NPeers;
                int remote_rank = chan_idx < Rank ? chan_idx : chan_idx + 1;
                size_t offset = shape_width * remote_rank + i * stride +
                                offset_in_width + base;
                ARK_SM_CHANS[chan_idx].get(peer_offsets[chan_idx] + offset,
                                           ori_offset + offset, unit_size, tid,
                                           UnitOp::NumThreads);
            }
        }
        if (base < bytes_per_width) {
#pragma unroll
            for (int j = 0; j < NPeers; ++j) {
                int chan_idx = (Rank + j) % NPeers;
                int remote_rank = chan_idx < Rank ? chan_idx : chan_idx + 1;
                size_t offset = shape_width * remote_rank + i * stride +
                                offset_in_width + base;
                ARK_SM_CHANS[chan_idx].get(
                    peer_offsets[chan_idx] + offset, ori_offset + offset,
                    bytes_per_width - base, tid, UnitOp::NumThreads);
            }
        }
    }
}

template <typename Dims, typename Shape, typename UnitOutDims, int NumWarps,
          unsigned int NPeers, unsigned int Rank, unsigned long long Stride>
DEVICE void gather_from_peers(size_t ori_offset, size_t target_offset_0,
                              size_t target_offset_1, size_t target_offset_2,
                              size_t target_offset_3, size_t target_offset_4,
                              size_t target_offset_5, size_t target_offset_6,
                              ark::fp16 *, int uop_idx, int) {
#if defined(ARK_TARGET_CUDA_ARCH)
    return ring_gather_from_peers<Dims, Shape, UnitOutDims, NumWarps, NPeers,
                                  Rank, Stride>(
        ori_offset, target_offset_0, target_offset_1, target_offset_2,
        target_offset_3, target_offset_4, target_offset_5, target_offset_6,
        nullptr, uop_idx, 0);
#else   // !defined(ARK_TARGET_CUDA_ARCH)
    return parallel_gather_from_peers<Dims, Shape, UnitOutDims, NumWarps,
                                      NPeers, Rank, Stride>(
        ori_offset, target_offset_0, target_offset_1, target_offset_2,
        target_offset_3, target_offset_4, target_offset_5, target_offset_6,
        nullptr, uop_idx, 0);
#endif  // !defined(ARK_TARGET_CUDA_ARCH)
}

template <typename Dims, typename Shape, typename UnitOutDims, int NumWarps,
          unsigned int NPeers, unsigned int NElemsPerRank, unsigned int Rank,
          unsigned long long RemoteDstOffset, unsigned long long ScratchOffset,
          int Flag>
DEVICE void reduce_and_write_packet(ark::fp16 *dst, ark::fp16 *src,
                                    void *scratch, size_t peer_offset_0,
                                    size_t peer_offset_1, size_t peer_offset_2,
                                    size_t peer_offset_3, size_t peer_offset_4,
                                    size_t peer_offset_5, size_t peer_offset_6,
                                    int uop_idx, int) {
    // All channels have the same src_, so we can use any channel to get dst
    using UnitOp = UnitOp<Dims, Shape, UnitOutDims, NumWarps, 0>;
    constexpr int total_tiles =
        math::div_up<Shape::NCHW, UnitOutDims::NCHW>::value;
    constexpr int total_threads = total_tiles * UnitOp::NumThreads;
    constexpr int npackets_per_rank =
        NElemsPerRank * sizeof(ark::fp16) / (sizeof(mscclpp::LLPacket) / 2);
    uint8_t *scratch_base = (uint8_t *)scratch + ScratchOffset;
    const int tid = uop_idx * UnitOp::NumThreads + UnitOp::thread_id();
    size_t peer_offsets[] = {peer_offset_0, peer_offset_1, peer_offset_2,
                             peer_offset_3, peer_offset_4, peer_offset_5,
                             peer_offset_6};
    for (int idx = tid; idx < npackets_per_rank; idx += total_threads) {
        BytesPack<8> data;
        data.u64 = *((uint64_t *)src + idx);
        for (int index = 0; index < NPeers; index++) {
            const int remote_rank = index < Rank ? index : index + 1;
            mscclpp::LLPacket *pkt = (mscclpp::LLPacket *)(scratch_base) +
                                     remote_rank * npackets_per_rank;
            uint2 val = pkt[idx].read(Flag);
            BytesPack<8> packet;
            packet.u64 = *reinterpret_cast<uint64_t *>(&val);
            add_half4(data, packet);
        }
        store((uint64_t *)dst + idx, data);
        for (int index = 0; index < NPeers; index++) {
            mscclpp::LLPacket *dst_pkt =
                (mscclpp::LLPacket *)((char *)ARK_SM_CHANS[index].dst_ +
                                      peer_offsets[index] + RemoteDstOffset);
            dst_pkt[idx + Rank * npackets_per_rank].write(data.u32[0],
                                                          data.u32[1], Flag);
        }
    }
}

template <typename Dims, typename Shape, typename UnitOutDims, int NumWarps,
          int NPacket, unsigned long long DstOffset,
          unsigned long long SrcOffset, int Flag>
DEVICE void get_from_packet(void *dst, void *src, int uop_idx, int) {
    using UnitOp = UnitOp<Dims, Shape, UnitOutDims, NumWarps, 0>;
    constexpr int total_tiles =
        math::div_up<Shape::NCHW, UnitOutDims::NCHW>::value;
    constexpr int total_threads = total_tiles * UnitOp::NumThreads;
    const int tid = uop_idx * UnitOp::NumThreads + UnitOp::thread_id();
    mscclpp::LLPacket *dst_pkt = (mscclpp::LLPacket *)((char *)src + SrcOffset);
    BytesPack<8> packet;
    uint64_t *dst_pkt_base = (uint64_t *)((char *)dst + DstOffset);
    for (int idx = tid; idx < NPacket; idx += total_threads) {
        uint2 data = dst_pkt[idx].read(Flag);
        packet.u64 = *reinterpret_cast<uint64_t *>(&data);
        store(dst_pkt_base + idx, packet);
    }
}

}  // namespace comm

template <comm::ChannelType ChanType, bool Signal, int RemoteRank,
          typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename DataType>
DEVICE void put(size_t dst_offset, size_t src_offset, int uop_idx, int) {
    using UnitOp = UnitOp<OutDims, OutShape, UnitOutDims, NumWarps, SmemBytes>;
    if constexpr (ChanType == comm::ChannelType::Sm) {
        comm::write<InDims, InShape, OutDims, OutShape, UnitOutDims, NumWarps,
                    SmemBytes, DataType>(RemoteRank, dst_offset, src_offset,
                                         uop_idx, 0);
        if constexpr (Signal) {
            if (UnitOp::thread_id() == 0) {
                comm::signal<ChanType>(RemoteRank);
            }
        }
    } else {
        // TODO: support multi-dimensional input/output.
        static_assert(InDims::W == InShape::W && InDims::H == InShape::H &&
                          InDims::C == InShape::C,
                      "multi-dimensional input is not supported");
        static_assert(OutDims::W == OutShape::W && OutDims::H == OutShape::H &&
                          OutDims::C == OutShape::C,
                      "multi-dimensional output is not supported");
        static_assert(InShape::NCHW == OutShape::NCHW,
                      "input and output sizes must be the same");
        if (UnitOp::thread_id() == 0) {
            constexpr size_t Bytes = sizeof(DataType) * InShape::NCHW;
            if constexpr (Signal) {
                comm::putWithSignal<ChanType>(RemoteRank, dst_offset,
                                              src_offset, Bytes);
            } else {
                comm::put<ChanType>(RemoteRank, dst_offset, src_offset, Bytes);
            }
        }
    }
}

template <comm::ChannelType ChanType, int RemoteRank>
DEVICE void flush(int, int) {
    if constexpr (ChanType != comm::ChannelType::Sm) {
        using UnitOp = UnitOp<ark::Vec<>, ark::Vec<>, ark::Vec<>, 1, 0>;
        if (UnitOp::thread_id() == 0) {
            comm::flush<ChanType>(RemoteRank);
        }
    }
}

template <comm::ChannelType ChanType, int RemoteRank, int64_t MaxSpinCount = -1>
DEVICE void wait(int, int) {
    using UnitOp = UnitOp<ark::Vec<>, ark::Vec<>, ark::Vec<>, 1, 0>;
    if (UnitOp::thread_id() == 0) {
        comm::wait<ChanType, MaxSpinCount>(RemoteRank);
    }
}

template <int RemoteRank, typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename PacketType, int Flag>
DEVICE void write_packet(size_t dst_offset, size_t src_offset, int uop_idx,
                         int) {
    comm::writePacket<InDims, InShape, OutDims, OutShape, UnitOutDims, NumWarps,
                      SmemBytes, PacketType, Flag>(RemoteRank, dst_offset,
                                                   src_offset, uop_idx, 0);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename PacketType, int Flag>
DEVICE void read_packet(size_t dst_offset, size_t src_offset, int uop_idx,
                        int) {
    comm::readPacket<InDims, InShape, OutDims, OutShape, UnitOutDims, NumWarps,
                     SmemBytes, PacketType, Flag>(dst_offset, src_offset,
                                                  uop_idx, 0);
}

}  // namespace ark

#endif  // ARK_KERNELS_COMM_H_
