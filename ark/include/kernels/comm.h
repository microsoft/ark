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
#include "reduce.h"

extern __constant__ mscclpp::SimpleProxyChannelDeviceHandle ARK_PROXY_CHANS[];
extern __constant__ mscclpp::SimpleProxyChannelDeviceHandle
    ARK_PROXY_SECONDARY_CHANS[];
extern __constant__ mscclpp::SmChannelDeviceHandle ARK_SM_CHANS[];

namespace ark {
namespace comm {

template <typename InDataType, typename OutDataType, typename PacketType,
          bool WritePacket, bool ReadPacket, uint32_t Flag>
struct PacketIntrinsic {
    using InputType = InDataType;
    using OutputType = OutDataType;
    using Payload = typename PacketType::Payload;

    // Each thread deal with one packet at a time
    static constexpr int NelemPerThread = 1;
    static_assert(
        !WritePacket || std::is_same<InputType, Payload>::value,
        "InputType must be the same as Payload when WritePacket is true");
    static_assert(
        !ReadPacket || std::is_same<OutputType, Payload>::value,
        "OutputType must be the same as Payload when ReadPacket is true");

    static DEVICE void compute(OutputType *out, const InputType *in) {
        if constexpr (WritePacket) {
            InputType stage;
            ark::load<sizeof(InputType), false>(&stage, in);
            out->write(stage, Flag);
        }
        if constexpr (ReadPacket) {
            OutDataType result = in->read(Flag);
            ark::store<sizeof(OutputType), false>(out, &result);
        }
    }
};

template <typename OutDims, typename OutShape, typename UnitOutDims,
          int NumWarps, int SmemBytes, typename PacketType, typename CompType>
struct PacketReduce {
    using UnitOp = UnitOp<OutDims, OutShape, UnitOutDims, NumWarps, SmemBytes>;
    using DataType = typename CompType::DataType;
    static const int NelemPerThread = CompType::NelemPerThread;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");
    static_assert(UnitOutDims::W % NelemPerThread == 0,
                  "UnitOutDims::W must be divisible by NelemPerThread");

    static DEVICE void run(DataType *out, DataType *in, PacketType *scratch,
                           void *args, int uop_idx) {
        int un = UnitOp::uop_idx_n(uop_idx);
        int uc = UnitOp::uop_idx_c(uop_idx);
        int uh = UnitOp::uop_idx_h(uop_idx);
        int uw = UnitOp::uop_idx_w(uop_idx);

        for (int tid = UnitOp::thread_id();; tid += UnitOp::NumThreads) {
            int tid_w = (tid * NelemPerThread) % UnitOutDims::W;
            int tid_h =
                ((tid * NelemPerThread) / UnitOutDims::W) % UnitOutDims::H;
            int tid_c =
                ((tid * NelemPerThread) / UnitOutDims::HW) % UnitOutDims::C;
            int tid_n = (tid * NelemPerThread) / UnitOutDims::CHW;

            if (tid_n >= UnitOutDims::N) {
                break;
            }

            int idx_n = tid_n + un * UnitOutDims::N;
            int idx_c = tid_c + uc * UnitOutDims::C;
            int idx_h = tid_h + uh * UnitOutDims::H;
            int idx_w = tid_w + uw * UnitOutDims::W;

            CompType::compute(out, in, idx_n, idx_c, idx_h, idx_w, args);
        }
    }
};

template <typename InDims, typename InShape, typename OutDims,
          typename PacketType, typename ReduceType, typename _DataType,
          int Rank, int NPeers, uint32_t NElemsPerRank, uint32_t Flag>
struct PacketReduceCompType {
    using DataType = _DataType;
    using Payload = typename PacketType::Payload;
    static const int NelemPerThread = sizeof(Payload) / sizeof(DataType);

    static DEVICE void compute(DataType *out, DataType *in, PacketType *scratch,
                               void *args, int idx_n, int idx_c, int idx_h,
                               int idx_w) {
        int idx = idx_n * OutDims::CHW + idx_c * OutDims::HW +
                      idx_h * OutDims::W + idx_w;
        utin32_t *output_offset = reinterpret_cast<uint32_t *>(args);

        DataType[NelemPerThread] reduced;
        ark::load<sizeof(Payload), false>(reduced, in + idx);
#pragma unroll
        for (int i = 0; i < NPeers; ++i) {
            int remote_rank = i < Rank ? i : i + 1;
            PacketType *pkg = scratch + remote_rank * NElemsPerRank + idx;
            Payload payload = pkg->read(Flag);
            ReduceType::template reduce<NelemPerThread>(
                reduced, reduced, reinterpret_cast<DataType *>(&payload));
        }
        ark::store<sizeof(Payload), false>(out + idx, reduced);
#pragma unroll
        for (int i = 0; i < NPeers; ++i) {
            int remote_rank = i < Rank ? i : i + 1;
            Payload *payload = reinterpret_cast<Payload *> reduced;
            PacketType pkg(payload, Flag);
            ARK_SM_CHANS[remote_rank]->write(
                output_offset[i] + remote_rank * NElemsPerRank + idx, pkg);
        }
    }
};

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
    using Payload = typename PacketType::Payload;
    const mscclpp::SmChannelDeviceHandle &chan = ARK_SM_CHANS[chan_id];
    char *local = reinterpret_cast<char *>(chan.src_) + local_offset;
    char *remote = reinterpret_cast<char *>(chan.dst_) + remote_offset;
    Payload *local_data = reinterpret_cast<Payload *>(local);
    PacketType *remote_data = reinterpret_cast<PacketType *>(remote);
    Broadcast1<InDims, InShape, OutDims, OutShape, UnitOutDims, NumWarps,
               SmemBytes,
               PacketIntrinsic<Payload, PacketType, PacketType, true, false,
                               Flag>>::run(remote_data, local_data, uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename PacketType, uint32_t Flag>
DEVICE void readPacket(size_t output_offset, size_t scratch_offset, int uop_idx,
                       [[maybe_unused]] int smem_per_warp) {
    using Payload = typename PacketType::Payload;
    char *base_addr = reinterpret_cast<char *>(ARK_SM_CHANS[0].src_);
    char *scratch = base_addr + scratch_offset;
    char *output = base_addr + output_offset;
    PacketType *scratch_data = reinterpret_cast<PacketType *>(scratch);
    Payload *output_data = reinterpret_cast<Payload *>(output);
    Broadcast1<InDims, InShape, OutDims, OutShape, UnitOutDims, NumWarps,
               SmemBytes,
               PacketIntrinsic<PacketType, Payload, PacketType, false, true,
                               Flag>>::run(output_data, scratch_data, uop_idx);
}

// template <int NBytes>
// union BytesPack {};

// template <>
// union BytesPack<16> {
//     uint16_t u16[8];
//     uint32_t u32[4];
//     uint64_t u64[2];
//     ulonglong2 u128;
// };

// template <>
// union BytesPack<8> {
//     uint16_t u16[4];
//     uint32_t u32[2];
//     uint64_t u64;
// };

// DEVICE void store(ulonglong2 *p, const BytesPack<16> &v) {
// #if defined(ARK_TARGET_CUDA_ARCH)
//     asm volatile("st.volatile.global.v2.b64 [%0], {%1,%2};"
//                  :
//                  : "l"(p), "l"(v.u64[0]), "l"(v.u64[1])
//                  : "memory");
// #else   // !defined(ARK_TARGET_CUDA_ARCH)
//     atomicStoreRelaxed(reinterpret_cast<uint64_t *>(&(p->x)), v.u64[0]);
//     atomicStoreRelaxed(reinterpret_cast<uint64_t *>(&(p->y)), v.u64[1]);
// #endif  // !defined(ARK_TARGET_CUDA_ARCH)
// }

// DEVICE void store(uint64_t *p, const BytesPack<8> &v) {
//     atomicStoreRelaxed(p, v.u64);
// }

// DEVICE void add_half8(BytesPack<16> &dst, BytesPack<16> &src) {
//     __half2 *pd = reinterpret_cast<__half2 *>(dst.u32);
//     __half2 *ps = reinterpret_cast<__half2 *>(src.u32);
// #pragma unroll
//     for (int i = 0; i < 4; ++i) {
//         union {
//             __half2 h2;
//             uint32_t u32;
//         } d, s;
//         d.h2 = pd[i];
//         s.h2 = ps[i];
//         pd[i] = __hadd2(d.h2, s.h2);
//     }
// }

// DEVICE void add_half4(BytesPack<8> &dst, BytesPack<8> &src) {
//     __half *pd = reinterpret_cast<__half *>(dst.u16);
//     __half *ps = reinterpret_cast<__half *>(src.u16);
// #pragma unroll
//     for (int i = 0; i < 2; ++i) {
//         __half2 d, s;
//         d.x = pd[i * 2];
//         d.y = pd[i * 2 + 1];
//         s.x = ps[i * 2];
//         s.y = ps[i * 2 + 1];
//         d = __hadd2(d, s);
//         pd[i * 2] = d.x;
//         pd[i * 2 + 1] = d.y;
//     }
// }

// template <unsigned int NRanks>
// DEVICE void device_sync(int, int) {
//     using UnitOp = UnitOp<ark::Vec<>, ark::Vec<>, ark::Vec<>, 1, 0>;
//     if (UnitOp::thread_id() != 0) {
//         return;
//     }
//     for (int i = 0; i < NRanks - 1; ++i) {
//         ARK_SM_CHANS[i].signal();
//     }
//     for (int i = 0; i < NRanks - 1; ++i) {
//         ARK_SM_CHANS[i].wait(-1);
//     }
// }

// // Do reduce scatter in a single node
// template <typename Dims, typename Shape, typename UnitOutDims, int NumWarps,
//           unsigned int NPeers, unsigned int Rank, unsigned long long Offset,
//           unsigned long long Length>
// DEVICE void ring_read_and_reduce(size_t src_offset_0, size_t src_offset_1,
//                                  size_t src_offset_2, size_t src_offset_3,
//                                  size_t src_offset_4, size_t src_offset_5,
//                                  size_t src_offset_6, ark::fp16 *src,
//                                  int uop_idx, int) {
//     // treat channel dst as src since we read from it, and reduce to local
//     // memory
//     using UnitOp = UnitOp<Dims, Shape, UnitOutDims, NumWarps, 0>;
//     constexpr int total_tiles =
//         math::div_up<Shape::NCHW, UnitOutDims::NCHW>::value;
//     constexpr int total_threads = total_tiles * UnitOp::NumThreads;
//     constexpr size_t nInt4 = Length / sizeof(int4);
//     const int tid = uop_idx * UnitOp::NumThreads + UnitOp::thread_id();
//     BytesPack<16> *dst =
//         reinterpret_cast<BytesPack<16> *>((uint8_t *)src + Offset);
//     size_t peer_offsets[] = {src_offset_0, src_offset_1, src_offset_2,
//                              src_offset_3, src_offset_4, src_offset_5,
//                              src_offset_6};
//     for (int i = 0; i < NPeers; ++i) {
//         int chan_idx = (Rank + i) % NPeers;
//         const size_t index_offset4 =
//             (peer_offsets[chan_idx] + Offset) / sizeof(int4);
//         union {
//             BytesPack<16> data;
//             int4 val;
//         } ret;
//         for (int idx = tid; idx < nInt4; idx += total_threads) {
//             BytesPack<16> tmp = dst[idx];
//             ret.val = ARK_SM_CHANS[chan_idx].read<int4>(index_offset4 + idx);
//             add_half8(tmp, ret.data);
//             store((ulonglong2 *)&dst[idx], tmp);
//         }
//     }
// }

// // Do reduce scatter in a single node with AMD
// template <typename Dims, typename Shape, typename UnitOutDims, int NumWarps,
//           unsigned int NPeers, unsigned int Rank, unsigned long long Offset,
//           unsigned long long Length>
// DEVICE void parallel_read_and_reduce(size_t src_offset_0, size_t src_offset_1,
//                                      size_t src_offset_2, size_t src_offset_3,
//                                      size_t src_offset_4, size_t src_offset_5,
//                                      size_t src_offset_6, ark::fp16 *src,
//                                      int uop_idx, int) {
//     // treat channel dst as src since we read from it, and reduce to local
//     // memory
//     using UnitOp = UnitOp<Dims, Shape, UnitOutDims, NumWarps, 0>;
//     constexpr int total_tiles =
//         math::div_up<Shape::NCHW, UnitOutDims::NCHW>::value;
//     constexpr int total_threads = total_tiles * UnitOp::NumThreads;
//     constexpr size_t nInt4 = Length / sizeof(int4);
//     const int tid = uop_idx * UnitOp::NumThreads + UnitOp::thread_id();
//     BytesPack<16> *dst =
//         reinterpret_cast<BytesPack<16> *>((uint8_t *)src + Offset);
//     size_t peer_offsets[] = {src_offset_0, src_offset_1, src_offset_2,
//                              src_offset_3, src_offset_4, src_offset_5,
//                              src_offset_6};
//     for (int idx = tid; idx < nInt4; idx += total_threads) {
//         BytesPack<16> tmp = dst[idx];
//         for (int i = 0; i < NPeers; ++i) {
//             int chan_idx = (Rank + i) % NPeers;
//             const size_t index_offset4 =
//                 (peer_offsets[chan_idx] + Offset) / sizeof(int4);
//             union {
//                 BytesPack<16> data;
//                 int4 val;
//             } ret;
//             ret.val = ARK_SM_CHANS[chan_idx].read<int4>(index_offset4 + idx);
//             add_half8(tmp, ret.data);
//         }
//         store((ulonglong2 *)&dst[idx], tmp);
//     }
// }

// template <typename Dims, typename Shape, typename UnitOutDims, int NumWarps,
//           unsigned int NPeers, unsigned int Rank, unsigned long long Offset,
//           unsigned long long Length>
// DEVICE void read_and_reduce(size_t src_offset_0, size_t src_offset_1,
//                             size_t src_offset_2, size_t src_offset_3,
//                             size_t src_offset_4, size_t src_offset_5,
//                             size_t src_offset_6, ark::fp16 *src, int uop_idx,
//                             int) {
//     // TODO: support length not multiple of 16
//     static_assert(Length % sizeof(int4) == 0, "Length must be multiple of 16");
// #if defined(ARK_TARGET_CUDA_ARCH)
//     return ring_read_and_reduce<Dims, Shape, UnitOutDims, NumWarps, NPeers,
//                                 Rank, Offset, Length>(
//         src_offset_0, src_offset_1, src_offset_2, src_offset_3, src_offset_4,
//         src_offset_5, src_offset_6, src, uop_idx, 0);
// #else   // !defined(ARK_TARGET_CUDA_ARCH)
//     return parallel_read_and_reduce<Dims, Shape, UnitOutDims, NumWarps, NPeers,
//                                     Rank, Offset, Length>(
//         src_offset_0, src_offset_1, src_offset_2, src_offset_3, src_offset_4,
//         src_offset_5, src_offset_6, src, uop_idx, 0);
// #endif  // !defined(ARK_TARGET_CUDA_ARCH)
// }

// template <typename Dims, typename Shape, typename UnitOutDims, int NumWarps,
//           unsigned int NPeers, unsigned int Rank, unsigned long long Stride>
// DEVICE void ring_gather_from_peers(
//     size_t ori_offset, size_t target_offset_0, size_t target_offset_1,
//     size_t target_offset_2, size_t target_offset_3, size_t target_offset_4,
//     size_t target_offset_5, size_t target_offset_6, ark::fp16 *, int uop_idx,
//     int) {
//     using UnitOp = UnitOp<Dims, Shape, UnitOutDims, NumWarps, 0>;
//     constexpr size_t shape_width = Shape::W * sizeof(ark::fp16);
//     constexpr size_t output_width = UnitOutDims::W * sizeof(ark::fp16);
//     constexpr size_t stride = Dims::W * sizeof(ark::fp16);
//     const int tid = UnitOp::thread_id();
//     const int tile_hid = UnitOp::uop_idx_h(uop_idx);
//     const int tile_wid = UnitOp::uop_idx_w(uop_idx);
//     const size_t offset_in_width =
//         tile_wid * UnitOutDims::W * sizeof(ark::fp16);
//     size_t bytes_per_width = UnitOutDims::W * sizeof(ark::fp16);
//     if (offset_in_width + output_width > shape_width) {
//         bytes_per_width = shape_width - offset_in_width;
//     }
//     size_t peer_offsets[] = {target_offset_0, target_offset_1, target_offset_2,
//                              target_offset_3, target_offset_4, target_offset_5,
//                              target_offset_6};
// #pragma unroll
//     for (int i = 0; i < NPeers; ++i) {
//         int chan_idx = (Rank + i) % NPeers;
//         int remote_rank = chan_idx < Rank ? chan_idx : chan_idx + 1;
//         for (int j = tile_hid * UnitOutDims::H;
//              j < tile_hid * UnitOutDims::H + UnitOutDims::H; ++j) {
//             size_t offset =
//                 shape_width * remote_rank + j * stride + offset_in_width;
//             ARK_SM_CHANS[chan_idx].get(peer_offsets[chan_idx] + offset,
//                                        ori_offset + offset, bytes_per_width,
//                                        tid, UnitOp::NumThreads);
//         }
//     }
// }

// template <typename Dims, typename Shape, typename UnitOutDims, int NumWarps,
//           unsigned int NPeers, unsigned int Rank, unsigned long long Stride>
// DEVICE void parallel_gather_from_peers(
//     size_t ori_offset, size_t target_offset_0, size_t target_offset_1,
//     size_t target_offset_2, size_t target_offset_3, size_t target_offset_4,
//     size_t target_offset_5, size_t target_offset_6, ark::fp16 *, int uop_idx,
//     int) {
//     using UnitOp = UnitOp<Dims, Shape, UnitOutDims, NumWarps, 0>;
//     constexpr size_t shape_width = Shape::W * sizeof(ark::fp16);
//     constexpr size_t output_width = UnitOutDims::W * sizeof(ark::fp16);
//     constexpr size_t stride = Dims::W * sizeof(ark::fp16);
//     const int tid = UnitOp::thread_id();
//     const int tile_hid = UnitOp::uop_idx_h(uop_idx);
//     const int tile_wid = UnitOp::uop_idx_w(uop_idx);
//     const size_t offset_in_width =
//         tile_wid * UnitOutDims::W * sizeof(ark::fp16);
//     size_t bytes_per_width = UnitOutDims::W * sizeof(ark::fp16);
//     if (offset_in_width + output_width > shape_width) {
//         bytes_per_width = shape_width - offset_in_width;
//     }
//     size_t peer_offsets[] = {target_offset_0, target_offset_1, target_offset_2,
//                              target_offset_3, target_offset_4, target_offset_5,
//                              target_offset_6};
//     const size_t unit_size = bytes_per_width >= (16 * UnitOp::NumThreads)
//                                  ? 16 * UnitOp::NumThreads
//                                  : bytes_per_width;
//     for (int i = tile_hid * UnitOutDims::H;
//          i < tile_hid * UnitOutDims::H + UnitOutDims::H; ++i) {
//         int base = 0;
//         for (; base < bytes_per_width; base += unit_size) {
// #pragma unroll
//             for (int j = 0; j < NPeers; ++j) {
//                 int chan_idx = (Rank + j) % NPeers;
//                 int remote_rank = chan_idx < Rank ? chan_idx : chan_idx + 1;
//                 size_t offset = shape_width * remote_rank + i * stride +
//                                 offset_in_width + base;
//                 ARK_SM_CHANS[chan_idx].get(peer_offsets[chan_idx] + offset,
//                                            ori_offset + offset, unit_size, tid,
//                                            UnitOp::NumThreads);
//             }
//         }
//         if (base < bytes_per_width) {
// #pragma unroll
//             for (int j = 0; j < NPeers; ++j) {
//                 int chan_idx = (Rank + j) % NPeers;
//                 int remote_rank = chan_idx < Rank ? chan_idx : chan_idx + 1;
//                 size_t offset = shape_width * remote_rank + i * stride +
//                                 offset_in_width + base;
//                 ARK_SM_CHANS[chan_idx].get(
//                     peer_offsets[chan_idx] + offset, ori_offset + offset,
//                     bytes_per_width - base, tid, UnitOp::NumThreads);
//             }
//         }
//     }
// }

// template <typename Dims, typename Shape, typename UnitOutDims, int NumWarps,
//           unsigned int NPeers, unsigned int Rank, unsigned long long Stride>
// DEVICE void gather_from_peers(size_t ori_offset, size_t target_offset_0,
//                               size_t target_offset_1, size_t target_offset_2,
//                               size_t target_offset_3, size_t target_offset_4,
//                               size_t target_offset_5, size_t target_offset_6,
//                               ark::fp16 *, int uop_idx, int) {
// #if defined(ARK_TARGET_CUDA_ARCH)
//     return ring_gather_from_peers<Dims, Shape, UnitOutDims, NumWarps, NPeers,
//                                   Rank, Stride>(
//         ori_offset, target_offset_0, target_offset_1, target_offset_2,
//         target_offset_3, target_offset_4, target_offset_5, target_offset_6,
//         nullptr, uop_idx, 0);
// #else   // !defined(ARK_TARGET_CUDA_ARCH)
//     return parallel_gather_from_peers<Dims, Shape, UnitOutDims, NumWarps,
//                                       NPeers, Rank, Stride>(
//         ori_offset, target_offset_0, target_offset_1, target_offset_2,
//         target_offset_3, target_offset_4, target_offset_5, target_offset_6,
//         nullptr, uop_idx, 0);
// #endif  // !defined(ARK_TARGET_CUDA_ARCH)
// }


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

// TODO: add reduce type in future
template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          unsigned int NPeers, unsigned int Rank, unsigned int NElemsPerRank,
          typename PacketType, typename DataType, int Flag>
DEVICE void read_reduce_and_write_packet(
    size_t dst_offset, size_t src_offset, size_t scratch_offset,
    size_t peer_offset_0, size_t peer_offset_1, size_t peer_offset_2,
    size_t peer_offset_3, size_t peer_offset_4, size_t peer_offset_5,
    size_t peer_offset_6, int uop_idx, int) {
    using UnitOp = UnitOp<OutDims, OutShape, UnitOutDims, NumWarps, SmemBytes>;
    using Payload = typename PacketType::Payload;
    constexpr int NelemPerThread = sizeof(Payload) / sizeof(DataType);
    size_t peer_offsets[] = {peer_offset_0, peer_offset_1, peer_offset_2,
                             peer_offset_3, peer_offset_4, peer_offset_5,
                             peer_offset_6};
    DataType *dst = reinterpret_cast<DataType *>(ARK_SM_CHANS[0].src_) +
                    dst_offset / sizeof(DataType);
    DataType *src = reinterpret_cast<DataType *>(ARK_SM_CHANS[0].src_) +
                    src_offset / sizeof(DataType);
    PacketType *scratch = reinterpret_cast<PacketType *>(ARK_SM_CHANS[0].src_) +
                          scratch_offset / sizeof(PacketType);
    comm::PacketReduce<
        OutDims, OutShape, UnitOutDims, NumWarps, SmemBytes, PacketType,
        comm::PacketReduceCompType<InDims, InShape, OutDims, PacketType,
                                   ReduceTypeSum, DataType, Rank, NPeers,
                                   NElemsPerRank, Flag>>::run(dst, src, scratch,
                                                              peer_offsets,
                                                              uop_idx);
}

}  // namespace ark

#endif  // ARK_KERNELS_COMM_H_
