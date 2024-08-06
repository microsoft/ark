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
            OutDataType result = in->read(Flag, -1);
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

            CompType::compute(out, in, scratch, args, idx_n, idx_c, idx_h,
                              idx_w);
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
        int idx = idx_n * InShape::CHW + idx_c * InShape::HW +
                  idx_h * InShape::W + idx_w;
        int idx_out = idx_n * OutDims::CHW + idx_c * OutDims::HW +
                      idx_h * OutDims::W + idx_w;
        int idx_in = idx_n * InDims::CHW + idx_c * InDims::HW +
                     idx_h * InDims::W + idx_w;
        uint32_t *output_offset = reinterpret_cast<uint32_t *>(args);

        DataType reduced[NelemPerThread];
        ark::load<sizeof(Payload), false>(reduced, in + idx_in);
#pragma unroll
        for (int i = 0; i < NPeers; ++i) {
            PacketType *pkg =
                scratch + (idx + i * NElemsPerRank) / NelemPerThread;
            Payload payload = pkg->read(Flag, -1);
            ReduceType::template reduce<NelemPerThread>(
                reduced, reduced, reinterpret_cast<DataType *>(&payload));
        }
        ark::store<sizeof(Payload), false>(out + idx_out, reduced);
#pragma unroll
        for (int i = 0; i < NPeers; ++i) {
            int remote_rank = i < Rank ? i : i + 1;
            Payload *payload = reinterpret_cast<Payload *>(reduced);
            char *output =
                reinterpret_cast<char *>(ARK_SM_CHANS[remote_rank].dst_) +
                output_offset[i];
            PacketType *pkg =
                reinterpret_cast<PacketType *>(output) + idx / NelemPerThread;
            pkg->write(*payload, Flag);
        }
    }
};

template <typename InDims, typename InShape, typename OutDims,
          typename ReduceType, typename _DataType, int _NelemPerThread,
          int Rank, int NPeers, uint32_t NElemsPerRank>
struct ReduceCompType {
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, DataType *scratch,
                               void *args, int idx_n, int idx_c, int idx_h,
                               int idx_w) {
        int idx = idx_n * InShape::CHW + idx_c * InShape::HW +
                  idx_h * InShape::W + idx_w;
        int idx_out = idx_n * OutDims::CHW + idx_c * OutDims::HW +
                      idx_h * OutDims::W + idx_w;
        int idx_in = idx_n * InDims::CHW + idx_c * InDims::HW +
                     idx_h * InDims::W + idx_w;
        uint32_t *output_offset = reinterpret_cast<uint32_t *>(args);

        DataType reduced[NelemPerThread];
        ark::load<sizeof(DataType) * NelemPerThread, false>(reduced,
                                                            in + idx_in);
#pragma unroll
        for (int i = 0; i < NPeers; ++i) {
            DataType *data = scratch + (idx + i * NElemsPerRank);
            ReduceType::template reduce<NelemPerThread>(reduced, reduced, data);
        }
        ark::store<sizeof(DataType) * NelemPerThread, false>(out + idx_out,
                                                             reduced);
#pragma unroll
        for (int i = 0; i < NPeers; ++i) {
            int remote_rank = i < Rank ? i : i + 1;
            char *output =
                reinterpret_cast<char *>(ARK_SM_CHANS[remote_rank].dst_) +
                output_offset[i];
            DataType *remote_out = reinterpret_cast<DataType *>(output) + idx;
            ark::store<sizeof(DataType) * NelemPerThread, false>(remote_out,
                                                                 reduced);
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
                      type::Identity, false, false, UnitOutDims, NumWarps,
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
                      type::Identity, false, false, UnitOutDims, NumWarps,
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
DEVICE void readPacket(int chan_id, size_t output_offset, size_t scratch_offset,
                       int uop_idx, [[maybe_unused]] int smem_per_warp) {
    using Payload = typename PacketType::Payload;
    char *base_addr = reinterpret_cast<char *>(ARK_SM_CHANS[chan_id].src_);
    char *scratch = base_addr + scratch_offset;
    char *output = base_addr + output_offset;
    PacketType *scratch_data = reinterpret_cast<PacketType *>(scratch);
    Payload *output_data = reinterpret_cast<Payload *>(output);
    Broadcast1<InDims, InShape, OutDims, OutShape, UnitOutDims, NumWarps,
               SmemBytes,
               PacketIntrinsic<PacketType, Payload, PacketType, false, true,
                               Flag>>::run(output_data, scratch_data, uop_idx);
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

template <comm::ChannelType ChanType, int RemoteRank, int64_t MaxSpinCount = -1,
          bool Wait = true>
DEVICE void wait(int, int) {
    if constexpr (!Wait) {
        return;
    }
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

template <int RemoteRank, typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename PacketType, int Flag>
DEVICE void read_packet(size_t dst_offset, size_t src_offset, int uop_idx,
                        int) {
    comm::readPacket<InDims, InShape, OutDims, OutShape, UnitOutDims, NumWarps,
                     SmemBytes, PacketType, Flag>(RemoteRank, dst_offset,
                                                  src_offset, uop_idx, 0);
}

// TODO: add reduce type in future
template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          unsigned int NPeers, unsigned int Rank, typename PacketType,
          typename DataType, int Flag = 1>
DEVICE void read_reduce_and_write(
    DataType *dst, DataType *src, void *scratch_base, uint32_t peer_offset_0,
    uint32_t peer_offset_1, uint32_t peer_offset_2, uint32_t peer_offset_3,
    uint32_t peer_offset_4, uint32_t peer_offset_5, uint32_t peer_offset_6,
    int uop_idx, int) {
    constexpr unsigned int nelems_per_rank = InShape::NCHW;
    uint32_t peer_offsets[] = {peer_offset_0, peer_offset_1, peer_offset_2,
                               peer_offset_3, peer_offset_4, peer_offset_5,
                               peer_offset_6};
    if constexpr (std::is_same_v<PacketType, DataType>) {
        DataType *scratch = reinterpret_cast<DataType *>(scratch_base);
        constexpr int NelemPerThread =
            DefaultNelemPerThread<OutDims, DataType, UnitOutDims>::value;
        comm::PacketReduce<
            OutDims, OutShape, UnitOutDims, NumWarps, SmemBytes, PacketType,
            comm::ReduceCompType<InDims, InShape, OutDims, ReduceTypeSum,
                                 DataType, NelemPerThread, Rank, NPeers,
                                 nelems_per_rank>>::run(dst, src, scratch,
                                                        peer_offsets, uop_idx);
    }
    else {
        PacketType *scratch = reinterpret_cast<PacketType *>(scratch_base);
        comm::PacketReduce<
            OutDims, OutShape, UnitOutDims, NumWarps, SmemBytes, PacketType,
            comm::PacketReduceCompType<
                InDims, InShape, OutDims, PacketType, ReduceTypeSum, DataType,
                Rank, NPeers, nelems_per_rank, Flag>>::run(dst, src, scratch,
                                                           peer_offsets,
                                                           uop_idx);
    }
}

template <comm::ChannelType ChanType, unsigned int NPeers, unsigned int Rank>
DEVICE void device_sync(int, int) {
    using UnitOp = UnitOp<ark::Vec<>, ark::Vec<>, ark::Vec<>, 1, 0>;
    int tid = UnitOp::thread_id();
    if (tid < NPeers) {
        int remote_rank = tid < Rank ? tid : tid + 1;
        comm::signal<ChanType>(remote_rank);
        comm::wait<ChanType>(remote_rank);
    }
}

}  // namespace ark

#endif  // ARK_KERNELS_COMM_H_
