// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_COMM_H_
#define ARK_KERNELS_COMM_H_

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/packet_device.hpp>
#include <mscclpp/port_channel_device.hpp>

#include "common/atomic.h"
#include "common/broadcast.h"
#include "common/fp16.h"
#include "common/type_intrinsics.h"
#include "common/unit_op.h"
#include "reduce.h"

extern __constant__ mscclpp::PortChannelDeviceHandle ARK_PROXY_CHANS[];
extern __constant__ mscclpp::PortChannelDeviceHandle
    ARK_PROXY_SECONDARY_CHANS[];
extern __constant__ mscclpp::MemoryChannelDeviceHandle ARK_SM_CHANS[];

// Device-wide barrier shared by the read-based all-reduce kernel. Reused across
// launches (self-resetting via 3 rotating counters); zero-initialized.
__device__ mscclpp::DeviceSyncer ARK_ALLREDUCE_SYNCER;

// Monotonic sequence number for the RSAG cross-rank (flag) barrier. Incremented
// once per barrier per rank by block 0's thread 0; all ranks advance it in
// lockstep so a given barrier uses the same value everywhere. Zero-initialized
// at module load, so the first barrier value is 1 and a zero-filled flag slot
// never false-matches.
__device__ uint32_t ARK_RSAG_BARRIER_SEQ;

// Per-block monotonically incrementing flag for the one-shot packet all-reduce
// (mscclpp LL protocol). Persists across rt.run() launches; zero-initialized at
// module load, so the first flag used is 1 and a zero-filled scratch never
// false-matches a valid packet. Sized for the max blocks any all-reduce plan
// uses (default_config caps NumTasks well below this).
#ifndef ARK_AR_MAX_BLOCKS
#define ARK_AR_MAX_BLOCKS 512
#endif
__device__ uint32_t ARK_AR_ONESHOT_FLAGS[ARK_AR_MAX_BLOCKS];

// Per-block monotonically incrementing flag for the all-pairs packet all-reduce
// (a port of mscclpp's `allreduceAllPairs`, the NCCL-API small-message path).
// Separate from the one-shot flags so the two packet kernels never share flag
// slots. Zero-initialized at module load, so the first flag used is 1 and a
// zero-filled scratch never false-matches a valid packet.
__device__ uint32_t ARK_AR_ALLPAIR_FLAGS[ARK_AR_MAX_BLOCKS];

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
    const mscclpp::MemoryChannelDeviceHandle &chan = ARK_SM_CHANS[ChanId];
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
    const mscclpp::MemoryChannelDeviceHandle &chan = ARK_SM_CHANS[ChanId];
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
    const mscclpp::MemoryChannelDeviceHandle &chan = ARK_SM_CHANS[chan_id];
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
    } else {
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

// Tile-local one-shot LL16-packet all-reduce (the only packet all-reduce
// kernel). Block `uop_idx` reduces ONLY its own [TileRows, TileCols]
// column-tile of the [Rows, Cols] output — the SAME tile a fused preceding
// matmul wrote on this block. The matmul→AR handoff is therefore intra-block: a
// cheap warp-group barrier (UnitOp::sync_threads) instead of a grid barrier, and
// each tile's NVLink exchange overlaps other tiles' matmul compute. Cross-rank
// ordering is carried by the per-packet LL flag; the scratch is a
// 2*WorldSize*NPkts double-buffer indexed by full-array packet position, so
// every rank's block-for-tile-t writes/reads identical scratch slots. A tile
// grid covering the whole tensor gives the plain one-shot all-reduce behavior.
//
//   Rows, Cols         — output shape (compile-time).
//   TileRows, TileCols — column-tile shape matching the fused matmul's
//                        UnitOutDims. TileCols must divide Cols and be a
//                        multiple of ElemsPerPkt (= 8 / dtype_bytes = 4).
template <int NPeers, int Rank, int NumProcs, int NumWarps, typename PacketType,
          typename DataType, int Rows, int Cols, int TileRows, int TileCols>
DEVICE void allreduce_packet(DataType *output, DataType *input,
                             DataType *scratch, uint64_t scratch_offset,
                             int uop_idx, int /*sram_per_warp*/) {
    static_assert(NPeers >= 1, "Need at least one peer");
    static_assert(sizeof(DataType) == 2,
                  "packet allreduce supports 2-byte types (fp16/bf16)");
    constexpr int WorldSize = NPeers + 1;
    using Payload = typename PacketType::Payload;                    // uint2 (8B)
    constexpr int ElemsPerPkt = sizeof(Payload) / sizeof(DataType);  // 4 (bf16)
    static_assert(TileCols % ElemsPerPkt == 0,
                  "TileCols must be divisible by ElemsPerPkt");
    static_assert(Cols % ElemsPerPkt == 0,
                  "Cols must be divisible by ElemsPerPkt");
    static_assert(Cols % TileCols == 0, "TileCols must divide Cols");
    constexpr int PktsPerRow = Cols / ElemsPerPkt;          // full-row packets
    constexpr int TilePktsPerRow = TileCols / ElemsPerPkt;  // tile-row packets
    constexpr int NColTiles = Cols / TileCols;
    constexpr int NPkts = Rows * PktsPerRow;                // full-array packets
    constexpr uint64_t HalfPkts = static_cast<uint64_t>(WorldSize) * NPkts;

    // This block's column-tile — matches the fused matmul's uop_idx layout
    // (col-major within a row-tile; one row-tile when Rows <= TileRows, which
    // holds for the T<=64 down_proj prefill shape).
    const int col_tile = uop_idx % NColTiles;
    const int row_tile = uop_idx / NColTiles;
    const int col_pkt0 = col_tile * TilePktsPerRow;
    const int row0 = row_tile * TileRows;
    const int row1 = (row0 + TileRows < Rows) ? (row0 + TileRows) : Rows;
    const int tile_rows = (row1 > row0) ? (row1 - row0) : 0;
    const int tile_pkts = tile_rows * TilePktsPerRow;

    // Order this block's matmul tile writes (to `input` in HBM) before the
    // block's own AR reads below, using the op's warp-group barrier. NumWarps
    // MUST equal the fused matmul's NumWarps so the block holds exactly that
    // many warps and the named barrier is used consistently across the two ops
    // (a wider AR barrier would collide with the matmul's still-in-flight
    // warp-group barrier -> illegal instruction). barrier.sync also makes the
    // block's global writes visible across the group, so no __threadfence is
    // needed. Intra-block only — no grid barrier.
    using Uop = UnitOp<Vec<>, Vec<>, Vec<>, NumWarps, 0>;
    Uop::sync_threads();

    const int tid = Uop::thread_id();       // warp-group-local thread index
    const int nThreads = Uop::NumThreads;   // = NumWarps * ThreadsPerWarp
    const uint32_t flag = ARK_AR_ONESHOT_FLAGS[blockIdx.x] + 1u;
    const uint64_t half_off = static_cast<uint64_t>(flag & 1u) * HalfPkts;

    Payload *in_pl = reinterpret_cast<Payload *>(input);
    Payload *out_pl = reinterpret_cast<Payload *>(output);
    PacketType *local_scratch =
        reinterpret_cast<PacketType *>(scratch) + half_off;

    using D2 = typename std::conditional<std::is_same<DataType, fp16>::value,
                                         fp16x2, bf16x2>::type;

    // ----- step 1: write my tile's packets into every peer's scratch slot
    //       [Rank] (at the packets' natural full-array positions).
    for (int t = tid; t < tile_pkts; t += nThreads) {
        const int r = t / TilePktsPerRow;
        const int cp = t % TilePktsPerRow;
        const int idx = (row0 + r) * PktsPerRow + col_pkt0 + cp;
        Payload val = in_pl[idx];
#pragma unroll
        for (int i = 0; i < NPeers; ++i) {
            const int rr = (i < Rank) ? i : i + 1;
            PacketType *peer_scratch = reinterpret_cast<PacketType *>(
                reinterpret_cast<char *>(ARK_SM_CHANS[rr].dst_) +
                scratch_offset);
            (peer_scratch + half_off + static_cast<uint64_t>(Rank) * NPkts + idx)
                ->write(val, flag);
        }
    }

    // ----- step 2: reduce my tile's local input + every peer's packet (fp32)
    //       -> output tile.
    for (int t = tid; t < tile_pkts; t += nThreads) {
        const int r = t / TilePktsPerRow;
        const int cp = t % TilePktsPerRow;
        const int idx = (row0 + r) * PktsPerRow + col_pkt0 + cp;
        Payload mine = in_pl[idx];
        D2 *mp = reinterpret_cast<D2 *>(&mine);
        float2 a0 = type::Cast::compute<float2>(mp[0]);
        float2 a1 = type::Cast::compute<float2>(mp[1]);
#pragma unroll
        for (int i = 0; i < NPeers; ++i) {
            const int rr = (i < Rank) ? i : i + 1;
            Payload v =
                (local_scratch + static_cast<uint64_t>(rr) * NPkts + idx)
                    ->read(flag, -1);
            D2 *vp = reinterpret_cast<D2 *>(&v);
            float2 f0 = type::Cast::compute<float2>(vp[0]);
            float2 f1 = type::Cast::compute<float2>(vp[1]);
            a0.x += f0.x; a0.y += f0.y;
            a1.x += f1.x; a1.y += f1.y;
        }
        Payload outv;
        D2 *op = reinterpret_cast<D2 *>(&outv);
        op[0] = type::Cast::compute<D2>(a0);
        op[1] = type::Cast::compute<D2>(a1);
        out_pl[idx] = outv;
    }

    if (threadIdx.x == 0) {
        ARK_AR_ONESHOT_FLAGS[blockIdx.x] = flag;
    }
}

// All-pairs one-shot LL-packet all-reduce, a direct port of mscclpp's
// `allreduceAllPairs` (the NCCL-API small-message path, message size <= 16 KB).
// Standalone grid op (its own processor group): every block owns a grid-strided
// stripe of the whole [Nelems] array (NOT a tile), exactly like mscclpp's
// 28-block / 512-thread launch. Phase 1 -- one warp per peer writes this block's
// stripe as LL packets into that peer's scratch slot [Rank]. Phase 2 -- all
// warps reduce their share of the stripe (local input + every peer's packet,
// fp32 accum) into the output. No intra-block barrier is needed because the AR
// is out-of-place (output is a distinct buffer from input, as the free function
// always allocates): phase 1 writes only to peers' scratch, phase 2 reads only
// local input + peers' packets (cross-rank ordering carried by the per-packet LL
// flag) and writes only the distinct output. The scratch is a 2*WorldSize*NPkts
// double buffer indexed by flag parity. NumBlocks = the op's NumProcs (the
// planner overrides it to the actual processor count; see planner.cpp).
template <int NPeers, int Rank, int NumWarps, typename PacketType,
          typename DataType, int Nelems, int NumBlocks>
DEVICE void allreduce_allpair_packet(DataType *output, DataType *input,
                                     DataType *scratch, uint64_t scratch_offset,
                                     int /*uop_idx*/, int /*sram_per_warp*/) {
    static_assert(NPeers >= 1, "Need at least one peer");
    static_assert(sizeof(DataType) == 2,
                  "packet allreduce supports 2-byte types (fp16/bf16)");
    static_assert(NumWarps > NPeers,
                  "allpair packet needs at least one warp per peer");
    constexpr int WorldSize = NPeers + 1;
    using Payload = typename PacketType::Payload;  // uint32 (LL8) / uint2 (LL16)
    constexpr int ElemsPerPkt = sizeof(Payload) / sizeof(DataType);  // 2 or 4
    static_assert(Nelems % ElemsPerPkt == 0,
                  "Nelems must be divisible by ElemsPerPkt");
    constexpr int NPkts = Nelems / ElemsPerPkt;  // whole-array packet count
    constexpr int NPairs = ElemsPerPkt / 2;      // bf16x2 lanes per packet (1/2)
    constexpr uint64_t HalfPkts = static_cast<uint64_t>(WorldSize) * NPkts;

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gstride = NumBlocks * 32;  // grid-wide warp stride (in packets)

    const uint32_t flag = ARK_AR_ALLPAIR_FLAGS[bid] + 1u;
    const uint64_t half_off = static_cast<uint64_t>(flag & 1u) * HalfPkts;

    Payload *in_pl = reinterpret_cast<Payload *>(input);
    Payload *out_pl = reinterpret_cast<Payload *>(output);
    PacketType *local_scratch =
        reinterpret_cast<PacketType *>(scratch) + half_off;

    using D2 = typename std::conditional<std::is_same<DataType, fp16>::value,
                                         fp16x2, bf16x2>::type;

    // ----- phase 1: one warp per peer writes this block's grid-strided stripe
    //       into that peer's scratch slot [Rank].
    if (warp < NPeers) {
        const int rr = (warp < Rank) ? warp : warp + 1;
        PacketType *peer_scratch = reinterpret_cast<PacketType *>(
            reinterpret_cast<char *>(ARK_SM_CHANS[rr].dst_) + scratch_offset);
        PacketType *peer_slot =
            peer_scratch + half_off + static_cast<uint64_t>(Rank) * NPkts;
        for (int idx = lane + bid * 32; idx < NPkts; idx += gstride) {
            (peer_slot + idx)->write(in_pl[idx], flag);
        }
    }

    // ----- phase 2: reduce this block's stripe (all warps split it). Each
    //       output packet = local input + every peer's packet, fp32 accum.
    for (int idx = lane + bid * 32 + warp * gstride; idx < NPkts;
         idx += NumWarps * gstride) {
        Payload mine = in_pl[idx];
        D2 *mp = reinterpret_cast<D2 *>(&mine);
        float2 acc[NPairs];
#pragma unroll
        for (int j = 0; j < NPairs; ++j) {
            acc[j] = type::Cast::compute<float2>(mp[j]);
        }
#pragma unroll
        for (int i = 0; i < NPeers; ++i) {
            const int rr = (i < Rank) ? i : i + 1;
            Payload v =
                (local_scratch + static_cast<uint64_t>(rr) * NPkts + idx)
                    ->read(flag, -1);
            D2 *vp = reinterpret_cast<D2 *>(&v);
#pragma unroll
            for (int j = 0; j < NPairs; ++j) {
                float2 f = type::Cast::compute<float2>(vp[j]);
                acc[j].x += f.x;
                acc[j].y += f.y;
            }
        }
        Payload outv;
        D2 *op = reinterpret_cast<D2 *>(&outv);
#pragma unroll
        for (int j = 0; j < NPairs; ++j) {
            op[j] = type::Cast::compute<D2>(acc[j]);
        }
        out_pl[idx] = outv;
    }

    if (tid == 0) {
        ARK_AR_ALLPAIR_FLAGS[bid] = flag;
    }
}

// Read-based Reduce-Scatter + All-Gather all-reduce for LARGE messages.
// Bandwidth-optimal (O(N) inter-GPU traffic, int4-coalesced) vs the one-shot
// packet AR's O(N*peers). Mirrors mscclpp's allreduce_rsag. Three phases, each
// separated by a grid-wide (ARK_ALLREDUCE_SYNCER) + cross-rank barrier:
//   1. scatter:        copy my input -> my (peer-visible) scratch.
//   2. reduce-scatter: reduce ONLY my 1/WorldSize chunk from every peer's
//      scratch, push the reduced chunk to every peer's scratch + my output.
//   3. all-gather:     copy peers' reduced chunks from my scratch -> my output.
// The reduced chunk is accumulated in fp32 (bf16x2/fp16x2 -> float2).
// Requires Nelems % (WorldSize * ElemsPerInt4) == 0 (holds for [T, H] with H a
// multiple of 64 on 8 ranks), so there is no partial-chunk/remainder handling.
// The op is dispatched to processor blocks [0, NumBlocks), one uop (block)
// each, so blockIdx.x is the block's 0-based index and ARK_ALLREDUCE_SYNCER
// .sync(NumBlocks) scopes exactly the participating blocks. NumBlocks is the
// op's NumProcs (NOT gridDim.x, which is sized to the max across all ops).
//
// WRITE-BASED (push) reduce-scatter + all-gather, mirroring mscclpp
// allreduce_fullmesh (the A100/no-NVLS large-message path). All cross-GPU
// traffic is REMOTE WRITES (channel.write); all reads are LOCAL. Measured at
// NVLink NCCL-parity for 16-33MB (~158 GB/s busbw on 8xA100 NVSwitch). Steps:
//   1. Push: for each peer, write my contribution to that peer's chunk into the
//      peer's scratch at my rank's slot (remote write).
//   2. Reduce: my chunk = my local input + every peer's contribution read from
//      MY scratch (local reads), reduced in fp32; write to my output and push
//      to every peer's output (remote write) -- the all-gather.
// The scratch holds WorldSize*nInt4PerRank int4 (== Nelems elems) of received
// contributions, followed by the WorldSize-slot uint32 flag region.
template <int NPeers, int Rank, int NumWarps, typename DataType, int Nelems,
          int NumBlocks>
DEVICE void allreduce_rsag(DataType *output, DataType *input, DataType *scratch,
                           uint64_t output_offset, uint64_t scratch_offset,
                           int /*uop_idx*/, int /*sram_per_warp*/) {
    static_assert(sizeof(DataType) == 2,
                  "rsag allreduce supports 2-byte types (fp16/bf16)");
    constexpr int WorldSize = NPeers + 1;
    constexpr int ElemsPerInt4 = sizeof(int4) / sizeof(DataType);  // 8 (bf16)
    static_assert(Nelems % (WorldSize * ElemsPerInt4) == 0,
                  "Nelems must be divisible by WorldSize * ElemsPerInt4");
    constexpr int nInt4Total = Nelems / ElemsPerInt4;
    constexpr int nInt4PerRank = nInt4Total / WorldSize;
    // Flag region: WorldSize uint32 slots just past the data.
    constexpr uint64_t flag_byte_off =
        static_cast<uint64_t>(Nelems) * sizeof(DataType);

    constexpr int nBlocks = NumBlocks;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int nThreads = blockDim.x;

    int4 *in4 = reinterpret_cast<int4 *>(input);
    int4 *out4 = reinterpret_cast<int4 *>(output);
    int4 *scr4 = reinterpret_cast<int4 *>(scratch);
    uint32_t *my_flags = reinterpret_cast<uint32_t *>(
        reinterpret_cast<char *>(scratch) + flag_byte_off);

    // This block's contiguous slice within a chunk (load-balanced).
    int per = nInt4PerRank / nBlocks;
    int rem = nInt4PerRank % nBlocks;
    int off4 = bid * per + (bid < rem ? bid : rem);
    if (bid < rem) per += 1;

    using D2 = typename std::conditional<std::is_same<DataType, fp16>::value,
                                         fp16x2, bf16x2>::type;

    // Grid-wide + cross-rank barrier. All blocks call the two grid syncs; block
    // 0 (threads 0..NPeers-1) does the flag exchange in between: publish my
    // writes, write seq to peer rr's flag[Rank], then spin until my flag[rr]
    // reaches seq. Monotonic seq avoids stale matches; fences order data writes
    // before the flag. seq is warp-shuffle broadcast (no static shared memory).
    auto barrier = [&]() {
        ARK_ALLREDUCE_SYNCER.sync(nBlocks);
        if (bid == 0) {
            uint32_t seq = 0;
            if (tid == 0) {
                seq = atomicAdd(&ARK_RSAG_BARRIER_SEQ, 1u) + 1u;
            }
            if (tid < 32) {
                seq = __shfl_sync(0xffffffffu, seq, 0);
            }
            if (tid < NPeers) {
                int rr = (tid < Rank) ? tid : tid + 1;
                __threadfence_system();
                uint32_t *peer_flags = reinterpret_cast<uint32_t *>(
                    reinterpret_cast<char *>(ARK_SM_CHANS[rr].dst_) +
                    scratch_offset + flag_byte_off);
                peer_flags[Rank] = seq;
                __threadfence_system();
                volatile uint32_t *mf = my_flags;
                while (mf[rr] != seq) {
                }
            }
            __threadfence_system();
        }
        ARK_ALLREDUCE_SYNCER.sync(nBlocks);
    };

    // Barrier 1: every rank's input buffer is fully computed and visible.
    barrier();

    // Step 1 -- push: write my contribution to every peer's chunk into that
    // peer's scratch at MY rank's slot. Remote WRITE (fire-and-forget). The
    // peer order is staggered by blockIdx.x ((p+bid)%NPeers) so different blocks
    // target different peers concurrently (mirrors fullmesh; spreads writes over
    // the NVLink links). NOTE: the stagger was not itself the bottleneck -- an
    // earlier apparent ~30 GB/s cap was a benchmark artifact (a per-call input-
    // staging copy from a placeholder input, not present with an internal input).
    for (int p = 0; p < NPeers; ++p) {
        int pp = (p + bid) % NPeers;
        int rr = (pp < Rank) ? pp : pp + 1;
        int4 *peer_scr = reinterpret_cast<int4 *>(
            reinterpret_cast<char *>(ARK_SM_CHANS[rr].dst_) + scratch_offset);
        int src_base = rr * nInt4PerRank + off4;    // my data for peer rr's chunk
        int dst_base = Rank * nInt4PerRank + off4;  // peer rr's scratch slot Rank
        for (int i = tid; i < per; i += nThreads) {
            peer_scr[dst_base + i] = in4[src_base + i];
        }
    }
    barrier();

    // Step 2 -- reduce my chunk locally, then push the result to peers.
    int mybase = Rank * nInt4PerRank + off4;
    for (int i = tid; i < per; i += nThreads) {
        int oi = mybase + i;
        int4 acc = in4[oi];  // my own contribution
        D2 *a = reinterpret_cast<D2 *>(&acc);
#pragma unroll
        for (int p = 0; p < NPeers; ++p) {
            int rr = (p < Rank) ? p : p + 1;
            int4 val = scr4[rr * nInt4PerRank + off4 + i];  // LOCAL read
            D2 *b = reinterpret_cast<D2 *>(&val);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 fa = type::Cast::compute<float2>(a[k]);
                float2 fb = type::Cast::compute<float2>(b[k]);
                fa.x += fb.x;
                fa.y += fb.y;
                a[k] = type::Cast::compute<D2>(fa);
            }
        }
        out4[oi] = acc;  // my output chunk (local write)
#pragma unroll
        for (int p = 0; p < NPeers; ++p) {
            int pp = (p + bid) % NPeers;  // stagger -> spread across NVLink links
            int rr = (pp < Rank) ? pp : pp + 1;
            int4 *peer_out = reinterpret_cast<int4 *>(
                reinterpret_cast<char *>(ARK_SM_CHANS[rr].dst_) + output_offset);
            peer_out[oi] = acc;  // push reduced chunk to peer output (remote write)
        }
    }

    // Barrier 2: every peer has written its chunk into my output.
    barrier();
}

}  // namespace ark

#endif  // ARK_KERNELS_COMM_H_
