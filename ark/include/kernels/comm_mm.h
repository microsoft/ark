// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_COMM_MM_H_
#define ARK_COMM_MM_H_

#include "common.h"
#include "load_store.h"

namespace ark {
namespace comm {

union alignas(16) DataPacketLL {
    // on 64bits machine, the access to the 64bits data is atomic, so we combine
    // the 32bits data and flag into one 64bits data to insure the GPU receive
    // complete data
    struct {
        uint32_t data1;
        uint32_t flag1;
        uint32_t data2;
        uint32_t flag2;
    };
    longlong2 l2;
    int4 i4;
};

// the tensor's data is float16 data,so the matrix size is TDM * TDN *
// sizeof(float16). the data is stored in column major, and LDM is the
// major dimension. TDM and TDN is the tile size.
template <int FLAG>
DEVICE uint64_t readLL(DataPacketLL *recv_buf) {
    DataPacketLL pkt;
    do {
        pkt.l2 = load_volatile_128b(recv_buf);
    } while ((pkt.flag1 != FLAG) || (pkt.flag2 != FLAG));
    return pkt.data1 + (((uint64_t)pkt.data2) << sizeof(uint32_t));
}

template <int FLAG>
DEVICE void storeLL(DataPacketLL *data_dst, uint64_t val) {
    DataPacketLL pkt;
    pkt.data1 = val & 0xFFFFFFFF;
    pkt.flag1 = FLAG;
    pkt.data2 = val >> sizeof(uint32_t);
    pkt.flag2 = FLAG;
    store_volatile_128b(data_dst, pkt.l2);
}

template <int NumWarps>
DEVICE void pre_send_mm_op(volatile int *send_ready_flag, int uop_idx) {
    if (threadIdx.x % (NumWarps * Arch::ThreadsPerWarp) == 0) {
        while (send_ready_flag[uop_idx] != 0) {
        }
        send_ready_flag[uop_idx] = 1;
    }
    sync_warps<NumWarps>();
}

template <int NumWarps>
DEVICE void post_recv_mm_op(volatile int *send_ready_flag, int uop_idx) {
    sync_warps<NumWarps>();
    if (threadIdx.x % (NumWarps * Arch::ThreadsPerWarp) == 0) {
        // reset the send_ready_flag to 0
        send_ready_flag[uop_idx] = 0;
    }
    sync_warps<1>();
}

template <int LDM, int LDN, int NumWarps, int SmemBytes, int TDM, int TDN,
          int FLAG = 1>
// send a tile of the tensor from data_src to recv_buff
DEVICE void sendLL(void *recv_buff, fp16 *data_src,
                   volatile int *send_ready_flag, int uop_idx, int) {
    using UnitOp = UnitOp<Vec<1, 1, LDN, LDM>, Vec<1, 1, LDN, LDM>,
                          Vec<1, 1, TDN, TDM>, NumWarps, SmemBytes>;

    DataPacketLL *recv_buff_ptr = reinterpret_cast<DataPacketLL *>(recv_buff);

    // elementwise copy, a thread copies 4 float16 data (64 bits)
    // in a loop, so the ElePerLoop is 4
    constexpr int ElePerLoop = 4;

    constexpr int MNumPerLoop =
        math::min<TDM, ElePerLoop * UnitOp::NumThreads>::value;
    constexpr int NNumPerLoop =
        math::max<1, ElePerLoop * UnitOp::NumThreads / TDM>::value;

    constexpr int IterMNum = math::div_up<TDM, MNumPerLoop>::value;
    constexpr int IterNNum = TDN / NNumPerLoop;

    int t0 = UnitOp::uop_idx_w(uop_idx);
    int t1 = UnitOp::uop_idx_h(uop_idx);
    int midx = TDM * t0 + math::mod<TDM>(ElePerLoop * UnitOp::thread_id());
    int nidx = TDN * t1 + math::div<TDM>(ElePerLoop * UnitOp::thread_id());
    pre_send_mm_op<NumWarps>(send_ready_flag, uop_idx);
#pragma unroll
    for (int i = 0; i < IterNNum; ++i) {
#pragma unroll
        for (int j = 0; j < IterMNum; ++j) {
            int idx = midx + j * MNumPerLoop + (nidx + i * NNumPerLoop) * LDM;
            // in ARK the src and dst store the float16 data, so a
            // thread copied 4 float16 (64 bits) data in a loop
            storeLL<FLAG>(recv_buff_ptr + math::div<4>(idx),
                          *(uint64_t *)&((fp16 *)data_src)[idx]);
        }
    }
}

// recv a tile of the tensor from recv_buff to data_dst
template <int LDM, int LDN, int NumWarps, int SmemBytes, int TDM, int TDN,
          int FLAG = 1>
DEVICE void recvLL(void *recv_buff, fp16 *data_dst,
                   volatile int *send_ready_flag, int uop_idx, int) {
    using UnitOp = UnitOp<Vec<1, 1, LDN, LDM>, Vec<1, 1, LDN, LDM>,
                          Vec<1, 1, TDN, TDM>, NumWarps, SmemBytes>;

    DataPacketLL *recv_buff_ptr = reinterpret_cast<DataPacketLL *>(recv_buff);

    // elementwise copy, a thread copies 4 float16 data (64 bits)
    // in a loop, so the ElePerLoop is 4
    constexpr int ElePerLoop = 4;

    constexpr int MNumPerLoop =
        math::min<TDM, ElePerLoop * UnitOp::NumThreads>::value;
    constexpr int NNumPerLoop =
        math::max<1, ElePerLoop * UnitOp::NumThreads / TDM>::value;

    constexpr int IterMNum = math::div_up<TDM, MNumPerLoop>::value;
    constexpr int IterNNum = TDN / NNumPerLoop;

    int t0 = UnitOp::uop_idx_w(uop_idx);
    int t1 = UnitOp::uop_idx_h(uop_idx);
    int midx = TDM * t0 + math::mod<TDM>(ElePerLoop * UnitOp::thread_id());
    int nidx = TDN * t1 + math::div<TDM>(ElePerLoop * UnitOp::thread_id());
#pragma unroll
    for (int i = 0; i < IterNNum; ++i) {
#pragma unroll
        for (int j = 0; j < IterMNum; ++j) {
            int idx = midx + j * MNumPerLoop + (nidx + i * NNumPerLoop) * LDM;
            // in ARK the src and dst store the float16 data, so a
            // thread copied 4 float16 (64 bits) data in a loop
            *(uint64_t *)&(((fp16 *)data_dst)[idx]) =
                readLL<FLAG>(recv_buff_ptr + math::div<4>(idx));
            (recv_buff_ptr + math::div<4>(idx))->i4 = make_int4(0, 0, 0, 0);
        }
    }
    post_recv_mm_op<NumWarps>(send_ready_flag, uop_idx);
}

}  // namespace comm
}  // namespace ark

#endif  // ARK_COMM_MM_H_
