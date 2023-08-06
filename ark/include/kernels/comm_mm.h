// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_COMM_MM_H_
#define ARK_COMM_MM_H_

#include "common.h"

namespace ark {
namespace comm {

union alignas(16) DataPacketLL {
    // on 64bits machine, the access to the 64bits data is atomic, so we combine
    // the 32bits data and flag into one 64bits data to insure the GPU receive
    // complete data
    struct
    {
        uint32_t data1;
        uint32_t flag1;
        uint32_t data2;
        uint32_t flag2;
    };
    uint64_t v[2];
    int4 i4;
};

// the tensor's data is float16 data,so the matrix size is TDM * TDN *
// sizeof(float16). the data is stored in column major, and LDM is the
// major dimension. The TN is the thread number, TDM and TDN is the tile
// size.
template <int FLAG> DEVICE uint64_t readLL(DataPacketLL *recv_buf)
{
    uint32_t data1, flag1, data2, flag2;
    do {
        asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(data1), "=r"(flag1), "=r"(data2), "=r"(flag2)
                     : "l"(&recv_buf->i4));
    } while ((flag1 != FLAG) || (flag2 != FLAG));
    uint64_t val64 = data1 + (((uint64_t)data2) << 32);
    return val64;
}
template <int FLAG> DEVICE void storeLL(DataPacketLL *data_dst, uint64_t val)
{
    asm volatile(
        "st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(&data_dst->i4),
        "r"((uint32_t)val), "r"(FLAG), "r"((uint32_t)(val >> 32)), "r"(FLAG));
}

template <int TN>
DEVICE void pre_send_mm_op(volatile int *send_ready_flag, int uop_idx)
{
    if (threadIdx.x % TN == 0) {
        while (send_ready_flag[uop_idx] != 0) {
        }
        send_ready_flag[uop_idx] = 1;
    }
    sync_warps<TN>();
}

template <int TN>
DEVICE void post_recv_mm_op(volatile int *send_ready_flag, int uop_idx)
{
    sync_warps<TN>();
    if (threadIdx.x % TN == 0) {
        // reset the send_ready_flag to 0
        send_ready_flag[uop_idx] = 0;
    }
    __syncwarp();
}

template <int LDM, int LDN, int TN, int SmemBytes, int TDM, int TDN,
          int FLAG = 1>
// send a tile of the tensor from data_src to recv_buff
DEVICE void sendLL(void *recv_buff, ark::half *data_src,
                   volatile int *send_ready_flag, int uop_idx, int)
{
    using UnitOp = UnitOp<Vec<1, 1, LDN, LDM>, Vec<1, 1, LDN, LDM>,
                          Vec<1, 1, TDN, TDM>, TN, SmemBytes>;

    DataPacketLL *recv_buff_ptr = reinterpret_cast<DataPacketLL *>(recv_buff);

    // elementwise copy, a thread copies 4 float16 data (64 bits)
    // in a loop, so the ElePerLoop is 4
    constexpr int ElePerLoop = 4;

    constexpr int MNumPerLoop = math::min<TDM, ElePerLoop * TN>::value;
    constexpr int NNumPerLoop = math::max<1, ElePerLoop * TN / TDM>::value;

    constexpr int IterMNum = math::div_up<TDM, MNumPerLoop>::value;
    constexpr int IterNNum = TDN / NNumPerLoop;

    int t0 = UnitOp::uop_idx_w(uop_idx);
    int t1 = UnitOp::uop_idx_h(uop_idx);
    int midx = TDM * t0 + math::mod<TDM>(ElePerLoop * UnitOp::thread_id());
    int nidx = TDN * t1 + math::div<TDM>(ElePerLoop * UnitOp::thread_id());
    pre_send_mm_op<TN>(send_ready_flag, uop_idx);
#pragma unroll
    for (int i = 0; i < IterNNum; ++i) {
#pragma unroll
        for (int j = 0; j < IterMNum; ++j) {
            int idx = midx + j * MNumPerLoop + (nidx + i * NNumPerLoop) * LDM;
            // in ARK the src and dst store the float16 data, so a
            // thread copied 4 float16 (64 bits) data in a loop
            storeLL<FLAG>(recv_buff_ptr + math::div<4>(idx),
                          *(uint64_t *)&((ark::half *)data_src)[idx]);
        }
    }
}

// recv a tile of the tensor from recv_buff to data_dst
template <int LDM, int LDN, int TN, int SmemBytes, int TDM, int TDN,
          int FLAG = 1>
DEVICE void recvLL(void *recv_buff, ark::half *data_dst,
                   volatile int *send_ready_flag, int uop_idx, int)
{
    using UnitOp = UnitOp<Vec<1, 1, LDN, LDM>, Vec<1, 1, LDN, LDM>,
                          Vec<1, 1, TDN, TDM>, TN, SmemBytes>;

    DataPacketLL *recv_buff_ptr = reinterpret_cast<DataPacketLL *>(recv_buff);

    // elementwise copy, a thread copies 4 float16 data (64 bits)
    // in a loop, so the ElePerLoop is 4
    constexpr int ElePerLoop = 4;

    constexpr int MNumPerLoop = math::min<TDM, ElePerLoop * TN>::value;
    constexpr int NNumPerLoop = math::max<1, ElePerLoop * TN / TDM>::value;

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
            *(uint64_t *)&(((ark::half *)data_dst)[idx]) =
                readLL<FLAG>(recv_buff_ptr + math::div<4>(idx));
            (recv_buff_ptr + math::div<4>(idx))->i4 = make_int4(0, 0, 0, 0);
        }
    }
    post_recv_mm_op<TN>(send_ready_flag, uop_idx);
}

} // namespace comm
} // namespace ark

#endif // ARK_COMM_MM_H_
