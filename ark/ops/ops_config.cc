// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_config.h"

namespace ark {

bool operator<(const OpConfigKey &ops1, const OpConfigKey &ops2)
{
    if (ops1.op_type != ops2.op_type) {
        return ops1.op_type < ops2.op_type;
    } else if (ops1.arch_type != ops2.arch_type) {
        return ops1.arch_type < ops2.arch_type;
    } else {
        return ops1.prec_type < ops2.prec_type;
    }
}

bool operator==(const OpConfigKey &ops1, const OpConfigKey &ops2)
{
    return ops1.op_type == ops2.op_type && ops1.arch_type == ops2.arch_type &&
           ops1.prec_type == ops2.prec_type;
}

void to_json(nlohmann::json &j, const OpConfig &cfg)
{
    // j = nlohmann::json{
    //     {"num_warps", cfg.num_warps},
    //     {"smem_bytes", cfg.smem_bytes},
    //     {"in_deps_tiles", cfg.in_deps_tiles},
    //     {"out_deps_tiles", cfg.out_deps_tiles},
    //     {"sync_pre", cfg.sync_pre},
    //     {"sync_post", cfg.sync_post},
    // };
}
void from_json(const nlohmann::json &j, OpConfig &cfg)
{
}

// OpConfig for virtual ops.
const OpConfig ARK_OP_CONFIG_VIRT;

// Map from OpConfigKey to a list of OpConfigs.
const std::map<OpConfigKey, std::vector<OpConfig>> ARK_OP_CONFIG_MAP = {
    {{OP_ADD, OP_ARCH_CUDA_70, OP_PREC_FP32},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 0, {{128, 256}, {128, 256}}, {{128, 256}}, false, false},
         {8, 0, {{256, 128}, {256, 128}}, {{256, 128}}, false, false},
         {8, 0, {{128, 128}, {128, 128}}, {{128, 128}}, false, false},
         {4, 0, {{64, 64}, {64, 64}}, {{64, 64}}, false, false},
         {2, 0, {{32, 64}, {32, 64}}, {{32, 64}}, false, false},
         {1, 0, {{16, 64}, {16, 64}}, {{16, 64}}, false, false},
         {1, 0, {{8, 64}, {8, 64}}, {{8, 64}}, false, false},
         {1, 0, {{2, 128}, {2, 128}}, {{2, 128}}, false, false},
         {1, 0, {{4, 64}, {4, 64}}, {{4, 64}}, false, false},
         {1, 0, {{2, 64}, {2, 64}}, {{2, 64}}, false, false},
         {1, 0, {{1, 64}, {1, 64}}, {{1, 64}}, false, false},
         {1, 0, {{1, 32}, {1, 32}}, {{1, 32}}, false, false},
     }},
    {{OP_ADD, OP_ARCH_CUDA_70, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {4, 0, {{64, 64}, {64, 64}}, {{64, 64}}, false, false},
         {2, 0, {{32, 64}, {32, 64}}, {{32, 64}}, false, false},
         {1, 0, {{16, 64}, {16, 64}}, {{16, 64}}, false, false},
         {1, 0, {{8, 64}, {8, 64}}, {{8, 64}}, false, false},
         {1, 0, {{2, 128}, {2, 128}}, {{2, 128}}, false, false},
         {1, 0, {{4, 64}, {4, 64}}, {{4, 64}}, false, false},
         {1, 0, {{2, 64}, {2, 64}}, {{2, 64}}, false, false},
         {1, 0, {{1, 64}, {1, 64}}, {{1, 64}}, false, false},
     }},
    {{OP_ADD, OP_ARCH_CUDA_80, OP_PREC_FP32},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 0, {{128, 256}, {128, 256}}, {{128, 256}}, false, false},
         {8, 0, {{256, 128}, {256, 128}}, {{256, 128}}, false, false},
         {8, 0, {{128, 128}, {128, 128}}, {{128, 128}}, false, false},
         {4, 0, {{64, 64}, {64, 64}}, {{64, 64}}, false, false},
         {2, 0, {{32, 64}, {32, 64}}, {{32, 64}}, false, false},
         {1, 0, {{16, 64}, {16, 64}}, {{16, 64}}, false, false},
         {1, 0, {{8, 64}, {8, 64}}, {{8, 64}}, false, false},
         {1, 0, {{2, 128}, {2, 128}}, {{2, 128}}, false, false},
         {1, 0, {{4, 64}, {4, 64}}, {{4, 64}}, false, false},
         {1, 0, {{2, 64}, {2, 64}}, {{2, 64}}, false, false},
         {1, 0, {{1, 64}, {1, 64}}, {{1, 64}}, false, false},
         {1, 0, {{1, 32}, {1, 32}}, {{1, 32}}, false, false},
     }},
    {{OP_ADD, OP_ARCH_CUDA_80, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 0, {{128, 256}, {128, 256}}, {{128, 256}}, false, false},
         {8, 0, {{256, 128}, {256, 128}}, {{256, 128}}, false, false},
         {8, 0, {{128, 128}, {128, 128}}, {{128, 128}}, false, false},
         {4, 0, {{64, 64}, {64, 64}}, {{64, 64}}, false, false},
         {2, 0, {{32, 64}, {32, 64}}, {{32, 64}}, false, false},
         {1, 0, {{16, 64}, {16, 64}}, {{16, 64}}, false, false},
         {1, 0, {{8, 64}, {8, 64}}, {{8, 64}}, false, false},
         {1, 0, {{2, 128}, {2, 128}}, {{2, 128}}, false, false},
         {1, 0, {{4, 64}, {4, 64}}, {{4, 64}}, false, false},
         {1, 0, {{2, 64}, {2, 64}}, {{2, 64}}, false, false},
         {1, 0, {{1, 64}, {1, 64}}, {{1, 64}}, false, false},
     }},
    {{OP_MUL, OP_ARCH_CUDA_70, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {4, 0, {{64, 64}, {64, 64}}, {{64, 64}}, false, false},
         {2, 0, {{32, 64}, {32, 64}}, {{32, 64}}, false, false},
         {1, 0, {{16, 64}, {16, 64}}, {{16, 64}}, false, false},
         {1, 0, {{8, 64}, {8, 64}}, {{8, 64}}, false, false},
         {1, 0, {{2, 128}, {2, 128}}, {{2, 128}}, false, false},
         {1, 0, {{4, 64}, {4, 64}}, {{4, 64}}, false, false},
         {1, 0, {{2, 64}, {2, 64}}, {{2, 64}}, false, false},
         {1, 0, {{1, 64}, {1, 64}}, {{1, 64}}, false, false},
     }},
    {{OP_MUL, OP_ARCH_CUDA_80, OP_PREC_FP32},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 0, {{128, 256}, {128, 256}}, {{128, 256}}, false, false},
         {8, 0, {{256, 128}, {256, 128}}, {{256, 128}}, false, false},
         {8, 0, {{128, 128}, {128, 128}}, {{128, 128}}, false, false},
         {4, 0, {{64, 64}, {64, 64}}, {{64, 64}}, false, false},
         {2, 0, {{32, 64}, {32, 64}}, {{32, 64}}, false, false},
         {1, 0, {{16, 64}, {16, 64}}, {{16, 64}}, false, false},
         {1, 0, {{8, 64}, {8, 64}}, {{8, 64}}, false, false},
         {1, 0, {{2, 128}, {2, 128}}, {{2, 128}}, false, false},
         {1, 0, {{4, 64}, {4, 64}}, {{4, 64}}, false, false},
         {1, 0, {{2, 64}, {2, 64}}, {{2, 64}}, false, false},
         {1, 0, {{1, 64}, {1, 64}}, {{1, 64}}, false, false},
     }},
    {{OP_MUL, OP_ARCH_CUDA_80, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 0, {{128, 256}, {128, 256}}, {{128, 256}}, false, false},
         {8, 0, {{256, 128}, {256, 128}}, {{256, 128}}, false, false},
         {8, 0, {{128, 128}, {128, 128}}, {{128, 128}}, false, false},
         {4, 0, {{64, 64}, {64, 64}}, {{64, 64}}, false, false},
         {2, 0, {{32, 64}, {32, 64}}, {{32, 64}}, false, false},
         {1, 0, {{16, 64}, {16, 64}}, {{16, 64}}, false, false},
         {1, 0, {{8, 64}, {8, 64}}, {{8, 64}}, false, false},
         {1, 0, {{2, 128}, {2, 128}}, {{2, 128}}, false, false},
         {1, 0, {{4, 64}, {4, 64}}, {{4, 64}}, false, false},
         {1, 0, {{2, 64}, {2, 64}}, {{2, 64}}, false, false},
         {1, 0, {{1, 64}, {1, 64}}, {{1, 64}}, false, false},
     }},
    {{OP_IM2COL, OP_ARCH_CUDA_70, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 0, {{1, 1}}, {{128, 128}}, true, false},
         {4, 0, {{1, 1}}, {{64, 128}}, true, false},
         {4, 0, {{1, 1}}, {{128, 64}}, true, false},
         {4, 0, {{1, 1}}, {{64, 64}}, true, false},
     }},
    {{OP_IM2COL, OP_ARCH_CUDA_80, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 0, {{1, 1}}, {{128, 128}}, true, false},
         {4, 0, {{1, 1}}, {{64, 128}}, true, false},
         {4, 0, {{1, 1}}, {{128, 64}}, true, false},
         {4, 0, {{1, 1}}, {{64, 64}}, true, false},
     }},
    {{OP_TRANSPOSE, OP_ARCH_CUDA_70, OP_PREC_FP32},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 0, {{1, 1}}, {{128, 128}}, true, false},
         {4, 0, {{1, 1}}, {{64, 128}}, true, false},
         {4, 0, {{1, 1}}, {{128, 64}}, true, false},
         {4, 0, {{1, 1}}, {{64, 64}}, true, false},
         {2, 0, {{1, 1}}, {{32, 32}}, true, false},
     }},
    {{OP_TRANSPOSE, OP_ARCH_CUDA_80, OP_PREC_FP32},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 0, {{1, 1}}, {{128, 128}}, true, false},
         {4, 0, {{1, 1}}, {{64, 128}}, true, false},
         {4, 0, {{1, 1}}, {{128, 64}}, true, false},
         {4, 0, {{1, 1}}, {{64, 64}}, true, false},
         {2, 0, {{1, 1}}, {{32, 32}}, true, false},
     }},
    {{OP_TRANSPOSE, OP_ARCH_CUDA_70, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 0, {{1, 1}}, {{128, 128}}, true, false},
         {4, 0, {{1, 1}}, {{64, 128}}, true, false},
         {4, 0, {{1, 1}}, {{128, 64}}, true, false},
         {4, 0, {{1, 1}}, {{64, 64}}, true, false},
         {2, 0, {{1, 1}}, {{32, 32}}, true, false},
         {1, 0, {{1, 1}}, {{16, 16}}, true, false},
         {1, 0, {{1, 1}}, {{8, 16}}, true, false},
     }},
    {{OP_TRANSPOSE, OP_ARCH_CUDA_80, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 0, {{1, 1}}, {{128, 128}}, true, false},
         {4, 0, {{1, 1}}, {{64, 128}}, true, false},
         {4, 0, {{1, 1}}, {{128, 64}}, true, false},
         {4, 0, {{1, 1}}, {{64, 64}}, true, false},
         {2, 0, {{1, 1}}, {{32, 32}}, true, false},
         {1, 0, {{1, 1}}, {{16, 16}}, true, false},
         {1, 0, {{1, 1}}, {{8, 16}}, true, false},
         {1, 0, {{1, 1}}, {{4, 8}}, true, false},
     }},
    {{OP_MATMUL, OP_ARCH_CUDA_70, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 49152, {{128, 32}, {32, 128}}, {{128, 128}}, true, false},
         {4, 24576, {{64, 32}, {32, 128}}, {{64, 128}}, true, false},
         {4, 24576, {{128, 32}, {32, 64}}, {{128, 64}}, true, false},
         {4, 24576, {{64, 32}, {32, 64}}, {{64, 64}}, true, false},
     }},
    {{OP_MATMUL, OP_ARCH_CUDA_80, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 166912, {{128, 64}, {64, 256}}, {{128, 256}}, true, false},
         {8, 166912, {{256, 64}, {64, 128}}, {{256, 128}}, true, false},
         {8, 166912, {{128, 64}, {64, 128}}, {{128, 128}}, true, false},
         {4, 83456, {{64, 64}, {64, 64}}, {{64, 64}}, true, false},
     }},
    {{OP_REDUCE, OP_ARCH_CUDA_70, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 128, {{32, 1}}, {{32, 1}}, false, false},
         {1, 128, {{16, 1}}, {{16, 1}}, false, false},
         {1, 128, {{8, 1}}, {{8, 1}}, false, false},
         {1, 128, {{4, 1}}, {{4, 1}}, false, false},
         {1, 128, {{2, 1}}, {{2, 1}}, false, false},
         {1, 128, {{1, 1}}, {{1, 1}}, false, false},
         {4, 128, {{1, 1}}, {{1, 1}}, false, false},
         {8, 128, {{1, 1}}, {{1, 1}}, false, false},
     }},
    {{OP_REDUCE, OP_ARCH_CUDA_80, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 128, {{32, 1}}, {{32, 1}}, false, false},
         {1, 128, {{16, 1}}, {{16, 1}}, false, false},
         {1, 128, {{8, 1}}, {{8, 1}}, false, false},
         {1, 128, {{4, 1}}, {{4, 1}}, false, false},
         {1, 128, {{2, 1}}, {{2, 1}}, false, false},
         {1, 128, {{1, 1}}, {{1, 1}}, false, false},
         {4, 128, {{1, 1}}, {{1, 1}}, false, false},
         {8, 128, {{1, 1}}, {{1, 1}}, false, false},
     }},
    {{OP_REDUCE, OP_ARCH_CUDA_70, OP_PREC_FP32},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 128, {{32, 1}}, {{32, 1}}, false, false},
         {1, 128, {{16, 1}}, {{16, 1}}, false, false},
         {1, 128, {{8, 1}}, {{8, 1}}, false, false},
         {1, 128, {{4, 1}}, {{4, 1}}, false, false},
         {1, 128, {{2, 1}}, {{2, 1}}, false, false},
         {1, 128, {{1, 1}}, {{1, 1}}, false, false},
         {4, 128, {{1, 1}}, {{1, 1}}, false, false},
         {8, 128, {{1, 1}}, {{1, 1}}, false, false},
     }},
    {{OP_REDUCE, OP_ARCH_CUDA_80, OP_PREC_FP32},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 128, {{32, 1}}, {{32, 1}}, false, false},
         {1, 128, {{16, 1}}, {{16, 1}}, false, false},
         {1, 128, {{8, 1}}, {{8, 1}}, false, false},
         {1, 128, {{4, 1}}, {{4, 1}}, false, false},
         {1, 128, {{2, 1}}, {{2, 1}}, false, false},
         {1, 128, {{1, 1}}, {{1, 1}}, false, false},
         {4, 128, {{1, 1}}, {{1, 1}}, false, false},
         {8, 128, {{1, 1}}, {{1, 1}}, false, false},
     }},
    {{OP_LAYERNORM, OP_ARCH_CUDA_70, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 128, {{32, 1}}, {{32, 1}}, false, false},
         {1, 128, {{16, 1}}, {{16, 1}}, false, false},
         {1, 128, {{8, 1}}, {{8, 1}}, false, false},
         {1, 128, {{4, 1}}, {{4, 1}}, false, false},
         {1, 128, {{2, 1}}, {{2, 1}}, false, false},
         {1, 128, {{1, 1}}, {{1, 1}}, false, false},
         {4, 128, {{1, 1}}, {{1, 1}}, false, false},
         {8, 128, {{1, 1}}, {{1, 1}}, false, false},
     }},
    {{OP_LAYERNORM, OP_ARCH_CUDA_80, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 128, {{32, 1}}, {{32, 1}}, false, false},
         {1, 128, {{16, 1}}, {{16, 1}}, false, false},
         {1, 128, {{8, 1}}, {{8, 1}}, false, false},
         {1, 128, {{4, 1}}, {{4, 1}}, false, false},
         {1, 128, {{2, 1}}, {{2, 1}}, false, false},
         {1, 128, {{1, 1}}, {{1, 1}}, false, false},
         {4, 128, {{1, 1}}, {{1, 1}}, false, false},
         {8, 128, {{1, 1}}, {{1, 1}}, false, false},
     }},
    {{OP_LAYERNORM, OP_ARCH_CUDA_70, OP_PREC_FP32},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 128, {{32, 1}}, {{32, 1}}, false, false},
         {1, 128, {{16, 1}}, {{16, 1}}, false, false},
         {1, 128, {{8, 1}}, {{8, 1}}, false, false},
         {1, 128, {{4, 1}}, {{4, 1}}, false, false},
         {1, 128, {{2, 1}}, {{2, 1}}, false, false},
         {1, 128, {{1, 1}}, {{1, 1}}, false, false},
         {4, 128, {{1, 1}}, {{1, 1}}, false, false},
         {8, 128, {{1, 1}}, {{1, 1}}, false, false},
     }},
    {{OP_LAYERNORM, OP_ARCH_CUDA_80, OP_PREC_FP32},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 128, {{32, 1}}, {{32, 1}}, false, false},
         {1, 128, {{16, 1}}, {{16, 1}}, false, false},
         {1, 128, {{8, 1}}, {{8, 1}}, false, false},
         {1, 128, {{4, 1}}, {{4, 1}}, false, false},
         {1, 128, {{2, 1}}, {{2, 1}}, false, false},
         {1, 128, {{1, 1}}, {{1, 1}}, false, false},
         {4, 128, {{1, 1}}, {{1, 1}}, false, false},
         {8, 128, {{1, 1}}, {{1, 1}}, false, false},
     }},
    {{OP_SOFTMAX, OP_ARCH_CUDA_70, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 128, {{32, 1}}, {{32, 1}}, false, false},
         {1, 128, {{16, 1}}, {{16, 1}}, false, false},
         {1, 128, {{8, 1}}, {{8, 1}}, false, false},
         {1, 128, {{4, 1}}, {{4, 1}}, false, false},
         {1, 128, {{2, 1}}, {{2, 1}}, false, false},
         {1, 128, {{1, 1}}, {{1, 1}}, false, false},
         {4, 128, {{1, 1}}, {{1, 1}}, false, false},
         {8, 128, {{1, 1}}, {{1, 1}}, false, false},
     }},
    {{OP_SOFTMAX, OP_ARCH_CUDA_80, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 128, {{32, 1}}, {{32, 1}}, false, false},
         {1, 128, {{16, 1}}, {{16, 1}}, false, false},
         {1, 128, {{8, 1}}, {{8, 1}}, false, false},
         {1, 128, {{4, 1}}, {{4, 1}}, false, false},
         {1, 128, {{2, 1}}, {{2, 1}}, false, false},
         {1, 128, {{1, 1}}, {{1, 1}}, false, false},
         {4, 128, {{1, 1}}, {{1, 1}}, false, false},
         {8, 128, {{1, 1}}, {{1, 1}}, false, false},
     }},
    {{OP_SOFTMAX, OP_ARCH_CUDA_70, OP_PREC_FP32},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 128, {{32, 1}}, {{32, 1}}, false, false},
         {1, 128, {{16, 1}}, {{16, 1}}, false, false},
         {1, 128, {{8, 1}}, {{8, 1}}, false, false},
         {1, 128, {{4, 1}}, {{4, 1}}, false, false},
         {1, 128, {{2, 1}}, {{2, 1}}, false, false},
         {1, 128, {{1, 1}}, {{1, 1}}, false, false},
         {4, 128, {{1, 1}}, {{1, 1}}, false, false},
         {8, 128, {{1, 1}}, {{1, 1}}, false, false},
     }},
    {{OP_SOFTMAX, OP_ARCH_CUDA_80, OP_PREC_FP32},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 128, {{32, 1}}, {{32, 1}}, false, false},
         {1, 128, {{16, 1}}, {{16, 1}}, false, false},
         {1, 128, {{8, 1}}, {{8, 1}}, false, false},
         {1, 128, {{4, 1}}, {{4, 1}}, false, false},
         {1, 128, {{2, 1}}, {{2, 1}}, false, false},
         {1, 128, {{1, 1}}, {{1, 1}}, false, false},
         {4, 128, {{1, 1}}, {{1, 1}}, false, false},
         {8, 128, {{1, 1}}, {{1, 1}}, false, false},
     }},
    {{OP_SCALE, OP_ARCH_CUDA_70, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {4, 0, {{64, 64}}, {{64, 64}}, false, false},
         {2, 0, {{32, 64}}, {{32, 64}}, false, false},
         {1, 0, {{16, 64}}, {{16, 64}}, false, false},
         {1, 0, {{8, 64}}, {{8, 64}}, false, false},
         {1, 0, {{2, 128}}, {{2, 128}}, false, false},
         {1, 0, {{4, 64}}, {{4, 64}}, false, false},
         {1, 0, {{2, 64}}, {{2, 64}}, false, false},
         {1, 0, {{1, 64}}, {{1, 64}}, false, false},
     }},
    {{OP_SCALE, OP_ARCH_CUDA_80, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 0, {{128, 256}}, {{128, 256}}, false, false},
         {8, 0, {{256, 128}}, {{256, 128}}, false, false},
         {8, 0, {{128, 128}}, {{128, 128}}, false, false},
         {4, 0, {{64, 64}}, {{64, 64}}, false, false},
         {2, 0, {{32, 64}}, {{32, 64}}, false, false},
         {1, 0, {{16, 64}}, {{16, 64}}, false, false},
         {1, 0, {{8, 64}}, {{8, 64}}, false, false},
         {1, 0, {{2, 128}}, {{2, 128}}, false, false},
         {1, 0, {{4, 64}}, {{4, 64}}, false, false},
         {1, 0, {{2, 64}}, {{2, 64}}, false, false},
         {1, 0, {{1, 64}}, {{1, 64}}, false, false},
     }},
    {{OP_GELU, OP_ARCH_CUDA_70, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {4, 0, {{64, 64}}, {{64, 64}}, false, false},
         {2, 0, {{32, 64}}, {{32, 64}}, false, false},
         {1, 0, {{16, 64}}, {{16, 64}}, false, false},
         {1, 0, {{8, 64}}, {{8, 64}}, false, false},
         {1, 0, {{2, 128}}, {{2, 128}}, false, false},
         {1, 0, {{4, 64}}, {{4, 64}}, false, false},
         {1, 0, {{2, 64}}, {{2, 64}}, false, false},
         {1, 0, {{1, 64}}, {{1, 64}}, false, false},
     }},
    {{OP_GELU, OP_ARCH_CUDA_80, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 0, {{128, 256}}, {{128, 256}}, false, false},
         {8, 0, {{256, 128}}, {{256, 128}}, false, false},
         {8, 0, {{128, 128}}, {{128, 128}}, false, false},
         {4, 0, {{64, 64}}, {{64, 64}}, false, false},
         {2, 0, {{32, 64}}, {{32, 64}}, false, false},
         {1, 0, {{16, 64}}, {{16, 64}}, false, false},
         {1, 0, {{8, 64}}, {{8, 64}}, false, false},
         {1, 0, {{2, 128}}, {{2, 128}}, false, false},
         {1, 0, {{4, 64}}, {{4, 64}}, false, false},
         {1, 0, {{2, 64}}, {{2, 64}}, false, false},
         {1, 0, {{1, 64}}, {{1, 64}}, false, false},
     }},
    {{OP_SEND_MM, OP_ARCH_CUDA_70, OP_PREC_NONE},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {4, 0, {{64, 64}}, {{64, 64}}, false, false},
         {2, 0, {{32, 64}}, {{32, 64}}, false, false},
         {1, 0, {{16, 64}}, {{16, 64}}, false, false},
         {1, 0, {{8, 64}}, {{8, 64}}, false, false},
         {1, 0, {{2, 128}}, {{2, 128}}, false, false},
         {1, 0, {{4, 64}}, {{4, 64}}, false, false},
         {1, 0, {{2, 64}}, {{2, 64}}, false, false},
     }},
    {{OP_SEND_MM, OP_ARCH_CUDA_80, OP_PREC_NONE},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {4, 0, {{64, 64}}, {{64, 64}}, false, false},
         {2, 0, {{32, 64}}, {{32, 64}}, false, false},
         {1, 0, {{16, 64}}, {{16, 64}}, false, false},
         {1, 0, {{8, 64}}, {{8, 64}}, false, false},
         {1, 0, {{2, 128}}, {{2, 128}}, false, false},
         {1, 0, {{4, 64}}, {{4, 64}}, false, false},
         {1, 0, {{2, 64}}, {{2, 64}}, false, false},
     }},
    {{OP_RECV_MM, OP_ARCH_CUDA_70, OP_PREC_NONE},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {4, 0, {{64, 64}}, {{64, 64}}, false, false},
         {2, 0, {{32, 64}}, {{32, 64}}, false, false},
         {1, 0, {{16, 64}}, {{16, 64}}, false, false},
         {1, 0, {{8, 64}}, {{8, 64}}, false, false},
         {1, 0, {{2, 128}}, {{2, 128}}, false, false},
         {1, 0, {{4, 64}}, {{4, 64}}, false, false},
         {1, 0, {{2, 64}}, {{2, 64}}, false, false},
     }},
    {{OP_RECV_MM, OP_ARCH_CUDA_80, OP_PREC_NONE},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {4, 0, {{64, 64}}, {{64, 64}}, false, false},
         {2, 0, {{32, 64}}, {{32, 64}}, false, false},
         {1, 0, {{16, 64}}, {{16, 64}}, false, false},
         {1, 0, {{8, 64}}, {{8, 64}}, false, false},
         {1, 0, {{2, 128}}, {{2, 128}}, false, false},
         {1, 0, {{4, 64}}, {{4, 64}}, false, false},
         {1, 0, {{2, 64}}, {{2, 64}}, false, false},
     }},
    {{OP_SEND, OP_ARCH_CUDA_70, OP_PREC_NONE},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 0, {{1, 1}}, {{1, 1}}, true, true},
     }},
    {{OP_SEND, OP_ARCH_CUDA_80, OP_PREC_NONE},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 0, {{1, 1}}, {{1, 1}}, true, true},
     }},
    {{OP_SEND_DONE, OP_ARCH_CUDA_70, OP_PREC_NONE},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 0, {{1, 1}}, {{1, 1}}, true, true},
     }},
    {{OP_SEND_DONE, OP_ARCH_CUDA_80, OP_PREC_NONE},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 0, {{1, 1}}, {{1, 1}}, true, true},
     }},
    {{OP_RECV, OP_ARCH_CUDA_70, OP_PREC_NONE},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 0, {{1, 1}}, {{1, 1}}, true, true},
     }},
    {{OP_RECV, OP_ARCH_CUDA_80, OP_PREC_NONE},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 0, {{1, 1}}, {{1, 1}}, true, true},
     }},
};

} // namespace ark
