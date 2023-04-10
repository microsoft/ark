// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#ifndef ARK_OPS_COMMON_H_
#define ARK_OPS_COMMON_H_

#include "ark/tensor.h"
#include "third_party/json/json.h"

namespace ark {

// Type of operation.
typedef enum
{
    OP_UNKNOWN = 0,
    OP_TENSOR,
    OP_REFER,
    OP_RESHAPE,
    OP_MERGE,
    OP_REDUCE,
    OP_SCALE,
    OP_MATMUL,
    OP_MAX_POOL,
    OP_ADD,
    OP_MUL,
    OP_IM2COL,
    OP_TRANSPOSE,
    OP_SEND,
    OP_SEND_DONE,
    OP_RECV,
    OP_SEND_MM,
    OP_RECV_MM,
} OpType;

// clang-format off
NLOHMANN_JSON_SERIALIZE_ENUM(OpType, {
    {OP_UNKNOWN, ""},
    {OP_TENSOR, "tensor"},
    {OP_REFER, "refer"},
    {OP_RESHAPE, "reshape"},
    {OP_MERGE, "merge"},
    {OP_REDUCE, "reduce"},
    {OP_MATMUL, "matmul"},
    {OP_MAX_POOL, "max_pool"},
    {OP_ADD, "add"},
    {OP_MUL, "mul"},
    {OP_IM2COL, "im2col"},
    {OP_TRANSPOSE, "transpose"},
    {OP_SEND, "send"},
    {OP_SEND_DONE, "send_done"},
    {OP_RECV, "recv"},
})
// clang-format on

// Type of precision of operation.
typedef enum
{
    OP_PREC_NONE,
    OP_PREC_FP16,
    OP_PREC_FP32,
} OpPrecType;

// Type of hardware architecture support.
typedef enum
{
    OP_ARCH_CUDA_70,
    OP_ARCH_CUDA_80,
} OpArchType;

// clang-format off
NLOHMANN_JSON_SERIALIZE_ENUM(OpPrecType, {
    {OP_PREC_NONE, ""},
    {OP_PREC_FP16, "fp16"},
    {OP_PREC_FP32, "fp32"},
})
// clang-format on

// Type of operation argument.
typedef enum
{
    OP_ARG_INT,
    OP_ARG_UINT64,
    OP_ARG_BOOL,
    OP_ARG_FLOAT
} OpArgType;

// clang-format off
NLOHMANN_JSON_SERIALIZE_ENUM(OpArgType, {
    {OP_ARG_INT, "i32"},
    {OP_ARG_UINT64, "i64"},
    {OP_ARG_BOOL, "bool"},
    {OP_ARG_FLOAT, "f32"},
})
// clang-format on

// Stores an arbitrary type of argument given to an operation.
struct OpArg
{
    OpArg(int arg);
    OpArg(uint64_t arg);
    OpArg(bool arg);
    OpArg(float arg);
    OpArg(const OpArg &);
    ~OpArg();
    //
    OpArgType type;
    void *val;

    friend bool operator<(const OpArg &oa1, const OpArg &oa2);
    friend bool operator==(const OpArg &oa1, const OpArg &oa2);
};

void to_json(nlohmann::json &j, const OpArg &oparg);
void from_json(const nlohmann::json &j, OpArg &oparg);

//
struct Op
{
    Op(const OpType &type, const OpPrecType &prec_type,
       const std::vector<Tensor *> &in_deps,
       const std::vector<Tensor *> &out_deps, const std::vector<OpArg> &args,
       const std::string &name, int gran_lev);
    Op(const Op &) = default;
    //
    OpType type;
    OpPrecType prec_type;
    std::vector<Tensor *> in_deps;
    std::vector<Tensor *> out_deps;
    std::vector<OpArg> args;
    std::string name;
    int gran_lev;

    friend bool operator<(const Op &op1, const Op &op2);
    friend bool operator==(const Op &op1, const Op &op2);
};

void to_json(nlohmann::json &j, const Op &op);
void from_json(const nlohmann::json &j, Op &op);

} // namespace ark

#endif // ARK_OPS_COMMON_H_
