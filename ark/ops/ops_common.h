// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_COMMON_H_
#define ARK_OPS_COMMON_H_

#include <ostream>
#include <vector>

namespace ark {

// Type of operator argument.
typedef enum
{
    OP_ARG_INT,
    OP_ARG_INT64,
    OP_ARG_UINT64,
    OP_ARG_BOOL,
    OP_ARG_FLOAT
} OpArgType;

// Stores an arbitrary type of argument given to an operator.
struct OpArg
{
    OpArg(int arg);
    OpArg(long long int arg);
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

// Type of operator.
typedef enum
{
    OP_UNKNOWN = 0,
    OP_TENSOR,
    OP_REFER,
    OP_RESHAPE,
    OP_MERGE,
    OP_REDUCE_E_SUM,
    OP_REDUCE_E_MEAN,
    OP_REDUCE_E_MAX,
    OP_REDUCE_W_SUM,
    OP_REDUCE_W_MEAN,
    OP_REDUCE_W_MAX,
    OP_LAYERNORM,
    OP_SOFTMAX,
    OP_SCALE,
    OP_GELU,
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

// Type of precision of operator.
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

struct Tensor;

// The operator of a model.
struct Op
{
    Op(const OpType &type, const OpPrecType &prec_type,
       const std::vector<Tensor *> &in_deps,
       const std::vector<Tensor *> &out_deps, const std::vector<OpArg> &args,
       const std::string &name, int gran_lev);
    Op(const Op &) = default;
    //
    OpType type;
    // Precision type of the operator.
    OpPrecType prec_type;
    // The input tensors of the operator.
    std::vector<Tensor *> in_deps;
    // The output tensors of the operator.
    std::vector<Tensor *> out_deps;
    // Additional arguments of the operator.
    std::vector<OpArg> args;
    std::string name;
    int gran_lev;

    friend bool operator<(const Op &op1, const Op &op2);
    friend bool operator==(const Op &op1, const Op &op2);
};

std::ostream &operator<<(std::ostream &os, const OpType &s);

} // namespace ark

#endif // ARK_OPS_COMMON_H_
