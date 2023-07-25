// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_COMMON_H_
#define ARK_OPS_COMMON_H_

#include "include/ark.h"
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
    OP_ARG_FLOAT,
    OP_ARG_DIMS
} OpArgType;

// Stores an arbitrary type of argument given to an operator.
struct OpArg
{
    OpArg(int arg);
    OpArg(long long int arg);
    OpArg(uint64_t arg);
    OpArg(bool arg);
    OpArg(float arg);
    OpArg(const Dims &arg);
    OpArg(const OpArg &);
    ~OpArg();

    //
    OpArgType type;
    void *val;

    friend bool operator<(const OpArg &oa1, const OpArg &oa2);
    friend bool operator==(const OpArg &oa1, const OpArg &oa2);
};

class Op;

class OpArgs
{
  public:
    OpArgs(const std::vector<OpArg> &args = {});
    OpArgs(const OpArgs &) = default;

    OpArgs &operator=(const OpArgs &opargs);

    void get(int *arg, size_t idx) const;
    void get(long long int *arg, size_t idx) const;
    void get(uint64_t *arg, size_t idx) const;
    void get(bool *arg, size_t idx) const;
    void get(float *arg, size_t idx) const;
    void get(Dims *arg, size_t idx) const;

  protected:
    std::vector<OpArg> args;

    friend class Op;
    friend bool operator<(const OpArgs &opargs1, const OpArgs &opargs2);
    friend bool operator==(const OpArgs &opargs1, const OpArgs &opargs2);
    friend bool operator!=(const OpArgs &opargs1, const OpArgs &opargs2);
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
    OP_RELU,
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

// 2-dimensional op tile
struct OpTile
{
    DimType x;
    DimType y;
};

// Configurations for execution of an operation.
struct OpConfig
{
    int num_warps = 0;
    int smem_bytes = 0;
    std::vector<OpTile> in_deps_tiles;
    std::vector<OpTile> out_deps_tiles;
    bool sync_pre = false;
    bool sync_post = false;
};

// The operator of a model.
class Op
{
  public:
    Op() = default;
    Op(const OpType &type, const OpPrecType &prec_type,
       const std::vector<Tensor *> &in_deps,
       const std::vector<Tensor *> &out_deps, const OpArgs& args,
       const std::string &name, int gran_lev);
    Op(const Op &) = default;

    virtual std::string function_string(const OpConfig &cfg) const { return ""; };

    //
    OpType type;
    // Precision type of the operator.
    OpPrecType prec_type;
    // The input tensors of the operator.
    std::vector<Tensor *> in_deps;
    // The output tensors of the operator.
    std::vector<Tensor *> out_deps;
    // Additional arguments of the operator.
    OpArgs args;
    std::string name;
    int gran_lev;

    friend bool operator<(const Op &op1, const Op &op2);
    friend bool operator==(const Op &op1, const Op &op2);

  protected:
    std::string function_name(const std::string &kernel_name,
                              const OpArgs &template_args) const;
};

std::ostream &operator<<(std::ostream &os, const OpType &s);

} // namespace ark

#endif // ARK_OPS_COMMON_H_
