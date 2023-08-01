// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_COMMON_H_
#define ARK_OPS_COMMON_H_

#include "include/ark.h"
#include <ostream>
#include <vector>

namespace ark {

/// Return the output shape of broadcasting between two shapes.
/// Follow NumPy rules.
/// https://numpy.org/doc/stable/user/basics.broadcasting.html
/// @param dims1 The first shape.
/// @param dims2 The second shape.
Dims broadcast(const Dims &dims1, const Dims &dims2);

/// Type of operator argument.
typedef enum
{
    OP_ARG_INT,
    OP_ARG_INT64,
    OP_ARG_UINT64,
    OP_ARG_BOOL,
    OP_ARG_FLOAT,
    OP_ARG_DIMS,
    OP_ARG_TENSOR,
} OpArgType;

/// Stores an arbitrary type of argument given to an operator.
struct OpArg
{
    OpArg(int arg);
    OpArg(long long int arg);
    OpArg(uint64_t arg);
    OpArg(bool arg);
    OpArg(float arg);
    OpArg(const Dims &arg);
    OpArg(Tensor *arg);
    OpArg(const OpArg &);
    ~OpArg();

    void get(int *arg) const;
    void get(long long int *arg) const;
    void get(uint64_t *arg) const;
    void get(bool *arg) const;
    void get(float *arg) const;
    void get(Dims *arg) const;
    void get(Tensor **arg) const;

    OpArgType type;
    void *val;

    friend bool operator<(const OpArg &oa1, const OpArg &oa2);
    friend bool operator==(const OpArg &oa1, const OpArg &oa2);
};

class Op;

/// Stores a list of @ref OpArg.
class OpArgs
{
  public:
    OpArgs(const std::vector<OpArg> &args = {});
    OpArgs(const OpArgs &) = default;
    ~OpArgs(){};

    OpArgs &operator=(const OpArgs &opargs);

    void put(const OpArg &arg);

    void get(int *arg, size_t idx) const;
    void get(long long int *arg, size_t idx) const;
    void get(uint64_t *arg, size_t idx) const;
    void get(bool *arg, size_t idx) const;
    void get(float *arg, size_t idx) const;
    void get(Dims *arg, size_t idx) const;
    void get(Tensor **arg, size_t idx) const;

    const std::vector<OpArg> &get_args() const;

  protected:
    std::vector<OpArg> args;

    friend class Op;
    friend bool operator<(const OpArgs &opargs1, const OpArgs &opargs2);
    friend bool operator==(const OpArgs &opargs1, const OpArgs &opargs2);
    friend bool operator!=(const OpArgs &opargs1, const OpArgs &opargs2);
};

/// Type of @ref Op.
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

/// Type of precision of @ref Op.
typedef enum
{
    OP_PREC_NONE,
    OP_PREC_FP16,
    OP_PREC_FP32,
} OpPrecType;

/// Type of hardware architecture support.
typedef enum
{
    OP_ARCH_CUDA_70,
    OP_ARCH_CUDA_80,
} OpArchType;

struct Tensor;

/// 2-dimensional op tile
struct OpTile
{
    DimType x;
    DimType y;
};

/// Configurations for execution of a @ref Op.
struct OpConfig
{
    int num_warps = 0;
    int smem_bytes = 0;
    std::vector<OpTile> input_tiles;
    std::vector<OpTile> output_tiles;
    bool sync_pre = false;
    bool sync_post = false;
};

/// Key to find a list of OpConfigs from OpConfigMap.
struct OpConfigKey
{
    OpArchType arch_type;
    OpPrecType prec_type;
};

bool operator<(const OpConfigKey &ops1, const OpConfigKey &ops2);

bool operator==(const OpConfigKey &ops1, const OpConfigKey &ops2);

/// Map from OpConfigKey to a list of OpConfigs.
using OpConfigMap = std::map<OpConfigKey, std::vector<OpConfig>>;

/// Operator.
class Op
{
  public:
    /// Construct an operator.
    Op() = default;

    /// Construct an operator.
    /// @param type the type of the @ref Op.
    /// @param prec_type the precision type of the @ref Op.
    /// @param inputs the input tensors of the @ref Op, including execution
    /// dependencies.
    /// @param output_refs the output reference tensors of the @ref Op. Output
    /// tensors are created based on these references.
    /// @param args the arguments of the @ref Op.
    /// @param name the name of the @ref Op.
    /// @param cfg_map the configuration map of the @ref Op
    /// @param gran_lev the granularity level of the @ref Op. Larger values
    /// should indicate finer-grained Ops. If it is -1, the granularity level
    /// will be automatically determined by the scheduler.
    /// @param force_inline whether to force inline the kernel of @ref Op.
    Op(const OpType &type, const OpPrecType &prec_type,
       const std::vector<Tensor *> &inputs,
       const std::vector<Tensor *> &output_refs, const OpArgs &args,
       const std::string &name, const OpConfigMap *cfg_map = nullptr,
       int gran_lev = -1, bool force_inline = false);

    /// Construct an operator.
    Op(const Op &) = default;

    /// Destruct the operator.
    ~Op(){};

    /// Return the kernel function name of the operator. Includes the template
    /// arguments of the kernel, if any.
    /// @param cfg the configuration of the operator.
    /// @return the kernel function name of the operator.
    std::string function_name(const OpConfig &) const;

    /// Return the kernel function's runtime arguments of the operator.
    /// @param cfg the configuration of the operator.
    /// @return the runtime arguments of the kernel function.
    OpArgs function_call_args(const OpConfig &) const;

    /// Returns true if the operator is virtual (i.e., performs no computation).
    bool is_virtual() const;

    /// Returns true if the operator is a communication operator.
    bool is_comm() const;

    /// Type of the operator.
    OpType type;
    /// Precision type of the operator.
    OpPrecType prec_type;
    /// The input tensors of the operator.
    std::vector<Tensor *> inputs;
    /// The output tensors of the operator.
    std::vector<Tensor *> outputs;
    /// The reference tensors of the output tensors.
    std::vector<Tensor *> output_refs;
    /// Additional arguments of the operator.
    OpArgs args;
    /// Name of the operator.
    std::string name;
    /// Map from OpConfigKey to a list of OpConfigs.
    const OpConfigMap *cfg_map;
    /// Granularity level of the operator.
    int gran_lev;
    /// Force inlining of the operator kernel.
    bool force_inline;

    friend bool operator<(const Op &op1, const Op &op2);
    friend bool operator==(const Op &op1, const Op &op2);

  protected:
    static std::string function_name(const std::string &kernel_name,
                                     const OpArgs &template_args);
};

std::ostream &operator<<(std::ostream &os, const OpType &s);

/// List all operator classes below.

class AddOp : public Op
{
  public:
    AddOp(OpPrecType prec_type, Tensor *input, Tensor *other, Tensor *output,
          const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class GeluOp : public Op
{
  public:
    GeluOp(OpPrecType prec_type, Tensor *input, Tensor *output,
           const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class Im2colOp : public Op
{
  public:
    Im2colOp(OpPrecType prec_type, Tensor *input, Tensor *output,
             int kernel_height, int kernel_width, int stride_height,
             int stride_width, int pad_height, int pad_width,
             int dilation_height, int dilation_width, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class LayernormOp : public Op
{
  public:
    LayernormOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class MatmulOp : public Op
{
  public:
    MatmulOp(OpPrecType prec_type, Tensor *mat_a, Tensor *mat_b, Tensor *mat_y,
             Dims nca, Dims ncb, Dims problem_size, Dims leading_dims,
             bool is_column_a, bool is_column_b, bool is_relu,
             const std::string &name, int gran_lev);
    std::string function_name(const OpConfig &cfg) const;
};

class MaxPoolOp : public Op
{
  public:
    MaxPoolOp(OpPrecType prec_type, Tensor *input, Tensor *output,
              DimType kernel_size, DimType stride, const std::string &name);
};

class MulOp : public Op
{
  public:
    MulOp(OpPrecType prec_type, Tensor *input, Tensor *other, Tensor *output,
          const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class ReduceOp : public Op
{
  public:
    ReduceOp(const OpType &type, const OpPrecType &prec_type,
             const std::vector<Tensor *> &inputs,
             const std::vector<Tensor *> &outputs, const OpArgs &args,
             const std::string &name, const OpConfigMap *cfg_map, int gran_lev);

  protected:
    std::string function_name(const OpConfig &cfg,
                              const std::string &type) const;
};

class ReduceWSumOp : public ReduceOp
{
  public:
    ReduceWSumOp(OpPrecType prec_type, Tensor *input, Tensor *output, int axis,
                 const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class ReduceESumOp : public ReduceOp
{
  public:
    ReduceESumOp(OpPrecType prec_type, Tensor *input, Tensor *output, int axis,
                 const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class ReduceWMaxOp : public ReduceOp
{
  public:
    ReduceWMaxOp(OpPrecType prec_type, Tensor *input, Tensor *output, int axis,
                 const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class ReduceEMaxOp : public ReduceOp
{
  public:
    ReduceEMaxOp(OpPrecType prec_type, Tensor *input, Tensor *output, int axis,
                 const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class ReduceWMeanOp : public ReduceOp
{
  public:
    ReduceWMeanOp(OpPrecType prec_type, Tensor *input, Tensor *output, int axis,
                  const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class ReduceEMeanOp : public ReduceOp
{
  public:
    ReduceEMeanOp(OpPrecType prec_type, Tensor *input, Tensor *output, int axis,
                  const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class ReluOp : public Op
{
  public:
    ReluOp(OpPrecType prec_type, Tensor *input, Tensor *output,
           const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class ReshapeOp : public Op
{
  public:
    ReshapeOp(OpPrecType prec_type, Tensor *input, Tensor *output,
              const std::string &name);
};

class ScaleOp : public Op
{
  public:
    ScaleOp(OpPrecType prec_type, Tensor *input, Tensor *output, float val,
            const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
    OpArgs function_call_args(const OpConfig &) const;
};

class SendMMOp : public Op
{
  public:
    SendMMOp(OpPrecType prec_type, Tensor *input, Tensor *recvbuf,
             Tensor *send_ready_flag, Tensor *output, int id, int gpu_dst,
             size_t bytes, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class RecvMMOp : public Op
{
  public:
    RecvMMOp(OpPrecType prec_type, Tensor *input, Tensor *recvbuf,
             Tensor *send_ready_flag, Tensor *output, int id, int gpu_src,
             size_t bytes, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class SendOp : public Op
{
  public:
    SendOp(OpPrecType prec_type, Tensor *input, Tensor *output, int sid,
           int rank, int dst_rank, size_t bytes, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
    OpArgs function_call_args(const OpConfig &cfg) const;
};

class SendDoneOp : public Op
{
  public:
    SendDoneOp(OpPrecType prec_type, Tensor *input, Tensor *output, int sid,
               int rank, int dst_rank, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
    OpArgs function_call_args(const OpConfig &cfg) const;
};

class RecvOp : public Op
{
  public:
    RecvOp(OpPrecType prec_type, Tensor *input, Tensor *output, int sid,
           int rank, int src_rank, size_t bytes, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
    OpArgs function_call_args(const OpConfig &cfg) const;
};

class SoftmaxOp : public Op
{
  public:
    SoftmaxOp(OpPrecType prec_type, Tensor *input, Tensor *output,
              const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class TensorOp : public Op
{
  public:
    TensorOp(const std::vector<Tensor *> &deps, Tensor *output,
             const std::string &name);
};

class TransposeOp : public Op
{
  public:
    TransposeOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                int tp_type, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

} // namespace ark

#endif // ARK_OPS_COMMON_H_
