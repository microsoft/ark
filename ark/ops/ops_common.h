// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_COMMON_H_
#define ARK_OPS_COMMON_H_

#include <map>
#include <ostream>
#include <vector>

#include "include/ark.h"

namespace ark {

/// Return the output shape of broadcasting between two shapes.
/// Follow NumPy rules.
/// https://numpy.org/doc/stable/user/basics.broadcasting.html
/// @param dims1 The first shape.
/// @param dims2 The second shape.
Dims broadcast(const Dims &dims1, const Dims &dims2);

/// Type of operator argument.
struct OpArgType {
    OpArgType(size_t id, const std::string &name) : id(id), name(name) {}
    size_t id;
    std::string name;
};

bool operator==(const OpArgType &lhs, const OpArgType &rhs);

bool operator!=(const OpArgType &lhs, const OpArgType &rhs);

const OpArgType OP_ARG_INT(0, "int");
const OpArgType OP_ARG_INT64(1, "int64");
const OpArgType OP_ARG_UINT64(2, "uint64");
const OpArgType OP_ARG_BOOL(3, "bool");
const OpArgType OP_ARG_FLOAT(4, "float");
const OpArgType OP_ARG_DIMS(5, "dims");
const OpArgType OP_ARG_TENSOR(6, "tensor");

/// Stores an arbitrary type of argument given to an operator.
struct OpArg {
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
class OpArgs {
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
struct OpType {
    OpType(size_t id, const std::string &name) : id(id), name(name) {}
    const size_t id;
    std::string name;
};

bool operator==(const OpType &lhs, const OpType &rhs);

const OpType OP_UNKNOWN(0, "unknown");
const OpType OP_TENSOR(1, "tensor");
const OpType OP_REFER(2, "refer");
const OpType OP_RESHAPE(3, "reshape");
const OpType OP_MERGE(4, "merge");
const OpType OP_REDUCE_E_SUM(5, "reduce_e_sum");
const OpType OP_REDUCE_E_MEAN(6, "reduce_e_mean");
const OpType OP_REDUCE_E_MAX(7, "reduce_e_max");
const OpType OP_REDUCE_W_SUM(8, "reduce_w_sum");
const OpType OP_REDUCE_W_MEAN(9, "reduce_w_mean");
const OpType OP_REDUCE_W_MAX(10, "reduce_w_max");
const OpType OP_LAYERNORM(11, "layernorm");
const OpType OP_SCALE(12, "scale");
const OpType OP_RELU(13, "relu");
const OpType OP_COPY(14, "copy");
const OpType OP_GELU(15, "gelu");
const OpType OP_SIGMOID(16, "sigmoid");
const OpType OP_EXP(17, "exp");
const OpType OP_SQRT(18, "sqrt");
const OpType OP_RSQRT(19, "rsqrt");
const OpType OP_MATMUL(20, "matmul");
const OpType OP_MAX_POOL(21, "max_pool");
const OpType OP_ADD(22, "add");
const OpType OP_SUB(23, "sub");
const OpType OP_MUL(24, "mul");
const OpType OP_DIV(25, "div");
const OpType OP_ROPE(26, "rope");
const OpType OP_IM2COL(27, "im2col");
const OpType OP_TRANSPOSE(28, "transpose");
const OpType OP_SEND(29, "send");
const OpType OP_SEND_DONE(30, "send_done");
const OpType OP_RECV(31, "recv");
const OpType OP_EMBEDDING(32, "embedding");
const OpType OP_DEVICE_SYNC(33, "device_sync");
const OpType OP_READ_AND_REDUCE(34, "read_and_reduce");
const OpType OP_GATHER_FROM_PEERS(35, "gather_from_peers");
const OpType OP_CAST(36, "cast");
const OpType OP_PUT_PACKET(37, "put_packet");
const OpType OP_REDUCE_AND_WRITE_PACKET(38, "reduce_and_write_packet");
const OpType OP_GET_FROM_PACKET(39, "get_from_packet");

/// Type of hardware architecture support.
typedef enum {
    OP_ARCH_UNKNOWN = 0,
    OP_ARCH_CUDA_60 = 0x1,
    OP_ARCH_CUDA_70 = 0x2,
    OP_ARCH_CUDA_80 = 0x4,
    OP_ARCH_CUDA_90 = 0x8,
    OP_ARCH_CUDA_ANY = 0x0f,
    OP_ARCH_ROCM_90A = 0x10,
    OP_ARCH_ROCM_942 = 0x20,
    OP_ARCH_ROCM_ANY = 0xf0,
    OP_ARCH_ANY = -1,
} OpArchType;

OpArchType op_arch_from_string(const std::string &arch);

class Tensor;

/// 2-dimensional op tile
struct OpTile {
    DimType x;
    DimType y;
};

/// Configurations for execution of a @ref Op.
struct OpConfig {
    int num_warps = 0;
    int smem_bytes = 0;
    std::vector<OpTile> input_tiles;
    std::vector<OpTile> output_tiles;
    bool sync_pre = false;
    bool sync_post = false;
};

/// Key to find a list of OpConfigs from OpConfigMap.
struct OpConfigKey {
    OpArchType arch_type;
    std::string prec_type;
};

bool operator<(const OpConfigKey &ops1, const OpConfigKey &ops2);

bool operator==(const OpConfigKey &ops1, const OpConfigKey &ops2);

/// Map from OpConfigKey to a list of OpConfigs.
class OpConfigMap {
   public:
    OpConfigMap(std::initializer_list<
                std::pair<const OpConfigKey, const std::vector<OpConfig>>>
                    ilist);
    ~OpConfigMap(){};

    const std::vector<OpConfig> &get(const OpConfigKey &key) const;

   private:
    const std::map<OpConfigKey, const std::vector<OpConfig>> cfg_map;
};

/// Operator.
class Op {
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
    Op(const OpType &type, const std::string &prec_type,
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
    std::string prec_type;
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

/// List all operator classes below.

class ArithmeticOp : public Op {
   public:
    ArithmeticOp(const OpType &type, const std::string &prec_type,
                 Tensor *input, Tensor *other, Tensor *output,
                 const std::string &name);

   protected:
    std::string function_name(const OpConfig &cfg,
                              const std::string &type) const;
};

class AddOp : public ArithmeticOp {
   public:
    AddOp(const std::string &prec_type, Tensor *input, Tensor *other,
          Tensor *output, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class SubOp : public ArithmeticOp {
   public:
    SubOp(const std::string &prec_type, Tensor *input, Tensor *other,
          Tensor *output, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class MulOp : public ArithmeticOp {
   public:
    MulOp(const std::string &prec_type, Tensor *input, Tensor *other,
          Tensor *output, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class DivOp : public ArithmeticOp {
   public:
    DivOp(const std::string &prec_type, Tensor *input, Tensor *other,
          Tensor *output, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class MathOp : public Op {
   public:
    MathOp(const OpType &type, const std::string &prec_type, Tensor *input,
           Tensor *output, const std::string &name);

   protected:
    std::string function_name(const OpConfig &cfg,
                              const std::string &type) const;
};

class GeluOp : public MathOp {
   public:
    GeluOp(const std::string &prec_type, Tensor *input, Tensor *output,
           const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class ExpOp : public MathOp {
   public:
    ExpOp(const std::string &prec_type, Tensor *input, Tensor *output,
          const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class ReluOp : public MathOp {
   public:
    ReluOp(const std::string &prec_type, Tensor *input, Tensor *output,
           const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class RsqrtOp : public MathOp {
   public:
    RsqrtOp(const std::string &prec_type, Tensor *input, Tensor *output,
            const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class SigmoidOp : public MathOp {
   public:
    SigmoidOp(const std::string &prec_type, Tensor *input, Tensor *output,
              const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class SqrtOp : public MathOp {
   public:
    SqrtOp(const std::string &prec_type, Tensor *input, Tensor *output,
           const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class RopeOp : public Op {
   public:
    RopeOp(const std::string &prec_type, Tensor *input, Tensor *other,
           Tensor *output, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class Im2colOp : public Op {
   public:
    Im2colOp(const std::string &prec_type, Tensor *input, Tensor *output,
             int kernel_height, int kernel_width, int stride_height,
             int stride_width, int pad_height, int pad_width,
             int dilation_height, int dilation_width, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class LayernormOp : public Op {
   public:
    LayernormOp(const std::string &prec_type, Tensor *input, Tensor *output,
                const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class MatmulOp : public Op {
   public:
    MatmulOp(const std::string &prec_type, Tensor *mat_a, Tensor *mat_b,
             Tensor *mat_y, Dims nca, Dims ncb, Dims problem_size,
             Dims leading_dims, bool is_column_a, bool is_column_b,
             const std::string &name, int gran_lev);
    std::string function_name(const OpConfig &cfg) const;
};

class MaxPoolOp : public Op {
   public:
    MaxPoolOp(const std::string &prec_type, Tensor *input, Tensor *output,
              DimType kernel_size, DimType stride, const std::string &name);
};

class ReduceOp : public Op {
   public:
    ReduceOp(const OpType &type, const std::string &prec_type,
             const std::vector<Tensor *> &inputs,
             const std::vector<Tensor *> &outputs, const OpArgs &args,
             const std::string &name, const OpConfigMap *cfg_map, int gran_lev);

   protected:
    std::string function_name(const OpConfig &cfg,
                              const std::string &type) const;
};

class ReduceWSumOp : public ReduceOp {
   public:
    ReduceWSumOp(const std::string &prec_type, Tensor *input, Tensor *output,
                 int axis, bool keepdims, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class ReduceESumOp : public ReduceOp {
   public:
    ReduceESumOp(const std::string &prec_type, Tensor *input, Tensor *output,
                 int axis, bool keepdims, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class ReduceWMaxOp : public ReduceOp {
   public:
    ReduceWMaxOp(const std::string &prec_type, Tensor *input, Tensor *output,
                 int axis, bool keepdims, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class ReduceEMaxOp : public ReduceOp {
   public:
    ReduceEMaxOp(const std::string &prec_type, Tensor *input, Tensor *output,
                 int axis, bool keepdims, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class ReduceWMeanOp : public ReduceOp {
   public:
    ReduceWMeanOp(const std::string &prec_type, Tensor *input, Tensor *output,
                  int axis, bool keepdims, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class ReduceEMeanOp : public ReduceOp {
   public:
    ReduceEMeanOp(const std::string &prec_type, Tensor *input, Tensor *output,
                  int axis, bool keepdims, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class CopyOp : public Op {
   public:
    CopyOp(const std::string &prec_type, Tensor *input, Tensor *output,
           const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class ReshapeOp : public Op {
   public:
    ReshapeOp(const std::string &prec_type, Tensor *input, Tensor *output,
              const std::string &name);
};

class ScaleOp : public Op {
   public:
    ScaleOp(const std::string &prec_type, Tensor *input, Tensor *output,
            float val, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
    OpArgs function_call_args(const OpConfig &) const;
};

class SendOp : public Op {
   public:
    SendOp(const std::string &prec_type, Tensor *input, Tensor *recvbuf,
           int sid, int rank, int dst_rank, size_t bytes,
           const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
    // The args determined by the scheduler.
    OpArgs function_call_args(const OpConfig &cfg) const;
};

class RecvOp : public Op {
   public:
    RecvOp(const std::string &prec_type, Tensor *output, int sid, int rank,
           int src_rank, size_t bytes, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
    OpArgs function_call_args(const OpConfig &cfg) const;
};

class SendDoneOp : public Op {
   public:
    SendDoneOp(const std::string &prec_type, Tensor *input, int rank,
               int dst_rank, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
    OpArgs function_call_args(const OpConfig &cfg) const;
};

class DeviceSyncOp : public Op {
   public:
    DeviceSyncOp(const std::string &prec_type, Tensor *input, Tensor *output,
                 int nranks, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
    OpArgs function_call_args(const OpConfig &cfg) const;
};

class ReadAndReduceOp : public Op {
   public:
    ReadAndReduceOp(const std::string &prec_type, Tensor *local_buf,
                    Tensor *cal_region_local, std::vector<Tensor *> remote_bufs,
                    int sid, int rank, int npeers, size_t offset, size_t bytes,
                    const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
    OpArgs function_call_args(const OpConfig &cfg) const;
};

class GatherFromPeersOp : public Op {
   public:
    GatherFromPeersOp(const std::string &prec_type, Tensor *local_buf,
                      Tensor *trans_region_local,
                      std::vector<Tensor *> remote_bufs, int sid, int rank,
                      int npeers, size_t stride, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
    OpArgs function_call_args(const OpConfig &cfg) const;
};

class PutPacketOp : public Op {
   public:
    PutPacketOp(const std::string &prec_type, Tensor *input,
                Tensor *local_tmp_buf, Tensor *recv_buf, int id, int rank,
                int dst_rank, size_t dst_offset, int flag,
                const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
    OpArgs function_call_args(const OpConfig &cfg) const;
};

class ReduceAndWritePacketOp : public Op {
   public:
    ReduceAndWritePacketOp(const std::string &prec_type,
                           std::vector<Tensor *> inputs, Tensor *output, int id,
                           int rank, int npeers, size_t elems_per_rank,
                           size_t scratch_offset, size_t remote_dst_offset,
                           int flag, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
    OpArgs function_call_args(const OpConfig &cfg) const;
};

class GetFromPacketOp : public Op {
   public:
    GetFromPacketOp(const std::string &prec_type, Tensor *input, Tensor *output,
                    size_t src_offset, size_t dst_offset, size_t npackets,
                    int flag, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
    OpArgs function_call_args(const OpConfig &cfg) const;
};

class TensorOp : public Op {
   public:
    TensorOp(const std::vector<Tensor *> &deps, Tensor *output,
             const std::string &name);
};

class TransposeOp : public Op {
   public:
    TransposeOp(const std::string &prec_type, Tensor *input, Tensor *output,
                int tp_type, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class EmbeddingOp : public Op {
   public:
    EmbeddingOp(const std::string &prec_type, Tensor *input, Tensor *weight,
                Tensor *output, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

class CastOp : public Op {
   public:
    CastOp(Tensor *input, Tensor *output, const std::string &name);
    std::string function_name(const OpConfig &cfg) const;
};

}  // namespace ark

#endif  // ARK_OPS_COMMON_H_
