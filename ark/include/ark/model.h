// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_H
#define ARK_MODEL_H

#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>

#include "dims.h"

namespace ark {

class Tensor;
class CodeGenerator;
class BaseScheduler;
class SchedOp;
class Model;

/// Type of tensor data.
class TensorType {
   private:
    const std::string name_;
    const int bytes_;
    const std::string type_str_;

   public:
    TensorType(const std::string &name = "none", int bytes = 0,
               const std::string &type_str = "void *");

    bool operator==(const TensorType &other) const;
    bool operator!=(const TensorType &other) const;

    int bytes() const;
    const std::string &name() const;
    const std::string &type_str() const;
};

const TensorType NONE;

std::ostream &operator<<(std::ostream &os, const TensorType &type);

#define REGISTER_TENSOR_TYPE(_type_name, _bytes, _type_str) \
    class TensorType_##_type_name : public TensorType {     \
       public:                                              \
        TensorType_##_type_name()                           \
            : TensorType{#_type_name, _bytes, _type_str} {} \
    };                                                      \
    const TensorType_##_type_name _type_name;

REGISTER_TENSOR_TYPE(FP32, 4, "float")
REGISTER_TENSOR_TYPE(FP16, 2, "ark::fp16")
REGISTER_TENSOR_TYPE(BF16, 2, "ark::bf16")
REGISTER_TENSOR_TYPE(INT32, 4, "int32_t")
REGISTER_TENSOR_TYPE(UINT32, 4, "uint32_t")
REGISTER_TENSOR_TYPE(INT8, 1, "int8_t")
REGISTER_TENSOR_TYPE(UINT8, 1, "uint8_t")
REGISTER_TENSOR_TYPE(BYTE, 1, "unsigned char")

class GpuBuffer;
// TensorBuf refers to a data array that can be shared by multiple tensors.
class TensorBuf {
   public:
    TensorBuf(const DimType &bytes = 0, int id = -1);
    TensorBuf(const TensorBuf &) = default;

    size_t get_buf_offset() const;

    DimType bytes;
    int id;
    bool immutable = true;

   protected:
    std::shared_ptr<GpuBuffer> buf = nullptr;

    friend class Tensor;
    friend class DefaultScheduler;
};

/// Tensor is a view of a TensorBuf.
///
/// Illustration of a single axis of a tensor:
///
/// 0           off                                                        ldim
/// |------------|-------------shape-------------|---------------------------|
///       ^       <----------------------------->                ^
///       |          data range of this tensor                   |
///       +------------------------------------------+-----------+
///                                                  |
///                                        We call these "padding".
///
class Tensor {
   public:
    /// Tensor constructor.
    Tensor(const Dims &shape, const TensorType &type, TensorBuf *buf,
           const Dims &ldims, const Dims &offs, const Dims &pads, bool exported,
           int imported_rank, int id, const std::string &name);
    Tensor(const Tensor &) = default;

    /// Copy contiguous data from a host buffer to the given tensor's (possibly
    /// non-contiguous) data range.
    ///
    /// For example, say the tensor is a 2D float tensor with shape [2, 3],
    /// ldims [2, 4], offs [0, 0], and pads [1, 1], then the data in the host
    /// buffer is 0, 1, ..., 5. After writing, the data in the tensor will be:
    ///
    ///     [[0, 1, 2, ?],
    ///      [3, 4, 5, ?]]
    ///
    /// where ? means the original unmodified value.
    ///
    /// @param buf The host buffer to copy from. The buffer must be large enough
    /// to hold the data.
    ///
    void write(const void *buf);

    /// Copy (possibly non-contiguous) data from a tensor on GPU to a contiguous
    /// host buffer.
    ///
    /// The given number of bytes is copied, in order of appearance on the
    /// memory. This function assumes that @p buf is large enough to hold the
    /// data. For example, say the tensor is a 2D float tensor with shape [2,
    /// 3], ldims [2, 4], offs [0, 0], and pads [1, 1], then the data in the
    /// tensor is:
    ///
    ///     [[0, 1, 2, 3],
    ///      [4, 5, 6, 7]]
    ///
    /// After read, the data in the host buffer will be 0, 1, 2, 4, 5, 6.
    ///
    /// @param buf The host buffer to copy to. The buffer must be large enough
    /// to hold the data. If @p buf is nullptr, a new buffer will be allocated.
    /// @return The host buffer that holds the data.
    ///
    void *read(void *buf = nullptr);

    /// Copy all the underlying buffer data (including padding) to a contiguous
    /// host buffer.
    ///
    /// This function is mainly for debugging purposes.
    ///
    /// @param buf The host buffer to copy to. The buffer must be large enough
    /// to hold the data. If @p buf is nullptr, a new buffer will be allocated.
    /// @return The host buffer that holds the data.
    ///
    void *read_raw(void *buf = nullptr);

    /// Set all bytes of the tensor buffer to 0.
    void clear();

    /// Offset to the element [i0][i1][i2][i3] of this tensor in the TensorBuf.
    /// @param i0, i1, i2, i3 The indices of the element.
    /// @return The offset in the number of elements.
    DimType offset(DimType i0 = 0, DimType i1 = 0, DimType i2 = 0,
                   DimType i3 = 0) const;

    /// Number of elements in the tensor excluding padding.
    /// @return The number of elements.
    DimType size() const;

    /// Number of dimensions of the tensor.
    /// @return The number of dimensions.
    int ndims() const;

    /// Number of bytes of each element in the tensor.
    /// @return The number of bytes.
    int type_bytes() const;

    /// Number of bytes in the tensor's data range.
    /// @return The number of bytes.
    DimType shape_bytes() const;

    /// Equivalent as the number of bytes of the underlying @ref TensorBuf.
    /// @return The number of bytes.
    DimType ldims_bytes() const;

    /// Offset in bytes.
    /// @param i0, i1, i2, i3 The indices of the element.
    /// @return The offset in bytes.
    DimType offset_bytes(DimType i0 = 0, DimType i1 = 0, DimType i2 = 0,
                         DimType i3 = 0) const;

    /// Checks if the tensor has the actually memory allocated.
    /// @return True if the tensor has the memory allocated.
    bool is_alloced() const;

    /// Checks if the tensor's data range is sequential in memory.
    /// @return True if the tensor is sequential in memory.
    bool is_sequential() const;

    /// TensorBuf that this tensor is associated with
    TensorBuf *buf;
    /// Data type of each element in the tensor
    TensorType type;
    /// Shape of the tensor
    Dims shape;
    /// Leading dimensions of the underlying data array
    Dims ldims;
    /// Offset of the tensor in the underlying data array
    Dims offs;
    /// Unit dimensions of the underlying data array. ldims[x] should be always
    /// divided by pads[x].
    Dims pads;
    /// Whether this tensor is local and accessed by remote devices.
    bool exported;
    /// If `imported_rank` is non-negative, the tensor is imported from another
    /// rank and don't need to allocate a TensorBuf for it.
    int imported_rank;
    /// Unique id of this tensor
    int id;
    /// Name of this tensor
    const std::string name;

   protected:
    bool update_pads(const Dims &tile, const Tensor *ref_tensor = nullptr,
                     const Dims &ref_orig_ldims = {});

    friend class DefaultScheduler;
    friend class SchedOp;
};

/// Type of operator argument.
struct OpArgType {
    OpArgType(size_t id, const std::string &name) : id(id), name(name) {}
    size_t id;
    std::string name;
};

bool operator==(const OpArgType &lhs, const OpArgType &rhs);

bool operator!=(const OpArgType &lhs, const OpArgType &rhs);

std::ostream &operator<<(std::ostream &os, const OpArgType &type);

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

/// A node of @ref Model.
class OpNode {
   public:
    /// Construct an empty @ref OpNode.
    OpNode(){};

    /// Destruct an @ref OpNode.
    ~OpNode(){};

    /// The list of @ref Op that this @ref OpNode contains. Sorted in the
    /// execution order.
    std::vector<Op *> ops;

    /// The list of @ref OpNode that depends on this @ref OpNode.
    std::set<OpNode *> users;

    /// The list of @ref OpNode that this @ref OpNode depends on.
    std::set<OpNode *> producers;

    /// Remove this @ref OpNode from the graph.
    void remove_self();

    /// Get the name of this @ref OpNode.
    std::string get_name() const;
};

class Model {
   public:
    // Constructors.
    Model(int rank_ = 0);
    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;

    ~Model();

    /// Verify if this model is valid.
    /// @return true if the model is valid, false otherwise.
    bool verify() const;

    void create_nodes();
    void clear_nodes();

    /// Get the @ref OpNode list.
    /// @return The @ref OpNode list.
    const std::list<std::unique_ptr<OpNode>> &get_nodes() const;

    /// Break a @ref OpNode into two @ref OpNode.
    ///
    /// The original node will have the first @p op_idx ops, and the new node
    /// will have the rest.
    ///
    /// @param node The @ref OpNode to break.
    /// @param op_idx The index of the first op in the new @ref OpNode.
    /// @return The new @ref OpNode.
    OpNode *break_node(OpNode *node, int op_idx);

    /// Check dependencies between two @ref OpNode.
    ///
    /// @param node1 The first @ref OpNode.
    /// @param node2 The second @ref OpNode.
    /// @return True if @p node1 depends on @p node2.
    bool depends_on(OpNode *node1, OpNode *node2) const;

    std::string serialize(int indent = -1) const;

    /// Returns a tensor object.
    ///
    /// @param shape Shape of the tensor, where the data of interest is.
    /// @param ttype Type of the tensor data.
    /// @param buf The @ref TensorBuf that holds the entire data including the
    /// padding.
    /// @param ldims Leading dimensions (ldim) of the tensor, which may be
    /// different from the shape. @p ldims can be considered as the actual shape
    /// of the underlying data buffer (@ref TensorBuf).
    /// @param offs Offsets of the tensor. The data of interest starts at
    /// @p offs and ends at @p offs + @p shape.
    /// @param pads If a dimension of @p pads is set to larger than 1, the
    /// corresponding ldim will be set to the minimum multiple of @p pads that
    /// is larger than or equal to the previous ldim. Padding is accumulated
    /// across all tensors that share the same @ref TensorBuf. For example, if
    /// one tensor sets the last dimension of @p pads to 2, and another tensor
    /// sets the last dimension of @p pads to 3, then the corresponding ldim
    /// will be the minimum multiple of 2x3=6 that is larger than or equal to
    /// the corresponding dimension of @p offs + @p shape.
    /// @param exported Whether the tensor is exported to other processes. This
    /// should be set to true if the tensor is used as an input or output of a
    /// remote process.
    /// @param imported_rank The rank of the process that exports the tensor.
    /// If @p imported_rank is set to a non-negative value, the tensor will be
    /// considered as a remote tensor, hence no memory will be allocated for it
    /// on the local. @p imported_rank should be set to -1 if the tensor resides
    /// on the local.
    /// @param name Name of the tensor.
    /// @return Pointer to a tensor object.
    ///
    Tensor *tensor(const Dims &shape, const TensorType &ttype,
                   TensorBuf *buf = nullptr, const Dims &ldims = {},
                   const Dims &offs = {}, const Dims &pads = {},
                   const std::vector<Tensor *> &deps = {},
                   bool exported = false, int imported_rank = -1,
                   const std::string &name = "tensor");

    Tensor *reshape(Tensor *input, const Dims &shape, bool allowzero = false,
                    Tensor *output = nullptr,
                    const std::string &name = "reshape");
    Tensor *reshape(Tensor *input, const std::initializer_list<DimType> &shape,
                    bool allowzero = false, Tensor *output = nullptr,
                    const std::string &name = "reshape");
    // Reshape `input` to `shape`. If one dimension of `shape` is -1, it will be
    // inferred from the `input`. If one dimension of `shape` is 0, by default
    // (`allowzero` is false), that dimension is unchanged from the
    // corresponding one of `input`. If `allowzero` is true, that dimension is
    // set to 0, which means that the reshaped tensor is an empty tensor, i.e.,
    // `input` should also be an empty tensor. If `allowzero` is true, `shape`
    // should not include both 0 and -1 at the same time. If `shape` is an empty
    // vector, `input` will be converted to a scalar.
    Tensor *reshape(Tensor *input, const std::vector<DimType> &shape,
                    bool allowzero = false, Tensor *output = nullptr,
                    const std::string &name = "reshape");
    // Returns an identical tensor of `input` with execution dependencies
    // `deps`.
    Tensor *identity(Tensor *input, const std::vector<Tensor *> &deps = {},
                     const std::string &name = "identity");

    // Shard `input` along `axis` into `dim_per_shard`-dimensional shards.
    std::vector<Tensor *> sharding(Tensor *input, DimType axis,
                                   DimType dim_per_shard,
                                   const std::string &name = "sharding");
    // Performs reduction along the `axis` of the `input` tensor and stores the
    // result in `output`.
    // Currently, only reduction along the last dimension is supported.
    template <typename ReduceOpType>
    Tensor *reduce(Tensor *input, int axis, bool keepdims = true,
                   Tensor *output = nullptr,
                   const std::string &name = "reduce");
    Tensor *reduce_sum(Tensor *input, int axis, bool keepdims = true,
                       Tensor *output = nullptr,
                       const std::string &name = "reduce_sum");
    Tensor *reduce_mean(Tensor *input, int axis, bool keepdims = true,
                        Tensor *output = nullptr,
                        const std::string &name = "reduce_mean");
    Tensor *reduce_max(Tensor *input, int axis, bool keepdims = true,
                       Tensor *output = nullptr,
                       const std::string &name = "reduce_max");
    // Applies layer normalization to the `input` tensor and returns the
    // normalized tensor as `output`.
    Tensor *layernorm(Tensor *input, Tensor *output = nullptr,
                      const std::string &name = "layernorm");
    // Transposes the `input` tensor according to the given `perm` permutation.
    // For example, transpose(input, {0, 1 ,3, 2}) will swap the last two
    // dimensions of the input tensor. Currently, only 4D tensors are supported.
    Tensor *transpose(Tensor *input, Dims perm, Tensor *output = nullptr,
                      const std::string &name = "transpose");
    // Performs matrix multiplication between the `input` tensor and another
    // `other` tensor, storing the result in `output`.
    Tensor *matmul(Tensor *input, Tensor *other, Tensor *output = nullptr,
                   DimType splitk = 1, bool trans_input = false,
                   bool trans_other = false, const std::string &name = "matmul",
                   int gran_lev = -1);
    // Implements the 'im2col' method for 2D convolution layers, which takes an
    // `input` tensor and reshapes it to a 2D matrix by extracting image patches
    // from the input tensor based on the provided parameters.
    Tensor *im2col(Tensor *input, int kernel_height, int kernel_width,
                   int stride_height, int stride_width, int pad_height,
                   int pad_width, int dilation_height, int dilation_width,
                   Tensor *output = nullptr,
                   const std::string &name = "im2col");
    // Applies max-pooling on the `input` tensor using `kernel_size` and
    // `stride`, reducing its spatial size. The output shape is calculated based
    // on the input tensor's shape and the stride value as follows: {is[0],
    // (is[1] + stride - 1) / stride, (is[2] + stride - 1) / stride, is[3]},
    // where 'is' represents the input tensor's shape.
    Tensor *max_pool(Tensor *input, DimType kernel_size, DimType stride,
                     Tensor *output = nullptr,
                     const std::string &name = "max_pool");
    // Multiplies the `input` tensor by a scalar `val`, element-wise.
    Tensor *scale(Tensor *input, float val, Tensor *output = nullptr,
                  const std::string &name = "scale");
    //
    template <typename MathOpType>
    Tensor *math(Tensor *input, Tensor *output = nullptr,
                 const std::string &name = "math");
    // Calculates the exponential of the `input` tensor, element-wise.
    Tensor *exp(Tensor *input, Tensor *output = nullptr,
                const std::string &name = "exp");
    // Calculates the square root of the `input` tensor, element-wise.
    Tensor *sqrt(Tensor *input, Tensor *output = nullptr,
                 const std::string &name = "sqrt");
    // Calculates the reverse square root of the `input` tensor, element-wise.
    Tensor *rsqrt(Tensor *input, Tensor *output = nullptr,
                  const std::string &name = "rsqrt");
    // ReLU activation
    Tensor *relu(Tensor *input, Tensor *output = nullptr,
                 const std::string &name = "relu");
    // Copy the `input` tensor to `output` tensor
    Tensor *copy(Tensor *input, Tensor *output = nullptr,
                 const std::string &name = "copy");
    // Applies the Gaussian Error Linear Unit (GELU) activation function to the
    // `input` tensor, element-wise. GELU is a smooth approximation of the
    // rectifier function and is widely used in deep learning models.
    Tensor *gelu(Tensor *input, Tensor *output = nullptr,
                 const std::string &name = "gelu");
    // Sigmoid activation
    Tensor *sigmoid(Tensor *input, Tensor *output = nullptr,
                    const std::string &name = "sigmoid");
    // Performs rotary position embedding (RoPE) on the `input` tensor
    Tensor *rope(Tensor *input, Tensor *other, Tensor *output = nullptr,
                 const std::string &name = "rope");
    // Template for broadcated arithmetic operators.
    template <typename ArithmeticOpType>
    Tensor *arithmetic(Tensor *input, Tensor *other, Tensor *output = nullptr,
                       const std::string &name = "arithmeitc");
    // Performs an element-wise addition operator between the `input` tensor
    // and the `other` tensor
    Tensor *add(Tensor *input, Tensor *other, Tensor *output = nullptr,
                const std::string &name = "add");
    // Performs an element-wise subtraction operator between the `input` tensor
    // and the `other` tensor
    Tensor *sub(Tensor *input, Tensor *other, Tensor *output = nullptr,
                const std::string &name = "sub");
    // Performs an element-wise multiplication operator between the `input`
    // tensor and the `other` tensor,
    Tensor *mul(Tensor *input, Tensor *other, Tensor *output = nullptr,
                const std::string &name = "mul");
    // Performs an element-wise division operator between the `input`
    // tensor and the `other` tensor,
    Tensor *div(Tensor *input, Tensor *other, Tensor *output = nullptr,
                const std::string &name = "div");
    /// Sends a tensor to a destination rank (@p dst_rank). Multiple tensors can
    /// be sent to the same rank,so an identifier `id` is required to
    /// distinguish the tensor. Each 'send' operator must have a corresponding
    /// 'recv' operator that have the same id in another rank's model.
    ///
    /// @param input
    /// @param id
    /// @param dst_rank Rank of the GPU to send to.
    /// @param bytes
    /// @param name
    /// @return
    Tensor *send(Tensor *input, int sid, int dst_rank, std::size_t bytes = 0,
                 const std::string &name = "send");
    // Blocks the execution until the corresponding 'send' operator with the
    // specified `id` is completed.
    Tensor *send_done(Tensor *input, int sid, int dst_rank,
                      const std::string &name = "send_done");
    // Receives a tensor from a source rank (@p src_rank), identified by the
    // `id` parameter. Blocks the execution until the corresponding 'recv'
    // operator is completed.
    Tensor *recv(int sid, int src_rank, std::size_t bytes = 0,
                 Tensor *output = nullptr, const std::string &name = "recv");
    //
    Tensor *put_packet(Tensor *input, Tensor *local_tmp_buf, Tensor *recv_buf,
                       int id, int rank, int dst_rank, size_t dst_offset,
                       int flag, const std::string &name = "put_packet");
    // Performs an all-reduce operator across all ranks, aggregating the input
    // tensors. Takes the `input` tensor, the current GPU's rank, and the
    // total number of ranks `rank_num`.
    Tensor *all_reduce(Tensor *input, int rank, int rank_num,
                       Tensor *output = nullptr,
                       const std::string &name = "all_reduce");
    // Performs an all-gather operator across all ranks, aggregating the input
    // tensors. Takes the `input` tensor, the current GPU's rank, and the
    // total number of ranks `rank_num`. Returns a vector of tensors, each
    // containing the aggregated data from all ranks.
    std::vector<Tensor *> all_gather(Tensor *input, int rank, int rank_num,
                                     const std::vector<Tensor *> &output = {},
                                     const std::string &name = "all_gather");
    /// Embedding layer.
    Tensor *embedding(Tensor *input, Tensor *weight, Tensor *output = nullptr,
                      const std::string &name = "embedding");
    /// Tensor type casting.
    Tensor *cast(Tensor *input, const TensorType &ttype,
                 Tensor *output = nullptr, const std::string &name = "cast");

    // sync across multi devices
    Tensor *device_sync(Tensor *input, int npeers,
                        const std::string &name = "device_sync");

    // local reduce scatter
    Tensor *local_reduce_scatter(
        Tensor *input, int gpu_id, int ngpus_per_node,
        const std::string &name = "local_reduce_scatter");

    // local all gather
    Tensor *local_all_gather(Tensor *input, int gpu_id, int ngpus_per_node,
                             int axis = 0,
                             const std::string &name = "local_all_gather");
    // read data from remote and reduce to current buffer
    Tensor *read_and_reduce(Tensor *input, int sid, int npeers, size_t offset,
                            size_t bytes,
                            const std::string &name = "read_and_reduce");
    // gather from peers
    Tensor *gather_from_peers(Tensor *input, Tensor *tile, int sid, int npeers,
                              size_t chunkBytes,
                              const std::string &name = "gather_from_peers");

    Tensor *local_all_reduce(Tensor *input, int gpu_id, int gpu_num,
                             const std::string &name = "local_all_reduce");
    Tensor *local_all_reduce_packet(
        Tensor *input, int gpu_id, int gpu_num,
        const std::string &name = "local_all_reduce_packet");

    Tensor *reduce_and_write_packet(
        Tensor *input, Tensor *scratch, Tensor *output,
        const std::vector<Tensor *> &remote_peer_bufs, int id, int rank,
        int npeers, size_t elems_per_rank, size_t scratch_offset,
        size_t remote_dst_offset, int flag,
        const std::string &name = "reduce_and_write_packet");
    Tensor *get_packet(Tensor *input, Tensor *output, size_t src_offset,
                       size_t dst_offset, size_t npackets, int flag,
                       const std::string &name = "get_packet");

   protected:
    class Impl;
    friend class DefaultScheduler;

   private:
    std::unique_ptr<Impl> impl;
};

}  // namespace ark

#endif  // ARK_MODEL_H
