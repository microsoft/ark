// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_H
#define ARK_H
// clang-format off
#include "vector_types.h"
#include "cutlass/half.h"
// clang-format on
#include "third_party/json/json.h"
#include <array>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace ark {

typedef long long int DimType;

enum
{
    DIMS_LEN = 4,
    NO_DIM = -1
};

const std::string shape_str(const std::vector<DimType> &shape);

// Up-to-`DIMS_LEN`-dimensional vector.
struct Dims
{
    Dims(DimType d0 = NO_DIM, DimType d1 = NO_DIM, DimType d2 = NO_DIM,
         DimType d3 = NO_DIM);
    Dims(const Dims &dims_);
    Dims(const std::vector<DimType> &vec);

    DimType size() const;
    int ndims() const;
    Dims dims4() const;
    bool is_no_dim() const;
    bool is_invalid() const;

    DimType &operator[](DimType idx)
    {
        return data[idx];
    }
    const DimType &operator[](DimType idx) const
    {
        return data[idx];
    }

    constexpr Dims &operator=(const Dims &) = default;

    friend bool operator==(const Dims &a, const Dims &b);
    friend bool operator!=(const Dims &a, const Dims &b);

    friend std::ostream &operator<<(std::ostream &os, const Dims &dims);

    DimType data[DIMS_LEN];
};

void to_json(nlohmann::json &j, const Dims &dims);
void from_json(const nlohmann::json &j, Dims &dims);

// TensorBuf refers to a data array that
// can be shared by multiple tensors.
struct TensorBuf
{
    TensorBuf(const DimType &bytes = 0, int id = -1);
    TensorBuf(const TensorBuf &) = default;

    DimType bytes;
    int id;
    bool immutable = false;
};

void to_json(nlohmann::json &j, const TensorBuf &tbuf);
void from_json(const nlohmann::json &j, TensorBuf &tbuf);

// Type of tensor data.
typedef enum
{
    FP16,
    FP32,
    INT32,
} TensorType;

// clang-format off
NLOHMANN_JSON_SERIALIZE_ENUM(TensorType,
{
    {FP16, "f16"},
    {FP32, "f32"},
    {INT32, "i32"},
})
// clang-format on

// Tensor is a view of a TensorBuf.
//
// Illustration of a single axis of a
// tensor:
//
// 0           off ldim
// |------------|-------------shape-------------|---------------------------|
//               <----------------------------->
//                  data range of this
//                  tensor
//
struct Tensor
{
    Tensor(const Dims &shape, TensorType type, TensorBuf *buf,
           const Dims &ldims, const Dims &offs, const Dims &pads, bool exported,
           bool imported, int id, const std::string &name);
    Tensor(const Tensor &) = default;

    void update_pads(const std::vector<DimType> &pads);
    DimType offset(DimType i0 = 0, DimType i1 = 0, DimType i2 = 0,
                   DimType i3 = 0) const;
    DimType size() const;
    int ndims() const;
    Dims padded_shape() const;
    unsigned int type_bytes() const;
    DimType shape_bytes() const;
    DimType ldims_bytes() const;
    DimType offset_bytes(DimType i0 = 0, DimType i1 = 0, DimType i2 = 0,
                         DimType i3 = 0) const;

    bool is_sequential() const;

    // TensorBuf that this tensor is associated with
    TensorBuf *buf;
    // Data type of each element in the tensor
    TensorType type;
    // Shape of the tensor
    Dims shape;
    // Leading dimensions of the underlying data array
    Dims ldims;
    // Offset of the tensor in the underlying data array
    Dims offs;
    // Unit dimensions of the underlying data array. ldims[x] should be always
    // divided by udims[x].
    Dims pads;
    // Whether this tensor is accessed by remote devices
    bool exported;
    // if imported is true, the tensor is imported from another GPU and don't
    // need to allocate a TensorBuf for it.
    bool imported;
    // Unique id of this tensor
    int id;
    // Name of this tensor
    const std::string name;
};
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

// Type of operation argument.
typedef enum
{
    OP_ARG_INT,
    OP_ARG_INT64,
    OP_ARG_UINT64,
    OP_ARG_BOOL,
    OP_ARG_FLOAT
} OpArgType;
// Stores an arbitrary type of argument given to an operation.
struct OpArg
{
    OpArg(int arg);
    OpArg(DimType arg);
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

class Model
{
  public:
    // Constructors.
    Model()
    {
    }
    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;

    Tensor *tensor(const Dims &shape, TensorType dtype,
                   TensorBuf *buf = nullptr, const Dims &ldims = {},
                   const Dims &offs = {}, const Dims &pads = {},
                   const std::vector<Tensor *> &deps = {},
                   bool exported = false, bool imported = false,
                   const std::string &name = "tensor");

    Tensor *reshape(Tensor *input, const Dims &shape, bool allowzero,
                    Tensor *output, const std::string &name);
    Tensor *reshape(Tensor *input, std::initializer_list<DimType> shape,
                    bool allowzero = false, Tensor *output = nullptr,
                    const std::string &name = "reshape");

    Tensor *identity(Tensor *input, const std::vector<Tensor *> &deps = {},
                     Tensor *output = nullptr,
                     const std::string &name = "identity");
    std::vector<Tensor *> sharding(Tensor *input, DimType axis,
                                   DimType dim_per_shard,
                                   const std::string &name = "sharding");

    Tensor *reduce(Tensor *input, DimType axis, Tensor *output = nullptr,
                   bool is_relu = false, const std::string &name = "reduce");
    Tensor *transpose(Tensor *input, Dims perm, Tensor *output = nullptr,
                      const std::string &name = "transpose");
    Tensor *matmul(Tensor *input, Tensor *other, Tensor *output = nullptr,
                   DimType splitk = 1, bool trans_input = false,
                   bool trans_other = false, bool is_relu = false,
                   const std::string &name = "matmul", int gran_lev = -1);
    Tensor *linear(Tensor *input, DimType out_features, bool bias = true,
                   Tensor *output = nullptr, DimType splitk = 1,
                   bool is_relu = false, const std::string &name = "linear",
                   int gran_lev = -1);
    Tensor *im2col(Tensor *input, DimType kernel_height, DimType kernel_width,
                   DimType stride_height, DimType stride_width,
                   DimType pad_height, DimType pad_width,
                   DimType dilation_height, DimType dilation_width,
                   Tensor *output = nullptr,
                   const std::string &name = "im2col");
    Tensor *conv2d(Tensor *input, DimType in_channels, DimType out_channels,
                   DimType kernel_size, DimType stride, DimType padding,
                   bool bias = false, Tensor *output = nullptr,
                   const std::string &name = "conv2d");
    Tensor *max_pool(Tensor *input, DimType kernel_size, DimType stride,
                     Tensor *output = nullptr,
                     const std::string &name = "max_pool");
    Tensor *scale(Tensor *input, float val, Tensor *output = nullptr,
                  const std::string &name = "scale");
    Tensor *add(Tensor *input, Tensor *other, Tensor *output = nullptr,
                const std::string &name = "add");
    Tensor *mul(Tensor *input, Tensor *other, Tensor *output = nullptr,
                const std::string &name = "mul");
    Tensor *send(Tensor *input, int id, int gpu_dst, std::size_t bytes = 0,
                 Tensor *output = nullptr, const std::string &name = "send");
    Tensor *send_done(Tensor *input, int id, Tensor *output = nullptr,
                      const std::string &name = "send_done");
    Tensor *recv(Tensor *input, int id, int gpu_src, std::size_t bytes = 0,
                 Tensor *output = nullptr, const std::string &name = "recv");

    Tensor *send_mm(Tensor *input, int id, int gpu_dst, std::size_t bytes = 0,
                    Tensor *output = nullptr,
                    const std::string &name = "send_mm");
    Tensor *recv_mm(Tensor *input, int id, int gpu_src, std::size_t bytes = 0,
                    Tensor *output = nullptr,
                    const std::string &name = "recv_mm");
    Tensor *all_reduce(Tensor *input, int gpu_id, int gpu_num,
                       Tensor *output = nullptr,
                       const std::string &name = "all_reduce");

    TensorBuf *create_tensor_buf(const DimType bytes = 0);
    void destroy_tensor_buf(const TensorBuf *buf);
    Op *create_op(const OpType &type, const OpPrecType &prec_type,
                  const std::vector<Tensor *> &in_deps,
                  const std::vector<Tensor *> &out_deps,
                  const std::vector<OpArg> &args, const std::string &name,
                  int gran_lev = -1);

    const std::list<std::unique_ptr<TensorBuf>> &get_tensor_bufs() const
    {
        return tns_bufs_storage;
    };
    const std::list<std::unique_ptr<Tensor>> &get_tensors() const
    {
        return tns_storage;
    };
    const std::list<std::unique_ptr<Op>> &get_ops() const
    {
        return ops_storage;
    };
    const Op *get_gen_op(Tensor *tns) const;
    const std::set<Op *> &get_ref_ops(Tensor *tns) const;
    bool is_no_ref(Tensor *tns) const;

  private:
    std::list<std::unique_ptr<TensorBuf>> tns_bufs_storage;
    std::list<std::unique_ptr<Tensor>> tns_storage;
    std::list<std::unique_ptr<Op>> ops_storage;

    std::map<Tensor *, Op *> gen_op;
    std::map<Tensor *, std::set<Op *>> ref_ops;
    int next_eid = 0;
    std::map<std::string, int> name_cnts;
};

void to_json(nlohmann::json &j, const Model &model);
void from_json(const nlohmann::json &j, Model &model);

// class GpuMgrCtx;
// class SchedulerBase;
// class GpuLoopKernel;
// class GpuStream;
// class GpuBuf;
class ExecutorMember;
class GpuBuf;
// Convenience class for executing a
// model.
class Executor
{
  public:
    Executor(const int gpu_id_, int rank_, int world_size_, const Model &model,
             const std::string &name);
    ~Executor();

    void compile();
    void launch();
    void run(int iter);
    void wait();
    float stop();
    GpuBuf *get_gpu_buf(Tensor *tns) const;
    Tensor *get_tensor(Tensor *tns) const;
    void tensor_memcpy(Tensor *tns, const void *src, size_t bytes);
    void tensor_memcpy(void *dst, Tensor *src, size_t bytes);
    void tensor_clear(Tensor *tns);

  private:
    const int gpu_id;
    const int rank;
    const int world_size;
    std::unique_ptr<ExecutorMember> member;
    // GpuMgrCtx *ctx;
    // SchedulerBase *sched;
    // GpuLoopKernel *glk = nullptr;
    // GpuStream stream = nullptr;
};

} // namespace ark

#endif // ARK_H