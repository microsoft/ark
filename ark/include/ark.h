// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_H
#define ARK_H

#include <array>
#include <cassert>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace ark {

// set random seed
void srand(int seed = -1);
int rand();

// Init an ark program. Call this function to clean up the shared memory
// directory
void init();

// Data type for dimension.
typedef long long int DimType;

enum
{
    DIMS_LEN = 4,
    NO_DIM = -1
};

// Up-to-`DIMS_LEN`-dimensional vector.
struct Dims
{
    // Construct with given four dimensions.
    Dims(DimType d0 = NO_DIM, DimType d1 = NO_DIM, DimType d2 = NO_DIM,
         DimType d3 = NO_DIM);
    // Copy another Dims object.
    Dims(const Dims &dims_);
    // Construct from a vector. If the vector is shorter than DIMS_LEN, put
    // following NO_DIMs. Raise an error if the vector is longer than DIMS_LEN.
    Dims(const std::vector<DimType> &vec);

    // Return the volume of dimensions. If the dimensions are invalid, return
    // -1.
    DimType size() const;
    // Return the number of valid dimensions.
    int ndims() const;
    // Return a new Dims object with 4 valid dimensions by prepending 1s.
    Dims dims4() const;
    // Return true if the dimensions are empty.
    bool is_no_dim() const;
    // Return true if the dimensions are invalid.
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

// Type of tensor data.
typedef enum
{
    FP16,
    FP32,
    INT32,
} TensorType;

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
    // Tensor constructor
    Tensor(const Dims &shape, TensorType type, TensorBuf *buf,
           const Dims &ldims, const Dims &offs, const Dims &pads, bool exported,
           bool imported, int id, const std::string &name);
    Tensor(const Tensor &) = default;

    void update_pads(const std::vector<DimType> &pads);
    // Offset to the element [i0][i1][i2][i3] of this tensor in the TensorBuf.
    DimType offset(DimType i0 = 0, DimType i1 = 0, DimType i2 = 0,
                   DimType i3 = 0) const;
    // Number of elements in the tensor excluding padding.
    DimType size() const;
    // Number of dimensions in the tensor.
    int ndims() const;
    // Shape of the tensor including padding.
    Dims padded_shape() const;
    // Number of bytes of each element in the tensor.
    unsigned int type_bytes() const;
    // Number of bytes of the tensor.
    DimType shape_bytes() const;
    // Should be the same as the number of bytes of the TensorBuf.
    DimType ldims_bytes() const;
    // Offset in bytes.
    DimType offset_bytes(DimType i0 = 0, DimType i1 = 0, DimType i2 = 0,
                         DimType i3 = 0) const;
    // TODO: deprecate this function.
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
    // Reshape `input` to `shape`. If one dimension of `shape` is -1, it will be
    // inferred from the `input`. If one dimension of `shape` is 0, by default
    // (`allowzero` is false), that dimension is unchanged from the
    // corresponding one of `input`. If `allowzero` is true, that dimension is
    // set to 0, which means that the reshaped tensor is an empty tensor, i.e.,
    // `input` should also be an empty tensor. If `allowzero` is true, `shape`
    // should not include both 0 and -1 at the same time. If `shape` is an empty
    // vector, `input` will be converted to a scalar.
    Tensor *reshape(Tensor *input, std::initializer_list<DimType> shape,
                    bool allowzero = false, Tensor *output = nullptr,
                    const std::string &name = "reshape");
    // Returns an identical tensor of `input` with execution dependencies
    // `deps`.
    Tensor *identity(Tensor *input, const std::vector<Tensor *> &deps = {},
                     Tensor *output = nullptr,
                     const std::string &name = "identity");

    // Shard `input` along `axis` into `dim_per_shard`-dimensional shards.
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
    // Multiply `input` by `val`.
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
    // send data from src to dst of id
    Tensor *send_mm(Tensor *input, int id, int gpu_dst, std::size_t bytes = 0,
                    Tensor *output = nullptr,
                    const std::string &name = "send_mm");
    Tensor *recv_mm(Tensor *input, int id, int gpu_src, std::size_t bytes = 0,
                    Tensor *output = nullptr,
                    const std::string &name = "recv_mm");
    Tensor *all_reduce(Tensor *input, int gpu_id, int gpu_num,
                       Tensor *output = nullptr,
                       const std::string &name = "all_reduce");
    // Create a new TensorBuf object with `bytes` bytes.
    // A common usage is setting `bytes` to 0 during declaring a model and let
    // the scheduler determine the value after the model is completely defined.
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

class ExecutorMember;
class GpuBuf;
// Convenience class for executing a model.
class Executor
{
  public:
    // Constructor.

    Executor(const int gpu_id_, int rank_, int world_size_, const Model &model,
             const std::string &name);
    ~Executor();
    // Compile the model. This must be called before `launch()`.
    void compile();
    // Launch the model (not running yet). This must be called after
    // `compile()`.
    void launch();
    // Run the model for `iter` iterations.
    void run(int iter);
    // Wait for the previous run to finish.
    void wait();
    // Stop the model and return the elapsed time in milliseconds.
    // Once this is called, we need to call `launch()` again to run the model
    // again.
    float stop();
    // Get the corresponding GPU buffer of the executor from the given model
    // tensor.
    GpuBuf *get_gpu_buf(Tensor *tns) const;
    // Get the corresponding tensor of the executor from the given model tensor.
    // Both tensors may be different if the scheduler creates an optimized model
    // out of the original one.
    Tensor *get_tensor(Tensor *tns) const;
    // Copy contiguous data from a host buffer to the given tensor's (possibly
    // non-contiguous) data range on GPU.
    void tensor_memcpy(Tensor *tns, const void *src, size_t bytes);
    // Copy (possibly non-contiguous) data from a tensor on GPU to a contiguous
    // host buffer. The given number of bytes is copied, in order of appearance
    // on the memory. This function assumes that `dst` is large enough to hold
    // the data.
    void tensor_memcpy(void *dst, Tensor *src, size_t bytes);
    // Set all bytes of `tns` into zero.
    void tensor_clear(Tensor *tns);

  private:
    const int gpu_id;
    const int rank;
    const int world_size;
    std::unique_ptr<ExecutorMember> member;
};

} // namespace ark

#endif // ARK_H