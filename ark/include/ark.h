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

// DIMS_LEN is the maximum number of dimensions of a tensor. If a tensor
// has less than DIMS_LEN dimensions, the remaining dimensions will be NO_DIM.
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
    // Erase the dimension at the given index and return the erased dimension.
    DimType erase(int idx);

    DimType &operator[](int idx);

    const DimType &operator[](int idx) const;

    constexpr Dims &operator=(const Dims &) = default;

    friend bool operator==(const Dims &a, const Dims &b);
    friend bool operator!=(const Dims &a, const Dims &b);

    friend std::ostream &operator<<(std::ostream &os, const Dims &dims);

    DimType data[DIMS_LEN];
};

// TensorBuf refers to a data array that can be shared by multiple tensors.
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
    BYTE,
} TensorType;

// Tensor is a view of a TensorBuf.
//
// Illustration of a single axis of a tensor:
//
// 0           off                                                        ldim
// |------------|-------------shape-------------|---------------------------|
//               <----------------------------->
//                  data range of this tensor
//
struct Tensor
{
    // Tensor constructor
    Tensor(const Dims &shape, TensorType type, TensorBuf *buf,
           const Dims &ldims, const Dims &offs, const Dims &pads, bool exported,
           int imported_rank, int id, const std::string &name);
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
    // divided by pads[x].
    Dims pads;
    // Whether this tensor is accessed by remote devices
    bool exported;
    // If imported_rank is non-negative, the tensor is imported from another GPU
    // and don't need to allocate a TensorBuf for it.
    int imported_rank;
    // Unique id of this tensor
    int id;
    // Name of this tensor
    const std::string name;
};

class Model
{
  public:
    // Constructors.
    Model(int rank_ = 0);
    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;

    ~Model();

    // construct a tensor with given shape and data type.
    Tensor *tensor(const Dims &shape, TensorType dtype,
                   TensorBuf *buf = nullptr, const Dims &ldims = {},
                   const Dims &offs = {}, const Dims &pads = {},
                   const std::vector<Tensor *> &deps = {},
                   bool exported = false, int imported_rank = -1,
                   const std::string &name = "tensor");

    Tensor *reshape(Tensor *input, const Dims &shape, bool allowzero = false,
                    Tensor *output = nullptr,
                    const std::string &name = "reshape");
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
    // Performs reduction along the `axis` of the `input` tensor and stores the
    // result in `output`.
    // Currently, only reduction along the last dimension is supported.
    Tensor *reduce_sum(Tensor *input, int axis, Tensor *output = nullptr,
                       const std::string &name = "reduce_sum");
    Tensor *reduce_mean(Tensor *input, int axis, Tensor *output = nullptr,
                        const std::string &name = "reduce_mean");
    Tensor *reduce_max(Tensor *input, int axis, Tensor *output = nullptr,
                       const std::string &name = "reduce_max");
    // Applies layer normalization to the `input` tensor and returns the
    // normalized tensor as `output`.

    Tensor *layernorm(Tensor *input, Tensor *output = nullptr,
                      const std::string &name = "layernorm");
    // Applies softmax activation to the `input` tensor, with the softmax
    // operator
    // being performed on the last dimension of the input tensor.
    Tensor *softmax(Tensor *input, Tensor *output = nullptr,
                    const std::string &name = "softmax");
    // Transposes the `input` tensor according to the given `perm` permutation.
    // For example, transpose(input, {0, 1 ,3, 2}) will swap the last two
    // dimensions of the input tensor. Currently, only 4D tensors are supported.
    Tensor *transpose(Tensor *input, Dims perm, Tensor *output = nullptr,
                      const std::string &name = "transpose");
    // Performs matrix multiplication between the `input` tensor and another
    // `other` tensor, storing the result in `output`.
    // Optional parameters allow controlling the behavior of the multiplication,
    // such as transposing the input tensors and applying a ReLU activation.
    Tensor *matmul(Tensor *input, Tensor *other, Tensor *output = nullptr,
                   DimType splitk = 1, bool trans_input = false,
                   bool trans_other = false, bool is_relu = false,
                   const std::string &name = "matmul", int gran_lev = -1);
    // Implements the 'im2col' method for 2D convolution layers, which takes an
    // `input` tensor and reshapes it to a 2D matrix by extracting image patches
    // from the input tensor based on the provided parameters.
    Tensor *im2col(Tensor *input, int kernel_height, int kernel_width,
                   int stride_height, int stride_width, int pad_height,
                   int pad_width, int dilation_height, int dilation_width,
                   Tensor *output = nullptr,
                   const std::string &name = "im2col");
    // Implements a 2D convolution layer using the 'im2col' method.
    Tensor *conv2d(Tensor *input, DimType in_channels, DimType out_channels,
                   DimType kernel_size, DimType stride, DimType padding,
                   bool bias = false, Tensor *output = nullptr,
                   const std::string &name = "conv2d");
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
    // ReLU activation
    Tensor *relu(Tensor *input, Tensor *output = nullptr,
                 const std::string &name = "relu");
    // Applies the Gaussian Error Linear Unit (GELU) activation function to the
    // `input` tensor, element-wise. GELU is a smooth approximation of the
    // rectifier function and is widely used in deep learning models.
    Tensor *gelu(Tensor *input, Tensor *output = nullptr,
                 const std::string &name = "gelu");
    // Performs an element-wise addition operator between the `input` tensor
    // and the `other` tensor
    Tensor *add(Tensor *input, Tensor *other, Tensor *output = nullptr,
                const std::string &name = "add");
    // Performs an element-wise multiplication operator between the `input`
    // tensor and the `other` tensor,
    Tensor *mul(Tensor *input, Tensor *other, Tensor *output = nullptr,
                const std::string &name = "mul");
    // Performs an element-wise division operator between the `input`
    // tensor and the `other` tensor,
    Tensor *div(Tensor *input, Tensor *other, Tensor *output = nullptr,
                const std::string &name = "div");
    /// Sends a tensor to a destination GPU (@p dst_rank). Multiple tensors can
    /// be sent to the same GPU,so an identifier `id` is required to distinguish
    /// the tensor. Each 'send' operator must have a corresponding 'recv'
    /// operator that have the same id in another GPU's model.
    ///
    /// @param input
    /// @param id
    /// @param dst_rank Rank of the GPU to send to.
    /// @param bytes
    /// @param output
    /// @param name
    /// @return
    Tensor *send(Tensor *input, int id, int dst_rank, std::size_t bytes = 0,
                 Tensor *output = nullptr, const std::string &name = "send");
    // Blocks the execution until the corresponding 'send' operator with the
    // specified `id` is completed.
    Tensor *send_done(Tensor *input, int id, int dst_rank,
                      Tensor *output = nullptr,
                      const std::string &name = "send_done");
    // Receives a tensor from a source GPU (@p src_rank), identified by the `id`
    // parameter. Blocks the execution until the corresponding 'recv' operator
    // is completed.
    Tensor *recv(Tensor *input, int id, int src_rank, std::size_t bytes = 0,
                 Tensor *output = nullptr, const std::string &name = "recv");
    // Similar to the 'send_done' function, but implemented using CUDA in-stream
    // RDMA copy and Low Latency (LL) protocol.
    Tensor *send_mm(Tensor *input, int id, int gpu_dst, std::size_t bytes = 0,
                    Tensor *output = nullptr,
                    const std::string &name = "send_mm");
    // Similar to the 'recv' function, but implemented using CUDA in-stream RDMA
    // copy and Low Latency (LL) protocol.
    Tensor *recv_mm(Tensor *input, int id, int gpu_src, std::size_t bytes = 0,
                    Tensor *output = nullptr,
                    const std::string &name = "recv_mm");
    // Performs an all-reduce operator across all GPUs, aggregating the input
    // tensors. Takes the `input` tensor, the current GPU's `gpu_id`, and the
    // total number of GPUs `gpu_num`.
    Tensor *all_reduce(Tensor *input, int gpu_id, int gpu_num,
                       Tensor *output = nullptr,
                       const std::string &name = "all_reduce");
    // Performs an all-gather operator across all GPUs, aggregating the input
    // tensors. Takes the `input` tensor, the current GPU's `gpu_id`, and the
    // total number of GPUs `gpu_num`. Returns a vector of tensors, each
    // containing the aggregated data from all GPUs.
    std::vector<Tensor *> all_gather(Tensor *input, int gpu_id, int gpu_num,
                                     std::vector<Tensor *> output,
                                     const std::string &name);

    /// Verify if this model is valid.
    /// @return true if the model is valid, false otherwise.
    bool verify() const;

  protected:
    class Impl;
    friend class OpGraph;
    friend class SimpleScheduler;
    friend class DefaultScheduler;

  private:
    std::unique_ptr<Impl> impl;
};

class GpuBuf;

// Convenience class for executing a model.
class Executor
{
  public:
    // Constructor.
    Executor(const int gpu_id_, int rank_, int world_size_, Model &model,
             const std::string &name, int num_warps_per_sm_ = 16);
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
    // Print the content of `tns` to stdout.
    void print_tensor(Tensor *tns);

  protected:
    class Impl;

  private:
    const int gpu_id;
    const int rank;
    const int world_size;
    std::unique_ptr<Impl> impl;
};

} // namespace ark

#endif // ARK_H
