// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_H
#define ARK_H

#include <memory>
#include <ostream>
#include <string>
#include <vector>

#define ARK_MAJOR 0
#define ARK_MINOR 3
#define ARK_PATCH 0
#define ARK_VERSION (ARK_MAJOR * 10000 + ARK_MINOR * 100 + ARK_PATCH)

namespace ark {

/// Return a version string.
std::string version();

// set random seed
void srand(int seed = -1);

// get random number
int rand();

/// Initialize the ARK runtime.
///
/// This function should be called by the user before any other functions are
/// called. It is safe to call this function multiple times.
void init();

// Data type for dimension.
typedef long long int DimType;

// DIMS_LEN is the maximum number of dimensions of a tensor. If a tensor
// has less than DIMS_LEN dimensions, the remaining dimensions will be NO_DIM.
enum { DIMS_LEN = 4, NO_DIM = -1 };

// Up-to-`DIMS_LEN`-dimensional vector.
struct Dims {
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

    Dims &operator=(const Dims &) = default;

    friend bool operator==(const Dims &a, const Dims &b);
    friend bool operator!=(const Dims &a, const Dims &b);

    friend std::ostream &operator<<(std::ostream &os, const Dims &dims);

    DimType data[DIMS_LEN];
};

class Tensor;
class CodeGenerator;
class BaseScheduler;
class SchedOp;

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
REGISTER_TENSOR_TYPE(FP16, 2, "ark::half")
REGISTER_TENSOR_TYPE(BF16, 2, "ark::bfloat16")
REGISTER_TENSOR_TYPE(INT32, 4, "int32_t")
REGISTER_TENSOR_TYPE(UINT32, 4, "uint32_t")
REGISTER_TENSOR_TYPE(INT8, 1, "int8_t")
REGISTER_TENSOR_TYPE(UINT8, 1, "uint8_t")
REGISTER_TENSOR_TYPE(BYTE, 1, "unsigned char")

// TensorBuf refers to a data array that can be shared by multiple tensors.
class TensorBuf {
   public:
    TensorBuf(const DimType &bytes = 0, int id = -1);
    TensorBuf(const TensorBuf &) = default;

    size_t get_buf_offset() const;

    DimType bytes;
    int id;
    bool immutable = false;

   protected:
    void *buf = nullptr;

    friend class Tensor;
    friend class BaseScheduler;
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
    void update_pads(const Dims &tile, const Dims &ref_ldims = {},
                     const Dims &ref_padded_ldims = {});

    friend class DefaultScheduler;
    friend class SchedOp;
};

class Model {
   public:
    // Constructors.
    Model(int rank_ = 0);
    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;

    ~Model();

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
    Tensor *reduce(Tensor *input, int axis, Tensor *output = nullptr,
                   const std::string &name = "reduce");
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
    // Applies RMS (Root Mean Square Layer Normalization) normalization to the
    // `input` tensor and returns the normalized tensor as `output`.
    Tensor *rmsnorm(Tensor *input, Tensor *output = nullptr,
                    const std::string &name = "rmsnorm");
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
    // Calculates the exponential of the `input` tensor, element-wise.
    Tensor *exp(Tensor *input, Tensor *output = nullptr,
                const std::string &name = "exp");
    // Calculates the square root of the `input` tensor, element-wise.
    Tensor *sqrt(Tensor *input, Tensor *output = nullptr,
                 const std::string &name = "sqrt");
    // ReLU activation
    Tensor *relu(Tensor *input, Tensor *output = nullptr,
                 const std::string &name = "relu");
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
    Tensor *send(Tensor *input, int id, int dst_rank, std::size_t bytes = 0,
                 const std::string &name = "send");
    // Blocks the execution until the corresponding 'send' operator with the
    // specified `id` is completed.
    Tensor *send_done(Tensor *input, int id, int dst_rank,
                      const std::string &name = "send_done");
    // Receives a tensor from a source rank (@p src_rank), identified by the
    // `id` parameter. Blocks the execution until the corresponding 'recv'
    // operator is completed.
    Tensor *recv(int id, int src_rank, std::size_t bytes = 0,
                 Tensor *output = nullptr, const std::string &name = "recv");
    // Similar to the 'send_done' function, but implemented using in-stream
    // RDMA copy and Low Latency (LL) protocol.
    Tensor *send_mm(Tensor *input, int id, int gpu_dst, std::size_t bytes = 0,
                    Tensor *output = nullptr,
                    const std::string &name = "send_mm");
    // Similar to the 'recv' function, but implemented using in-stream RDMA
    // copy and Low Latency (LL) protocol.
    Tensor *recv_mm(Tensor *input, int id, int gpu_src, std::size_t bytes = 0,
                    Tensor *output = nullptr,
                    const std::string &name = "recv_mm");
    //
    Tensor *send_msll(Tensor *input, int sid, int dst_rank,
                      std::size_t bytes = 0,
                      const std::string &name = "send_msll");
    //
    Tensor *send_done_msll(Tensor *input, int dst_rank,
                           const std::string &name = "send_done_msll");
    //
    Tensor *recv_msll(int sid, int src_rank, std::size_t bytes = 0,
                      Tensor *output = nullptr,
                      const std::string &name = "recv_msll");
    //
    Tensor *put_packet_msll(Tensor *input, Tensor *local_tmp_buf,
                            Tensor *recv_buf, int id, int rank, int dst_rank,
                            size_t dst_offset, int flag,
                            const std::string &name = "put_packet_msll");
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
    Tensor *device_sync_msll(Tensor *input, int npeers,
                             const std::string &name = "device_sync_msll");

    // local reduce scatter
    Tensor *local_reduce_scatter_msll(
        Tensor *input, int gpu_id, int ngpus_per_node,
        const std::string &name = "local_reduce_scatter_msll");

    // local all gather
    Tensor *local_all_gather_msll(
        Tensor *input, int gpu_id, int ngpus_per_node,
        const std::string &name = "local_all_gather_msll");
    // read data from remote and reduce to current buffer
    Tensor *read_and_reduce_msll(
        Tensor *input, int sid, int npeers, size_t offset, size_t bytes,
        const std::string &name = "read_and_reduce_msll");
    // gather from peers
    Tensor *gather_from_peers_msll(
        Tensor *input, int sid, int npeers, size_t chunkBytes,
        const std::string &name = "gather_from_peers_msll");

    Tensor *local_all_reduce_msll(
        Tensor *input, int gpu_id, int gpu_num,
        const std::string &name = "local_all_reduce_msll");
    Tensor *local_all_reduce_packet_msll(
        Tensor *input, int gpu_id, int gpu_num,
        const std::string &name = "local_all_reduce_packet_msll");

    Tensor *reduce_and_write_packet_msll(
        Tensor *input, Tensor *scratch, Tensor *output,
        const std::vector<Tensor *> &remote_peer_bufs, int id, int rank,
        int npeers, size_t elems_per_rank, size_t scratch_offset,
        size_t remote_dst_offset, int flag,
        const std::string &name = "reduce_and_write_packet_msll");
    Tensor *get_packet_msll(Tensor *input, Tensor *output, size_t src_offset,
                            size_t dst_offset, size_t npackets, int flag,
                            const std::string &name = "get_packet_msll");
    /// Verify if this model is valid.
    /// @return true if the model is valid, false otherwise.
    bool verify() const;

   protected:
    class Impl;
    friend class OpGraph;
    friend class DefaultScheduler;

   private:
    std::unique_ptr<Impl> impl;
};

/// Convenience class for executing a model.
class Executor {
   public:
    /// Constructor.
    Executor(int rank, int world_size, Model &model, const std::string &name,
             int num_warps_per_sm = 16);
    ~Executor();
    /// Compile the model. This must be called before `launch()`.
    void compile();
    /// Launch the model (not running yet). This must be called after
    /// `compile()`.
    void launch();
    /// Run the model for `iter` iterations.
    void run(int iter);
    /// Wait for the previous run to finish.
    void wait();
    /// Stop the model and return the elapsed time in milliseconds.
    /// Once this is called, we need to call `launch()` again to run the model
    /// again.
    float stop();

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ark

#endif  // ARK_H
