// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_HPP
#define ARK_MODEL_HPP

#include <string>
#include <vector>

#include "dims.hpp"
#include "model_graph.hpp"
#include "model_ref.hpp"
#include "tensor.hpp"

namespace ark {

class ModelDataT;
using ModelDataType = std::shared_ptr<ModelDataT>;

extern const ModelDataType NONE;
extern const ModelDataType FP32;
extern const ModelDataType FP16;
extern const ModelDataType BF16;
extern const ModelDataType INT32;
extern const ModelDataType UINT32;
extern const ModelDataType INT8;
extern const ModelDataType UINT8;
extern const ModelDataType BYTE;

class Model : public ModelGraph {
   private:
    int rank_;
    int world_size_;

   public:
    Model(int rank = 0, int world_size = 1)
        : rank_(rank), world_size_(world_size) {}
    Model(const Model &other) : ModelGraph(other), rank_(other.rank()) {}
    ~Model() {}

    Model &operator=(const Model &other) = default;

    int rank() const { return rank_; }

    int world_size() const { return world_size_; }

    Model compress() const;

    void noop(Tensor input, const std::string &name = "");

    /// Returns a tensor object.
    ///
    /// @param shape Shape of the tensor, where the data of interest is.
    /// @param dtype Type of the tensor data.
    /// @param strides Leading dimensions (ldim) of the tensor, which may be
    /// different from the shape. @p strides can be considered as the actual
    /// shape of the underlying data buffer (@ref TensorBuf).
    /// @param offsets Offsets of the tensor. The data of interest starts at
    /// @p offsets and ends at @p offsets + @p shape.
    /// @param pads If a dimension of @p pads is set to larger than 1, the
    /// corresponding ldim will be set to the minimum multiple of @p pads that
    /// is larger than or equal to the previous ldim. Padding is accumulated
    /// across all tensors that share the same @ref TensorBuf. For example, if
    /// one tensor sets the last dimension of @p pads to 2, and another tensor
    /// sets the last dimension of @p pads to 3, then the corresponding ldim
    /// will be the minimum multiple of 2x3=6 that is larger than or equal to
    /// the corresponding dimension of @p offsets + @p shape.
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
    Tensor tensor(const Dims &shape, ModelDataType data_type,
                  const Dims &strides = {}, const Dims &offsets = {},
                  const Dims &pads = {}, bool exported = false,
                  int imported_rank = -1, const std::string &name = "");

    Tensor refer(Tensor input, const Dims &shape = {}, const Dims &strides = {},
                 const Dims &offsets = {}, const Dims &pads = {},
                 const std::string &name = "");

    Tensor reshape(Tensor input, const Dims &shape, bool allowzero = false,
                   const std::string &name = "");
    Tensor reshape(Tensor input, const std::initializer_list<DimType> &shape,
                   bool allowzero = false, const std::string &name = "");
    // Reshape `input` to `shape`. If one dimension of `shape` is -1, it will be
    // inferred from the `input`. If one dimension of `shape` is 0, by default
    // (`allowzero` is false), that dimension is unchanged from the
    // corresponding one of `input`. If `allowzero` is true, that dimension is
    // set to 0, which means that the reshaped tensor is an empty tensor, i.e.,
    // `input` should also be an empty tensor. If `allowzero` is true, `shape`
    // should not include both 0 and -1 at the same time. If `shape` is an empty
    // vector, `input` will be converted to a scalar.
    Tensor reshape(Tensor input, const std::vector<DimType> &shape,
                   bool allowzero = false, const std::string &name = "");
    // Returns an identical tensor of `input` with execution dependencies
    // `deps`.
    Tensor identity(Tensor input, const std::vector<Tensor> &deps = {},
                    const std::string &name = "");

    // Shard `input` along `axis` into `dim_per_shard`-dimensional shards.
    std::vector<Tensor> sharding(Tensor input, DimType axis,
                                 DimType dim_per_shard,
                                 const std::string &name = "");
    // Performs reduction along the `axis` of the `input` tensor and stores the
    // result in `output`.
    // Currently, only reduction along the last dimension is supported.
    Tensor reduce_sum(Tensor input, int axis, bool keepdims = true,
                      Tensor output = NoneTensor, const std::string &name = "");
    Tensor reduce_mean(Tensor input, int axis, bool keepdims = true,
                       Tensor output = NoneTensor,
                       const std::string &name = "");
    Tensor reduce_max(Tensor input, int axis, bool keepdims = true,
                      Tensor output = NoneTensor, const std::string &name = "");
    // Applies layer normalization to the `input` tensor and returns the
    // normalized tensor as `output`.
    Tensor layernorm(Tensor input, Tensor output = NoneTensor,
                     const std::string &name = "");
    // Transposes the `input` tensor according to the given `permutation`.
    // For example, transpose(input, {0, 1 ,3, 2}) will swap the last two
    // dimensions of the input tensor. Currently, only 4D tensors are supported.
    Tensor transpose(Tensor input, const std::vector<int64_t> &permutation,
                     Tensor output = NoneTensor, const std::string &name = "");
    // Performs matrix multiplication between the `input` tensor and another
    // `other` tensor, storing the result in `output`.
    Tensor matmul(Tensor input, Tensor other, Tensor output = NoneTensor,
                  bool trans_input = false, bool trans_other = false,
                  const std::string &name = "");
    // Implements the 'im2col' method for 2D convolution layers, which takes an
    // `input` tensor and reshapes it to a 2D matrix by extracting image patches
    // from the input tensor based on the provided parameters.
    Tensor im2col(Tensor input, int kernel_height, int kernel_width,
                  int stride_height, int stride_width, int pad_height,
                  int pad_width, int dilation_height, int dilation_width,
                  Tensor output = NoneTensor, const std::string &name = "");
    // Applies max-pooling on the `input` tensor using `kernel_size` and
    // `stride`, reducing its spatial size. The output shape is calculated based
    // on the input tensor's shape and the stride value as follows: {is[0],
    // (is[1] + stride - 1) / stride, (is[2] + stride - 1) / stride, is[3]},
    // where 'is' represents the input tensor's shape.
    Tensor max_pool(Tensor input, DimType kernel_size, DimType stride,
                    Tensor output = NoneTensor, const std::string &name = "");
    // Multiplies the `input` tensor by a scalar `val`, element-wise.
    Tensor scale(Tensor input, float val, Tensor output = NoneTensor,
                 const std::string &name = "");
    // Calculates the exponential of the `input` tensor, element-wise.
    Tensor exp(Tensor input, Tensor output = NoneTensor,
               const std::string &name = "");
    // Calculates the square root of the `input` tensor, element-wise.
    Tensor sqrt(Tensor input, Tensor output = NoneTensor,
                const std::string &name = "");
    // Calculates the reverse square root of the `input` tensor, element-wise.
    Tensor rsqrt(Tensor input, Tensor output = NoneTensor,
                 const std::string &name = "");
    // ReLU activation
    Tensor relu(Tensor input, Tensor output = NoneTensor,
                const std::string &name = "");
    // Copy the `input` tensor to `output` tensor
    Tensor copy(Tensor input, Tensor output = NoneTensor,
                const std::string &name = "");
    // Applies the Gaussian Error Linear Unit (GELU) activation function to the
    // `input` tensor, element-wise. GELU is a smooth approximation of the
    // rectifier function and is widely used in deep learning models.
    Tensor gelu(Tensor input, Tensor output = NoneTensor,
                const std::string &name = "");
    // Sigmoid activation
    Tensor sigmoid(Tensor input, Tensor output = NoneTensor,
                   const std::string &name = "");
    // Performs rotary position embedding (RoPE) on the `input` tensor
    Tensor rope(Tensor input, Tensor other, Tensor output = NoneTensor,
                const std::string &name = "");

    // Performs an element-wise addition operator between the `input` tensor
    // and the `other` tensor
    Tensor add(Tensor input, Tensor other, Tensor output = NoneTensor,
               const std::string &name = "");
    // Performs an element-wise subtraction operator between the `input` tensor
    // and the `other` tensor
    Tensor sub(Tensor input, Tensor other, Tensor output = NoneTensor,
               const std::string &name = "");
    // Performs an element-wise multiplication operator between the `input`
    // tensor and the `other` tensor,
    Tensor mul(Tensor input, Tensor other, Tensor output = NoneTensor,
               const std::string &name = "");
    // Performs an element-wise division operator between the `input`
    // tensor and the `other` tensor,
    Tensor div(Tensor input, Tensor other, Tensor output = NoneTensor,
               const std::string &name = "");
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
    Tensor send(Tensor input, int sid, int dst_rank, DimType bytes = 0,
                const std::string &name = "");
    // Blocks the execution until the corresponding 'send' operator with the
    // specified `id` is completed.
    Tensor send_done(Tensor input, int sid, int dst_rank,
                     const std::string &name = "");
    // Receives a tensor from a source rank (@p src_rank), identified by the
    // `id` parameter. Blocks the execution until the corresponding 'recv'
    // operator is completed.
    Tensor recv(int sid, int src_rank, DimType bytes = 0,
                Tensor output = NoneTensor, const std::string &name = "");
    //
    Tensor put_packet(Tensor input, Tensor local_tmp_buf, Tensor recv_buf,
                      int id, int rank, int dst_rank, size_t dst_offset,
                      int flag, const std::string &name = "");
    // Performs an all-reduce operator across all ranks, aggregating the input
    // tensors. Takes the `input` tensor, the current GPU's rank, and the
    // total number of ranks `rank_num`.
    Tensor all_reduce(Tensor input, int rank, int rank_num,
                      Tensor output = NoneTensor, const std::string &name = "");
    // Performs an all-gather operator across all ranks, aggregating the input
    // tensors. Takes the `input` tensor, the current GPU's rank, and the
    // total number of ranks `rank_num`. Returns a vector of tensors, each
    // containing the aggregated data from all ranks.
    std::vector<Tensor> all_gather(Tensor input, int rank, int rank_num,
                                   const std::vector<Tensor> &output = {},
                                   const std::string &name = "");
    /// Embedding layer.
    Tensor embedding(Tensor input, Tensor weight, Tensor output = NoneTensor,
                     const std::string &name = "");
    /// Tensor type casting.
    Tensor cast(Tensor input, ModelDataType data_type,
                Tensor output = NoneTensor, const std::string &name = "");

    // sync across multi devices
    Tensor device_sync(Tensor input, int npeers, const std::string &name = "");

    // local reduce scatter
    Tensor local_reduce_scatter(Tensor input, int gpu_id, int ngpus_per_node,
                                const std::string &name = "");

    // local all gather
    Tensor local_all_gather(Tensor input, int gpu_id, int ngpus_per_node,
                            int axis = 0, const std::string &name = "");
    // read data from remote and reduce to current buffer
    Tensor read_and_reduce(Tensor input, int sid, int npeers, size_t offset,
                           size_t bytes, const std::string &name = "");
    // gather from peers
    Tensor gather_from_peers(Tensor input, Tensor tile, int sid, int npeers,
                             size_t chunkBytes, const std::string &name = "");

    Tensor local_all_reduce(Tensor input, int gpu_id, int gpu_num,
                            const std::string &name = "");
    Tensor local_all_reduce_packet(Tensor input, int gpu_id, int gpu_num,
                                   const std::string &name = "");

    Tensor reduce_and_write_packet(Tensor input, Tensor scratch, Tensor output,
                                   const std::vector<Tensor> &remote_peer_bufs,
                                   int id, int rank, int npeers,
                                   size_t elems_per_rank, size_t scratch_offset,
                                   size_t remote_dst_offset, int flag,
                                   const std::string &name = "");
    Tensor get_packet(Tensor input, Tensor output, size_t src_offset,
                      size_t dst_offset, size_t npackets, int flag,
                      const std::string &name = "");
};

}  // namespace ark

#endif  // ARK_MODEL_HPP
