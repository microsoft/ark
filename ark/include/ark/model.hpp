// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_HPP
#define ARK_MODEL_HPP

#include <string>
#include <vector>

#include "dims.hpp"
#include "model_graph.hpp"
#include "model_ref.hpp"

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

    void noop(ModelTensorRef input, const std::string &name = "");

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
    ModelTensorRef tensor(const Dims &shape, ModelDataType data_type,
                          const Dims &strides = {}, const Dims &offsets = {},
                          const Dims &pads = {}, bool exported = false,
                          int imported_rank = -1, const std::string &name = "");

    ModelTensorRef refer(ModelTensorRef input, const Dims &shape = {},
                         const Dims &strides = {}, const Dims &offsets = {},
                         const Dims &pads = {}, const std::string &name = "");

    ModelTensorRef reshape(ModelTensorRef input, const Dims &shape,
                           bool allowzero = false,
                           ModelTensorRef output = nullptr,
                           const std::string &name = "");
    ModelTensorRef reshape(ModelTensorRef input,
                           const std::initializer_list<DimType> &shape,
                           bool allowzero = false,
                           ModelTensorRef output = nullptr,
                           const std::string &name = "");
    // Reshape `input` to `shape`. If one dimension of `shape` is -1, it will be
    // inferred from the `input`. If one dimension of `shape` is 0, by default
    // (`allowzero` is false), that dimension is unchanged from the
    // corresponding one of `input`. If `allowzero` is true, that dimension is
    // set to 0, which means that the reshaped tensor is an empty tensor, i.e.,
    // `input` should also be an empty tensor. If `allowzero` is true, `shape`
    // should not include both 0 and -1 at the same time. If `shape` is an empty
    // vector, `input` will be converted to a scalar.
    ModelTensorRef reshape(ModelTensorRef input,
                           const std::vector<DimType> &shape,
                           bool allowzero = false,
                           ModelTensorRef output = nullptr,
                           const std::string &name = "");
    // Returns an identical tensor of `input` with execution dependencies
    // `deps`.
    ModelTensorRef identity(ModelTensorRef input,
                            const std::vector<ModelTensorRef> &deps = {},
                            const std::string &name = "");

    // Shard `input` along `axis` into `dim_per_shard`-dimensional shards.
    std::vector<ModelTensorRef> sharding(ModelTensorRef input, DimType axis,
                                         DimType dim_per_shard,
                                         const std::string &name = "");
    // Performs reduction along the `axis` of the `input` tensor and stores the
    // result in `output`.
    // Currently, only reduction along the last dimension is supported.
    template <typename ReduceOpType>
    ModelTensorRef reduce(ModelTensorRef input, int axis, bool keepdims = true,
                          ModelTensorRef output = nullptr,
                          const std::string &name = "");
    ModelTensorRef reduce_sum(ModelTensorRef input, int axis,
                              bool keepdims = true,
                              ModelTensorRef output = nullptr,
                              const std::string &name = "");
    ModelTensorRef reduce_mean(ModelTensorRef input, int axis,
                               bool keepdims = true,
                               ModelTensorRef output = nullptr,
                               const std::string &name = "");
    ModelTensorRef reduce_max(ModelTensorRef input, int axis,
                              bool keepdims = true,
                              ModelTensorRef output = nullptr,
                              const std::string &name = "");
    // Applies layer normalization to the `input` tensor and returns the
    // normalized tensor as `output`.
    ModelTensorRef layernorm(ModelTensorRef input,
                             ModelTensorRef output = nullptr,
                             const std::string &name = "");
    // Transposes the `input` tensor according to the given `perm` permutation.
    // For example, transpose(input, {0, 1 ,3, 2}) will swap the last two
    // dimensions of the input tensor. Currently, only 4D tensors are supported.
    ModelTensorRef transpose(ModelTensorRef input, Dims perm,
                             ModelTensorRef output = nullptr,
                             const std::string &name = "");
    // Performs matrix multiplication between the `input` tensor and another
    // `other` tensor, storing the result in `output`.
    ModelTensorRef matmul(ModelTensorRef input, ModelTensorRef other,
                          ModelTensorRef output = nullptr,
                          bool trans_input = false, bool trans_other = false,
                          const std::string &name = "");
    // Implements the 'im2col' method for 2D convolution layers, which takes an
    // `input` tensor and reshapes it to a 2D matrix by extracting image patches
    // from the input tensor based on the provided parameters.
    ModelTensorRef im2col(ModelTensorRef input, int kernel_height,
                          int kernel_width, int stride_height, int stride_width,
                          int pad_height, int pad_width, int dilation_height,
                          int dilation_width, ModelTensorRef output = nullptr,
                          const std::string &name = "");
    // Applies max-pooling on the `input` tensor using `kernel_size` and
    // `stride`, reducing its spatial size. The output shape is calculated based
    // on the input tensor's shape and the stride value as follows: {is[0],
    // (is[1] + stride - 1) / stride, (is[2] + stride - 1) / stride, is[3]},
    // where 'is' represents the input tensor's shape.
    ModelTensorRef max_pool(ModelTensorRef input, DimType kernel_size,
                            DimType stride, ModelTensorRef output = nullptr,
                            const std::string &name = "");
    // Multiplies the `input` tensor by a scalar `val`, element-wise.
    ModelTensorRef scale(ModelTensorRef input, float val,
                         ModelTensorRef output = nullptr,
                         const std::string &name = "");
    //
    template <typename MathOpType>
    ModelTensorRef math(ModelTensorRef input, ModelTensorRef output = nullptr,
                        const std::string &name = "");
    // Calculates the exponential of the `input` tensor, element-wise.
    ModelTensorRef exp(ModelTensorRef input, ModelTensorRef output = nullptr,
                       const std::string &name = "");
    // Calculates the square root of the `input` tensor, element-wise.
    ModelTensorRef sqrt(ModelTensorRef input, ModelTensorRef output = nullptr,
                        const std::string &name = "");
    // Calculates the reverse square root of the `input` tensor, element-wise.
    ModelTensorRef rsqrt(ModelTensorRef input, ModelTensorRef output = nullptr,
                         const std::string &name = "");
    // ReLU activation
    ModelTensorRef relu(ModelTensorRef input, ModelTensorRef output = nullptr,
                        const std::string &name = "");
    // Copy the `input` tensor to `output` tensor
    ModelTensorRef copy(ModelTensorRef input, ModelTensorRef output = nullptr,
                        const std::string &name = "");
    // Applies the Gaussian Error Linear Unit (GELU) activation function to the
    // `input` tensor, element-wise. GELU is a smooth approximation of the
    // rectifier function and is widely used in deep learning models.
    ModelTensorRef gelu(ModelTensorRef input, ModelTensorRef output = nullptr,
                        const std::string &name = "");
    // Sigmoid activation
    ModelTensorRef sigmoid(ModelTensorRef input,
                           ModelTensorRef output = nullptr,
                           const std::string &name = "");
    // Performs rotary position embedding (RoPE) on the `input` tensor
    ModelTensorRef rope(ModelTensorRef input, ModelTensorRef other,
                        ModelTensorRef output = nullptr,
                        const std::string &name = "");

    // Performs an element-wise addition operator between the `input` tensor
    // and the `other` tensor
    ModelTensorRef add(ModelTensorRef input, ModelTensorRef other,
                       ModelTensorRef output = nullptr,
                       const std::string &name = "");
    // Performs an element-wise subtraction operator between the `input` tensor
    // and the `other` tensor
    ModelTensorRef sub(ModelTensorRef input, ModelTensorRef other,
                       ModelTensorRef output = nullptr,
                       const std::string &name = "");
    // Performs an element-wise multiplication operator between the `input`
    // tensor and the `other` tensor,
    ModelTensorRef mul(ModelTensorRef input, ModelTensorRef other,
                       ModelTensorRef output = nullptr,
                       const std::string &name = "");
    // Performs an element-wise division operator between the `input`
    // tensor and the `other` tensor,
    ModelTensorRef div(ModelTensorRef input, ModelTensorRef other,
                       ModelTensorRef output = nullptr,
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
    ModelTensorRef send(ModelTensorRef input, int sid, int dst_rank,
                        DimType bytes = 0, const std::string &name = "");
    // Blocks the execution until the corresponding 'send' operator with the
    // specified `id` is completed.
    ModelTensorRef send_done(ModelTensorRef input, int sid, int dst_rank,
                             const std::string &name = "");
    // Receives a tensor from a source rank (@p src_rank), identified by the
    // `id` parameter. Blocks the execution until the corresponding 'recv'
    // operator is completed.
    ModelTensorRef recv(int sid, int src_rank, DimType bytes = 0,
                        ModelTensorRef output = nullptr,
                        const std::string &name = "");
    //
    ModelTensorRef put_packet(ModelTensorRef input,
                              ModelTensorRef local_tmp_buf,
                              ModelTensorRef recv_buf, int id, int rank,
                              int dst_rank, size_t dst_offset, int flag,
                              const std::string &name = "");
    // Performs an all-reduce operator across all ranks, aggregating the input
    // tensors. Takes the `input` tensor, the current GPU's rank, and the
    // total number of ranks `rank_num`.
    ModelTensorRef all_reduce(ModelTensorRef input, int rank, int rank_num,
                              ModelTensorRef output = nullptr,
                              const std::string &name = "");
    // Performs an all-gather operator across all ranks, aggregating the input
    // tensors. Takes the `input` tensor, the current GPU's rank, and the
    // total number of ranks `rank_num`. Returns a vector of tensors, each
    // containing the aggregated data from all ranks.
    std::vector<ModelTensorRef> all_gather(
        ModelTensorRef input, int rank, int rank_num,
        const std::vector<ModelTensorRef> &output = {},
        const std::string &name = "");
    /// Embedding layer.
    ModelTensorRef embedding(ModelTensorRef input, ModelTensorRef weight,
                             ModelTensorRef output = nullptr,
                             const std::string &name = "");
    /// Tensor type casting.
    ModelTensorRef cast(ModelTensorRef input, ModelDataType data_type,
                        ModelTensorRef output = nullptr,
                        const std::string &name = "");

    // sync across multi devices
    ModelTensorRef device_sync(ModelTensorRef input, int npeers,
                               const std::string &name = "");

    // local reduce scatter
    ModelTensorRef local_reduce_scatter(ModelTensorRef input, int gpu_id,
                                        int ngpus_per_node,
                                        const std::string &name = "");

    // local all gather
    ModelTensorRef local_all_gather(ModelTensorRef input, int gpu_id,
                                    int ngpus_per_node, int axis = 0,
                                    const std::string &name = "");
    // read data from remote and reduce to current buffer
    ModelTensorRef read_and_reduce(ModelTensorRef input, int sid, int npeers,
                                   size_t offset, size_t bytes,
                                   const std::string &name = "");
    // gather from peers
    ModelTensorRef gather_from_peers(ModelTensorRef input, ModelTensorRef tile,
                                     int sid, int npeers, size_t chunkBytes,
                                     const std::string &name = "");

    ModelTensorRef local_all_reduce(ModelTensorRef input, int gpu_id,
                                    int gpu_num, const std::string &name = "");
    ModelTensorRef local_all_reduce_packet(ModelTensorRef input, int gpu_id,
                                           int gpu_num,
                                           const std::string &name = "");

    ModelTensorRef reduce_and_write_packet(
        ModelTensorRef input, ModelTensorRef scratch, ModelTensorRef output,
        const std::vector<ModelTensorRef> &remote_peer_bufs, int id, int rank,
        int npeers, size_t elems_per_rank, size_t scratch_offset,
        size_t remote_dst_offset, int flag, const std::string &name = "");
    ModelTensorRef get_packet(ModelTensorRef input, ModelTensorRef output,
                              size_t src_offset, size_t dst_offset,
                              size_t npackets, int flag,
                              const std::string &name = "");
};

}  // namespace ark

#endif  // ARK_MODEL_HPP
