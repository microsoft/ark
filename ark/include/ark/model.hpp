// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_HPP
#define ARK_MODEL_HPP

#include <ark/data_type.hpp>
#include <ark/dims.hpp>
#include <ark/model_graph.hpp>
#include <ark/model_ref.hpp>
#include <ark/tensor.hpp>
#include <set>
#include <string>
#include <vector>

namespace ark {

class Model : public ModelGraph {
   private:
    size_t id_;
    std::set<int> tags_;

   public:
    Model(int rank = 0, int world_size = 1);

    Model(const Model &other);

    ~Model() {}

    Model &operator=(const Model &other) = default;

    /// Get the unique identifier of the model.
    size_t id() const;

    Model compress() const;

    int unique_tag();

    Tensor constant(float val, const Dims &shape, DataType data_type,
                    const std::string &name = "");

    /// No operation.
    ///
    /// This operator can be used to prevent unused tensors from being optimized
    /// out by the compiler.
    ///
    /// @param input Input tensor.
    /// @param name Name of the operator.
    ///
    void noop(Tensor input, const std::string &name = "");

    /// Returns a tensor object.
    ///
    /// @param shape Shape of the tensor, where the data of interest is.
    /// @param dtype Type of the tensor data.
    /// @param strides Strides of each dimensions of the tensor, which may be
    /// different from the shape. @p strides can be considered as the actual
    /// shape of the underlying data buffer (@ref ModelBuffer).
    /// @param offsets Offsets of the tensor. The data of interest starts at
    /// @p offsets and ends at @p offsets + @p padded_shape.
    /// @param padded_shape Padded shape of the tensor. Padding is used to
    /// reserve extra space for the tensor when computation requires it.
    /// Data on the padded region is allowed to be accessed by computation,
    /// but it is not considered as the data of interest. The padded region is
    /// initialized to zero only once when the Executor is launched. The padded
    /// shape should be greater than or equal to the @p shape, and the
    /// @p strides should be greater than or equal to the padded shape. If the
    /// @p strides are not provided, they are set to the padded shape. If the
    /// padded shape is not provided, it is set to the @p shape.
    /// @param name Name of the tensor.
    /// @return Pointer to a tensor object.
    ///
    Tensor tensor(const Dims &shape, const DataType &data_type,
                  const Dims &strides = {}, const Dims &offsets = {},
                  const Dims &padded_shape = {}, const std::string &name = "");
    Tensor tensor(std::shared_ptr<ModelBuffer> buffer, const Dims &shape,
                  const DataType &data_type, const Dims &strides = {},
                  const Dims &offsets = {}, const Dims &padded_shape = {},
                  const std::string &name = "");

    Tensor refer(Tensor input, const Dims &shape = {}, const Dims &strides = {},
                 const Dims &offsets = {}, const Dims &padded_shape = {},
                 const std::string &name = "");

    // Reshape `input` to `shape`. If one dimension of `shape` is -1, it will be
    // inferred from the `input`. If one dimension of `shape` is 0, by default
    // (`allowzero` is false), that dimension is unchanged from the
    // corresponding one of `input`. If `allowzero` is true, that dimension is
    // set to 0, which means that the reshaped tensor is an empty tensor, i.e.,
    // `input` should also be an empty tensor. If `allowzero` is true, `shape`
    // should not include both 0 and -1 at the same time. If `shape` is an empty
    // vector, `input` will be converted to a scalar.
    Tensor reshape(Tensor input, const Dims &shape, bool allowzero = false,
                   const std::string &name = "");
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
                      Tensor output = NullTensor, const std::string &name = "");
    Tensor reduce_mean(Tensor input, int axis, bool keepdims = true,
                       Tensor output = NullTensor,
                       const std::string &name = "");
    Tensor reduce_max(Tensor input, int axis, bool keepdims = true,
                      Tensor output = NullTensor, const std::string &name = "");

    // Transposes the `input` tensor according to the given `permutation`.
    // For example, transpose(input, {0, 1 ,3, 2}) will swap the last two
    // dimensions of the input tensor. Currently, only 4D tensors are supported.
    Tensor transpose(Tensor input, const std::vector<int64_t> &permutation,
                     Tensor output = NullTensor, const std::string &name = "");
    // Performs matrix multiplication between the `input` tensor and another
    // `other` tensor, storing the result in `output`.
    Tensor matmul(Tensor input, Tensor other, Tensor output = NullTensor,
                  bool trans_input = false, bool trans_other = false,
                  const std::string &name = "");
    // Implements the 'im2col' method for 2D convolution layers, which takes an
    // `input` tensor and reshapes it to a 2D matrix by extracting image patches
    // from the input tensor based on the provided parameters.
    Tensor im2col(Tensor input, int kernel_height, int kernel_width,
                  int stride_height, int stride_width, int pad_height,
                  int pad_width, int dilation_height, int dilation_width,
                  Tensor output = NullTensor, const std::string &name = "");
    // Applies max-pooling on the `input` tensor using `kernel_size` and
    // `stride`, reducing its spatial size. The output shape is calculated based
    // on the input tensor's shape and the stride value as follows: {is[0],
    // (is[1] + stride - 1) / stride, (is[2] + stride - 1) / stride, is[3]},
    // where 'is' represents the input tensor's shape.
    Tensor max_pool(Tensor input, DimType kernel_size, DimType stride,
                    Tensor output = NullTensor, const std::string &name = "");
    // Calculates the exponential of the `input` tensor, element-wise.
    Tensor exp(Tensor input, Tensor output = NullTensor,
               const std::string &name = "");
    // Calculates the square root of the `input` tensor, element-wise.
    Tensor sqrt(Tensor input, Tensor output = NullTensor,
                const std::string &name = "");
    // Calculates the reverse square root of the `input` tensor, element-wise.
    Tensor rsqrt(Tensor input, Tensor output = NullTensor,
                 const std::string &name = "");
    // ReLU activation
    Tensor relu(Tensor input, Tensor output = NullTensor,
                const std::string &name = "");
    // Copy the `input` tensor to `output` tensor
    Tensor copy(Tensor input, Tensor output = NullTensor,
                const std::string &name = "");
    Tensor copy(float val, Tensor output = NullTensor,
                const std::string &name = "");
    // Applies the Gaussian Error Linear Unit (GELU) activation function to the
    // `input` tensor, element-wise. GELU is a smooth approximation of the
    // rectifier function and is widely used in deep learning models.
    Tensor gelu(Tensor input, Tensor output = NullTensor,
                const std::string &name = "");
    // Sigmoid activation
    Tensor sigmoid(Tensor input, Tensor output = NullTensor,
                   const std::string &name = "");
    // Performs rotary position embedding (RoPE) on the `input` tensor
    Tensor rope(Tensor input, Tensor other, Tensor output = NullTensor,
                const std::string &name = "");

    // Performs an element-wise addition operator between the `input` tensor
    // and the `other` tensor
    Tensor add(Tensor input, Tensor other, Tensor output = NullTensor,
               const std::string &name = "");
    Tensor add(Tensor input, float value, Tensor output = NullTensor,
               const std::string &name = "");
    // Performs an element-wise subtraction operator between the `input` tensor
    // and the `other` tensor
    Tensor sub(Tensor input, Tensor other, Tensor output = NullTensor,
               const std::string &name = "");
    Tensor sub(Tensor input, float value, Tensor output = NullTensor,
               const std::string &name = "");
    // Performs an element-wise multiplication operator between the `input`
    // tensor and the `other` tensor,
    Tensor mul(Tensor input, Tensor other, Tensor output = NullTensor,
               const std::string &name = "");
    Tensor mul(Tensor input, float value, Tensor output = NullTensor,
               const std::string &name = "");
    // Performs an element-wise division operator between the `input`
    // tensor and the `other` tensor,
    Tensor div(Tensor input, Tensor other, Tensor output = NullTensor,
               const std::string &name = "");
    Tensor div(Tensor input, float value, Tensor output = NullTensor,
               const std::string &name = "");

    Tensor send(Tensor input, int remote_rank, int tag,
                Tensor output = NullTensor, const std::string &name = "");
    // Blocks the execution until the corresponding 'send' operator with the
    // specified `id` is completed.
    Tensor send_done(Tensor input, const std::string &name = "");
    // Receives a tensor from a source rank (@p src_rank), identified by the
    // `id` parameter. Blocks the execution until the corresponding 'recv'
    // operator is completed.
    Tensor recv(Tensor output, int remote_rank, int tag,
                const std::string &name = "");
    Tensor send_packet(Tensor input, int remote_rank, int tag, int flag,
                       Tensor output = NullTensor,
                       const std::string &name = "");
    Tensor recv_packet(Tensor output, int remote_rank, int tag, int flag,
                       Tensor scratch = NullTensor,
                       const std::string &name = "");
    Tensor recv_reduce_send_packet(
        Tensor input, const std::vector<int> &remote_ranks, int recv_tag,
        int output_tag, unsigned int flag, Tensor output = NullTensor,
        std::vector<Tensor> peer_outputs = {}, Tensor scratch = NullTensor,
        const std::string &name = "");
    Tensor recv_reduce_send(Tensor input, const std::vector<int> &remote_ranks,
                            int recv_tag, int output_tag,
                            Tensor output = NullTensor,
                            std::vector<Tensor> peer_outputs = {},
                            Tensor scratch = NullTensor,
                            const std::string &name = "");
    // Performs an all-reduce operator across all ranks, aggregating the input
    // tensors. Takes the `input` tensor, the current GPU's rank, and the
    // total number of ranks `rank_num`.
    Tensor all_reduce(Tensor input, int rank, int rank_num,
                      Tensor output = NullTensor, const std::string &name = "");
    // Performs an all-gather operator across all ranks, aggregating the input
    // tensors. Takes the `input` tensor, the current GPU's rank, and the
    // total number of ranks `rank_num`. Returns a vector of tensors, each
    // containing the aggregated data from all ranks.
    std::vector<Tensor> all_gather(Tensor input, int rank, int rank_num,
                                   const std::vector<Tensor> &output = {},
                                   const std::string &name = "");
    /// Embedding layer.
    Tensor embedding(Tensor input, Tensor weight, Tensor output = NullTensor,
                     const std::string &name = "");
    /// Tensor type casting.
    Tensor cast(Tensor input, const DataType &data_type,
                Tensor output = NullTensor, const std::string &name = "");

    // sync across multi devices
    Tensor device_sync(Tensor input, int rank, int rank_num,
                       const std::string &name = "");

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

};

}  // namespace ark

#endif  // ARK_MODEL_HPP
