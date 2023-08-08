# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ._ark_core import _Model, TensorBuf, TensorType, Dims
from .tensor import Tensor
import logging
from typing import List


def tensor(
    shape,
    dtype: TensorType = TensorType.FP32,
    buf: TensorBuf = None,
    ldims: Dims = Dims(),
    offs: Dims = Dims(),
    pads: Dims = Dims(),
    deps: list = [],
    exported: bool = False,
    imported_rank: int = -1,
    name: str = "tensor",
) -> Tensor:
    """
    Construct a tensor with given shape and data type.
    Usage:
    tensor = ark.tensor([1, 2, 3, 4], dtype=TensorType.FP32)
    tensor = ark.tensor(ark.Dims(1, 2), dtype=TensorType.FP16)
    """
    # if shape is a list of integers, convert it to a Dims object
    if isinstance(shape, list):
        # only support tensors with up to 4 dimensions
        if len(shape) > 4:
            logging.error("Only support tensors with up to 4 dimensions")
            raise ValueError("Only support tensors with up to 4 dimensions")
        shape = Dims(*shape)
    _tensor = Model.get_global_model().tensor(
        shape,
        dtype,
        buf,
        ldims,
        offs,
        pads,
        deps,
        exported,
        imported_rank,
        name,
    )
    return Tensor(_tensor)


def reshape(
    input: Tensor,
    shape: list,
    allowzero: bool = False,
    output: Tensor = None,
    name: str = "reshape",
) -> Tensor:
    """
    Reshape `input` to `shape`. If one dimension of `shape` is -1, it will
    be inferred from the `input`. If one dimension of `shape` is 0,
    by default (`allowzero` is false), that dimension is unchanged from
    the corresponding one of `input`. If `allowzero` is true, that dimension
    is set to 0, which means that the reshaped tensor is an empty tensor,
    i.e., `input` should also be an empty tensor. If `allowzero` is true,
    `shape` should not include both 0 and -1 at the same time. If `shape`
    is an empty vector, `input` will be converted to a scalar.
    Usage:
    # tensors shape is [128, 64]
    tensor = ark.reshape(tensor, [2, 64, 64])
    """
    if isinstance(shape, list):
        # only support tensors with up to 4 dimensions
        if len(shape) > 4:
            logging.error("Only support tensors with up to 4 dimensions")
            raise ValueError("Only support tensors with up to 4 dimensions")
        shape = Dims(*shape)
    if output is not None:
        output = output._tensor
    input = input._tensor
    _tensor = Model.get_global_model().reshape(
        input, shape, allowzero, output, name
    )
    return Tensor(_tensor)


def identity(
    input: Tensor,
    deps: List[Tensor] = [],
    output: Tensor = None,
    name: str = "identity",
) -> Tensor:
    """
    Returns an identical tensor of `input` with execution dependencies `deps`.
    Usage:
    tensor_identity = ark.identity(tensor, deps=[tensor1, tensor2])
    """
    dep_tensor = []
    for dep in deps:
        if not isinstance(dep, Tensor):
            logging.error("All dependencies should be tensors")
            raise TypeError("All dependencies should be tensors")
        dep_tensor.append(dep._tensor)
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().identity(
        input._tensor, dep_tensor, output, name
    )
    return Tensor(_tensor)


def sharding(
    input: Tensor,
    axis: int,
    dim_per_shard: int,
    name: str = "sharding",
) -> List[Tensor]:
    """
    Shard `input` along `axis` into `dim_per_shard`-dimensional shards.
    Usage:
    # tensors shape is [64, 128]
    tensor_sharding = ark.sharding(tensor, axis=1, dim_per_shard=64)
    # tensor_sharding is a list of 2 tensors, each of which has shape [64, 64]
    # The first tensor's buffer is the same as the first 64 columns of tensor
    # The second tensor's buffer is the same as the last 64 columns of tensor
    """
    _tensor_list = Model.get_global_model().sharding(
        input._tensor, axis, dim_per_shard, name
    )
    tensor_list = []
    for _tensor in _tensor_list:
        tensor_list.append(Tensor(_tensor))
    return tensor_list


def reduce_sum(
    input: Tensor,
    axis: int,
    output: Tensor = None,
    name: str = "reduce_sum",
) -> Tensor:
    """
    Performs reduction along the `axis` of the `input` tensor and
    stores the result in `output`.
    Usage:
    # tensors shape is [64, 128]
    tensor_reduce_sum = ark.reduce_sum(tensor, axis=1)
    # tensor_reduce_sum is a tensor with shape [64, 1]
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().reduce_sum(
        input._tensor, axis, output, name
    )
    return Tensor(_tensor)


def reduce_mean(
    input: Tensor,
    axis: int,
    output: Tensor = None,
    name: str = "reduce_mean",
) -> Tensor:
    """
    Performs reduction along the `axis` of the `input` tensor and
    stores the result in `output`.
    Usage:
    tensor_reduce_mean = ark.reduce_mean(tensor, axis=1)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().reduce_mean(
        input._tensor, axis, output, name
    )
    return Tensor(_tensor)


def reduce_max(
    input: Tensor,
    axis: int,
    output: Tensor = None,
    name: str = "reduce_max",
) -> Tensor:
    """
    Performs reduction along the `axis` of the `input` tensor and
    stores the result in `output`.
    Usage:
    tensor_reduce_max = ark.reduce_max(tensor, axis=1)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().reduce_max(
        input._tensor, axis, output, name
    )
    return Tensor(_tensor)


def layernorm(
    input: Tensor,
    output: Tensor = None,
    name: str = "layernorm",
) -> Tensor:
    """
    Applies layer normalization to the `input` tensor and returns
    the normalized tensor as `output`.
    Usage:
    tensor_layernorm = ark.layernorm(tensor)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().layernorm(input._tensor, output, name)
    return Tensor(_tensor)


def softmax(
    input: Tensor,
    output: Tensor = None,
    name: str = "softmax",
) -> Tensor:
    """
    Applies softmax  to the `input` tensor on the last dimension.
    Usage:
    tensor_softmax = ark.softmax(tensor)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().softmax(input._tensor, output, name)
    return Tensor(_tensor)


def transpose(
    input: Tensor,
    perm: list,
    output: Tensor = None,
    name: str = "transpose",
) -> Tensor:
    """
    Transposes the `input` tensor according to the given `perm` permutation.
    For example, transpose(input, [0, 1 ,3, 2]) will swap the last two
    dimensions of the input tensor. Currently, only 4D tensors are supported.
    Usage:
    # tensors shape is [1, 64, 128, 32]
    tensor_transpose = ark.transpose(tensor, perm=[0, 1, 3, 2])
    # tensor_transpose is a tensor with shape [1, 64, 32, 128]
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().transpose(
        input._tensor, perm, output, name
    )
    return Tensor(_tensor)


def matmul(
    input: Tensor,
    other: Tensor,
    output: Tensor = None,
    splitk: int = 1,
    transpose_a: bool = False,
    transpose_b: bool = False,
    is_relu: bool = False,
    name: str = "matmul",
    gran_lev: int = -1,
) -> Tensor:
    """
    Performs matrix multiplication between the `input` tensor and
    `other` tensor, storing the result in `output`. Optional
    parameters allow controlling the behavior of the multiplication,
    such as transposing the input tensors and applying a ReLU
    activation.
    Usage:
    tensor_matmul = ark.matmul(tensor1, tensor2)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().matmul(
        input._tensor,
        other._tensor,
        output,
        splitk,
        transpose_a,
        transpose_b,
        is_relu,
        name,
        gran_lev,
    )
    return Tensor(_tensor)


def im2col(
    input: Tensor,
    kernel_height: int,
    kernel_width: int,
    stride_height: int,
    stride_width: int,
    pad_height: int,
    pad_width: int,
    dilation_height: int,
    dilation_width: int,
    output: Tensor = None,
    name: str = "im2col",
) -> Tensor:
    """
    Implements the 'im2col' method for 2D convolution layers, which
    takes an `input` tensor and reshapes it to a 2D matrix by
    extracting image patches from the input tensor based on the
    provided parameters.
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().im2col(
        input._tensor,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        pad_height,
        pad_width,
        dilation_height,
        dilation_width,
        output,
        name,
    )
    return Tensor(_tensor)


def conv2d(
    input: Tensor,
    in_channels: int,
    out_channels: int,
    kernel_size: list,
    stride: list,
    padding: list,
    bias: bool = False,
    output: Tensor = None,
    name: str = "conv2d",
) -> Tensor:
    """
    Implements a 2D convolution layer using the 'im2col' method.
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().conv2d(
        input._tensor,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        output,
        name,
    )
    return Tensor(_tensor)


def max_pool(
    input: Tensor,
    kernel_size: int,
    stride: int,
    output: Tensor = None,
    name: str = "max_pool",
) -> Tensor:
    """
    Applies max-pooling on the `input` tensor using `kernel_size`
    and `stride`, reducing its spatial size. The output shape is
    calculated based on the input tensor's shape and the stride
    value as follows: {is[0], (is[1] + stride - 1) / stride, (is[2]
    + stride - 1) / stride, is[3]}, where 'is' represents the input
    tensor's shape.
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().max_pool(
        input._tensor,
        kernel_size,
        stride,
        output,
        name,
    )
    return Tensor(_tensor)


def scale(
    input: Tensor,
    val: float,
    output: Tensor = None,
    name: str = "scale",
) -> Tensor:
    """
    Multiplies the `input` tensor by a scalar `val`, element-wise.
    Usage:
    tensor_scale = ark.scale(tensor, 1.6)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().scale(
        input._tensor,
        val,
        output,
        name,
    )
    return Tensor(_tensor)


def sqrt(
    input: Tensor,
    output: Tensor = None,
    name: str = "sqrt",
) -> Tensor:
    """
    Calculates the square root of the `input` tensor, element-wise.
    Usage:
    tensor_sqrt = ark.sqrt(tensor)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().sqrt(input._tensor, output, name)
    return Tensor(_tensor)


def relu(
    input: Tensor,
    output: Tensor = None,
    name: str = "relu",
) -> Tensor:
    """
    Applies the ReLU activation function to the `input` tensor,
    element-wise.
    Usage:
    tensor_relu = ark.relu(tensor)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().relu(input._tensor, output, name)
    return Tensor(_tensor)


def gelu(
    input: Tensor,
    output: Tensor = None,
    name: str = "gelu",
) -> Tensor:
    """
    Applies the Gaussian Error Linear Unit (GELU) activation
    function to the `input` tensor, element-wise. GELU is a smooth
    approximation of the rectifier function and is widely used in
    deep learning models.
    Usage:
    tensor_gelu = ark.gelu(tensor)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().gelu(input._tensor, output, name)
    return Tensor(_tensor)


def sigmoid(
    input: Tensor,
    output: Tensor = None,
    name: str = "sigmoid",
) -> Tensor:
    """
    Applies the Sigmoid activation function to the `input` tensor,
    element-wise.
    Usage:
    tensor_sigmoid = ark.sigmoid(tensor)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().sigmoid(input._tensor, output, name)
    return Tensor(_tensor)


def add(
    input: Tensor,
    other: Tensor,
    output: Tensor = None,
    name: str = "add",
) -> Tensor:
    """
    Performs an element-wise addition operator between the `input`
    tensor and the `other` tensor.
    Usage:
    tensor_add = ark.add(tensor1, tensor2)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().add(
        input._tensor, other._tensor, output, name
    )
    return Tensor(_tensor)


def sub(
    input: Tensor,
    other: Tensor,
    output: Tensor = None,
    name: str = "sub",
) -> Tensor:
    """
    Performs an element-wise addition operator between the `input`
    tensor and the `other` tensor.
    Usage:
    tensor_add = ark.sub(tensor1, tensor2)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().sub(
        input._tensor, other._tensor, output, name
    )
    return Tensor(_tensor)


def mul(
    input: Tensor,
    other: Tensor,
    output: Tensor = None,
    name: str = "mul",
) -> Tensor:
    """
    Performs an element-wise multiplication operator between the
    `input` tensor and the `other` tensor.
    Usage:
    tensor_mul = ark.mul(tensor1, tensor2)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().mul(
        input._tensor, other._tensor, output, name
    )
    return Tensor(_tensor)


def div(
    input: Tensor,
    other: Tensor,
    output: Tensor = None,
    name: str = "div",
) -> Tensor:
    """
    Performs an element-wise division operator between the
    `input` tensor and the `other` tensor.
    Usage:
    tensor_mul = ark.div(tensor1, tensor2)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().div(
        input._tensor, other._tensor, output, name
    )
    return Tensor(_tensor)


def send(
    input: Tensor,
    id: int,
    dst_rank: int,
    bytes: int = 0,
    output: Tensor = None,
    name: str = "send",
) -> Tensor:
    """
    Sends a tensor to a destination GPU (`dst_rank`). Multiple
    tensors can be sent to the same GPU, so an identifier `id` is
    required to distinguish the tensor. Each 'send' operator must
    have a corresponding 'recv' operator that have the same id in
    another GPU's model.
    Usage:
    # on GPU0:
    ark.send(tensor_send, 1, 1)
    ark.send_done(tensor_send, 1, 1)
    # on GPU1:
    ark.recv(tensor, 1, 0)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().send(
        input._tensor,
        id,
        dst_rank,
        bytes,
        output,
        name,
    )
    return Tensor(_tensor)


def send_done(
    input: Tensor,
    id: int,
    dst_rank: int,
    output: Tensor = None,
    name: str = "send_done",
) -> Tensor:
    """
    Blocks the execution until the corresponding 'send' operator
    with the specified `id` is completed.
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().send_done(
        input._tensor,
        id,
        dst_rank,
        output,
        name,
    )
    return Tensor(_tensor)


def recv(
    input: Tensor,
    id: int,
    src_rank: int,
    bytes: int = 0,
    output: Tensor = None,
    name: str = "recv",
) -> Tensor:
    """
    Receives a tensor from a source GPU (`src_rank`), identified by
    the `id` parameter. Blocks the execution until the corresponding
    'recv' operator is completed.
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().recv(
        input._tensor,
        id,
        src_rank,
        bytes,
        output,
        name,
    )
    return Tensor(_tensor)


def send_mm(
    input: Tensor,
    id: int,
    gpu_dst: int,
    bytes: int = 0,
    output: Tensor = None,
    name: str = "send_mm",
) -> Tensor:
    """
    Similar to the 'send_done' function, but implemented using CUDA
    in-stream RDMA copy and Low Latency (LL) protocol.
    Usage:
    # on GPU0:
    ark.send_mm(tensor_send, 1, 1)
    # on GPU1:
    ark.recv_mm(tensor, 1, 0)
    """
    if output is not None:
        output = output._tensor

    _tensor = Model.get_global_model().send_mm(
        input._tensor,
        id,
        gpu_dst,
        bytes,
        output,
        name,
    )
    return Tensor(_tensor)


def recv_mm(
    input: Tensor,
    id: int,
    gpu_src: int,
    bytes: int = 0,
    output: Tensor = None,
    name: str = "recv_mm",
) -> Tensor:
    """
    Similar to the 'recv' function, but implemented using CUDA
    in-stream RDMA copy and Low Latency (LL) protocol.
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().recv_mm(
        input._tensor,
        id,
        gpu_src,
        bytes,
        output,
        name,
    )
    return Tensor(_tensor)


def all_gather(
    input: Tensor,
    gpu_id: int,
    gpu_num: int,
    output: List[Tensor] = [],
    name: str = "all_gather",
) -> List[Tensor]:
    """
    Performs an all-gather operator across all GPUs.
    Usage:
    # all-gather
    ark.init(rank, world_size)
    input_tensor = ark.tensor([tensor_len], ark.TensorType.FP16)
    # The all_gather operation will create the recv tensor shards and return
    them as a list. The allgather_result[rank] is the same as input_tensor
    allgather_result = ark.all_gather(input_tensor, rank, world_size)

    # in-place all-gather
    ark.init(rank, world_size)
    output_tensor = ark.tensor(
        [tensor_len * world_size], ark.TensorType.FP16
    )
    # Shard the output tensor into world_size shards
    output_shard = ark.sharding(output_tensor, 0, tensor_len)
    # The input tensor is the rank'th shard of the output tensor
    input_tensor = output_shard[rank]
    allgather_result = ark.all_gather(
        input_tensor, rank, world_size, output_shard
    )
    """
    for output_shard in output:
        if output_shard is not None:
            output_shard = output_shard._tensor
    tensor_shards = Model.get_global_model().all_gather(
        input._tensor,
        gpu_id,
        gpu_num,
        output,
        name,
    )
    return [Tensor(_tensor) for _tensor in tensor_shards]


def all_reduce(
    input: Tensor,
    gpu_id: int,
    gpu_num: int,
    output: Tensor = None,
    name: str = "all_reduce",
) -> Tensor:
    """
    Performs an all-reduce operator across all GPUs, aggregating the
    input tensors. Takes the `input` tensor, the current GPU's
    `gpu_id`, and the total number of GPUs `gpu_num`.
    Usage:
    ark.init(rank, world_size)
    input_tensor = ark.tensor([tensor_len], ark.TensorType.FP16)
    allreduce_result = ark.all_reduce(input_tensor, rank, world_size)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_global_model().all_reduce(
        input._tensor,
        gpu_id,
        gpu_num,
        output,
        name,
    )
    return Tensor(_tensor)


class Model(_Model):
    """
    The Model class will record the all operators and tensors defined
    by the user.
    """

    # A global model object
    global_model = None

    @staticmethod
    def get_global_model():
        if Model.global_model is None:
            logging.error("Model is not initialized")
            raise RuntimeError("Model is not initialized")
        return Model.global_model
