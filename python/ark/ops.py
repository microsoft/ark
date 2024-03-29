# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Iterable

from .tensor import Dims, Tensor, TensorBuf, Parameter
from .data_type import DataType, fp32
from .model import Model


def _is_list_or_tuple(obj):
    return isinstance(obj, list) or isinstance(obj, tuple)


def tensor(
    shape: Iterable[int],
    dtype: DataType = fp32,
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
    tensor = ark.tensor([1, 2, 3, 4], dtype=ark.fp32)
    tensor = ark.tensor([1, 2], dtype=ark.fp16)
    """
    if not _is_list_or_tuple(shape):
        raise ValueError("shape should be a list or tuple of integers")
    # only support tensors with up to 4 dimensions
    if len(shape) > 4:
        raise ValueError("Only support tensors with up to 4 dimensions")
    _tensor = Model.get_model().tensor(
        Dims(*shape),
        dtype.ttype(),
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


def parameter(
    shape: Iterable[int],
    dtype: DataType = fp32,
    buf: TensorBuf = None,
    ldims: Dims = Dims(),
    offs: Dims = Dims(),
    pads: Dims = Dims(),
    deps: list = [],
    exported: bool = False,
    imported_rank: int = -1,
    name: str = "parameter",
) -> Parameter:
    """
    Construct a parameter with given shape and data type.
    """
    if not _is_list_or_tuple(shape):
        raise ValueError("shape should be a list or tuple of integers")
    # only support tensors with up to 4 dimensions
    if len(shape) > 4:
        raise ValueError("Only support tensors with up to 4 dimensions")
    _tensor = Model.get_model().tensor(
        Dims(*shape),
        dtype.ttype(),
        buf,
        ldims,
        offs,
        pads,
        deps,
        exported,
        imported_rank,
        name,
    )
    return Parameter(_tensor)


def reshape(
    input: Tensor,
    shape: Iterable[int],
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
    if not _is_list_or_tuple(shape):
        raise ValueError("shape should be a list or tuple of integers")
    # only support tensors with up to 4 dimensions
    if len(shape) > 4:
        raise ValueError("Only support tensors with up to 4 dimensions")
    if output is not None:
        output = output._tensor
    input = input._tensor
    _tensor = Model.get_model().reshape(input, shape, allowzero, output, name)
    return Tensor(_tensor)


def identity(
    input: Tensor, deps: List[Tensor] = [], name: str = "identity"
) -> Tensor:
    """
    Returns an identical tensor of `input` with execution dependencies `deps`.
    Usage:
    tensor_identity = ark.identity(tensor, deps=[tensor1, tensor2])
    """
    dep_tensor = []
    for dep in deps:
        if not isinstance(dep, Tensor):
            raise TypeError("All dependencies should be a tensor")
        dep_tensor.append(dep._tensor)
    _tensor = Model.get_model().identity(input._tensor, dep_tensor, name)
    return Tensor(_tensor)


def sharding(
    input: Tensor, axis: int, dim_per_shard: int, name: str = "sharding"
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
    _tensor_list = Model.get_model().sharding(
        input._tensor, axis, dim_per_shard, name
    )
    tensor_list = []
    for _tensor in _tensor_list:
        tensor_list.append(Tensor(_tensor))
    return tensor_list


def reduce_sum(
    input: Tensor,
    axis: int,
    keepdims: bool = True,
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
    _tensor = Model.get_model().reduce_sum(
        input._tensor, axis, keepdims, output, name
    )
    return Tensor(_tensor)


def reduce_mean(
    input: Tensor,
    axis: int,
    keepdims: bool = True,
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
    _tensor = Model.get_model().reduce_mean(
        input._tensor, axis, keepdims, output, name
    )
    return Tensor(_tensor)


def reduce_max(
    input: Tensor,
    axis: int,
    keepdims: bool = True,
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
    _tensor = Model.get_model().reduce_max(
        input._tensor, axis, keepdims, output, name
    )
    return Tensor(_tensor)


def layernorm(
    input: Tensor, output: Tensor = None, name: str = "layernorm"
) -> Tensor:
    """
    Applies layer normalization to the `input` tensor and returns
    the normalized tensor as `output`.
    Usage:
    tensor_layernorm = ark.layernorm(tensor)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_model().layernorm(input._tensor, output, name)
    return Tensor(_tensor)


def softmax(
    input: Tensor, output: Tensor = None, name: str = "softmax"
) -> Tensor:
    """
    Applies softmax  to the `input` tensor on the last dimension.
    Usage:
    tensor_softmax = ark.softmax(tensor)
    """
    max = reduce_max(input, axis=-1)
    output = sub(input, max, output=output)
    output = exp(output, output=output)
    sum = reduce_sum(output, axis=-1)
    output = div(output, sum, output=output)
    return output


def transpose(
    input: Tensor, perm: list, output: Tensor = None, name: str = "transpose"
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
    if not _is_list_or_tuple(perm):
        raise ValueError("perm should be a list or tuple of integers")
    # only support tensors with up to 4 dimensions
    if len(perm) > 4:
        raise ValueError("Only support perm up to 4 dimensions")
    _tensor = Model.get_model().transpose(
        input._tensor, Dims(*perm), output, name
    )
    return Tensor(_tensor)


def matmul(
    input: Tensor,
    other: Tensor,
    output: Tensor = None,
    splitk: int = 1,
    transpose_input: bool = False,
    transpose_other: bool = False,
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
    _tensor = Model.get_model().matmul(
        input._tensor,
        other._tensor,
        output,
        splitk,
        transpose_input,
        transpose_other,
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
    _tensor = Model.get_model().im2col(
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


def scale(
    input: Tensor, val: float, output: Tensor = None, name: str = "scale"
) -> Tensor:
    """
    Multiplies the `input` tensor by a scalar `val`, element-wise.
    Usage:
    tensor_scale = ark.scale(tensor, 1.6)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_model().scale(input._tensor, val, output, name)
    return Tensor(_tensor)


def exp(input: Tensor, output: Tensor = None, name: str = "exp") -> Tensor:
    """
    Calculates the exponential of the `input` tensor, element-wise.
    Usage:
    tensor_exp = ark.exp(tensor)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_model().exp(input._tensor, output, name)
    return Tensor(_tensor)


def sqrt(input: Tensor, output: Tensor = None, name: str = "sqrt") -> Tensor:
    """
    Calculates the square root of the `input` tensor, element-wise.
    Usage:
    tensor_sqrt = ark.sqrt(tensor)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_model().sqrt(input._tensor, output, name)
    return Tensor(_tensor)


def rsqrt(input: Tensor, output: Tensor = None, name: str = "rsqrt") -> Tensor:
    """
    Calculates the square root of the `input` tensor, element-wise.
    Usage:
    tensor_rsqrt = ark.rsqrt(tensor)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_model().rsqrt(input._tensor, output, name)
    return Tensor(_tensor)


def rope(
    input: Tensor, other: Tensor, output: Tensor = None, name: str = "rope"
) -> Tensor:
    """
    Performs rotary position embedding (RoPE) on the `input` tensor
    Usage:
    tensor_mul = ark.rope(tensor1, tensor2)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_model().rope(input._tensor, other._tensor, output, name)
    return Tensor(_tensor)


def relu(input: Tensor, output: Tensor = None, name: str = "relu") -> Tensor:
    """
    Applies the ReLU activation function to the `input` tensor,
    element-wise.
    Usage:
    tensor_relu = ark.relu(tensor)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_model().relu(input._tensor, output, name)
    return Tensor(_tensor)


def gelu(input: Tensor, output: Tensor = None, name: str = "gelu") -> Tensor:
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
    _tensor = Model.get_model().gelu(input._tensor, output, name)
    return Tensor(_tensor)


def sigmoid(
    input: Tensor, output: Tensor = None, name: str = "sigmoid"
) -> Tensor:
    """
    Applies the Sigmoid activation function to the `input` tensor,
    element-wise.
    Usage:
    tensor_sigmoid = ark.sigmoid(tensor)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_model().sigmoid(input._tensor, output, name)
    return Tensor(_tensor)


def add(
    input: Tensor, other: Tensor, output: Tensor = None, name: str = "add"
) -> Tensor:
    """
    Performs an element-wise addition operator between the `input`
    tensor and the `other` tensor.
    Usage:
    tensor_add = ark.add(tensor1, tensor2)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_model().add(input._tensor, other._tensor, output, name)
    return Tensor(_tensor)


def sub(
    input: Tensor, other: Tensor, output: Tensor = None, name: str = "sub"
) -> Tensor:
    """
    Performs an element-wise addition operator between the `input`
    tensor and the `other` tensor.
    Usage:
    tensor_add = ark.sub(tensor1, tensor2)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_model().sub(input._tensor, other._tensor, output, name)
    return Tensor(_tensor)


def mul(
    input: Tensor, other: Tensor, output: Tensor = None, name: str = "mul"
) -> Tensor:
    """
    Performs an element-wise multiplication operator between the
    `input` tensor and the `other` tensor.
    Usage:
    tensor_mul = ark.mul(tensor1, tensor2)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_model().mul(input._tensor, other._tensor, output, name)
    return Tensor(_tensor)


def div(
    input: Tensor, other: Tensor, output: Tensor = None, name: str = "div"
) -> Tensor:
    """
    Performs an element-wise division operator between the
    `input` tensor and the `other` tensor.
    Usage:
    tensor_mul = ark.div(tensor1, tensor2)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_model().div(input._tensor, other._tensor, output, name)
    return Tensor(_tensor)


def send(
    input: Tensor,
    sid: int,
    dst_rank: int,
    bytes: int = 0,
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
    ark.recv(1, 0, 0, tensor_recv)
    """
    _tensor = Model.get_model().send(
        input._tensor,
        sid,
        dst_rank,
        bytes,
        name,
    )
    return Tensor(_tensor)


def send_done(
    input: Tensor,
    sid: int,
    dst_rank: int,
    name: str = "send_done",
) -> Tensor:
    """
    Blocks the execution until the corresponding 'send' operator
    with the specified `id` is completed.
    """
    _tensor = Model.get_model().send_done(
        input._tensor,
        sid,
        dst_rank,
        name,
    )
    return Tensor(_tensor)


def recv(
    sid: int,
    src_rank: int,
    bytes: int,
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
    _tensor = Model.get_model().recv(
        sid,
        src_rank,
        bytes,
        output,
        name,
    )
    return Tensor(_tensor)


def all_gather(
    input: Tensor,
    rank: int,
    world_size: int,
    output: List[Tensor] = [],
    name: str = "all_gather",
) -> List[Tensor]:
    """
    Performs an all-gather operator across all GPUs.
    Usage:
    # all-gather
    ark.init(rank, world_size)
    input_tensor = ark.tensor([tensor_len], ark.fp16)
    # The all_gather operation will create the recv tensor shards and return
    them as a list. The allgather_result[rank] is the same as input_tensor
    allgather_result = ark.all_gather(input_tensor, rank, world_size)

    # in-place all-gather
    ark.init(rank, world_size)
    output_tensor = ark.tensor(
        [tensor_len * world_size], ark.fp16
    )
    # Shard the output tensor into world_size shards
    output_shard = ark.sharding(output_tensor, 0, tensor_len)
    # The input tensor is the rank'th shard of the output tensor
    input_tensor = output_shard[rank]
    allgather_result = ark.all_gather(
        input_tensor, rank, world_size, output_shard
    )
    """
    output = [output_shard._tensor for output_shard in output]
    tensor_shards = Model.get_model().all_gather(
        input._tensor, rank, world_size, output, name
    )
    return [Tensor(_tensor) for _tensor in tensor_shards]


def local_all_gather(
    input: Tensor,
    rank: int,
    ranks_per_node: int,
    axis: int,
    name: str = "local_all_gather",
) -> Tensor:
    """
    Performs an all-gather operator across local node GPUs.
    Usage:
    # all-gather
    ark.init(rank, world_size)
    input_tensor = ark.tensor([tensor_len], ark.fp16)
    allgather_result = ark.local_all_gather(input_tensor, rank, ranks_per_node)
    """
    _tensor = Model.get_model().local_all_gather(
        input._tensor,
        rank,
        ranks_per_node,
        axis,
        name,
    )
    return Tensor(_tensor)


def local_reduce_scatter(
    input: Tensor,
    rank: int,
    ranks_per_node: int,
    name: str = "local_reduce_scatter",
) -> Tensor:
    """
    Performs an reduce-scatter operator across local node GPUs.
    Usage:
    # reduce-scatter
    ark.init(rank, world_size)
    input_tensor = ark.tensor([tensor_len], ark.fp16)
    reduce_scatter_result = ark.local_reduce_scatter(input_tensor, rank, ranks_per_node)
    """
    _tensor = Model.get_model().local_reduce_scatter(
        input._tensor,
        rank,
        ranks_per_node,
        name,
    )
    return Tensor(_tensor)


def all_reduce(
    input: Tensor,
    rank: int,
    world_size: int,
    output: Tensor = None,
    name: str = "all_reduce",
) -> Tensor:
    """
    Performs an all-reduce operator across all GPUs, aggregating the
    input tensors. Takes the `input` tensor, the current GPU's
    `rank`, and the total number of GPUs `world_size`.
    Usage:
    ark.init(rank, world_size)
    input_tensor = ark.tensor([tensor_len], ark.fp16)
    allreduce_result = ark.all_reduce(input_tensor, rank, world_size)
    """
    if output is not None:
        output = output._tensor
    _tensor = Model.get_model().all_reduce(
        input._tensor, rank, world_size, output, name
    )
    return Tensor(_tensor)


def local_all_reduce(
    input: Tensor,
    rank: int,
    ranks_per_node: int,
    name: str = "local_all_reduce",
) -> Tensor:
    """
    Performs an all-reduce operator across local GPUs, aggregating the
    input tensors. Takes the `input` tensor, the current GPU's
    `rank`, and the total number of GPUs in a node`ranks_per_node`.
    Usage:
    ark.init(rank, world_size)
    input_tensor = ark.tensor([tensor_len], ark.fp16)
    allreduce_result = ark.local_all_reduce(input_tensor, rank, ranks_per_node)
    """
    _tensor = Model.get_model().local_all_reduce(
        input._tensor,
        rank,
        ranks_per_node,
        name,
    )
    return Tensor(_tensor)


def local_all_reduce_packet(
    input: Tensor,
    rank: int,
    ranks_per_node: int,
    name: str = "local_all_reduce",
) -> Tensor:
    """
    Performs an all-reduce operator across local GPUs with LL algo, aggregating the
    input tensors. Takes the `input` tensor, the current GPU's
    `rank`, and the total number of GPUs in a node`ranks_per_node`.
    Usage:
    ark.init(rank, world_size)
    input_tensor = ark.tensor([tensor_len], ark.fp16)
    allreduce_result = ark.local_all_reduce_packet(input_tensor, rank, ranks_per_node)
    """
    _tensor = Model.get_model().local_all_reduce_packet(
        input._tensor,
        rank,
        ranks_per_node,
        name,
    )
    return Tensor(_tensor)


def embedding(
    input: Tensor,
    weight: Tensor,
    output: Tensor = None,
    name: str = "embedding",
) -> Tensor:
    """Embedding layer."""
    if output is not None:
        output = output._tensor
    _tensor = Model.get_model().embedding(
        input._tensor, weight._tensor, output, name
    )
    return Tensor(_tensor)


def cast(
    input: Tensor, dtype: DataType, output: Tensor = None, name: str = "cast"
) -> Tensor:
    """Type casting."""
    if output is not None:
        output = output._tensor
    _tensor = Model.get_model().cast(input._tensor, dtype.ttype(), output, name)
    return Tensor(_tensor)
