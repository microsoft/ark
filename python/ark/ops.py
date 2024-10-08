# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Iterable, Union

from .tensor import Dims, Tensor, Parameter, NullTensor
from .data_type import DataType, fp32
from .model import Model
from . import log


__all__ = [
    "tensor",
    "parameter",
    "reshape",
    "identity",
    "sharding",
    "reduce_sum",
    "reduce_mean",
    "reduce_max",
    "layernorm",
    "softmax",
    "transpose",
    "matmul",
    "exp",
    "sqrt",
    "rsqrt",
    "rope",
    "relu",
    "gelu",
    "sigmoid",
    "add",
    "sub",
    "mul",
    "div",
    "all_reduce",
    "embedding",
    "cast",
    "copy",
    "constant",
    "ones",
    "zeros",
]


def is_list_or_tuple(obj):
    return isinstance(obj, list) or isinstance(obj, tuple)


def _tensor(
    shape: Iterable[int],
    dtype: DataType = fp32,
    strides: Iterable[int] = [],
    offsets: Iterable[int] = [],
    padded_shape: Iterable[int] = [],
    rank: int = -1,
    name: str = "",
) -> Tensor:
    if not is_list_or_tuple(shape):
        raise log.InvalidUsageError(
            "shape should be a list or tuple of integers"
        )
    if not is_list_or_tuple(strides):
        raise log.InvalidUsageError(
            "strides should be a list or tuple of integers"
        )
    if not is_list_or_tuple(offsets):
        raise log.InvalidUsageError(
            "offsets should be a list or tuple of integers"
        )
    if not is_list_or_tuple(padded_shape):
        raise log.InvalidUsageError(
            "padded_shape should be a list or tuple of integers"
        )
    # only support tensors with up to 4 dimensions
    if (
        len(shape) > 4
        or len(strides) > 4
        or len(offsets) > 4
        or len(padded_shape) > 4
    ):
        raise log.InvalidUsageError(
            "Only support tensors with up to 4 dimensions"
        )
    return Model.get_model().tensor(
        Dims(shape),
        dtype.ctype(),
        Dims(strides),
        Dims(offsets),
        Dims(padded_shape),
        rank,
        name,
    )


def add(
    input: Union[Tensor, float],
    other: Union[Tensor, float],
    output: Tensor = NullTensor,
    name: str = "add",
) -> Union[Tensor, float]:
    """ """
    if isinstance(input, Tensor) and isinstance(other, Tensor):
        a = input._tensor
        b = other._tensor
    elif isinstance(input, Tensor):
        a = input._tensor
        b = other
    elif isinstance(other, Tensor):
        a = other._tensor
        b = input
    elif output is NullTensor:
        return input + other
    else:
        return Tensor(
            Model.get_model().copy(input + other, output._tensor, name)
        )
    if output is not NullTensor:
        output = output._tensor
    return Tensor(Model.get_model().add(a, b, output, name))


def cast(
    input: Tensor,
    dtype: DataType,
    output: Tensor = NullTensor,
    name: str = "cast",
) -> Tensor:
    """ """
    if output is not NullTensor:
        output = output._tensor
    return Tensor(
        Model.get_model().cast(input._tensor, dtype.ctype(), output, name)
    )


def constant(
    value: float,
    shape: Iterable[int],
    dtype: DataType = fp32,
    name: str = "constant",
) -> Tensor:
    """ """
    return Tensor(
        Model.get_model().constant(value, Dims(shape), dtype.ctype(), name)
    )


def copy(
    input: Union[Tensor, float], output: Tensor = NullTensor, name: str = "copy"
) -> Tensor:
    """ """
    if output is not NullTensor:
        output = output._tensor
    if isinstance(input, Tensor):
        input = input._tensor
    return Tensor(Model.get_model().copy(input, output, name))


def div(
    input: Tensor,
    other: Union[Tensor, float],
    output: Tensor = NullTensor,
    name: str = "div",
) -> Tensor:
    """ """
    if output is not NullTensor:
        output = output._tensor
    if isinstance(other, Tensor):
        other = other._tensor
    return Tensor(Model.get_model().div(input._tensor, other, output, name))


def embedding(
    input: Tensor,
    weight: Tensor,
    output: Tensor = NullTensor,
    name: str = "embedding",
) -> Tensor:
    """ """
    if output is not NullTensor:
        output = output._tensor
    return Tensor(
        Model.get_model().embedding(input._tensor, weight._tensor, output, name)
    )


def exp(
    input: Tensor, output: Tensor = NullTensor, name: str = "exp"
) -> Tensor:
    """ """
    if output is not NullTensor:
        output = output._tensor
    return Tensor(Model.get_model().exp(input._tensor, output, name))


def gelu(
    input: Tensor, output: Tensor = NullTensor, name: str = "gelu"
) -> Tensor:
    """ """
    if output is not NullTensor:
        output = output._tensor
    return Tensor(Model.get_model().gelu(input._tensor, output, name))


def identity(
    input: Tensor, deps: List[Tensor] = [], name: str = "identity"
) -> Tensor:
    """ """
    dep_tensors = []
    for dep in deps:
        if not isinstance(dep, Tensor):
            raise log.InvalidUsageError("All dependencies should be a tensor")
        dep_tensors.append(dep._tensor)
    return Tensor(Model.get_model().identity(input._tensor, dep_tensors, name))


def matmul(
    input: Tensor,
    other: Tensor,
    output: Tensor = NullTensor,
    transpose_input: bool = False,
    transpose_other: bool = False,
    name: str = "matmul",
) -> Tensor:
    """ """
    if output is not NullTensor:
        output = output._tensor
    return Tensor(
        Model.get_model().matmul(
            input._tensor,
            other._tensor,
            output,
            transpose_input,
            transpose_other,
            name,
        )
    )


def mul(
    input: Tensor,
    other: Union[Tensor, float],
    output: Tensor = NullTensor,
    name: str = "mul",
) -> Tensor:
    """ """
    if output is not NullTensor:
        output = output._tensor
    if isinstance(other, Tensor):
        other = other._tensor
    return Tensor(Model.get_model().mul(input._tensor, other, output, name))


def noop(input: Tensor, name: str = "noop"):
    """ """
    Model.get_model().noop(input._tensor, name)


def reduce_max(
    input: Tensor,
    axis: int,
    keepdims: bool = True,
    output: Tensor = NullTensor,
    name: str = "reduce_max",
) -> Tensor:
    """ """
    if output is not NullTensor:
        output = output._tensor
    return Tensor(
        Model.get_model().reduce_max(
            input._tensor, axis, keepdims, output, name
        )
    )


def reduce_mean(
    input: Tensor,
    axis: int,
    keepdims: bool = True,
    output: Tensor = NullTensor,
    name: str = "reduce_mean",
) -> Tensor:
    """ """
    if output is not NullTensor:
        output = output._tensor
    return Tensor(
        Model.get_model().reduce_mean(
            input._tensor, axis, keepdims, output, name
        )
    )


def reduce_sum(
    input: Tensor,
    axis: int,
    keepdims: bool = True,
    output: Tensor = NullTensor,
    name: str = "reduce_sum",
) -> Tensor:
    """ """
    if output is not NullTensor:
        output = output._tensor
    return Tensor(
        Model.get_model().reduce_sum(
            input._tensor, axis, keepdims, output, name
        )
    )


def relu(
    input: Tensor, output: Tensor = NullTensor, name: str = "relu"
) -> Tensor:
    """ """
    if output is not NullTensor:
        output = output._tensor
    return Tensor(Model.get_model().relu(input._tensor, output, name))


def reshape(
    input: Tensor,
    shape: Iterable[int],
    allowzero: bool = False,
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
    if not is_list_or_tuple(shape):
        raise log.InvalidUsageError(
            "shape should be a list or tuple of integers"
        )
    # only support tensors with up to 4 dimensions
    if len(shape) > 4:
        raise log.InvalidUsageError(
            "Only support tensors with up to 4 dimensions"
        )
    return Tensor(
        Model.get_model().reshape(input._tensor, Dims(shape), allowzero, name)
    )


def rope(
    input: Tensor,
    other: Tensor,
    output: Tensor = NullTensor,
    name: str = "rope",
) -> Tensor:
    """ """
    if output is not NullTensor:
        output = output._tensor
    return Tensor(
        Model.get_model().rope(input._tensor, other._tensor, output, name)
    )


def rsqrt(
    input: Tensor, output: Tensor = NullTensor, name: str = "rsqrt"
) -> Tensor:
    """ """
    if output is not NullTensor:
        output = output._tensor
    return Tensor(Model.get_model().rsqrt(input._tensor, output, name))


def sharding(
    input: Tensor, axis: int, dim_per_shard: int, name: str = "sharding"
) -> List[Tensor]:
    """ """
    _tensor_list = Model.get_model().sharding(
        input._tensor, axis, dim_per_shard, name
    )
    return [Tensor(_tensor) for _tensor in _tensor_list]


def sigmoid(
    input: Tensor, output: Tensor = NullTensor, name: str = "sigmoid"
) -> Tensor:
    """ """
    if output is not NullTensor:
        output = output._tensor
    return Tensor(Model.get_model().sigmoid(input._tensor, output, name))


def sqrt(
    input: Tensor, output: Tensor = NullTensor, name: str = "sqrt"
) -> Tensor:
    """ """
    if output is not NullTensor:
        output = output._tensor
    return Tensor(Model.get_model().sqrt(input._tensor, output, name))


def sub(
    input: Tensor,
    other: Union[Tensor, float],
    output: Tensor = NullTensor,
    name: str = "sub",
) -> Tensor:
    """ """
    if output is not NullTensor:
        output = output._tensor
    if isinstance(other, Tensor):
        other = other._tensor
    return Tensor(Model.get_model().sub(input._tensor, other, output, name))


def tensor(
    shape: Iterable[int],
    dtype: DataType = fp32,
    strides: Iterable[int] = [],
    offsets: Iterable[int] = [],
    padded_shape: Iterable[int] = [],
    rank: int = -1,
    name: str = "",
) -> Tensor:
    """ """
    return Tensor(
        _tensor(shape, dtype, strides, offsets, padded_shape, rank, name)
    )


def transpose(
    input: Tensor,
    perm: Iterable[int],
    output: Tensor = NullTensor,
    name: str = "transpose",
) -> Tensor:
    """ """
    if output is not NullTensor:
        output = output._tensor
    if not is_list_or_tuple(perm):
        raise log.InvalidUsageError(
            "perm should be a list or tuple of integers"
        )
    # only support tensors with up to 4 dimensions
    if len(perm) > 4:
        raise log.InvalidUsageError("Only support perm up to 4 dimensions")
    return Tensor(
        Model.get_model().transpose(input._tensor, perm, output, name)
    )


################################################################################


def mean(
    input: Tensor,
    axis: int,
    keepdims: bool = True,
    output: Tensor = NullTensor,
    name: str = "mean",
) -> Tensor:
    """ """
    return reduce_mean(input, axis, keepdims, output, name)


def ones(
    shape: Iterable[int], dtype: DataType = fp32, name: str = "ones"
) -> Tensor:
    """ """
    return Tensor(
        Model.get_model().constant(1, Dims(shape), dtype.ctype(), name)
    )


def parameter(
    shape: Iterable[int],
    dtype: DataType = fp32,
    strides: Iterable[int] = [],
    offsets: Iterable[int] = [],
    padded_shape: Iterable[int] = [],
    name: str = "",
) -> Parameter:
    """ """
    return Parameter(
        _tensor(shape, dtype, strides, offsets, padded_shape, name)
    )


def softmax(
    input: Tensor, output: Tensor = NullTensor, name: str = "softmax"
) -> Tensor:
    """ """
    max = reduce_max(input, axis=-1)
    output = sub(input, max, output=output)
    output = exp(output, output=output)
    sum = reduce_sum(output, axis=-1)
    output = div(output, sum, output=output)
    return output


def layernorm(
    input: Tensor,
    eps: float = 1e-6,
    output: Tensor = NullTensor,
    name: str = "layernorm",
) -> Tensor:
    mean = reduce_mean(input, axis=-1)
    x = sub(input, mean)
    variance = reduce_mean(mul(x, x), axis=-1)
    output = mul(x, rsqrt(add(variance, eps)), output=output)
    return output


def zeros(
    shape: Iterable[int], dtype: DataType = fp32, name: str = "zeros"
) -> Tensor:
    """ """
    return Tensor(
        Model.get_model().constant(0, Dims(shape), dtype.ctype(), name)
    )


def all_reduce(
    input: Tensor,
    rank: int,
    world_size: int,
    output: Tensor = NullTensor,
    name: str = "all_reduce",
) -> Tensor:
    """
    Perform an all-reduce operation on the input tensor.

    Args:
        input (Tensor): The input tensor to be reduced.
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
        output (Tensor, optional): The output tensor. If provided, the result
            will be stored in this tensor. Defaults to NullTensor.
        name (str, optional): The name of the operation. Defaults to
            "all_reduce".

    Returns:
        Tensor: The reduced tensor.
    """
    if output is not NullTensor:
        output = output._tensor
    _tensor = Model.get_model().all_reduce(
        input._tensor, rank, world_size, output, name
    )
    return Tensor(_tensor)


# def im2col(
#     input: Tensor,
#     kernel_height: int,
#     kernel_width: int,
#     stride_height: int,
#     stride_width: int,
#     pad_height: int,
#     pad_width: int,
#     dilation_height: int,
#     dilation_width: int,
#     output: Tensor = NullTensor,
#     name: str = "im2col",
# ) -> Tensor:
#     """
#     Implements the 'im2col' method for 2D convolution layers, which
#     takes an `input` tensor and reshapes it to a 2D matrix by
#     extracting image patches from the input tensor based on the
#     provided parameters.
#     """
#     if output is not NullTensor:
#         output = output._tensor
#     _tensor = Model.get_model().im2col(
#         input._tensor,
#         kernel_height,
#         kernel_width,
#         stride_height,
#         stride_width,
#         pad_height,
#         pad_width,
#         dilation_height,
#         dilation_width,
#         output,
#         name,
#     )
#     return Tensor(_tensor)

# def rope(
#     input: Tensor, other: Tensor, output: Tensor = NullTensor, name: str = "rope"
# ) -> Tensor:
#     """
#     Performs rotary position embedding (RoPE) on the `input` tensor
#     Usage:
#     tensor_mul = ark.rope(tensor1, tensor2)
#     """
#     if output is not NullTensor:
#         output = output._tensor
#     _tensor = Model.get_model().rope(input._tensor, other._tensor, output, name)
#     return Tensor(_tensor)


# def send(
#     input: Tensor,
#     sid: int,
#     dst_rank: int,
#     bytes: int = 0,
#     name: str = "send",
# ) -> Tensor:
#     """
#     Sends a tensor to a destination GPU (`dst_rank`). Multiple
#     tensors can be sent to the same GPU, so an identifier `id` is
#     required to distinguish the tensor. Each 'send' operator must
#     have a corresponding 'recv' operator that have the same id in
#     another GPU's model.
#     Usage:
#     # on GPU0:
#     ark.send(tensor_send, 1, 1)
#     ark.send_done(tensor_send, 1, 1)
#     # on GPU1:
#     ark.recv(1, 0, 0, tensor_recv)
#     """
#     _tensor = Model.get_model().send(
#         input._tensor,
#         sid,
#         dst_rank,
#         bytes,
#         name,
#     )
#     return Tensor(_tensor)


# def send_done(
#     input: Tensor,
#     sid: int,
#     dst_rank: int,
#     name: str = "send_done",
# ) -> Tensor:
#     """
#     Blocks the execution until the corresponding 'send' operator
#     with the specified `id` is completed.
#     """
#     _tensor = Model.get_model().send_done(
#         input._tensor,
#         sid,
#         dst_rank,
#         name,
#     )
#     return Tensor(_tensor)


# def recv(
#     sid: int,
#     src_rank: int,
#     bytes: int,
#     output: Tensor = NullTensor,
#     name: str = "recv",
# ) -> Tensor:
#     """
#     Receives a tensor from a source GPU (`src_rank`), identified by
#     the `id` parameter. Blocks the execution until the corresponding
#     'recv' operator is completed.
#     """
#     if output is not NullTensor:
#         output = output._tensor
#     _tensor = Model.get_model().recv(
#         sid,
#         src_rank,
#         bytes,
#         output,
#         name,
#     )
#     return Tensor(_tensor)


# def all_gather(
#     input: Tensor,
#     rank: int,
#     world_size: int,
#     output: List[Tensor] = [],
#     name: str = "all_gather",
# ) -> List[Tensor]:
#     """
#     Performs an all-gather operator across all GPUs.
#     Usage:
#     # all-gather
#     ark.init(rank, world_size)
#     input_tensor = ark.tensor([tensor_len], ark.fp16)
#     # The all_gather operation will create the recv tensor shards and return
#     them as a list. The allgather_result[rank] is the same as input_tensor
#     allgather_result = ark.all_gather(input_tensor, rank, world_size)

#     # in-place all-gather
#     ark.init(rank, world_size)
#     output_tensor = ark.tensor(
#         [tensor_len * world_size], ark.fp16
#     )
#     # Shard the output tensor into world_size shards
#     output_shard = ark.sharding(output_tensor, 0, tensor_len)
#     # The input tensor is the rank'th shard of the output tensor
#     input_tensor = output_shard[rank]
#     allgather_result = ark.all_gather(
#         input_tensor, rank, world_size, output_shard
#     )
#     """
#     output = [output_shard._tensor for output_shard in output]
#     tensor_shards = Model.get_model().all_gather(
#         input._tensor, rank, world_size, output, name
#     )
#     return [Tensor(_tensor) for _tensor in tensor_shards]


# def local_all_gather(
#     input: Tensor,
#     rank: int,
#     ranks_per_node: int,
#     axis: int,
#     name: str = "local_all_gather",
# ) -> Tensor:
#     """
#     Performs an all-gather operator across local node GPUs.
#     Usage:
#     # all-gather
#     ark.init(rank, world_size)
#     input_tensor = ark.tensor([tensor_len], ark.fp16)
#     allgather_result = ark.local_all_gather(input_tensor, rank, ranks_per_node)
#     """
#     _tensor = Model.get_model().local_all_gather(
#         input._tensor,
#         rank,
#         ranks_per_node,
#         axis,
#         name,
#     )
#     return Tensor(_tensor)


# def local_reduce_scatter(
#     input: Tensor,
#     rank: int,
#     ranks_per_node: int,
#     name: str = "local_reduce_scatter",
# ) -> Tensor:
#     """
#     Performs an reduce-scatter operator across local node GPUs.
#     Usage:
#     # reduce-scatter
#     ark.init(rank, world_size)
#     input_tensor = ark.tensor([tensor_len], ark.fp16)
#     reduce_scatter_result = ark.local_reduce_scatter(input_tensor, rank, ranks_per_node)
#     """
#     _tensor = Model.get_model().local_reduce_scatter(
#         input._tensor,
#         rank,
#         ranks_per_node,
#         name,
#     )
#     return Tensor(_tensor)


# def local_all_reduce(
#     input: Tensor,
#     rank: int,
#     ranks_per_node: int,
#     name: str = "local_all_reduce",
# ) -> Tensor:
#     """
#     Performs an all-reduce operator across local GPUs, aggregating the
#     input tensors. Takes the `input` tensor, the current GPU's
#     `rank`, and the total number of GPUs in a node`ranks_per_node`.
#     Usage:
#     ark.init(rank, world_size)
#     input_tensor = ark.tensor([tensor_len], ark.fp16)
#     allreduce_result = ark.local_all_reduce(input_tensor, rank, ranks_per_node)
#     """
#     _tensor = Model.get_model().local_all_reduce(
#         input._tensor,
#         rank,
#         ranks_per_node,
#         name,
#     )
#     return Tensor(_tensor)


# def local_all_reduce_packet(
#     input: Tensor,
#     rank: int,
#     ranks_per_node: int,
#     name: str = "local_all_reduce",
# ) -> Tensor:
#     """
#     Performs an all-reduce operator across local GPUs with LL algo, aggregating the
#     input tensors. Takes the `input` tensor, the current GPU's
#     `rank`, and the total number of GPUs in a node`ranks_per_node`.
#     Usage:
#     ark.init(rank, world_size)
#     input_tensor = ark.tensor([tensor_len], ark.fp16)
#     allreduce_result = ark.local_all_reduce_packet(input_tensor, rank, ranks_per_node)
#     """
#     _tensor = Model.get_model().local_all_reduce_packet(
#         input._tensor,
#         rank,
#         ranks_per_node,
#         name,
#     )
#     return Tensor(_tensor)
