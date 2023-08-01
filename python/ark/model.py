# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ._ark_core import _Model, Tensor, TensorBuf, TensorType, Dims


def tensor(
    shape,
    dtype: TensorType = TensorType.FP32,
    buf: TensorBuf = None,
    ldims: Dims = Dims(),
    offs: Dims = Dims(),
    pads: Dims = Dims(),
    deps: list = [],
    exported: bool = False,
    imported: bool = False,
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
        assert len(shape) <= 4
        shape = Dims(*shape)
    return Model.global_model.tensor(
        shape, dtype, buf, ldims, offs, pads, deps, exported, imported, name
    )


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
        assert len(shape) <= 4
        shape = Dims(*shape)
    return Model.global_model.reshape(input, shape, allowzero, output, name)


def identity(
    input: Tensor,
    deps: list = [],
    output: Tensor = None,
    name: str = "identity",
) -> Tensor:
    """
    Returns an identical tensor of `input` with execution dependencies `deps`.
    Usage:
    tensor_identity = ark.identity(tensor, deps=[tensor1, tensor2])
    """
    return Model.global_model.identity(input, deps, output, name)


def sharding(
    input: Tensor,
    axis: int,
    dim_per_shard: int,
    name: str = "sharding",
) -> Tensor:
    """
    Shard `input` along `axis` into `dim_per_shard`-dimensional shards.
    Usage:
    # tensors shape is [64, 128]
    tensor_sharding = ark.sharding(tensor, axis=1, dim_per_shard=64)
    # tensor_sharding is a list of 2 tensors, each of which has shape [64, 64]
    # The first tensor's buffer is the same as the first 64 columns of tensor
    # The second tensor's buffer is the same as the last 64 columns of tensor
    """
    return Model.global_model.sharding(input, axis, dim_per_shard, name)


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
    return Model.global_model.reduce_sum(input, axis, output, name)


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
    return Model.global_model.reduce_mean(input, axis, output, name)


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
    return Model.global_model.reduce_max(input, axis, output, name)


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
    return Model.global_model.layernorm(input, output, name)


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
    return Model.global_model.softmax(input, output, name)


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
    return Model.global_model.transpose(input, perm, output, name)


def matmul(
    input: Tensor,
    other: Tensor,
    output: Tensor = None,
    splitk: int = 1,
    tran_input: bool = False,
    tran_other: bool = False,
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
    return Model.global_model.matmul(
        input,
        other,
        output,
        splitk,
        tran_input,
        tran_other,
        is_relu,
        name,
        gran_lev,
    )


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
    return Model.global_model.im2col(
        input,
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
    return Model.global_model.conv2d(
        input,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        output,
        name,
    )


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
    return Model.global_model.max_pool(
        input,
        kernel_size,
        stride,
        output,
        name,
    )


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
    return Model.global_model.scale(
        input,
        val,
        output,
        name,
    )


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
    return Model.global_model.relu(input, output, name)


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
    return Model.global_model.gelu(input, output, name)


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
    return Model.global_model.add(input, other, output, name)


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
    return Model.global_model.mul(input, other, output, name)


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
    return Model.global_model.send(
        input,
        id,
        dst_rank,
        bytes,
        output,
        name,
    )


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
    return Model.global_model.send_done(
        input,
        id,
        dst_rank,
        output,
        name,
    )


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
    return Model.global_model.recv(
        input,
        id,
        src_rank,
        bytes,
        output,
        name,
    )


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
    return Model.global_model.send_mm(
        input,
        id,
        gpu_dst,
        bytes,
        output,
        name,
    )


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
    return Model.global_model.recv_mm(
        input,
        id,
        gpu_src,
        bytes,
        output,
        name,
    )


def all_gather(
    input: Tensor,
    gpu_id: int,
    gpu_num: int,
    output: Tensor = None,
    name: str = "all_gather",
) -> Tensor:
    """
    Performs an all-gather operator across all GPUs.
    Usage:
    # all-gather
    ark.init_model(rank, world_size)
    input_tensor = ark.tensor([tensor_len], ark.TensorType.FP16)
    # The all_gather operation will create the recv tensor shards and return them as a list. The allgather_result[rank] is the same as input_tensor
    allgather_result = ark.all_gather(input_tensor, rank, world_size)

    # in-place all-gather
    ark.init_model(rank, world_size)
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
    return Model.global_model.all_gather(
        input,
        gpu_id,
        gpu_num,
        output,
        name,
    )


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
    ark.init_model(rank, world_size)
    input_tensor = ark.tensor([tensor_len], ark.TensorType.FP16)
    allreduce_result = ark.all_reduce(input_tensor, rank, world_size)
    """
    return Model.global_model.all_reduce(
        input,
        gpu_id,
        gpu_num,
        output,
        name,
    )


class Model(_Model):
    # a global model object
    global_model = None

    def __init__(self, rank: int = 0):
        super().__init__(rank)
