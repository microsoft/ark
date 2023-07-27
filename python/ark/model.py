# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ._ark_core import _Model, Tensor, TensorBuf, TensorType, Dims


def tensor(
    shape: list,
    dtype: TensorType,
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
    Construct a tensor with given shape and data type."""
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
    """
    return Model.global_model.reshape(input, shape, allowzero, output, name)


def identity(
    input: Tensor,
    deps: list = [],
    output: Tensor = None,
    name: str = "identity",
) -> Tensor:
    """
    Returns an identical tensor of `input` with execution dependencies `deps`.
    """
    return Model.global_model.identity(input, deps, output, name)


def matmul(
    input: Tensor,
    other: Tensor,
    output: Tensor,
    splitk: int = 1,
    tran_input: bool = False,
    tran_other: bool = False,
    is_relu: bool = False,
    name: str = "matmul",
    gran_lev: int = -1,
) -> Tensor:
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


class Model(_Model):
    # a global model object
    global_model = None

    def __init__(self, rank: int = 0):
        super().__init__(rank)
        Model.global_model = self
