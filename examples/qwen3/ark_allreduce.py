# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Qwen3 TP all-reduce wrapper via ARK mscclpp fused-packet API.

Wraps ``ark.all_reduce_packet`` for 2-D Qwen3 tensor-parallel shapes.
Flattens to 1-D for the packet API and reshapes output back to the
original shape.  Includes alignment, dtype, and contiguity validation.

Qwen3-8B TP all-reduce sites (both attention output and MLP output):
  Prefill (B=1, S=2048): (2048, 4096) = 8,388,608 elements
  Decode  (B=1, S=1):    (1, 4096)    = 4,096 elements
Both divisible by 4 * world_size (32 for TP=8).
"""

import torch

import ark


def validate_allreduce_input(x: torch.Tensor, world_size: int) -> None:
    """Validate that *x* is suitable for ``ark.all_reduce_packet``.

    Checks:
      - dtype is float16 (packet API requirement)
      - tensor is contiguous
      - element count is divisible by ``4 * world_size``

    Raises:
        ValueError: on any failed check.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if x.dtype != torch.float16:
        raise ValueError(f"all_reduce_packet requires float16, got {x.dtype}")
    if not x.is_contiguous():
        raise ValueError("all_reduce_packet requires a contiguous tensor")
    divisor = 4 * world_size
    if x.numel() % divisor != 0:
        raise ValueError(
            f"element count {x.numel()} is not divisible by "
            f"4 * world_size = {divisor}"
        )


def ark_allreduce(
    x: torch.Tensor,
    rank: int,
    world_size: int,
) -> "ark.Tensor":
    """All-reduce a contiguous fp16 tensor via ARK fused-packet API.

    Flattens *x* to 1-D, calls ``ark.all_reduce_packet``, and returns
    an ARK tensor whose ``.to_torch()`` yields a torch tensor with the
    original shape restored.

    Args:
        x: fp16 contiguous CUDA tensor (any shape).
        rank: Rank of the current process (0-indexed).
        world_size: Total number of TP ranks.

    Returns:
        ARK tensor wrapping the all-reduced result.  Call ``.to_torch()``
        to materialise a torch tensor.  The original shape is already
        restored.
    """
    validate_allreduce_input(x, world_size)
    orig_shape = x.shape
    x_flat = x.reshape(-1)

    ark.set_rank(rank)
    ark.set_world_size(world_size)
    ark.init()
    result = ark.all_reduce_packet(x_flat, rank, world_size)
    # Reshape back to original shape via ark.reshape
    if len(orig_shape) > 1:
        result = ark.reshape(result, list(orig_shape))
    return result


def torch_allreduce(
    x: torch.Tensor,
) -> torch.Tensor:
    """All-reduce via ``torch.distributed`` (NCCL backend).

    Requires ``torch.distributed`` to be initialised.  Operates
    in-place and returns the result tensor.

    Args:
        x: Tensor on CUDA.  Modified in-place.

    Returns:
        The same tensor after in-place all-reduce.
    """
    import torch.distributed as dist

    dist.all_reduce(x)
    return x
