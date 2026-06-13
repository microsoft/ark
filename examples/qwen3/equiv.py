# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Equivalence-test helper: compare ARK output against torch reference.

Provides richer mismatch diagnostics (bad-element count, per-tensor stats)
than torch.testing.assert_close, useful for debugging ARK-vs-reference
numerical differences.
"""

import torch


def assert_close(
    ark_out: torch.Tensor,
    ref_out: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    msg: str = "",
) -> None:
    """Assert that two tensors are element-wise close.

    On mismatch, reports shape, max absolute error, relative error,
    and basic statistics for both tensors.

    Args:
        ark_out: Tensor produced by the ARK implementation.
        ref_out: Tensor produced by the torch reference.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        msg: Optional context message for the assertion.
    """
    if ark_out.shape != ref_out.shape:
        raise AssertionError(
            f"Shape mismatch: ark {ark_out.shape} vs ref {ref_out.shape}. {msg}"
        )

    ark_f = ark_out.float()
    ref_f = ref_out.float()

    abs_diff = (ark_f - ref_f).abs()
    max_abs = abs_diff.max().item()
    ref_abs = ref_f.abs().clamp(min=1e-12)
    max_rel = (abs_diff / ref_abs).max().item()

    close = abs_diff <= (atol + rtol * ref_abs)
    if close.all():
        return

    n_bad = (~close).sum().item()
    n_total = close.numel()

    detail = (
        f"Tensors not close. {n_bad}/{n_total} elements exceed tolerance "
        f"(atol={atol}, rtol={rtol}).\n"
        f"  max |diff|      = {max_abs:.6e}\n"
        f"  max |diff|/|ref|= {max_rel:.6e}\n"
        f"  ark  stats: mean={ark_f.mean().item():.4e}, "
        f"std={ark_f.std().item():.4e}, "
        f"min={ark_f.min().item():.4e}, max={ark_f.max().item():.4e}\n"
        f"  ref  stats: mean={ref_f.mean().item():.4e}, "
        f"std={ref_f.std().item():.4e}, "
        f"min={ref_f.min().item():.4e}, max={ref_f.max().item():.4e}"
    )
    if msg:
        detail = f"{msg}\n{detail}"

    raise AssertionError(detail)
