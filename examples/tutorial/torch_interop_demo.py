# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
PyTorch interop demo — shows how ARK tensors and torch tensors
interact seamlessly.

Demonstrates:
  - ``Tensor.from_torch()`` — zero-copy ARK view of a CUDA torch tensor.
  - ``Tensor.to_torch()``   — zero-copy torch view of an ARK tensor.
  - Implicit torch→ARK conversion when passing torch tensors directly
    to ``ark.*`` ops (no explicit conversion needed). The torch tensor
    must be contiguous and on a CUDA device.
  - ``Tensor.eval()``       — one-liner: build graph, run, return torch
    tensor.

Run: ``python examples/tutorial/torch_interop_demo.py``
"""

import numpy as np
import torch
import ark


def demo_from_to_torch():
    """Round-trip: torch → ARK → torch preserves data."""
    ark.init()

    t = torch.randn(4, 64, dtype=torch.float16, device="cuda")

    # torch → ARK (zero-copy; shares memory)
    a = ark.Tensor.from_torch(t)
    assert a.shape() == list(t.shape), "Shape mismatch"

    # Build a trivial identity graph so the runtime has work to do
    out = ark.add(a, 0.0)

    runtime = ark.Runtime()
    runtime.launch()
    runtime.run()

    # ARK → torch (zero-copy via DLPack)
    t2 = out.to_torch()
    assert t2.shape == t.shape, "Shape mismatch after round-trip"

    # Adding 0.0 to fp16 values is exact; atol=0 is intentional.
    np.testing.assert_allclose(
        t2.cpu().numpy(),
        t.cpu().numpy(),
        atol=0,
        err_msg="Round-trip data mismatch",
    )
    print("[from/to_torch]  Round-trip passed.")


def demo_implicit_conversion():
    """Pass torch tensors directly to ark ops (implicit conversion)."""
    ark.init()

    x = torch.randn(8, 32, dtype=torch.float16, device="cuda")
    y = torch.randn(8, 32, dtype=torch.float16, device="cuda")

    # ark.add accepts torch.Tensor inputs — implicit conversion
    z = ark.add(x, y)

    runtime = ark.Runtime()
    runtime.launch()
    runtime.run()

    result = z.to_torch()
    expected = x + y
    np.testing.assert_allclose(
        result.cpu().numpy(),
        expected.cpu().numpy(),
        atol=1e-3,
        err_msg="Implicit conversion result mismatch",
    )
    print("[implicit conv]  torch tensors accepted by ark.add — passed.")


def demo_eval():
    """Tensor.eval() compiles and runs the graph in one call."""
    ark.init()

    x = torch.randn(4, 64, dtype=torch.float16, device="cuda")

    # eval() returns a torch.Tensor directly — no manual Runtime needed
    result = ark.relu(x).eval()

    assert isinstance(result, torch.Tensor), "eval() should return torch.Tensor"
    expected = torch.relu(x)
    np.testing.assert_allclose(
        result.cpu().numpy(),
        expected.cpu().numpy(),
        atol=1e-3,
        err_msg="eval() result mismatch",
    )
    print("[eval]           One-liner eval passed.")


def main():
    demo_from_to_torch()
    demo_implicit_conversion()
    demo_eval()
    print("\nAll torch interop demos passed!")


if __name__ == "__main__":
    main()
