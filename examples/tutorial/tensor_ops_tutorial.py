# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tensor ops tutorial — covers tensor creation, elementwise ops,
reductions, and ``Runtime`` usage with the current ARK functional API.

Complements ``quickstart_tutorial.py`` by exercising more ops:
  - ``ark.tensor``, ``ark.ones``, ``ark.zeros``, ``ark.constant``
  - ``ark.add``, ``ark.sub``, ``ark.mul``, ``ark.div``
  - ``ark.exp``, ``ark.sqrt``, ``ark.rsqrt``
  - ``ark.relu``, ``ark.gelu``, ``ark.sigmoid``
  - ``ark.reduce_sum``, ``ark.reduce_mean``, ``ark.reduce_max``
  - ``ark.layernorm``
  - ``ark.reshape``, ``ark.transpose``

Each op result is validated against a NumPy reference.

Run: ``python examples/tutorial/tensor_ops_tutorial.py``
"""

import math

import numpy as np
import ark


def tensor_ops_tutorial():
    ark.init()

    M, N = 64, 128

    # ---- Tensor creation ----
    x = ark.tensor([M, N], ark.fp32)
    y = ark.tensor([M, N], ark.fp32)
    one = ark.ones([M, N], ark.fp32)
    zero = ark.zeros([M, N], ark.fp32)
    c5 = ark.constant(5.0, [M, N], ark.fp32)

    # ---- Elementwise arithmetic ----
    a_add = ark.add(x, y)
    a_sub = ark.sub(x, y)
    a_mul = ark.mul(x, y)
    a_div = ark.div(x, ark.add(y, one))  # avoid div-by-zero

    # ---- Unary math ----
    a_exp = ark.exp(x)
    a_sqrt = ark.sqrt(ark.add(x, c5))  # shift to positive
    a_rsqrt = ark.rsqrt(ark.add(x, c5))

    # ---- Activations ----
    a_relu = ark.relu(x)
    a_gelu = ark.gelu(x)
    a_sig = ark.sigmoid(x)

    # ---- Reductions ----
    r_sum = ark.reduce_sum(x, axis=-1)  # [M, 1]
    r_mean = ark.reduce_mean(x, axis=-1)
    r_max = ark.reduce_max(x, axis=-1)

    # ---- Layernorm ----
    a_ln = ark.layernorm(x)

    # ---- Reshape / transpose ----
    a_reshape = ark.reshape(x, [M * N])
    a_trans = ark.transpose(
        ark.reshape(x, [M, N // 2, 2]), [0, 2, 1]
    )  # [M, 2, N//2]

    # ---- Identity (constant tensors) ----
    a_one = ark.add(one, zero)  # should be all ones

    # ---- Launch runtime ----
    runtime = ark.Runtime()
    runtime.launch()

    rng = np.random.RandomState(0)
    x_np = (rng.randn(M, N) * 0.5).astype(np.float32)
    y_np = (rng.randn(M, N) * 0.5).astype(np.float32)

    x.from_numpy(x_np)
    y.from_numpy(y_np)

    runtime.run()

    # ---- Validate ----
    def check(name, ark_tensor, expected, atol=1e-5):
        got = ark_tensor.to_numpy()
        np.testing.assert_allclose(
            got, expected, atol=atol, rtol=1e-4, err_msg=name
        )

    # Elementwise
    check("add", a_add, x_np + y_np)
    check("sub", a_sub, x_np - y_np)
    check("mul", a_mul, x_np * y_np)
    check("div", a_div, x_np / (y_np + 1.0))

    # Unary
    check("exp", a_exp, np.exp(x_np), atol=1e-4)
    check("sqrt", a_sqrt, np.sqrt(x_np + 5.0), atol=1e-5)
    check("rsqrt", a_rsqrt, 1.0 / np.sqrt(x_np + 5.0), atol=1e-5)

    # Activations
    check("relu", a_relu, np.maximum(x_np, 0))

    # GELU: approximate check (ARK uses erff-based GELU)

    gelu_ref = 0.5 * x_np * (1 + np.vectorize(math.erf)(x_np / np.sqrt(2)))
    check("gelu", a_gelu, gelu_ref, atol=1e-4)

    sig_ref = 1.0 / (1.0 + np.exp(-x_np))
    check("sigmoid", a_sig, sig_ref, atol=1e-5)

    # Reductions (keepdims=True by default → [M, 1])
    check("reduce_sum", r_sum, x_np.sum(axis=-1, keepdims=True))
    check(
        "reduce_mean",
        r_mean,
        x_np.mean(axis=-1, keepdims=True),
        atol=1e-4,
    )
    check("reduce_max", r_max, x_np.max(axis=-1, keepdims=True))

    # Layernorm
    mu = x_np.mean(axis=-1, keepdims=True)
    var = ((x_np - mu) ** 2).mean(axis=-1, keepdims=True)
    ln_ref = (x_np - mu) / np.sqrt(var + 1e-6)
    check("layernorm", a_ln, ln_ref, atol=1e-4)

    # Reshape / transpose
    check("reshape", a_reshape, x_np.reshape(M * N))
    check(
        "transpose",
        a_trans,
        x_np.reshape(M, N // 2, 2).transpose(0, 2, 1),
    )

    # Constant tensors
    check("ones", a_one, np.ones((M, N), dtype=np.float32))

    print("Tensor ops tutorial passed — all ops validated!")


if __name__ == "__main__":
    tensor_ops_tutorial()
