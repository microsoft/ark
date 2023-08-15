# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark

import numpy as np
import unittest


def test_matmul_internal(
    m,
    n,
    k,
    bs_a,
    bs_b,
    split_k,
    trans_input=False,
    trans_other=False,
    gran_lev=-1,
    iter=1,
    data_type="float",
):
    runtime = ark.Runtime()
    if data_type == "float":
        ark_data_type = ark.TensorType.FP32
        numpy_data_type = np.float32
    elif data_type == "half":
        ark_data_type = ark.TensorType.FP16
        numpy_data_type = np.float16
    if trans_input:
        input_shape = [bs_a, k, m]
    else:
        input_shape = [bs_a, m, k]
    if trans_other:
        other_shape = [bs_b, n, k]
    else:
        other_shape = [bs_b, k, n]
    input_tensor = ark.tensor(input_shape, ark_data_type)
    other_tensor = ark.tensor(other_shape, ark_data_type)

    output_tensor = ark.matmul(
        input_tensor,
        other_tensor,
        None,
        split_k,
        trans_input,
        trans_other,
        "matmul",
        gran_lev,
    )
    runtime.launch()
    input_tensor_host = np.random.rand(*input_shape).astype(numpy_data_type)
    other_tensor_host = np.random.rand(*other_shape).astype(numpy_data_type)
    input_tensor.from_numpy(input_tensor_host)
    other_tensor.from_numpy(other_tensor_host)

    runtime.run(iter, async_run=True)

    elapsed = runtime.stop()

    output_tensor_host = output_tensor.to_numpy()

    if trans_input:
        input_tensor_host = np.transpose(input_tensor_host, (0, 2, 1))
    if trans_other:
        other_tensor_host = np.transpose(other_tensor_host, (0, 2, 1))

    gt = np.matmul(input_tensor_host, other_tensor_host)

    # test if the result is correct
    max_abs_error = np.max(np.abs(output_tensor_host - gt))
    mean_abs_error = np.mean(np.abs(output_tensor_host - gt))
    numeric_epsilon_half = np.finfo(np.float16).eps
    atol = 2 * numeric_epsilon_half * k
    np.testing.assert_allclose(output_tensor_host, gt, atol=atol)

    print(
        f"matmul test: data_type {data_type} bs_a {bs_a:6d} bs_b {bs_b:6d} "
        f"m {m:6d} n {n:6d} k {k:6d} (split_k={split_k}, gran_lev={gran_lev}) "
        f"trans_input {trans_input} trans_other {trans_other}"
        f"max_abs_error {max_abs_error:.5f} mse {mean_abs_error:.5f} elapsed "
        f"{elapsed:.5f} ms iter {iter} elapsed_per_iter {elapsed / iter:.5f} ms"
    )
    return True


# Test the correctness of matmul at small scale
def test_matmul_small_sizes(
    split_k, trans_input, trans_other, gran_lev, type_str="half", iter=1
):
    test_matmul_internal(
        64,
        64,
        32,
        1,
        1,
        split_k,
        trans_input,
        trans_other,
        gran_lev,
        iter,
        type_str,
    )
    test_matmul_internal(
        128,
        64,
        32,
        1,
        1,
        split_k,
        trans_input,
        trans_other,
        gran_lev,
        iter,
        type_str,
    )
    test_matmul_internal(
        64,
        128,
        32,
        1,
        1,
        split_k,
        trans_input,
        trans_other,
        gran_lev,
        iter,
        type_str,
    )
    test_matmul_internal(
        128,
        128,
        32,
        1,
        1,
        split_k,
        trans_input,
        trans_other,
        gran_lev,
        iter,
        type_str,
    )

    test_matmul_internal(
        64,
        64,
        64,
        1,
        1,
        split_k,
        trans_input,
        trans_other,
        gran_lev,
        iter,
        type_str,
    )
    test_matmul_internal(
        128,
        64,
        64,
        1,
        1,
        split_k,
        trans_input,
        trans_other,
        gran_lev,
        iter,
        type_str,
    )
    test_matmul_internal(
        64,
        128,
        64,
        1,
        1,
        split_k,
        trans_input,
        trans_other,
        gran_lev,
        iter,
        type_str,
    )
    test_matmul_internal(
        128,
        128,
        64,
        1,
        1,
        split_k,
        trans_input,
        trans_other,
        gran_lev,
        iter,
        type_str,
    )
    test_matmul_internal(
        256,
        128,
        64,
        1,
        1,
        split_k,
        trans_input,
        trans_other,
        gran_lev,
        iter,
        type_str,
    )

    test_matmul_internal(
        128,
        128,
        256,
        1,
        1,
        split_k,
        trans_input,
        trans_other,
        gran_lev,
        iter,
        type_str,
    )


class TestMatmul(unittest.TestCase):
    def test_matmul_gran_half(self):
        for gran_lev in range(0, 3):
            print("test_matmul_gran gran_lev=", gran_lev)
            test_matmul_small_sizes(1, False, False, gran_lev, "half")

    def test_matmul_split_half(self):
        print("test_matmul_split")
        for split_k in range(3, 8):
            for gran_lev in range(0, 3):
                test_matmul_internal(
                    128,
                    4096,
                    1024,
                    1,
                    1,
                    split_k,
                    False,
                    False,
                    gran_lev,
                    1,
                    "half",
                )

    def test_matmul_gran_float(self):
        for gran_lev in range(0, 3):
            print("test_matmul_gran gran_lev=", gran_lev)
            test_matmul_small_sizes(1, False, False, gran_lev, "float")

    def test_matmul_split_float(self):
        print("test_matmul_split")
        for split_k in range(3, 8):
            for gran_lev in range(0, 3):
                test_matmul_internal(
                    128,
                    4096,
                    1024,
                    1,
                    1,
                    split_k,
                    False,
                    False,
                    gran_lev,
                    1,
                    "float",
                )

    def test_matmul_transpose(self):
        test_matmul_small_sizes(1, False, False, 0, "half")
        test_matmul_small_sizes(1, True, True, 0, "half")
        test_matmul_small_sizes(1, False, True, 0, "half")
        test_matmul_small_sizes(1, True, False, 0, "half")
        test_matmul_small_sizes(1, False, False, 0, "float")
        test_matmul_small_sizes(1, True, True, 0, "float")
        test_matmul_small_sizes(1, False, True, 0, "float")
        test_matmul_small_sizes(1, True, False, 0, "float")


if __name__ == "__main__":
    unittest.main()
