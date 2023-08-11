import ark
import numpy as np
import multiprocessing as mp
import time

C_DIM = 8192


def main(rank, mat_b_data):
    rt = ark.Runtime(rank=rank, world_size=2)

    if rank == 0:
        mat_a = ark.tensor([256, C_DIM], ark.FP16)
        mat_b = ark.tensor([C_DIM, C_DIM], ark.FP16)

        mat_a_part0, mat_a_part1 = ark.sharding(
            mat_a, axis=0, dim_per_shard=128
        )
        assert mat_a_part0.shape == [128, C_DIM]
        assert mat_a_part1.shape == [128, C_DIM]

        send_tensor = ark.send_mscclpp(
            mat_a_part1, sid=0, dst_rank=1, bytes=mat_a_part1.shape_bytes()
        )
        mat_a_part0 = ark.identity(mat_a_part0, [send_tensor])

        output = ark.tensor([256, C_DIM], ark.FP16)
        output_part0, output_part1 = ark.sharding(
            output, axis=0, dim_per_shard=128
        )

        matmul_part0 = ark.matmul(mat_a_part0, mat_b, output=output_part0)
        output_part1 = ark.identity(output_part1, [matmul_part0])

        recv_tensor = ark.recv_mscclpp(
            output_part1, sid=1, src_rank=1, bytes=output_part1.shape_bytes()
        )
        send_tensor = ark.identity(send_tensor, [recv_tensor])
        send_done_tensor = ark.send_done_mscclpp(send_tensor, dst_rank=1)

        mat_c = ark.matmul(mat_a, mat_b)
    else:
        mat_a_part1 = ark.tensor([128, C_DIM], ark.FP16)
        mat_b = ark.tensor([C_DIM, C_DIM], ark.FP16)

        recv_tensor = ark.recv_mscclpp(
            mat_a_part1, sid=0, src_rank=0, bytes=mat_a_part1.shape_bytes()
        )
        mat_a_part1 = ark.identity(mat_a_part1, [recv_tensor])
        matmul_part1 = ark.matmul(mat_a_part1, mat_b)

        send_tensor = ark.send_mscclpp(
            matmul_part1, sid=1, dst_rank=0, bytes=matmul_part1.shape_bytes()
        )
        send_done_tensor = ark.send_done_mscclpp(send_tensor, dst_rank=0)

    rt.launch()

    if rank == 0:
        mat_a_data = np.random.uniform(
            low=-1.0, high=1.0, size=(256, C_DIM)
        ).astype(np.float16)
        mat_a.from_numpy(mat_a_data)
    mat_b.from_numpy(mat_b_data)

    rt.run(1)

    print(f"rank {rank} done")

    if rank == 0:
        result = output.to_numpy()
        ground_truth = mat_c.to_numpy()
        numeric_epsilon = np.finfo(np.float16).eps
        atol = 2 * numeric_epsilon * C_DIM

        for i in range(256):
            try:
                np.testing.assert_allclose(
                    result[i], ground_truth[i], rtol=0.01, atol=atol
                )
            except AssertionError as e:
                print(f"i={i}")
                print(mat_a_part1.to_numpy())
                raise

        print("correctness check passed")


if __name__ == "__main__":
    ark.init()

    mat_b_data = np.random.uniform(
        low=-1.0, high=1.0, size=(C_DIM, C_DIM)
    ).astype(np.float16)
    proc0 = mp.Process(target=main, args=(0, mat_b_data))
    proc1 = mp.Process(target=main, args=(1, mat_b_data))
    proc0.start()
    proc1.start()
    proc0.join()
    proc1.join()
