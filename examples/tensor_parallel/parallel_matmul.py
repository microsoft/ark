import ark
import numpy as np
import multiprocessing as mp
import time

C_DIM = 1024

def main(rank, mat_b_data):
    model = ark.Model(rank=rank)

    if rank == 0:
        mat_a = model.tensor(ark.Dims(128, C_DIM), ark.FP16)
        mat_b = model.tensor(ark.Dims(C_DIM, C_DIM), ark.FP16)

        mat_a_part0, mat_a_part1 = model.sharding(mat_a, axis=0, dim_per_shard=64)
        assert(mat_a_part0.shape == [64, C_DIM])
        assert(mat_a_part1.shape == [64, C_DIM])

        send_tensor = model.send(mat_a_part1, id=0, dst_rank=1, bytes=mat_a_part1.shape_bytes())
        mat_a_part0 = model.identity(mat_a_part0, [send_tensor])

        output = model.tensor(ark.Dims(128, C_DIM), ark.FP16)
        output_part0, output_part1 = model.sharding(output, axis=0, dim_per_shard=64)

        matmul_part0 = model.matmul(mat_a_part0, mat_b, output=output_part0)
        output_part1 = model.identity(output_part1, [matmul_part0])

        recv_tensor = model.recv(output_part1, id=1, src_rank=1, bytes=output_part1.shape_bytes())
    else:
        mat_a_part1 = model.tensor(ark.Dims(64, C_DIM), ark.FP16)
        mat_b = model.tensor(ark.Dims(C_DIM, C_DIM), ark.FP16)

        recv_tensor = model.recv(mat_a_part1, id=0, src_rank=0, bytes=mat_a_part1.shape_bytes())
        mat_a_part1 = model.identity(mat_a_part1, [recv_tensor])
        matmul_part1 = model.matmul(mat_a_part1, mat_b)

        send_tensor = model.send(matmul_part1, id=1, dst_rank=0, bytes=matmul_part1.shape_bytes())
        send_done_tensor = model.send_done(send_tensor, id=1, dst_rank=0)

    exe = ark.Executor(rank, rank, 2, model, "parallel_matmul")
    exe.compile()
    
    if rank == 0:
        mat_a_data = np.random.rand(128, C_DIM).astype(np.float16)
        exe.tensor_memcpy_host_to_device(mat_a, mat_a_data)
    exe.tensor_memcpy_host_to_device(mat_b, mat_b_data)

    exe.launch()
    exe.run(1)
    exe.stop()

    print(f"rank {rank} done")

    if rank == 0:
        result = np.zeros((128, C_DIM), dtype=np.float16)
        exe.tensor_memcpy_device_to_host(result, output)

        ground_truth = np.matmul(mat_a_data, mat_b_data)
        numeric_epsilon_half = np.finfo(np.float16).eps
        atol = 2 * numeric_epsilon_half * C_DIM
        np.testing.assert_allclose(result, ground_truth, atol=atol)

    time.sleep(5)


if __name__ == "__main__":
    ark.init()

    mat_b_data = np.random.rand(C_DIM, C_DIM).astype(np.float16)
    proc0 = mp.Process(target=main, args=(0, mat_b_data))
    proc1 = mp.Process(target=main, args=(1, mat_b_data))
    proc0.start()
    proc1.start()
    proc0.join()
    proc1.join()
