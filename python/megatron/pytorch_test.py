import torch
import torch.distributed as dist

# initialize the process group
dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

# create local tensors on each process
rank = dist.get_rank()
size = dist.get_world_size()
tensor = torch.ones(3).to(rank) * (rank + 1)

print(f"Before all_reduce: Rank {rank}, tensor = {tensor}")

# perform all-reduce on tensor
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

print(f"After all_reduce: Rank {rank}, tensor = {tensor}")

# destroy the process group
dist.destroy_process_group()
