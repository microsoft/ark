import torch  
import torch.distributed as dist  
  
# 初始化分布式进程组  
dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')  
  
# 创建一个张量并将其放在当前设备上  
rank = dist.get_rank()  
size = dist.get_world_size()  
tensor = torch.ones(3).to(rank) * (rank + 1)  
  
print(f"Before all_reduce: Rank {rank}, tensor = {tensor}")  
  
# 执行AllReduce操作  
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)  
  
print(f"After all_reduce: Rank {rank}, tensor = {tensor}")  
  
# 结束分布式进程组  
dist.destroy_process_group()  
