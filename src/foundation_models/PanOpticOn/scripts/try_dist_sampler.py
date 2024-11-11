from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch import distributed as dist
import torch
import time

dist.init_process_group(backend="nccl")
dist.barrier()

class TestDataset(Dataset):
    def __init__(self, n):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        return torch.tensor([idx])


ds = TestDataset(17)
sampler = DistributedSampler(ds, drop_last=True, shuffle=False)
dl = DataLoader(ds, batch_size=3, sampler=sampler, drop_last=True, shuffle=False)

if dist.get_rank() == 0:
    print(len(dl))
    for batch in dl:
        print(batch)

time.sleep(5)
if dist.get_rank() == 1:
    print(len(dl))
    for batch in dl:
        print(batch)