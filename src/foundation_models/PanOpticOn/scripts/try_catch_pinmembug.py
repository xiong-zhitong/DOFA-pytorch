import torch
from time import sleep
from torch.utils.data import Dataset, DataLoader


class WaitDataset(Dataset):
    def __init__(self, t: int, n: int, shape = (3, 224, 224)):
        self.n = n
        self.t = t
        self.shape = shape

    def __len__(self):
        return self.n

    def __getitem__(self, idx) -> torch.Tensor:
        sleep(self.t)
        return torch.full(self.shape, idx, dtype=torch.int16)
    
def fun(x: torch.Tensor):
    x = x.cuda(non_blocking=True)
    sleep(iter_time)
    return x.sum()

ds_size = 60
bsz = 5
eval_period = 10000
data_time = 0.1
iter_time = 0.5

longdl = DataLoader(WaitDataset(data_time, ds_size), 
                    batch_size=bsz, 
                    num_workers=4,
                    pin_memory=True,
                    prefetch_factor=2)
print('len of dataloader', len(longdl))


def pprint(iterable):
    print('iterable',iterable._tasks_outstanding, iterable._rcvd_idx, iterable._send_idx)

iterable = iter(longdl)
pprint(iterable)
print('---')

i = 0
for x in iterable:
    fun(x)

    print('--',i)
    pprint(iterable)
    sleep(data_time + 0.05)
    pprint(iterable)
    sleep(data_time + 0.05)
    pprint(iterable)
    sleep(2)
    pprint(iterable)
    i += 1

    if i % eval_period == 0:
        shortdl = DataLoader(WaitDataset(0.3, int(ds_size/2)), 
                             batch_size=5, 
                             num_workers=2,
                             pin_memory=True)
        # shortdl.
        for j, y in enumerate(shortdl):
            print(f"  {j}")
            fun(x)