import pandas as pd
import os
from tifffile import imread
import multiprocessing as mp
import numpy as np


def _check_exists(df):
    corrupt = []
    for paths in df['path']:
        for p in paths:
            if not os.path.exists(os.path.join(root, p)):
                corrupt.append(p)
    return corrupt

def _try_load(df):
    corrupt = []
    for paths in df['path']:
        for p in paths:
            try:
                imread(os.path.join(root, p))
            except:
                corrupt.append(p)
    return corrupt


df = pd.read_parquet('/data/panopticon/datasets/fmow/metadata_v2/fmow_iwm_onid_train_val_presorted.parquet')
root = '/data/panopticon/datasets'
num_workers = 64

# df = df.iloc[:10]

pool = mp.Pool(num_workers)
tasks = [[] for _ in range(num_workers)]
idxs = np.array_split(np.arange(len(df)), num_workers)
for i, idx in enumerate(idxs):
    tasks[i] = pool.apply_async(_try_load, args=(df.iloc[idx],))
pool.close()
pool.join()
tasks = [t.get() for t in tasks]
print(tasks)