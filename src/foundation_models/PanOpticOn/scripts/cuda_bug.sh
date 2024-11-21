
# the following produces the bug!!
# subset=10
# batch_size_per_gpu=10
# num_workers=1
# pin_memory=True
# epochs=2
# OFFICIAL_EPOCH_LENGTH=5
# PYTHONPATH=. python dinov2/train/train.py --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/pedri/cuda_bug.yaml --output-dir=/data/panopticon/logs/dino_logs/debug/cuda_bug/batch_size=${batch_size_per_gpu}_num_workers=${num_workers}_pin_memory=${pin_memory}_subset=${subset}_OFFICIAL_EPOCH_LENGTH=${OFFICIAL_EPOCH_LENGTH} train.batch_size_per_gpu=${batch_size_per_gpu} train.num_workerss=${num_workers} train.pin-memory=${pin_memory} train.dataset.subset=${subset} optim.epochs=${epochs} train.OFFICIAL_EPOCH_LENGTH=${OFFICIAL_EPOCH_LENGTH} 

# this also generates the bug!
PYTHONPATH=. python dinov2/train/train.py --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/pedri/cuda_bug_proven.yaml --output-dir=/data/panopticon/logs/dino_logs/debug/1


# *****************

# subset=10
# batch_size_per_gpu=10
# num_workers=1
# pin_memory=True
# epochs=2
# OFFICIAL_EPOCH_LENGTH=5
# PYTHONPATH=. python dinov2/train/train.py --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/pedri/cuda_bug_proven.yaml --output-dir=/data/panopticon/logs/dino_logs/debug/1/batch_size=${batch_size_per_gpu}_num_workers=${num_workers}_pin_memory=${pin_memory}_subset=${subset}_OFFICIAL_EPOCH_LENGTH=${OFFICIAL_EPOCH_LENGTH} train.batch_size_per_gpu=${batch_size_per_gpu} train.num_workerss=${num_workers} train.pin-memory=${pin_memory} train.dataset.subset=${subset} optim.epochs=${epochs} train.OFFICIAL_EPOCH_LENGTH=${OFFICIAL_EPOCH_LENGTH} 

# subset=80
# batch_size_per_gpu=80
# num_workers=1
# pin_memory=True
# epochs=2
# OFFICIAL_EPOCH_LENGTH=15
# PYTHONPATH=. python dinov2/train/train.py --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/pedri/cuda_bug.yaml --output-dir=/data/panopticon/logs/dino_logs/debug/cuda_bug/batch_size=${batch_size_per_gpu}_num_workers=${num_workers}_pin_memory=${pin_memory}_subset=${subset}_OFFICIAL_EPOCH_LENGTH=${OFFICIAL_EPOCH_LENGTH} train.batch_size_per_gpu=${batch_size_per_gpu} train.num_workerss=${num_workers} train.pin-memory=${pin_memory} train.dataset.subset=${subset} optim.epochs=${epochs} train.OFFICIAL_EPOCH_LENGTH=${OFFICIAL_EPOCH_LENGTH} 

# subset=80
# batch_size_per_gpu=80
# num_workers=8
# pin_memory=True
# epochs=2
# OFFICIAL_EPOCH_LENGTH=3
# PYTHONPATH=. python dinov2/train/train.py --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/pedri/cuda_bug.yaml --output-dir=/data/panopticon/logs/dino_logs/debug/cuda_bug/batch_size=${batch_size_per_gpu}_num_workers=${num_workers}_pin_memory=${pin_memory}_subset=${subset}_OFFICIAL_EPOCH_LENGTH=${OFFICIAL_EPOCH_LENGTH} train.batch_size_per_gpu=${batch_size_per_gpu} train.num_workerss=${num_workers} train.pin-memory=${pin_memory} train.dataset.subset=${subset} optim.epochs=${epochs} train.OFFICIAL_EPOCH_LENGTH=${OFFICIAL_EPOCH_LENGTH} 

# subset=80
# batch_size_per_gpu=80
# num_workers=8
# pin_memory=True
# epochs=2
# OFFICIAL_EPOCH_LENGTH=5
# PYTHONPATH=. python dinov2/train/train.py --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/pedri/cuda_bug.yaml --output-dir=/data/panopticon/logs/dino_logs/debug/cuda_bug/batch_size=${batch_size_per_gpu}_num_workers=${num_workers}_pin_memory=${pin_memory}_subset=${subset}_OFFICIAL_EPOCH_LENGTH=${OFFICIAL_EPOCH_LENGTH} train.batch_size_per_gpu=${batch_size_per_gpu} train.num_workerss=${num_workers} train.pin-memory=${pin_memory} train.dataset.subset=${subset} optim.epochs=${epochs} train.OFFICIAL_EPOCH_LENGTH=${OFFICIAL_EPOCH_LENGTH} 

# subset=80
# batch_size_per_gpu=80
# num_workers=8
# pin_memory=True
# epochs=2
# OFFICIAL_EPOCH_LENGTH=15
# PYTHONPATH=. python dinov2/train/train.py --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/pedri/cuda_bug.yaml --output-dir=/data/panopticon/logs/dino_logs/debug/cuda_bug/batch_size=${batch_size_per_gpu}_num_workers=${num_workers}_pin_memory=${pin_memory}_subset=${subset}_OFFICIAL_EPOCH_LENGTH=${OFFICIAL_EPOCH_LENGTH} train.batch_size_per_gpu=${batch_size_per_gpu} train.num_workerss=${num_workers} train.pin-memory=${pin_memory} train.dataset.subset=${subset} optim.epochs=${epochs} train.OFFICIAL_EPOCH_LENGTH=${OFFICIAL_EPOCH_LENGTH} 
