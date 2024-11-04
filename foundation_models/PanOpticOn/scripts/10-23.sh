

# only mmearth
# base_lr=1e-3
# lrmul=0.2
# warmup=3
# PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=50 dinov2/train/train.py \
#     --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/train/06_only_mmearth.yaml \
#     --output-dir=/data/panopticon/logs/dino_logs/comb/mmearth/ \
#     optim.base_lr=${base_lr} \
#     optim.warmup_epochs=${warmup} \
#     optim.lr_multiplier=backbone=${lrmul}


# only mmearth sar
# base_lr=1e-3
# lrmul=0.2
# warmup=3
# PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=5 dinov2/train/train.py \
#     --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/train/06_only_mmearth.yaml \
#     --output-dir=/data/panopticon/logs/dino_logs/comb/mmearth/ \
#     optim.base_lr=${base_lr} \
#     optim.warmup_epochs=${warmup} \
#     optim.lr_multiplier=backbone=${lrmul} \
#     train.dataset.modalities=MODALITY_S1

# only mmearth optical
# base_lr=1e-3
# lrmul=0.2
# warmup=3
# PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=10 dinov2/train/train.py \
#     --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/train/06_only_mmearth.yaml \
#     --output-dir=/data/panopticon/logs/dino_logs/comb/mmearth/ \
#     optim.base_lr=${base_lr} \
#     optim.warmup_epochs=${warmup} \
#     optim.lr_multiplier=backbone=${lrmul} \
#     train.dataset.modalities=MODALITY_S2

############# FROM HERE

# fmow_qb_s2_15k
base_lr=1e-3
lrmul=0.2
warmup=0
PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=10 dinov2/train/train.py \
    --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/train/04_randchns.yaml \
    --output-dir=/data/panopticon/logs/dino_logs/ds/fmow_qb_s2_15k/ \
    optim.base_lr=${base_lr} \
    optim.warmup_epochs=${warmup} \
    optim.lr_multiplier=backbone=${lrmul} \
    train.dataset.subset=15000 \
    train.epochs=20

env=/home/lewaldm/code/PanOpticOn/dinov2/configs/envs/yamal.yaml




# fmow_qb_s2_30k
base_lr=1e-3
lrmul=0.2
warmup=3
PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=10 dinov2/train/train.py \
    --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/train/04_randchns.yaml \
    --output-dir=/data/panopticon/logs/dino_logs/ds/fmow_qb_s2_30k/ --env=${env} \
    optim.base_lr=${base_lr} \
    optim.warmup_epochs=${warmup} \
    optim.lr_multiplier=backbone=${lrmul} \
    train.epochs=20


