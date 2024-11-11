# redo freeze
# base_lr=2e-3
# lrmul=0.2
# PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=0 dinov2/train/train.py \
#     --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/04_sweep.yaml \
#     --output-dir=/data/panopticon/logs/dino_logs/sattn/s_lr_rchns/ \
#     optim.base_lr=${base_lr} \
#     optim.warmup_epochs=0 \
#     optim.lr_multiplier=backbone=${lrmul} \
#     optim.freeze_weights='last_layer=1,patch_embed.patch_emb.=1000'


# 1e-3 with some warmup on combined dataset


# debug
base_lr=1e-3
lrmul=0.2
warmup=3
PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=50 dinov2/train/train.py \
    --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/train/05_combined.yaml \
    --output-dir=/data/panopticon/logs/dino_logs/comb/long/ \
    optim.base_lr=${base_lr} \
    optim.warmup_epochs=${warmup} \
    optim.lr_multiplier=backbone=${lrmul}
