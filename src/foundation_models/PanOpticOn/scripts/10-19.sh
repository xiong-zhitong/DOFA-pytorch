# influence of shuffling -> seems to make no real difference!
# PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=10 dinov2/train/train.py \
#     --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/lr_det.yaml \
#     --output-dir=/data/panopticon/logs/dino_logs/simplattn/det \
#     optim.base_lr=1e-3 \
#     optim.epochs=5 \
#     add_args=false

# PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=10 dinov2/train/train.py \
#     --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/lr_det_smasks.yaml \
#     --output-dir=/data/panopticon/logs/dino_logs/simplattn/det_smask \
#     optim.base_lr=1e-3 \
#     optim.epochs=5 \
#     add_args=false

# PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=10 dinov2/train/train.py \
#     --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/lr_nodet.yaml \
#     --output-dir=/data/panopticon/logs/dino_logs/simplattn/nodet \
#     optim.base_lr=1e-3 \
#     optim.epochs=5 \
#     add_args=false

# PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=10 dinov2/train/train.py \
#     --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/lr_nodet_smasks.yaml \
#     --output-dir=/data/panopticon/logs/dino_logs/simplattn/nodet_smask \
#     optim.base_lr=1e-3 \
#     optim.epochs=5 \
#     add_args=false


# long run with shuffling 
# PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=50 dinov2/train/train.py \
#     --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/train/04_randchns.yaml \
#     --output-dir=/data/panopticon/logs/dino_logs/simplattn/long_rchns/base_lr=1e-3_warmup=0 \
#     add_args=false

# PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=0 dinov2/train/train.py \
#     --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/train/04_randchns.yaml \
#     --output-dir=/data/panopticon/logs/dino_logs/simplattn/long_rchns/ \
#     optim.base_lr=1e-3 \
#     optim.warmup_epochs=0 \
#     optim.freeze_weights='last_layer=1,backbone=10'

PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=0 dinov2/train/train.py \
    --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/train/04_randchns.yaml \
    --output-dir=/data/panopticon/logs/dino_logs/simplattn/long_rchns/ \
    optim.base_lr=1e-3 \
    optim.warmup_epochs=0 \
    optim.freeze_weights='last_layer=1,backbone=10' \
    optim.lr_multiplier='backbone=0.1'