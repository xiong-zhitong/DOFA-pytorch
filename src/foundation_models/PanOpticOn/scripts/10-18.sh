
# PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=10 dinov2/train/train.py \
#     --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/lr_det.yaml \
#     --output-dir=/data/panopticon/logs/dino_logs/simplattn/debug/det \
#     optim.base_lr=1e-3 \
#     optim.epochs=2 \
#     add_args=false

# PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=10 dinov2/train/train.py \
#     --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/lr_det_smasks.yaml \
#     --output-dir=/data/panopticon/logs/dino_logs/simplattn/debug/det_smask \
#     optim.base_lr=1e-3 \
#     optim.epochs=2 \
#     add_args=false

# PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=10 dinov2/train/train.py \
#     --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/lr_nodet.yaml \
#     --output-dir=/data/panopticon/logs/dino_logs/simplattn/debug/nodet \
#     optim.base_lr=1e-3 \
#     optim.epochs=2 \
#     add_args=false

# PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=10 dinov2/train/train.py \
#     --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/lr_nodet_smasks.yaml \
#     --output-dir=/data/panopticon/logs/dino_logs/simplattn/debug/nodet_smask \
#     optim.base_lr=1e-3 \
#     optim.epochs=2 \
#     add_args=false



# logn run
PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=20 dinov2/train/train.py \
    --config-file=/data/panopticon/logs/dino_logs/simplattn/s_lr/lr=1e-3_warmup=0_fw=backbone=0_inp=false/before_rules_config.yaml \
    --output-dir=/data/panopticon/logs/dino_logs/simplattn/long/lr=1e-3_warmup=0 \
    optim.epochs=60 \
    optim.scaling_rule=sqrt_wrt_1024 \
    add_args=false > /data/panopticon/logs/dino_logs/simplattn/long/lr=1e-3_warmup=0/sysout