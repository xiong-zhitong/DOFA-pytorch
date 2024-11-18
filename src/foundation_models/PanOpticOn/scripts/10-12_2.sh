# which scheduler causes performance drop

config='/data/panopticon/logs/dino_logs/s_fixedchns/lr2/rm=all/config.yaml'
output_dir=/data/panopticon/logs/dino_logs/s_fixedchns/lr2


# lr=1e-3
# warmup=12
# epochs=14
# PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=3 dinov2/train/train.py --config-file=$config --output_dir=$output_dir/lr=${lr}_warmup=${warmup} optim.scaling_rule=sqrt_wrt_1024,find_lr optim.base_lr=${lr} optim.warmup_epochs=${warmup} optim.epochs=${epochs}

# lr=1e-4
# warmup=6
# epochs=8
# pelrmul=5.0
# PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=3 dinov2/train/train.py --config-file=$config --output_dir=$output_dir/lr=${lr}_warmup=${warmup}_pelrmul=${pelrmul} optim.scaling_rule=sqrt_wrt_1024,find_lr optim.base_lr=${lr} optim.warmup_epochs=${warmup} optim.epochs=${epochs} optim.patch_embed_lr_mult=${pelrmul}

lr=1e-3
warmup=12
epochs=60
PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=10 dinov2/train/train.py --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/wo.yaml --output_dir=$output_dir/long_lr=${lr}_warmup=${warmup} optim.scaling_rule=sqrt_wrt_1024 optim.base_lr=${lr} optim.warmup_epochs=${warmup} optim.epochs=${epochs}
