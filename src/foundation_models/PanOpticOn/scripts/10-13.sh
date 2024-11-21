
# freeze backbone
warmup=0
for lr in 1e-4 5e-4 1e-3
do
    for pelrmul in 1.0
    do
        PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=10 dinov2/train/train.py --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/wo.yaml --output_dir=/data/panopticon/logs/dino_logs/s_fixedchns/lr2/freezebb/lr=${lr}_warmup=${warmup}_pelrmul=${pelrmul} optim.base_lr=${lr} optim.warmup_epochs=${warmup} optim.patch_embed_lr_mult=${pelrmul} optim.scaling_rule=sqrt_wrt_1024,find_lr freeze_weights=backbone=1000
    done
done



# load dinov2 heads
warmup=0
for lr in 1e-4 5e-4 1e-3
do
    for pelrmul in 2.0 5.0
    do
        PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=10 dinov2/train/train.py --config-file=//home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/wo_wdinorgbheads.yaml --output_dir=/data/panopticon/logs/dino_logs/s_fixedchns/lr2/dinohead/lr=${lr}_warmup=${warmup}_pelrmul=${pelrmul} optim.base_lr=${lr} optim.warmup_epochs=${warmup} optim.patch_embed_lr_mult=${pelrmul} optim.scaling_rule=sqrt_wrt_1024,find_lr 
    done
done

