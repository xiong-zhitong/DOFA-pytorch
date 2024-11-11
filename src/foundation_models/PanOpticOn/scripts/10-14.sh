# separate heads
# config=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/wo.yaml
# output_dir=/data/panopticon/logs/dino_logs/s_fixedchns/lr2/sepheads

# for lr in 5e-4 1e-3 5e-5
# do
#     for lrmult in 0.5
#     do
#         PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=5 dinov2/train/train.py --config-file=${config} --output-dir=${output_dir}/lr=${lr}_lrmbb=${lrmult} ibot.separate_head=true optim.base_lr=${lr} optim.lr_multiplier=backbone=${lrmult}
#     done
# done    


# rgb with fixed backbone
# config=/data/panopticon/logs/dino_logs/dino_rgb/rgb_nopatchpos/before_rules_config.yaml
# output_dir=/data/panopticon/logs/dino_logs/dino_rgb/rgb_freezebb

# for lr in 1e-4 5e-4 1e-3 5e-5
# do
#     PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=5 dinov2/train/train.py --config-file=${config} --output-dir=${output_dir}/lr=${lr} optim.base_lr=${lr} optim.freeze_weights=last_layer=1,backbone=1000 optim.lr_multiplier=patch_emb=1.0 optim.scaling_rule=sqrt_wrt_1024,find_lr optim.epochs=10
# done    


# check p2p flag
# config=/data/panopticon/logs/dino_logs/dino_rgb/rgb_nopatchpos/before_rules_config.yaml
# output_dir=/data/panopticon/logs/dino_logs/debug/runtime
# lr=1e-4

# PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=0 dinov2/train/train.py --config-file=${config} --output-dir=${output_dir}/woflag optim.epochs=2 train.OFFICIAL_EPOCH_LENGTH=100 optim.warmup_epochs=0
# export NCCL_P2P_DISABLE=1
# PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=0 dinov2/train/train.py --config-file=${config} --output-dir=${output_dir}/wflag  optim.epochs=2 train.OFFICIAL_EPOCH_LENGTH=100 optim.warmup_epochs=0
    

# chn att with fixed backbone
config=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/wo.yaml
output_dir=/data/panopticon/logs/dino_logs/s_fixedchns/freezebb

for warmup in 0 10
do
    for lr in 1e-4 1e-5 1e-6
    do
        PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=10 dinov2/train/train.py --config-file=${config} --output-dir=${output_dir}/lr=${lr}_warmup=${warmup} optim.base_lr=${lr} optim.freeze_weights=last_layer=1,backbone=1000 optim.lr_multiplier=patch_emb=1.0 optim.epochs=15 optim.warmup_epochs=${warmup}
    done
done