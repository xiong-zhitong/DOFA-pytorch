
# chn att with correct freeze
config=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/wo.yaml
output_dir=/data/panopticon/logs/dino_logs/s_fixedchns/freezebb
envfile=/home/lewaldm/code/PanOpticOn/dinov2/configs/envs/yamal.yaml

for warmup in 0
do
    for lr in 1e-4 1e-5 1e-6
    do
        PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=10 dinov2/train/train.py --config-file=${config} --env=${envfile} --output-dir=${output_dir}/lr=${lr}_warmup=${warmup}_corr optim.base_lr=${lr} optim.freeze_weights=last_layer=1,backbone.block=1000,backbone.norm=1000 optim.lr_multiplier=patch_emb=1.0 optim.epochs=15 optim.warmup_epochs=${warmup}
    done
done


# chn att with fixed backbone & separate heads
config=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/wo.yaml
output_dir=/data/panopticon/logs/dino_logs/s_fixedchns/freezebb
envfile=/home/lewaldm/code/PanOpticOn/dinov2/configs/envs/yamal.yaml

for warmup in 0
do
    for lr in 1e-4 1e-5 1e-6
    do
        PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=10 dinov2/train/train.py --config-file=${config} --env=${envfile} --output-dir=${output_dir}/lr=${lr}_warmup=${warmup}_sephead optim.base_lr=${lr} optim.freeze_weights=last_layer=1,backbone.block=1000,backbone.norm=1000 optim.lr_multiplier=patch_emb=1.0 optim.epochs=15 optim.warmup_epochs=${warmup} ibot.separate_head=true
    done
done

# chn att with fixed backbone & ratio between patchemb and head
config=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/wo.yaml
output_dir=/data/panopticon/logs/dino_logs/s_fixedchns/freezebb
envfile=/home/lewaldm/code/PanOpticOn/dinov2/configs/envs/yamal.yaml

for warmup in 0
do
    for lr in 1e-4 1e-5 1e-6
    do
        PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=10 dinov2/train/train.py --config-file=${config} --env=${envfile} --output-dir=${output_dir}/lr=${lr}_warmup=${warmup}_patchembmul=0.2 optim.base_lr=${lr} optim.freeze_weights=last_layer=1,backbone.block=1000,backbone.norm=1000 optim.lr_multiplier=patch_emb=0.2 optim.epochs=15 optim.warmup_epochs=${warmup}
    done
done

