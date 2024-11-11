


pelrmul=1.0
for warmup in 0 8
do 
    for freeze_weights in backbone=1000 backbone=0
    do
        for lr in 1e-4 1e-3 1e-5
        do
            PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=2 dinov2/train/train.py --config-file=dinov2/configs/sweep/lr.yaml --output_dir=/data/panopticon/logs/dino_logs/s_fixedchns/lr/s_fchnlr=${lr}_warmup=${warmup}_pelrmul=${pelrmul}_fw=${freeze_weights} optim.base_lr=${lr} optim.warmup_epochs=${warmup} optim.patch_embed_lr_mult=${pelrmul} freeze_weights=${freeze_weights}
        done
    done
done