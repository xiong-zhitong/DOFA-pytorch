
# v1
# for lr in 1e-4 1e-3 5e-3
# do
#     for warmup in 0 8 15
#     do
#         for pelrmul in 0.5 1.0 2.0
#         do
#             PYTHONPATH=. torchrun --nproc_per_node=2 dinov2/train/train.py --config-file=dinov2/configs/train/02_fixedChns.yaml --output_dir=/data/panopticon/logs/dino_logs/s_fixedchns/s_fixedchns_lr=${lr}_warmup=${warmup}_pelrmul=${pelrmul} optim.base_lr=${lr} optim.warmup_epochs=${warmup} optim.patch_embed_lr_mult=${pelrmul}
#         done
#     done
# done


# v2: find minimal unstable lr
warmup=0
prefix=s_fixedchnslr
for lr in 1e-4 5e-4 1e-3
do
    for pelrmul in 0.5 1.0 2.0
    do
        PYTHONPATH=. torchrun --nproc_per_node=2 dinov2/train/train.py --config-file=dinov2/configs/train/02_fixedChns.yaml --output_dir=/data/panopticon/logs/dino_logs/s_fixedchns/lr/${prefix}_lr=${lr}_warmup=${warmup}_pelrmul=${pelrmul} optim.base_lr=${lr} optim.warmup_epochs=${warmup} optim.patch_embed_lr_mult=${pelrmul}
    done
done