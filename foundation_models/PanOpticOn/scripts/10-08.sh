
# Sweep lr=5e-5 over pelrmul
# warmup=0
# prefix=s_fixedchnslr
# for lr in 5e-5
# do
#     for pelrmul in 0.5 1.0 2.0
#     do
#         PYTHONPATH=. torchrun --nproc_per_node=2 dinov2/train/train.py --config-file=dinov2/configs/train/02_fixedChns.yaml --output_dir=/data/panopticon/logs/dino_logs/s_fixedchns/lr/${prefix}_lr=${lr}_warmup=${warmup}_pelrmul=${pelrmul} optim.base_lr=${lr} optim.warmup_epochs=${warmup} optim.patch_embed_lr_mult=${pelrmul}
#     done
# done


# # Warmup sweep 1e-3
# for lr in 1e-3
# do
#     for warmup in 8
#     do
#         for pelrmul in 1.0 2.0
#         do
#             PYTHONPATH=. torchrun --nproc_per_node=2 dinov2/train/train.py --config-file=dinov2/configs/train/02_fixedChns.yaml --output_dir=/data/panopticon/logs/dino_logs/s_fixedchns/s_fixedchns_lr=${lr}_warmup=${warmup}_pelrmul=${pelrmul} optim.base_lr=${lr} optim.warmup_epochs=${warmup} optim.patch_embed_lr_mult=${pelrmul} optim.epochs=10
#         done
#     done
# done

# Long run 1e-4 without warmup
warmup=0
lr=1e-4
pelrmul=2.0
PYTHONPATH=. torchrun --nproc_per_node=2 --max-restarts=3 dinov2/train/train.py --config-file=dinov2/configs/train/03_fixedChns_long.yaml --output_dir=/data/panopticon/logs/dino_logs/s_fixedchns/long_lr=${lr}_warmup=${warmup}_pelrmul=${pelrmul} optim.base_lr=${lr} optim.warmup_epochs=${warmup} optim.patch_embed_lr_mult=${pelrmul}


# Sweep lr=1e-4 over pelrmul even larger
# warmup=0
# prefix=s_fixedchnslr
# for lr in 1e-4
# do
#     for pelrmul in 3.0 5.0
#     do
#         PYTHONPATH=. torchrun --nproc_per_node=2 dinov2/train/train.py --config-file=dinov2/configs/train/02_fixedChns.yaml --output_dir=/data/panopticon/logs/dino_logs/s_fixedchns/lr/${prefix}_lr=${lr}_warmup=${warmup}_pelrmul=${pelrmul} optim.base_lr=${lr} optim.warmup_epochs=${warmup} optim.patch_embed_lr_mult=${pelrmul}
#     done
# done