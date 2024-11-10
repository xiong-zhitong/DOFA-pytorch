
# new lr
# for base_lr in 2e-3 5e-3
# do
#     for lrmul in 0.1 0.2 0.5 
#     do
#         PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=0 dinov2/train/train.py \
#             --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/04_sweep.yaml \
#             --output-dir=/data/panopticon/logs/dino_logs/sattn/s_lr_rchns/ \
#             optim.base_lr=${base_lr} \
#             optim.warmup_epochs=0 \
#             optim.lr_multiplier=backbone=${lrmul}
#     done
# done

# # 1e-3
# base_lr=1e-3
# for lrmul in 0.1 0.5
# do
#     PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=0 dinov2/train/train.py \
#         --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/04_sweep.yaml \
#         --output-dir=/data/panopticon/logs/dino_logs/sattn/s_lr_rchns/ \
#         optim.base_lr=${base_lr} \
#         optim.warmup_epochs=0 \
#         optim.lr_multiplier=backbone=${lrmul}
# done



# first check with MMEarth
# for base_lr in 1e-4 1e-3 5e-3
# do
#     for lrmul in 0.2
#     do
#         PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=3 dinov2/train/train.py \
#             --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/05_sweep.yaml \
#             --output-dir=/data/panopticon/logs/dino_logs/comb/s_lr/ \
#             optim.base_lr=${base_lr} \
#             optim.warmup_epochs=0 \
#             optim.lr_multiplier=backbone=${lrmul}
#     done
# done


# freeze pe
# base_lr=2e-3
# lrmul=0.2
# PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=0 dinov2/train/train.py \
#     --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/04_sweep.yaml \
#     --output-dir=/data/panopticon/logs/dino_logs/sattn/s_lr_rchns/ \
#     optim.base_lr=${base_lr} \
#     optim.warmup_epochs=0 \
#     optim.lr_multiplier=backbone=${lrmul} \
#     optim.freeze_weights='last_layer=1,chnattnblock.patch_emb=1000'


# # 1e-2 to test boundaries
# base_lr=1e-2
# for lrmul in 0.2
# do
#     for warmup in 0 10
#     do
#         PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=5 dinov2/train/train.py \
#             --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/04_sweep.yaml \
#             --output-dir=/data/panopticon/logs/dino_logs/sattn/s_lr_rchns/ \
#             optim.base_lr=${base_lr} \
#             optim.warmup_epochs=${warmup} \
#             optim.lr_multiplier=backbone=${lrmul}
#     done
# done


# # 1e-3 for completeness
# base_lr=1e-3
# lrmul=0.2
# PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=0 dinov2/train/train.py \
#     --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/04_sweep.yaml \
#     --output-dir=/data/panopticon/logs/dino_logs/sattn/s_lr_rchns/ \
#     optim.base_lr=${base_lr} \
#     optim.warmup_epochs=0 \
#     optim.lr_multiplier=backbone=${lrmul}


# long run MMEarth training
for base_lr in 1e-3
do
    for lrmul in 0.2
    do
        PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=10 dinov2/train/train.py \
            --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/train/05_combined.yaml \
            --output-dir=/data/panopticon/logs/dino_logs/comb/long/ \
            optim.base_lr=${base_lr} \
            optim.warmup_epochs=0 \
            optim.lr_multiplier=backbone=${lrmul}
    done
done