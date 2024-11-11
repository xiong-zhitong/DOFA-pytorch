#!/bin/bash
#SBATCH --job-name=pt
#SBATCH --account=fc_biosense
#SBATCH --partition=savio4_gpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A5000:4
#SBATCH --cpus-per-task=4

#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lewaldm@berkeley.edu


# ---------- SAVIO ------------
cd /global/home/users/lewaldm/code/PanOpticOn  # probably not necessary
export PYTHONPATH=.
export NCCL_P2P_DISABLE=1 # Need this on savio!
export $(cat /global/home/users/lewaldm/code/PanOpticOn/.env)

cmd='srun --export=ALL /global/home/users/lewaldm/.conda/envs/dinov2/bin/python /global/home/users/lewaldm/code/PanOpticOn/dinov2/train/train.py'
env=${CDIR}/envs/savio.yaml
# -----------------------------

# ---------- YAMAL ------------
# export PYTHONPATH='/home/lewaldm/code/PanOpticOn/'
# export $(cat /home/lewaldm/code/PanOpticOn/.env)

# cmd='torchrun --nproc_per_node=2 --max_restarts=0 dinov2/train/train.py'
# env=${CDIR}/envs/yamal.yaml
# -----------------------------

# fastdevrun='--fastdevrun'



# eval on some model_final

# srun --export=ALL /global/home/users/lewaldm/.conda/envs/dinov2/bin/python /global/home/users/lewaldm/code/PanOpticOn/dinov2/eval/main.py main \
#     --model-obj=/global/scratch/users/lewaldm/panopticon/dino_logs/ds3/fmow-wvs2/s_lr/base_lr=5e-3_warmup_epochs=0_lr_multiplier=backbone=0.2 \
#     --config-obj=/global/home/users/lewaldm/code/PanOpticOn/dinov2/configs/eval/offline_train_optical/eurosat_knn.yaml \
#     --output-dir=/global/scratch/users/lewaldm/panopticon/dino_logs/debug/2



# tryout new largest batchsize
# ${cmd} --env=${env} ${fastdevrun} \
#     --config-file=${CDIR}/train/06_big_crops.yaml \
#     --output-dir=${ODIR}/debug/bsz/ \
#     train.use_wandb=False \
#     train.dino_augm.global_crops_size=[13,196] \
#     train.dino_augm.local_crops_size=[6,84] \
#     train.batch_size_per_gpu=70

# ${cmd} --env=${env} ${fastdevrun} \
#     --config-file=${CDIR}/train/06_big_crops.yaml \
#     --output-dir=${ODIR}/debug/bsz/ \
#     train.batch_size_per_gpu=40
    
# ${cmd} --env=${env} ${fastdevrun} \
#     --config-file=${CDIR}/train/06_big_crops.yaml \
#     --output-dir=${ODIR}/debug/bsz/ \
#     train.batch_size_per_gpu=50

# ${cmd} --env=${env} ${fastdevrun} \
#     --config-file=${CDIR}/train/06_big_crops.yaml \
#     --output-dir=${ODIR}/debug/bsz/ \
#     train.batch_size_per_gpu=60

# ${cmd} --env=${env} ${fastdevrun} \
#     --config-file=${CDIR}/train/06_big_crops.yaml \
#     --output-dir=${ODIR}/debug/bsz/ \
#     train.batch_size_per_gpu=65


# tryout max with small crops

# ${cmd} --env=${env} ${fastdevrun} \
#     --config-file=${CDIR}/train/06_big_crops.yaml \
#     --output-dir=${ODIR}/debug/bsz/ \
#     train.use_wandb=False \
#     train.dino_augm.global_crops_size=[13,196] \
#     train.dino_augm.local_crops_size=[6,84] \
#     train.batch_size_per_gpu=80

# ${cmd} --env=${env} ${fastdevrun} \
#     --config-file=${CDIR}/train/06_big_crops.yaml \
#     --output-dir=${ODIR}/debug/bsz/ \
#     train.use_wandb=False \
#     train.dino_augm.global_crops_size=[13,196] \
#     train.dino_augm.local_crops_size=[6,84] \
#     train.batch_size_per_gpu=90

# ${cmd} --env=${env} ${fastdevrun} \
#     --config-file=${CDIR}/train/06_big_crops.yaml \
#     --output-dir=${ODIR}/debug/bsz/ \
#     train.use_wandb=False \
#     train.dino_augm.global_crops_size=[13,196] \
#     train.dino_augm.local_crops_size=[6,84] \
#     train.batch_size_per_gpu=100


# ablations on crop size
# warmup=0
# for base_lr in 1e-3  # 1e-4
# do
#     for lrmul in 0.05
#     do
#         ${cmd} --env=${env} ${fastdevrun} \
#             --config-file=${CDIR}/train/05_01_fmow.yaml \
#             --output-dir=${ODIR}/ds4/fmow-wvs2+bigcrops/ \
#             optim.base_lr=${base_lr} \
#             optim.warmup_epochs=${warmup} \
#             optim.lr_multiplier=backbone=${lrmul} \
#             train.batch_size_per_gpu=55 \
#             train.dino_augm.global_crops_size=[13,224] \
#             train.dino_augm.local_crops_size=[6,98] 
#     done
# done

# warmup=0
# for base_lr in 1e-3  # 1e-4
# do
#     for lrmul in 0.05
#     do
#         ${cmd} --env=${env} ${fastdevrun} \
#             --config-file=${CDIR}/train/05_01_fmow.yaml \
#             --output-dir=${ODIR}/ds4/fmow-wvs2+rgb+bigcrops/ \
#             optim.base_lr=${base_lr} \
#             optim.warmup_epochs=${warmup} \
#             optim.lr_multiplier=backbone=${lrmul} \
#             train.batch_size_per_gpu=55 \
#             train.dino_augm.global_crops_size=[13,224] \
#             train.dino_augm.local_crops_size=[6,98] \
#             train.dataset.keep_sensors=[wv23,s2,rgb]
#     done
# done



# freeze backbone on comb
base_lr=1e-3
lrmul=0.2

warmup=5
freezebb=5
${cmd} --env=${env} ${fastdevrun} \
    --config-file=${CDIR}/train/06_big_crops.yaml \
    --output-dir=${ODIR}/ds4/fmow-wvs2_mme-all/freezebb/ \
    optim.base_lr=${base_lr} \
    optim.warmup_epochs=${warmup} \
    optim.lr_multiplier=backbone=${lrmul} \
    optim.freeze_weights=last_layer=1,backbone=${freezebb} \

warmup=5
freezebb=10
${cmd} --env=${env} ${fastdevrun} \
    --config-file=${CDIR}/train/06_big_crops.yaml \
    --output-dir=${ODIR}/ds4/fmow-wvs2_mme-all/freezebb/ \
    optim.base_lr=${base_lr} \
    optim.warmup_epochs=${warmup} \
    optim.epochs=12 \
    optim.lr_multiplier=backbone=${lrmul} \
    optim.freeze_weights=last_layer=1,backbone=${freezebb} \

warmup=10
freezebb=10
${cmd} --env=${env} ${fastdevrun} \
    --config-file=${CDIR}/train/06_big_crops.yaml \
    --output-dir=${ODIR}/ds4/fmow-wvs2_mme-all/freezebb/ \
    optim.base_lr=${base_lr} \
    optim.warmup_epochs=${warmup} \
    optim.epochs=12 \
    optim.lr_multiplier=backbone=${lrmul} \
    optim.freeze_weights=last_layer=1,backbone=${freezebb} \


# just mmearth
# warmup=5
# freezebb=5
# base_lr=1e-3
# ${cmd} --env=${env} ${fastdevrun} \
#     --config-file=${CDIR}/train/06_big_crops.yaml \
#     --output-dir=${ODIR}/ds4/mme-all/freezebb/ \
#     optim.base_lr=${base_lr} \
#     optim.warmup_epochs=${warmup} \
#     optim.epochs=7 \
#     optim.lr_multiplier=backbone=${lrmul} \
#     optim.freeze_weights=last_layer=1,backbone=${freezebb} \