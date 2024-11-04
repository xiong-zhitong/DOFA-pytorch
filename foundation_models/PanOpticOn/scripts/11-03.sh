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
# cd /global/home/users/lewaldm/code/PanOpticOn  # probably not necessary
# export PYTHONPATH=.
# export NCCL_P2P_DISABLE=1 # Need this on savio!
# export $(cat /global/home/users/lewaldm/code/PanOpticOn/.env)

# cmd='srun --export=ALL /global/home/users/lewaldm/.conda/envs/dinov2/bin/python /global/home/users/lewaldm/code/PanOpticOn/dinov2/train/train.py'
# env=${CDIR}/envs/savio.yaml
# -----------------------------

# ---------- YAMAL ------------
export PYTHONPATH='/home/lewaldm/code/PanOpticOn/'
export $(cat /home/lewaldm/code/PanOpticOn/.env)

cmd='torchrun --nproc_per_node=2 --max_restarts=0 dinov2/train/train.py'
env=${CDIR}/envs/yamal.yaml
# -----------------------------

# fastdevrun='--fastdevrun'



# full spectra
warmup=0
lr=1e-4
lrmul=0.2
${cmd} --env=${env} ${fastdevrun} \
    --config-file=${CDIR}/train/07_fullspectra.yaml \
    --output-dir=${ODIR}/c7/fullspec=True/ \
    optim.base_lr=${lr} \
    optim.warmup_epochs=${warmup} \
    optim.lr_multiplier=backbone=${lrmul} \

warmup=0
lr=1e-4
lrmul=0.2
${cmd} --env=${env} ${fastdevrun} \
    --config-file=${CDIR}/train/07__augmv2.yaml \
    --output-dir=${ODIR}/c7/fullspec=False/ \
    optim.base_lr=${lr} \
    optim.warmup_epochs=${warmup} \
    optim.lr_multiplier=backbone=${lrmul} \
    train.dino_augm.global_modes_probs="[1,0,0]" \
    train.dino_augm.local_modes_probs="[1,0,0]" \



#### try recreating previous dino augm

# colorjitter=0, else stays same
warmup=0
lr=1e-4
lrmul=0.2
${cmd} --env=${env} ${fastdevrun} \
    --config-file=${CDIR}/train/07__augmv2.yaml \
    --output-dir=${ODIR}/c7/cjitter=0 \
    optim.base_lr=${lr} \
    optim.warmup_epochs=${warmup} \
    optim.lr_multiplier=backbone=${lrmul} \
    train.dino_augm.global_modes_probs="[1,0,0]" \
    train.dino_augm.local_modes_probs="[0,0,1]" \
    train.dino_augm.color_jitter_args.p=0.0 \

# hue & sat = 0
warmup=0
lr=1e-4
lrmul=0.2
${cmd} --env=${env} ${fastdevrun} \
    --config-file=${CDIR}/train/07__augmv2.yaml \
    --output-dir=${ODIR}/c7/hue=sat=0 \
    optim.base_lr=${lr} \
    optim.warmup_epochs=${warmup} \
    optim.lr_multiplier=backbone=${lrmul} \
    train.dino_augm.global_modes_probs="[1,0,0]" \
    train.dino_augm.local_modes_probs="[0,0,1]" \
    train.dino_augm.color_jitter_args.saturation=0 \
    train.dino_augm.color_jitter_args.hue=0 \


# local probs = single sensor
warmup=0
lr=1e-4
lrmul=0.2
${cmd} --env=${env} ${fastdevrun} \
    --config-file=${CDIR}/train/07__augmv2.yaml \
    --output-dir=${ODIR}/c7/lprobs=[1,0,0]/ \
    optim.base_lr=${lr} \
    optim.warmup_epochs=${warmup} \
    optim.lr_multiplier=backbone=${lrmul} \
    train.dino_augm.global_modes_probs="[1,0,0]" \
    train.dino_augm.local_modes_probs="[1,0,0]" \
    train.dino_augm.color_jitter_args.saturation=0.2 \
    train.dino_augm.color_jitter_args.hue=0.1 \


#### can we learn ben-lin (espc. with [0,1,0]) and big crops?
warmup=0
lr=1e-4
lrmul=0.2
for lcsz in "[1,6]" "[1,13]" "[6,6]"
do
    ${cmd} --env=${env} ${fastdevrun} \
        --config-file=${CDIR}/train/07__augmv2.yaml \
        --output-dir=${ODIR}/c7/learnboth/ \
        optim.base_lr=${lr} \
        optim.warmup_epochs=${warmup} \
        optim.lr_multiplier=backbone=${lrmul} \
        train.dino_augm.global_crops_spectral_size=[21,21] \
        train.dino_augm.global_modes_probs="[0,1,0]" \
        train.dino_augm.local_modes_probs="[0,1,0]" \
        train.dino_augm.local_crops_spectral_size=${lcsz} \
        train.batch_size_per_gpu=100

done

# warmup=0
# lr=1e-4
# lrmul=0.2
# for lcsz in "[1,6]" "[1,13]" "[6,6]"
# do
#     ${cmd} --env=${env} ${fastdevrun} \
#         --config-file=${CDIR}/train/07__augmv2.yaml \
#         --output-dir=${ODIR}/c7/learnboth/ \
#         optim.base_lr=${lr} \
#         optim.warmup_epochs=${warmup} \
#         optim.lr_multiplier=backbone=${lrmul} \
#         train.dino_augm.global_crops_spectral_size=[21,21] \
#         train.dino_augm.global_modes_probs="[0.5,0.5,0]" \
#         train.dino_augm.local_modes_probs="[0.5,0.5,0]" \
#         train.dino_augm.local_crops_spectral_size=${lcsz} \
#         train.batch_size_per_gpu=100

# done




#### saturate training

# previous augm with +rgbhead +ibot
# warmup=0
# lr=1e-4
# lrmul=0.2
# ${cmd} --env=${env} ${fastdevrun} \
#     --config-file=${CDIR}/train/07__augmv2.yaml \
#     --output-dir=${ODIR}/c7/long/ \
#     optim.base_lr=${lr} \
#     optim.warmup_epochs=${warmup} \
#     optim.lr_multiplier=backbone=${lrmul} \
#     train.dino_augm.global_crops_spectral_size=[13,13] \
#     train.dino_augm.local_crops_spectral_size=[3,6] \
#     train.dino_augm.global_modes_probs="[1,0,0]" \
#     train.dino_augm.local_modes_probs="[1,0,0]" \
#     train.dino_augm.color_jitter_args.p=0 \
#     optim.epochs=19 \
#     eval.eval_period_iterations=-1 \
#     eval.eval_period_epoch=3 \
#     eval.include_final_ckpt=True