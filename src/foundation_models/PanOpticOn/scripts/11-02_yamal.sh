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
# cmd='python dinov2/train/train.py'
env=${CDIR}/envs/yamal.yaml
# -----------------------------

# fastdevrun='--fastdevrun'



### testout new augmentation


# global: SENSOR_SINGLE
warmup=0
lr=1e-4
lrmul=0.2
${cmd} --env=${env} ${fastdevrun} \
    --config-file=${CDIR}/train/07__augmv2.yaml \
    --output-dir=${ODIR}/c7/probs \
    optim.base_lr=${lr} \
    optim.warmup_epochs=${warmup} \
    optim.lr_multiplier=backbone=${lrmul} \
    train.dino_augm.global_modes_probs="[1.0,0.0,0.0]" \

# global: CHN_MIX
warmup=0
lr=1e-4
lrmul=0.2
${cmd} --env=${env} ${fastdevrun} \
    --config-file=${CDIR}/train/07__augmv2.yaml \
    --output-dir=${ODIR}/c7/probs \
    optim.base_lr=${lr} \
    optim.warmup_epochs=${warmup} \
    optim.lr_multiplier=backbone=${lrmul} \
    train.dino_augm.global_modes_probs="[0.0, 0.0, 1.0]" \

# global: SENSOR_MULTI + CHN_MIX
warmup=0
lr=1e-4
lrmul=0.2
${cmd} --env=${env} ${fastdevrun} \
    --config-file=${CDIR}/train/07__augmv2.yaml \
    --output-dir=${ODIR}/c7/probs \
    optim.base_lr=${lr} \
    optim.warmup_epochs=${warmup} \
    optim.lr_multiplier=backbone=${lrmul} \
    train.dino_augm.global_modes_probs="[0.0, 0.5, 0.5]" \


# all lr sweep
warmup=0
lr=1e-4
lrmul=0.2
for lr in 5e-5 1e-4 1e-3
do
    ${cmd} --env=${env} ${fastdevrun} \
        --config-file=${CDIR}/train/07__augmv2.yaml \
        --output-dir=${ODIR}/c7/probs \
        optim.base_lr=${lr} \
        optim.warmup_epochs=${warmup} \
        optim.lr_multiplier=backbone=${lrmul} \
        train.dino_augm.global_modes_probs="[0.6, 0.2, 0.2]" \

done