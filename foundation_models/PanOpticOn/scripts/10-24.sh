#!/bin/bash
#SBATCH --job-name=pt
#SBATCH --account=fc_biosense
#SBATCH --partition=savio4_gpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A5000:4
#SBATCH --cpus-per-task=4

#SBATCH --time=15:00:00
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

cmd='torchrun --nproc_per_node=2 --max_restarts=3 dinov2/train/train.py'
env=${CDIR}/envs/yamal.yaml
# -----------------------------

# fastdevrun='--fastdevrun'




# fmow_qb_s2_30kpre
base_lr=1e-3
lrmul=0.2
warmup=0
${cmd} --env=${env} ${fastdevrun} \
    --config-file=${CDIR}/train/05_01_fmow.yaml \
    --output-dir=${ODIR}/ds/fmow_wvs2_30k+pre/ \
    optim.base_lr=${base_lr} \
    optim.warmup_epochs=${warmup} \
    optim.lr_multiplier=backbone=${lrmul} \
    train.dataset.split=fmow/metadata_v2/fmow_iwm_onid_train_val_presorted.parquet



base_lr=1e-3
warmup=0
lrmul=0.2
${cmd} --env=${env} ${fastdevrun} \
    --config-file=${CDIR}/train/05_01_fmow.yaml \
    --output-dir=${ODIR}/ds/fmow_wvs2_30k/ \
    optim.base_lr=${base_lr} \
    optim.warmup_epochs=${warmup} \
    optim.lr_multiplier=backbone=${lrmul} \
    train.dataset.subset=30000


# fmow_qb_s2_15kpre
base_lr=1e-3
lrmul=0.2
warmup=0
${cmd} --env=${env} ${fastdevrun} \
    --config-file=${CDIR}/train/05_01_fmow.yaml \
    --output-dir=${ODIR}/ds/fmow_wvs2_30k+pre/ \
    optim.base_lr=${base_lr} \
    optim.warmup_epochs=${warmup} \
    optim.lr_multiplier=backbone=${lrmul} \
    train.dataset.split=fmow/metadata_v2/fmow_iwm_onid_train_val_presorted.parquet \
    train.dataset.subset=15000



# base_lr=1e-3
# lrmul=0.2
# warmup=1
# ${cmd} --env=${env} ${fastdevrun} \
#     --config-file=${CDIR}/train/05_01_fmow.yaml \
#     --output-dir=${ODIR}/ds/fmow_wvs2_30k+rgb/  \
#     optim.base_lr=${base_lr} \
#     optim.warmup_epochs=${warmup} \
#     optim.lr_multiplier=backbone=${lrmul} \
#     train.dataset.subset=30000 \
#     train.dataset.keep_sensors=['wv23','s2','rgb']



# base_lr=1e-3
# lrmul=0.2
# warmup=1
# ${cmd} --env=${env} ${fastdevrun} \
#     --config-file=${CDIR}/train/05_01_fmow.yaml \
#     --output-dir=${ODIR}/ds/fmow_wvs2_30k+crops/  \
#     optim.base_lr=${base_lr} \
#     optim.warmup_epochs=${warmup} \
#     optim.lr_multiplier=backbone=${lrmul} \
#     train.dataset.subset=30000 \
#     train.dataset.min_crop=42 \
#     train.dataset.max_crop=196



# base_lr=1e-3
# lrmul=0.2
# warmup=1
# ${cmd} --env=${env} ${fastdevrun} \
#     --config-file=${CDIR}/train/05_01_fmow.yaml \
#     --output-dir=${ODIR}/ds/fmow_wvs2_30k+rgb+crops/  \
#     optim.base_lr=${base_lr} \
#     optim.warmup_epochs=${warmup} \
#     optim.lr_multiplier=backbone=${lrmul} \
#     train.dataset.subset=30000 \
#     train.dataset.min_crop=42 \
#     train.dataset.max_crop=196 \
#     train.dataset.keep_sensors=[wv23,s2,rgb]



# MMEARTH
