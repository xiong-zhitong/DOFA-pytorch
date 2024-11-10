#!/bin/bash
#SBATCH --job-name=pt
#SBATCH --account=fc_biosense
#SBATCH --partition=savio4_gpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A5000:4
#SBATCH --cpus-per-task=4

#SBATCH --time=05:00:00
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


# # speedtest

# base_lr=1e-3
# lrmul=0.2
# warmup=1
# ${cmd} --env=${env} ${fastdevrun} \
#     --config-file=${CDIR}/train/05_combined_norgb.yaml \
#     --output-dir=${ODIR}/speedtest/no_rgb \
#     optim.base_lr=${base_lr} \
#     optim.warmup_epochs=${warmup} \
#     optim.lr_multiplier=backbone=${lrmul}

# base_lr=1e-3
# lrmul=0.2
# warmup=1
# ${cmd} --env=${env} ${fastdevrun} \
#     --config-file=${CDIR}/train/05_combined.yaml \
#     --output-dir=${ODIR}/speedtest/with_rgb \
#     optim.base_lr=${base_lr} \
#     optim.warmup_epochs=${warmup} \
#     optim.lr_multiplier=backbone=${lrmul} \



# lr sweep on fmow-wvs2 + mmearth
# warmup=0
# for base_lr in 5e-3 #1e-3 1e-4 1e-5
# do
#     for lrmul in 0.2 #0.5 0.05
#     do
#         ${cmd} --env=${env} ${fastdevrun} \
#             --config-file=${CDIR}/train/05_combined.yaml \
#             --output-dir=${ODIR}/ds3/fmow-wvs2_mmearth-all/s_lr/ \
#             optim.base_lr=${base_lr} \
#             optim.warmup_epochs=${warmup} \
#             optim.lr_multiplier=backbone=${lrmul} 
#     done
# done


# lr sweep on fmow-wvs2
warmup=0
for base_lr in 1e-4
do
    for lrmul in 0.5
    do
        ${cmd} --env=${env} ${fastdevrun} \
            --config-file=${CDIR}/train/05_01_fmow.yaml \
            --output-dir=${ODIR}/ds3/fmow-wvs2/s_lr/ \
            optim.base_lr=${base_lr} \
            optim.warmup_epochs=${warmup} \
            optim.lr_multiplier=backbone=${lrmul} 
    done
done

# warmup=0
# for base_lr in 1e-3 1e-4 3e-5
# do
#     for lrmul in 0.2 0.05 0.5
#     do
#         ${cmd} --env=${env} ${fastdevrun} \
#             --config-file=${CDIR}/train/05_01_fmow.yaml \
#             --output-dir=${ODIR}/ds3/fmow-wvs2/s_lr/ \
#             optim.base_lr=${base_lr} \
#             optim.warmup_epochs=${warmup} \
#             optim.lr_multiplier=backbone=${lrmul} 
#     done
# done