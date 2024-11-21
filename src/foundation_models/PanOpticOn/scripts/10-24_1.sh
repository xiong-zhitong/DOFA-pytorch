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

cmd='torchrun --nproc_per_node=2 --max_restarts=1 dinov2/train/train.py'
env=${CDIR}/envs/yamal.yaml
# -----------------------------

# fastdevrun='--fastdevrun'


# fmow-all + mmearth-all: low lr
lrmul=0.2
warmup=0
for base_lr in 1e-4 1e-5
do
${cmd} --env=${env} ${fastdevrun} \
    --config-file=${CDIR}/train/05_combined.yaml \
    --output-dir=${ODIR}/ds/fmow-all_mmearth-s1-s2/ \
    optim.base_lr=${base_lr} \
    optim.warmup_epochs=${warmup} \
    optim.lr_multiplier=backbone=${lrmul}
done


# fmow-all + mmearth-all: high lr
lrmul=0.2
for base_lr in 1e-3 5e-3
do
    for warmup in 0 5
    do
        ${cmd} --env=${env} ${fastdevrun} \
            --config-file=${CDIR}/train/05_combined.yaml \
            --output-dir=${ODIR}/ds/fmow-all_mmearth-s1-s2/ \
            optim.base_lr=${base_lr} \
            optim.warmup_epochs=${warmup} \
            optim.lr_multiplier=backbone=${lrmul}
    done
done


# varying lrmul
