#!/bin/bash
#SBATCH --job-name=pt
#SBATCH --account=fc_biosense
#SBATCH --partition=savio4_gpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4

#SBATCH --time=00:10:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lewaldm@berkeley.edu


# ---------- SAVIO ------------
cd /global/home/users/lewaldm/code/PanOpticOn
export PYTHONPATH=.
export NCCL_P2P_DISABLE=1 # Need this on savio!
export $(cat /global/home/users/lewaldm/code/PanOpticOn/.env)

cmd='srun --export=ALL /global/home/users/lewaldm/.conda/envs/dinov2/bin/python /global/home/users/lewaldm/code/PanOpticOn/dinov2/train/train.py'
env=${CDIR}/envs/savio.yaml
# -----------------------------

# ---------- YAMAL ------------
# env=${CDIR}/envs/yamal.yaml
# cmd='PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=10 dinov2/train/train.py'
# -----------------------------

fastdevrun='--fastdevrun'


# command
${cmd} --env=${env} ${debug_flag} \
    --config-file=${CDIR}/train/05_01_fmow.yaml \
    --output-dir=${LDIR}/ds/fmow_wvs2_30k+rgb/  \
