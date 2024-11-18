#!/bin/bash
#SBATCH --job-name=t
#SBATCH --account=fc_biosense
#SBATCH --partition=savio4_gpu

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A5000:2
#SBATCH --cpus-per-task=4

#SBATCH --time=0:15:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lewaldm@berkeley.edu

# working: n0134
# not working: n0123, n0142


module load anaconda3
source activate dinov2 #/global/home/users/lewaldm/.conda/envs/dinov2
cd /global/home/users/lewaldm/PanOpticOn
python -c 'import sys; print(sys.executable)'

export PYTHONPATH=.
export NCCL_BLOCKING_WAIT=1
export NCCL_P2P_DISABLE=1 # Need this on savio!

srun --export=ALL python dinov2/train/train.py --config-file=/global/home/users/lewaldm/PanOpticOn/dinov2/configs/savio/train.yaml --output-dir=/global/home/users/lewaldm/out/debug/3
# srun --export=ALL python /global/home/users/lewaldm/test.py