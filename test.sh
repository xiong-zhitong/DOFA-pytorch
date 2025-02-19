#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
#SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%j-%x.out

#SBATCH --job-name=playground
#SBATCH --partition=dev_accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16       # default: 38
#SBATCH --time=00:10:00



export $(cat /home/hk-project-pai00028/tum_mhj8661/code/fm-playground/.env)
export PYTHONPATH='/home/hk-project-pai00028/tum_mhj8661/code/fm-playground:/home/hk-project-pai00028/tum_mhj8661/code/PanOpticOn'
cmd='/home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval/bin/python /home/hk-project-pai00028/tum_mhj8661/code/fm-playground/src/main.py'

all_tasks=(
    ######## classification
    'croma_s2 linear_probe geobench_eurosat_12b'
    'dofa_b linear_probe geobench_eurosat'
    'softcon_B13_b linear_probe geobench_eurosat'
    'stealth linear_probe geobench_eurosat'
    'dinov2 linear_probe geobench_eurosat_rgb'    
    'anysat linear_probe geobench_eurosat_rgb'
    'senpamae linear_probe geobench_eurosat'
)

for task in "${all_tasks[@]}"
do
    echo "Running Task: $task"

    set -- $task
    model=$1
    training_mode=$2
    ds=$3

    $cmd \
        model=$model \
        dataset=$ds \
        +model.training_mode=$training_mode \
        \
        lr=0.002 \
        epochs=1 \
        warmup_epochs=0 \
        \
        batch_size=64 \
        num_workers=8 \
        num_gpus=1 \
        seed=13 \

done