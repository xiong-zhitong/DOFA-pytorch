#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export GEO_BENCH_DIR=/home/zhitong/Datasets/geobench/
export DATA_CONFIG_DIR=/home/zhitong/OFALL/OFALL_baseline/mae/DOFA-pytorch/foundation_models/PanOpticOn/dinov2/configs/data/

model=$1        # Pass the model as the first argument
dataset=$2      # Pass the dataset as the second argument
task=$3         # Pass the task as the third argument
batch_size=$4   # Pass the batch size as the fourth argument
lr=$5           # Pass the learning rate as the fifth argument
epochs=$6       # Pass the number of epochs as the sixth argument

torchrun --nproc_per_node=1 --master_port=25673 src/main.py \
--output_dir logs/"${model}_${dataset}" \
--model $model \
--dataset $dataset \
--task $task \
--num_workers 8 \
--batch_size $batch_size \
--epochs $epochs \
--lr $lr \
--warmup_epochs 0 \
--seed 42