#!/bin/bash
echo "Contents of the current directory:"
ls -lah

export CUDA_VISIBLE_DEVICES=0
export $(cat /home/ando/fm-playground/.env)

echo "Output Directory": $ODIR

model=dinov2_cls_linear_probe
dataset=geobench_eurosat_rgb
batch_size=32
lr=0.002
epochs=1
warmup_epochs=1
task=classification
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

python /home/ando/fm-playground/src/main.py \
output_dir=${ODIR}/exps/${model}_${dataset} \
model=${model} \
dataset=${dataset} \
lr=${lr} \
task=${task} \
num_gpus=${num_gpus} \
num_workers=8 \
epochs=${epochs} \
warmup_epochs=${warmup_epochs} \
seed=13 \
