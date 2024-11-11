export CUDA_VISIBLE_DEVICES=0
export GEO_BENCH_DIR=/home/xshadow/Datasets/geobench/

model=softcon_seg
dataset=geobench_SAcrop_13
task=segmentation
batch_size=84
epochs=20
lr=0.005

torchrun --nproc_per_node=1 --master_port=25673 src/main.py \
--output_dir logs/"${model}_${dataset}_${lr}_${batch_size}_${epochs}" \
--model $model \
--dataset $dataset \
--task $task \
--num_workers 8 \
--batch_size $batch_size \
--epochs $epochs \
--lr $lr \
--warmup_epochs 3 \
--seed 42 \
--dist_eval
