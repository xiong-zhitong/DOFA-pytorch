export CUDA_VISIBLE_DEVICES=0
export GEO_BENCH_DIR=/home/zhitong/Datasets/geobench/

model=panopticon_seg
dataset=geobench_cashew
task=segmentation
batch_size=12
blr=0.001
epochs=20

torchrun --nproc_per_node=1 --master_port=25673 main.py \
--output_dir logs/"${model}_${dataset}_${blr}_${batch_size}_${epochs}" \
--log_dir logs/"${model}_${dataset}_${blr}_${batch_size}_${epochs}" \
--model $model \
--dataset $dataset \
--task $task \
--num_workers 8 \
--batch_size $batch_size \
--epochs $epochs \
--blr $blr \
--lr 0.001 \
--warmup_epochs 0 \
--seed 42 \
--dist_eval
