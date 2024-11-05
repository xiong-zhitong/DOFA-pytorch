export CUDA_VISIBLE_DEVICES=4
export GEO_BENCH_DIR=/home/zhitong/Datasets/geobench/

model=dinov2base_seg
dataset=geobench_pv4ger
batch_size=64
blr=0.1
epochs=20

torchrun --nproc_per_node=1 --master_port=25673 main.py \
--output_dir logs/"${model}_${dataset}_${blr}_${batch_size}_${epochs}" \
--log_dir logs/"${model}_${dataset}_${blr}_${batch_size}_${epochs}" \
--model $model \
--dataset $dataset \
--num_workers 4 \
--batch_size $batch_size \
--epochs $epochs \
--blr $blr \
--warmup_epochs 0 \
--seed 42 \
--dist_eval
