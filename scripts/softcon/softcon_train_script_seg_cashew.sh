export CUDA_VISIBLE_DEVICES=1
export GEO_BENCH_DIR=/home/zhitong/Datasets/geobench/
export DATA_CONFIG_DIR=/home/zhitong/OFALL/OFALL_baseline/mae/DOFA-pytorch/foundation_models/PanOpticOn/dinov2/configs/data/

model=softcon_seg
dataset=geobench_cashew_13
task=segmentation
batch_size=16
epochs=20
lr=0.01

torchrun --nproc_per_node=1 --master_port=25676 main.py \
--output_dir logs/"${model}_${dataset}_${lr}_${batch_size}_${epochs}" \
--model $model \
--dataset $dataset \
--task $task \
--num_workers 8 \
--batch_size $batch_size \
--epochs $epochs \
--warmup_epochs 3 \
--lr $lr \
--seed 42 
