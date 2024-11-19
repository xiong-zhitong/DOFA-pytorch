export CUDA_VISIBLE_DEVICES=7
export GEO_BENCH_DIR=/home/zhitong/Datasets/geobench/
export DATA_CONFIG_DIR=/home/zhitong/OFALL/OFALL_baseline/mae/DOFA-pytorch/foundation_models/PanOpticOn/dinov2/configs/data/

model=softcon_seg
dataset=geobench_SAcrop_13
task=segmentation
batch_size=84
epochs=20
lr=0.05

torchrun --nproc_per_node=1 --master_port=25670 main.py \
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