export CUDA_VISIBLE_DEVICES=0
export GEO_BENCH_DIR=/home/zhitong/Datasets/geobench/
export DATA_CONFIG_DIR=/home/zhitong/OFALL/OFALL_baseline/mae/DOFA-pytorch/foundation_models/PanOpticOn/dinov2/configs/data/

model=croma_cls
dataset=geobench_brick_kiln_12
task=classification
batch_size=256
blr=10
epochs=50

torchrun --nproc_per_node=1 --master_port=25673 main.py \
--output_dir logs/"${model}_${dataset}_${blr}_${batch_size}_${epochs}" \
--model $model \
--dataset $dataset \
--task $task \
--num_workers 8 \
--batch_size $batch_size \
--epochs $epochs \
--blr $blr \
--warmup_epochs 0 \
--seed 42
