export CUDA_VISIBLE_DEVICES=4
export GEO_BENCH_DIR=/home/zhitong/Datasets/geobench/

torchrun --nproc_per_node=1 --master_port=25673 main.py \
--output_dir logs/linear_geobench_croma \
--log_dir logs/linear_geobench_croma \
--model gfm_seg \
--dataset geobench_pv4ger \
--num_workers 4 \
--batch_size 64 \
--epochs 20 \
--blr 0.1 \
--warmup_epochs 0 \
--seed 42 \
--dist_eval
