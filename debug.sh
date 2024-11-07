export CUDA_VISIBLE_DEVICES=6
export GEO_BENCH_DIR=/home/zhitong/Datasets/geobench/
#export TORCH_DISTRIBUTED_DEBUG=INFO
python -m torch.distributed.launch --nproc_per_node=1 --master_port=15676 main_segment_geobench_ofa.py \
--data_path /home/zhitong/Datasets/geobench/ \
--output_dir logs/segment_geobench_ofa \
--log_dir logs/segment_geobench_ofa \
--model vit_large_patch16 \
--num_workers 8 \
--batch_size 64 \
--epochs 20 \
--seed 42 