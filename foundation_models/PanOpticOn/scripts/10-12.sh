


PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=3 dinov2/train/train.py --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/w.yaml --output_dir=/data/panopticon/logs/dino_logs/s_fixedchns/check_pe_w optim.epochs=8 
PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=3 dinov2/train/train.py --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/wo.yaml --output_dir=/data/panopticon/logs/dino_logs/s_fixedchns/check_pe_wo optim.epochs=8 
