

PYTHONPATH=. torchrun --nproc_per_node=2 dinov2/train/train.py --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/runtime.yaml --output_dir=/data/panopticon/logs/dino_logs/debug/runtime/1epoch optim.epochs=1 train.OFFICIAL_EPOCH_LENGTH=200
PYTHONPATH=. torchrun --nproc_per_node=2 dinov2/train/train.py --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/runtime.yaml --output_dir=/data/panopticon/logs/dino_logs/debug/runtime/2manyepochs
