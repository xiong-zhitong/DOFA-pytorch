
# sweep lr 
# warmup=0

# for lr in 1e-4 1e-3 1e-5 1e-2
# do
#     for fw in 'backbone=0' 'backbone=5'
#     do
#         PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=5 dinov2/train/train.py \
#             --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/lr.yaml \
#             --output-dir=/data/panopticon/logs/dino_logs/simplattn/s_lr_rchns/lr=${lr}_warmup=${warmup}_fw=${fw} \
#             optim.base_lr=${lr} optim.warmup_epochs=${warmup} optim.freeze_weights=${fw} \
#             optim.scaling_rule=sqrt_wrt_1024,find_lr
#     done
# done

 
# long training
lr=1e-3
warmup=5
fw='backbone=10'

PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=10 dinov2/train/train.py \
    --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/lr.yaml \
    --output-dir=/data/panopticon/logs/dino_logs/simplattn/long_rchns/lr=${lr}_warmup=${warmup}_fw=${fw} \
    optim.base_lr=${lr} optim.warmup_epochs=${warmup} optim.freeze_weights=${fw} \
    optim.epochs=40