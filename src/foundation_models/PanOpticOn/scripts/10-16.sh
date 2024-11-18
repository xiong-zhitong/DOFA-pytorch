# frozen bb
# imgs=1
# fw='backbone=2,last_layer=1'
# for warmup in 0 1
# do
#     for lr in 1e-6 1e-5 1e-4 1e-3
#     do
#         CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python dinov2/train/train.py --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/debug_viewimgs.yaml --output-dir=/data/panopticon/logs/dino_logs/debug/v2_img/img=${imgs}_lr=${lr}_warmup=${warmup} optim.base_lr=${lr} train.dataset.subset=${imgs} train.batch_size_per_gpu=${imgs} optim.warmup_epochs=${warmup} optim.freeze_weights=${fw} &
#     done    
# done

# unfrozen bb
# imgs=2
# fw='backbone=0,last_layer=1'
# for warmup in 0 1
# do
#     for lrmult in backbone=0.2 backbone=0.5
#     do
#         for lr in 1e-6 1e-5 1e-4 1e-3
#         do
#             CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python dinov2/train/train.py --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/debug_viewimgs.yaml --output-dir=/data/panopticon/logs/dino_logs/debug/img=${imgs}_lr=${lr}_warmup=${warmup}_lrmul=${lrmult} optim.base_lr=${lr} train.dataset.subset=${imgs} train.batch_size_per_gpu=${imgs} optim.warmup_epochs=${warmup} freeze_weights=backbone=0,last_layer=1 optim.lr_multiplier=${lrmult} optim.freeze_weights=${fw} &
#         done    
#     done
#     wait
# done

# rgb
# fw='backbone=2,last_layer=1'
# imgs=1
# for warmup in 0 1
# do
#     for lr in 1e-6 1e-5 1e-4 1e-3
#     do
#         CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python dinov2/train/train.py --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/debug_viewimgs.yaml --output-dir=/data/panopticon/logs/dino_logs/debug/v2_img_rgb/img=${imgs}_lr=${lr}_warmup=${warmup} optim.base_lr=${lr} train.dataset.subset=${imgs} train.batch_size_per_gpu=${imgs} optim.warmup_epochs=${warmup} student.embed_layer=PatchEmbed optim.freeze_weights=${fw} &
#     done    
# done

############## New version with 1 batch

# check that patch emb pretrained works
# lr=1e-3
# warmup=1

# for fw in 'backbone=0' 'backbone=1'
# do
#     PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=3 dinov2/train/train.py --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/debug_viewimgs.yaml --output-dir=/data/panopticon/logs/dino_logs/fewimgs_rgb/lr=${lr}_warmup=${warmup}_fw=${fw} optim.base_lr=${lr} optim.warmup_epochs=${warmup} student.embed_layer=PatchEmbed optim.freeze_weights=${fw}
# done

# check with old chn att
# lr=1e-4
# warmup=0
# for fw in 'backbone=0' 'backbone=1'
# do
#     PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=3 dinov2/train/train.py \
#         --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/debug_viewimgs.yaml \
#         --output-dir=/data/panopticon/logs/dino_logs/fewimgs_attn/lr=${lr}_warmup=${warmup}_fw=${fw} \
#         optim.base_lr=${lr} optim.warmup_epochs=${warmup} optim.freeze_weights=${fw} \
#         student.add_pe_args.id_attn_block=ChnAttnBlock
# done

# new simple chn att
# lr=1e-4
# warmup=0

# for norm_input in true false
# do
#     for skip_conn in true false
#     do
#         for norm_output in true false
#         do
#             for use_layer_scale in false true
#             do
#                 for fw in 'backbone=0' 'backbone=1'
#                 do
#                     PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=3 dinov2/train/train.py \
#                         --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/debug_viewimgs.yaml \
#                         --output-dir=/data/panopticon/logs/dino_logs/fewimgs_simplattn/lr=${lr}_warmup=${warmup}_fw=${fw}_inp=${norm_input}_skip=${skip_conn}_out=${norm_output}_scale=${use_layer_scale} \
#                         optim.base_lr=${lr} optim.warmup_epochs=${warmup} optim.freeze_weights=${fw} student.add_pe_args.id_attn_block=ChnAttnBlockSimple \
#                         student.add_pe_args.norm_input=${norm_input} \
#                         student.add_pe_args.skip_conn=${skip_conn} \
#                         student.add_pe_args.norm_output=${norm_output} \
#                         student.add_pe_args.use_layer_scale=${use_layer_scale}
#                 done
#             done
#         done   
#     done
# done


# ------------ training overnight
# sweep lr 
norm_output=false
use_layer_scale=false
skip_conn=true
warmup=0

for lr in 1e-4 1e-3 1e-5
do
    for norm_input in false # true
    do
        for fw in 'backbone=0' 'backbone=5'
        do
            PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=5 dinov2/train/train.py \
                --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/lr.yaml \
                --output-dir=/data/panopticon/logs/dino_logs/simplattn/s_lr/lr=${lr}_warmup=${warmup}_fw=${fw}_inp=${norm_input} \
                optim.base_lr=${lr} optim.warmup_epochs=${warmup} optim.freeze_weights=${fw} \
                student.add_pe_args.norm_input=${norm_input} \
                student.add_pe_args.use_layer_scale=${use_layer_scale} \
                student.add_pe_args.skip_conn=${skip_conn} \
                student.add_pe_args.norm_output=${norm_output} \
                optim.scaling_rule=sqrt_wrt_1024,find_lr

        done
    done
done

 
# long training
lr=1e-4
warmup=0
fw='backbone=5'

norm_input=false
use_layer_scale=false
skip_conn=true
norm_output=false

PYTHONPATH=. torchrun --nproc_per_node=2 --max_restarts=10 dinov2/train/train.py \
    --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/lr.yaml \
    --output-dir=/data/panopticon/logs/dino_logs/simplattn/long/lr=${lr}_warmup=${warmup}_fw=${fw} \
    optim.base_lr=${lr} optim.warmup_epochs=${warmup} optim.freeze_weights=${fw} \
    student.add_pe_args.norm_input=${norm_input} \
    student.add_pe_args.use_layer_scale=${use_layer_scale} \
    student.add_pe_args.skip_conn=${skip_conn} \
    student.add_pe_args.norm_output=${norm_output} \
    optim.epochs=50