#!/bin/bash
#SBATCH --job-name=pt
#SBATCH --account=fc_biosense
#SBATCH --partition=savio4_gpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A5000:4
#SBATCH --cpus-per-task=4

#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lewaldm@berkeley.edu


# ---------- SAVIO ------------
# cd /global/home/users/lewaldm/code/PanOpticOn  # probably not necessary
# export PYTHONPATH=.
# export NCCL_P2P_DISABLE=1 # Need this on savio!
# export $(cat /global/home/users/lewaldm/code/PanOpticOn/.env)

# cmd='srun --export=ALL /global/home/users/lewaldm/.conda/envs/dinov2/bin/python /global/home/users/lewaldm/code/PanOpticOn/dinov2/train/train.py'
# env=${CDIR}/envs/savio.yaml
# -----------------------------

# ---------- YAMAL ------------
export PYTHONPATH='/home/lewaldm/code/PanOpticOn/'
export $(cat /home/lewaldm/code/PanOpticOn/.env)

cmd='torchrun --nproc_per_node=2 --max_restarts=0 dinov2/train/train.py'
env=${CDIR}/envs/yamal.yaml
# -----------------------------

# fastdevrun='--fastdevrun'



############### Debugging ###############

# debug wandb name
# lr=1e-3
# warmup=0
# lrmul=0.2
# ${cmd} --env=${env} ${fastdevrun} \
#     --config-file=${CDIR}/debug.yaml \
#     --output-dir=${ODIR}/debug/1 \
#     optim.base_lr=${lr} \
#     optim.warmup_epochs=${warmup} \
#     optim.lr_multiplier=backbone=${lrmul} \
#     train.batch_size_per_gpu=55 \
#     train.dino_augm.global_crops_size=[13,224] \
#     train.dino_augm.local_crops_size=[6,98] \
#     train.use_wandb=True


# test big crops bsz yamal
# bsz=100
# ${cmd} --env=${env} ${fastdevrun} \
#     --config-file=${CDIR}/train/06_1_fmow.yaml \
#     --output-dir=${ODIR}/debug/ \
#     train.batch_size_per_gpu=${bsz} \
#     optim.epochs=1 \
#     train.OFFICIAL_EPOCH_LENGTH=40 

# bsz=110
# ${cmd} --env=${env} ${fastdevrun} \
#     --config-file=${CDIR}/train/06_1_fmow.yaml \
#     --output-dir=${ODIR}/debug \
#     train.batch_size_per_gpu=${bsz} \
#     optim.epochs=1 \
#     train.OFFICIAL_EPOCH_LENGTH=40 


# bsz=120
# ${cmd} --env=${env} ${fastdevrun} \
#     --config-file=${CDIR}/train/06_1_fmow.yaml \
#     --output-dir=${ODIR}/debug/ \
#     train.batch_size_per_gpu=${bsz} \
#     optim.epochs=1 \
#     train.OFFICIAL_EPOCH_LENGTH=40 






############### YAMAL  ###############


# mmearth only (with big crops)
warmup=0
for lr in 5e-4 1e-4 1e-5
do
    for lrmul in 0.2 0.05
    do
        ${cmd} --env=${env} ${fastdevrun} \
            --config-file=${CDIR}/train/06_1_mmearth.yaml \
            --output-dir=${ODIR}/ds4/mme-all/s_lr/ \
            optim.base_lr=${lr} \
            optim.warmup_epochs=${warmup} \
            optim.lr_multiplier=backbone=${lrmul} 
    done
done


# long run of best loss curve for fmow + mmearth
lr=1e-4
warmup=0
lrmul=0.2
${cmd} --env=${env} ${fastdevrun} \
    --config-file=${CDIR}/train/06_big_crops.yaml \
    --output-dir=${ODIR}/ds4/fmow-wvs2_mme-all/long/ \
    optim.base_lr=${lr} \
    optim.warmup_epochs=${warmup} \
    optim.lr_multiplier=backbone=${lrmul} \
    optim.epochs=30 \
    eval.eval_period_epoch=5



# different values for ibot loss
lr=1e-4
warmup=0
lrmul=0.2
for ibot in 0.1 0.5 2.0
do
    ${cmd} --env=${env} ${fastdevrun} \
        --config-file=${CDIR}/train/06_find_lr.yaml \
        --output-dir=${ODIR}/ds4/fmow-wvs2_mme-all/ibot/ \
        optim.base_lr=${lr} \
        optim.warmup_epochs=${warmup} \
        optim.lr_multiplier=backbone=${lrmul} \
        ibot.loss_weight=${ibot} \
done






############### SAVIO  ###############


# # sweep over big bsz
# warmup=0
# lrmul=0.2
# for lr in 5e-5 #1e-4 5e-4
# do
#     ${cmd} --env=${env} ${fastdevrun} \
#         --config-file=${CDIR}/train/06_findlr.yaml \
#         --output-dir=${ODIR}/ds4/fmow-wvs2_mme-all/bigbsz/ \
#         optim.base_lr=${lr} \
#         optim.warmup_epochs=${warmup} \
#         optim.lr_multiplier=backbone=${lrmul} \

# done


# # different values for dino loss
# lr=1e-4
# warmup=0
# lrmul=0.2
# for dino in 0.1 #0.5 2.0
# do
#     ${cmd} --env=${env} ${fastdevrun} \
#         --config-file=${CDIR}/train/06_findlr.yaml \
#         --output-dir=${ODIR}/ds4/fmow-wvs2_mme-all/dino/ \
#         optim.base_lr=${lr} \
#         optim.warmup_epochs=${warmup} \
#         optim.lr_multiplier=backbone=${lrmul} \
#         dino.loss_weight=${dino} 
# done




# # separate heads
# warmup=0
# lrmul=0.2
# for lr in 1e-4 1e-3
# do
#     ${cmd} --env=${env} ${fastdevrun} \
#         --config-file=${CDIR}/train/06_findlr.yaml \
#         --output-dir=${ODIR}/ds4/fmow-wvs2_mme-all/sepheads/ \
#         optim.base_lr=${lr} \
#         optim.warmup_epochs=${warmup} \
#         optim.lr_multiplier=backbone=${lrmul} \
#         ibot.separate_head=True  \

# done


# # load dino head from pretraining
# warmup=0
# lr=1e-4
# for lrmul in 0.2 0.05
# do
#     ${cmd} --env=${env} ${fastdevrun} \
#         --config-file=${CDIR}/train/06_load_rgbdinohead.yaml \
#         --output-dir=${ODIR}/ds4/fmow-wvs2_mme-all/rgbhead/ \
#         optim.base_lr=${lr} \
#         optim.warmup_epochs=${warmup} \
#         optim.lr_multiplier=backbone=${lrmul} \

# done


# # load mean patch_emb
# warmup=0
# lr=1e-4
# lrmul=0.2
# for frz in 0 10
# do
#     ${cmd} --env=${env} ${fastdevrun} \
#         --config-file=${CDIR}/train/06_load_rgbmeanpe.yaml \
#         --output-dir=${ODIR}/ds4/fmow-wvs2_mme-all/rgbmeanpe/ \
#         optim.base_lr=${lr} \
#         optim.warmup_epochs=${warmup} \
#         optim.lr_multiplier=backbone=${lrmul} \
#         optim.freeze_weights=last_layer=1,patch_embed.patch_emb.=${frz} \

# done



# # test find_lr vs not find_lr
# warmup=0
# lrmul=0.2
# lr=1e-4
# ${cmd} --env=${env} ${fastdevrun} \
#     --config-file=${CDIR}/train/06_findlr.yaml \
#     --output-dir=${ODIR}/ds4/fmow-wvs2_mme-all/no_findlr/ \
#     optim.base_lr=${lr} \
#     optim.warmup_epochs=${warmup} \
#     optim.lr_multiplier=backbone=${lrmul} \
#     optim.scaling_rule=sqrt_wrt_1024 \