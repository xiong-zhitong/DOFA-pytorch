
file=dinov2_rgb_notrain
PYTHONPATH=. python dinov2/eval/linear.py --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/eval/rgb/${file}.yaml --output-dir=/data/panopticon/logs/dino_logs/eval/rgb5/dinov2_rgb student.pretrained_weights=/data/panopticon/other_model_ckpts/dinov2/dinov2_vitb14_pretrain_wmodelkey.pth
PYTHONPATH=. python dinov2/eval/linear.py --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/eval/rgb/${file}.yaml --output-dir=/data/panopticon/logs/dino_logs/eval/rgb5/dinov2_rgb_rand-all &
PYTHONPATH=. python dinov2/eval/linear.py --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/eval/rgb/${file}.yaml --output-dir=/data/panopticon/logs/dino_logs/eval/rgb5/dinov2_rgb_rand-posemb student.pretrained_weights=/data/panopticon/other_model_ckpts/dinov2/dinov2_vitb14_pretrain_wmodelkey.pth student.drop_pretrained_weights=pos_embed &
PYTHONPATH=. python dinov2/eval/linear.py --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/eval/rgb/${file}.yaml --output-dir=/data/panopticon/logs/dino_logs/eval/rgb5/dinov2_rgb_rand-patchemb student.pretrained_weights=/data/panopticon/other_model_ckpts/dinov2/dinov2_vitb14_pretrain_wmodelkey.pth student.drop_pretrained_weights=patch_embed &
PYTHONPATH=. python dinov2/eval/linear.py --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/eval/rgb/${file}.yaml --output-dir=/data/panopticon/logs/dino_logs/eval/rgb5/dinov2_rgb_rand-posemb-patchemb student.pretrained_weights=/data/panopticon/other_model_ckpts/dinov2/dinov2_vitb14_pretrain_wmodelkey.pth student.drop_pretrained_weights=pos_embed,patch_embed &
wait()


# PYTHONPATH=. python dinov2/eval/linear.py --config-file=/home/lewaldm/code/PanOpticOn/dinov2/configs/eval/rgb/dinov2_rgb_notrain.yaml --output-dir=/data/panopticon/logs/dino_logs/eval/rgb3/dinov2_rgb student.pretrained_weights=/data/panopticon/other_model_ckpts/dinov2/dinov2_vitb14_pretrain_wmodelkey.pth
