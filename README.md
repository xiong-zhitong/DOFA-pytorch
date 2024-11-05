# Evaluation of Foundation Models for EO

### For pretraining and deployment of [DOFA](https://arxiv.org/abs/2403.15356), please refer to the [main DOFA repo](https://github.com/zhu-xlab/DOFA).

---


This repo is tested using the following versions:

```
pip install torch==2.1.2
pip install torchvision==0.16.2
pip install numpy==1.26.4
pip install openmim
mim install mmengine
mim install "mmcv==2.1.0"
mim install mmsegmentation
```

The model weights can be downloaded from [HF](https://huggingface.co/XShadow/GeoFMs).


The following models are included:
- ScaleMAE
- DOFA
- GFM
- CROMA
- Dinov2 large, base, w/ or w/o registers

---
The following datasets are supported:
- GeoBench
- Spectral Earth (on going)

How to add new model and dataset wrappers?

1) Add new model wrapper to the [foundation_models](foundation_models) folder and add it to __init__.py;
2) Add the model name to [factory.py](factory.py) for access by ```model_type``` name;
3) Add new dataset wrapper to the [datasets](datasets) folder;
4) Add the dataset name to [factory.py](factory.py) 
5) Write the configuration class in the [config.py](util/config.py).

---
How to run the evaluation?

```bash
sh scripts/train_linear_geobench_cls_croma.sh
```


