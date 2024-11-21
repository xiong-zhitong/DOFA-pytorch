# PanOpticOn
This codebase is a fork (git clone) from the official DINOv2 codebase (https://github.com/facebookresearch/dinov2).

## setup

### conda
```
conda create -n dinov2 python=3.9
conda activate dinov2
pip install -r requirements.txt
```

### bugfix in skmultilearn source code (NOT NEEDED ANYMORE as of Oct27)
In skmultilearn, you might need to fix the following bugs in `skmultilearn/adapt/mlknn` (best found by cmd+click in the import statement in the dinov2/eval/knn.py file):
- line 120: just super().__init__()
- line 166: pass self.k as kwarg n_neighbors=self.k

### env variables
The following env variables need to be put in `.env` file in the codebase root
```
WANDB_PROJECT=dino
GEO_BENCH_DIR=/path/to/geobench/dir
RDIR=/path/to/resource/dir
CDIR=/.../PanOpticOn/dinov2/configs
ODIR=path/to/output/dir
```
Datsets are saved in `$RDIR/datasets/` and we have `$ODIR=$RDIR/logs/`.


## training
single gpu:
```
PYTHONPATH=. python dinov2/train/train.py \
  --config-file=dinov2/configs/debug.yaml \
  --output_dir=.
```
multi-gpu on local machine:
```
PYTHONPATH=. torchrun \
  --nproc_per_node=2 dinov2/train/train.py \
  --config-file=dinov2/configs/debug.yaml --output_dir=.
```

### Pretraining Datasets

Here is how you should setup your `datasets` folder to use the datasets we use in our experiments.

```
datasets
├── fmow
|   └── train
│   └── val
│   └── test_gt
│   └── seq_gt
│   └── metadata_v2
└── fmow-sentinel
│   └── train
│   └── val
│   └── test_gt
│   └── seq_gt
└── mmearth
│   └── data_100k_v001
│   └── data_1M_v001
.....
```


#### FMOW

We use a combined fmow-fmow_sentinel dataset, which we call `fmow-mm`. 


In your .env set:
```
$RDIR = <path to your datasets folder>
```

To use this dataset, follows these steps:
1. Download the `fmow` dataset from [the official github repo](https://github.com/fMoW/dataset?tab=readme-ov-file). You only need `fmow-full` NOT `fmow-rgb`. Unzip to a your `dataset` folder. tl;dr `aws s3 ls s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-full/`. Caution: this is a LARGE dataset ~3.5TB, and will take some time to untar!

2. Download the `fmow-sentinel` dataset from [this link](https://purl.stanford.edu/vg497cb6002). Unzip to your `dataset` folder. Large-ish dataset ~80GB

3. Setup a folder called `metadata_v2` under `fmow/` and download the metadata files from [this link](https://drive.google.com/drive/folders/1nsTN-5v6jHusYm_NnKrVLN-TYo7RmUz8?usp=drive_link)


#### MMEarth

1. Download the `mmearth` dataset from the [links on the official github repo](https://github.com/vishalned/MMEarth-data). Please download atleast the MMEarth (full dataset). Unzip to your `dataset` folder. Large-ish dataset ~600GB

2. Download the metadata parquet file [from this link](https://drive.google.com/drive/folders/1LfTBRxJNzgDFIrW1yD9gPbV2Rl-ZV3-d?usp=drive_link) and place in the `mmearth/data_1M_v001` folder.


## eval

### Design
Each downstream task is defined by a yaml config, e.g. [`dinov2/configs/eval/debug/eurosat_knn.yaml`](https://github.com/ando-shah/PanOpticOn/blob/dino/dinov2/configs/eval/debug/eurosat_knn.yaml).
Default arguments are loaded depending on the task from `dinov2/configs/defaults/`, in our case from [`dinov2/configs/defaults/default_eval_knn.yaml`](https://github.com/ando-shah/PanOpticOn/blob/dino/dinov2/configs/defaults/default_eval_knn.yaml).

The main eval function `dinov2/eval/main.py/main` takes 3 args:
- model_obj: any of; run folder, config & ckpt path, identifier string for models used often
- config_obj: any of; single config, folder of configs
- output_dir

This function will evaluate all provided configs on the provided model with individual logs & aggregated results. If we, e.g., provide as config_obj the path to this folder
```
/dinov2/configs/eval/debug
├── benv2_multiknn.yaml
├── eurosat_knn.yaml
└── eurosat_linear.yaml
```
we will get an output_dir like this
```
output_dir
├── log
├── results.csv
├── benv2_multiknn
│   └── config.yaml
│   ├── log
│   └── results_eval_knn.json
├── eurosat_knn
│   └── ...
├── eurosat_linear
│   └── ...
```
with aggregated values in results.csv, individual logs in the subfolders and a final summary printout in `output_dir/log` like this
```
I20241024 15:25:29 2075111 eval main.py:193] Running task benv2_multiknn ...
I20241024 15:26:13 2075111 eval main.py:198] Finished in 44.10s
I20241024 15:26:13 2075111 eval main.py:193] Running task eurosat_knn ...
I20241024 15:26:19 2075111 eval main.py:198] Finished in 5.76s
I20241024 15:26:19 2075111 eval main.py:193] Running task eurosat_linear ...
I20241024 15:26:23 2075111 eval main.py:198] Finished in 4.19s
I20241024 15:26:23 2075111 eval main.py:201] All tasks finished in 54.06s
I20241024 15:26:23 2075111 eval main.py:217]
                                          value                                 best_classifier
task                     metric
benv2_multiknn/          acc_top-1_macro   5.26                                      (full, 10)
                         acc_top-1_micro   5.26                                      (full, 10)
                         acc_top-5_micro  26.08                                      (full, 10)
eurosat_knn/             acc_top-1_macro  10.00                                      (full, 10)
                         acc_top-1_micro  16.00                                      (full, 20)
                         acc_top-5_micro  48.00                                      (full, 20)
eurosat_linear/m-eurosat acc_top-1_macro  10.00  classifier_1_blocks_pooling_avgpool_lr_0_00000
```
where the best_classifier denotes the best classifier of all classifiers tried.

### Commands

To generate the results of previous section, navigate to the codebase root folder (& load env variables). Start a quick training run:
```
PYTHONPATH=. python dinov2/train/train.py \
  --config-file=dinov2/configs/debug.yaml \
  --output-dir=$ODIR/my_run_dir
```
To evaluate your training run, execute
```
PYTHONPATH=. python dinov2/eval/main.py main \
  --model-obj=$ODIR/my_run_dir \
  --config-obj=dinov2/configs/eval/debug \
  --output-dir=$ODIR/eval_showcase
```

### Add datasets, models, transforms, ...

#### datasets
Write your dataset class and add it in [`dinov2/data/loaders.py/make_dataset`](https://github.com/ando-shah/PanOpticOn/blob/45d5b43bc8f9b9242f68550e096de5aae51567a1/dinov2/data/loaders.py#L51)

#### transforms
Write your transform (to dinov2/data/augmentations.py) and add it in [`dinov2/data/augmentations.py/make_augmentation`](https://github.com/ando-shah/PanOpticOn/blob/45d5b43bc8f9b9242f68550e096de5aae51567a1/dinov2/data/augmentations.py#L29)

#### models
To add additional ViT architectures, you only need to:
- add if statement in `dinov2/eval/setup.py/build_model_for_eval`
- add a corresponding wrapper in `dinov2/eval/wrapper.py` that (1) extracts the input from a dict containing key 'imgs' (our default data structure) (2) returns the last n blocks of the vit backbone

Non ViT architecture & other extractions could be implemented by adding different feature extraction methods to `dinov2/eval/wrapper.py/backbone_to_features`. However, this will be more work.
