# Evaluation of Foundation Models for Earth Observation

This repository provides tools for evaluating various foundation models on Earth Observation tasks. For detailed instructions on **pretraining and deploying [DOFA](https://arxiv.org/abs/2403.15356)**, please refer to the [main DOFA repository](https://github.com/zhu-xlab/DOFA).

---

## Setup

To get started, you can install the required dependencies from the `pyproject.toml`:

For this navigate to the root directory of this repository and do:

```
pip install -e .
```

You currently do not need to install the ViT Adapter part below, as it is not used in the current version of the repository. It is optional, and relies on CUDA toolkit < 12

### To use [ViT Adapter](https://arxiv.org/abs/2205.08534)
```bash
cd src/foundation_models/modules/ops/
sh make.sh
```


### Model Weights
Pretrained model weights are available on [Hugging Face](https://huggingface.co/XShadow/GeoFMs).


### Set Up Your Environment Variables

You can set this environment variable in a .env in the root directory. The variables here are automatically exported and used by different scripts, so make sure to set the following variables:

```shell
MODEL_WEIGHTS_DIR=<path/to/your/where/you/want/to/store/weights>
TORCH_HOME=<path/to/your/where/you/want/to/store/torch/hub/weights>
DATASETS_DIR=<path/to/your/where/you/want/to/store/all/other/datasets>
GEO_BENCH_DIR=<path/to/your/where/you/want/to/store/GeoBench>
ODIR=<path/to/your/where/you/want/to/store/logs>
REPO_PATH=<path/to/this/repo>
```

When using any of the FMs, the init method will check whether it can find the pre-trained checkpoint of the respective FM in the above `MODEL_WEIGHTS_DIR` and download it there if not found. If you do not change the env
variable, the default will be `./fm_weights`.

Some models depend on [torch hub](https://pytorch.org/docs/stable/hub.html#where-are-my-downloaded-models-saved), which by default will load models to `~.cache/torch/hub`. If you would like to change the directory if this to
for example have a single place where all weights across the models are stored, you can also change


---

## Available Models

This repository includes the following models for evaluation:

- CROMA
- DOFA
- GFM
- RemoteCLIP
- SatMAE
- ScaleMAE
- Skyscript
- SoftCON
- AnySat

---

## Supported Datasets

The following datasets are currently supported:

- GeoBench
- BigEarthNetV2
- Resisc45

---

## Adding New Models and Datasets

To add a new model or dataset for evaluation, follow these steps:

1. **Add a Model Wrapper:**
   - Create a new model wrapper in the [`foundation_models`](foundation_models) folder.
   - Add the new model to `__init__.py` for integration.
   - Register the model in [`factory.py`](factory.py) by adding its name to make it accessible via the `model_type` parameter.

2. **Add a Dataset Wrapper:**
   - Create a new dataset wrapper in the [`datasets`](datasets) folder.
   - Register the dataset in [`factory.py`](factory.py) to ensure access.
   
3. **Configuration Setup:**
   This project is using [hydra](https://hydra.cc/docs/1.3/intro/) for experiment configuation:

   In the configs directory there is a subdirectory for models and dataset, where you need to add
   a config file for the new dataset and model

---

## Running Experiments

To run evaluation on any of the models, you can use the following example:


```bash
export $(cat .env)
export MODEL_SIZE=base #can be base or large
echo "Output Directory": $ODIR
echo "Model Size": $MODEL_SIZE

python src/main.py \
output_dir=${ODIR}/exps/dinov2_cls_linear_probe_benv2_rgb \
model=dinov2_cls_linear_probe \
dataset=benv2_rgb \
lr=0.002 \
task=classification \
num_gpus=0 \
num_workers=8 \
epochs=30 \
warmup_epochs=5 \
seed=13 \
```


The model and dataset arguments are the names of the config.yaml files specified under the `src/configs` directory. Additional arguments can be passed to the command: basically, anything in `src/main.py` that has `cfg.{something}` passing the argument with the command line command will overwrite the configs with the dedicated values.

There is a convenience script for generating such shell scripts for running experiments. 

```bash
scripts/generate_bash_scripts.py
```

You can modify this to your needs and it will generate a different shell script for every experiment you want to run stored in their own folders under `scripts/<dataset>/run_<model>_<dataset>.sh`


You can use the following command to run an experiment:
```bash
cd <path/to/this/repo>
sh scripts/<path/to/your/experiment>.sh
```

## Hyperparameter Tuning

There is also a script included that can optimize hyperparameters with [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) similar to the hydra setup above but with additional parameters for hparam tuning.

The python file `generate_bash_scripts_ray_tune.py` can generate bash scripts that execute the `src/hparam_ray_hdra.py` script with optimizing the learning rate and batch size. The additonal ray relevant parameters are `cfg.ray.{something}` inside that script. Some defaults are provided, but if you need more specific control over ray tune configuration, additional ray arguments can be passed to the command line or a script with the plus sign. For more information you can see how the `generate_bash_scripts_ray_tune.py` configures an experiment.

---

## Contributing

We welcome contributions! If you'd like to add new models, datasets, or evaluation scripts, please submit a pull request, and ensure that you have tested your changes.
