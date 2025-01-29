# Evaluation of Foundation Models for Earth Observation

This repository provides tools for evaluating various foundation models on Earth Observation tasks. For detailed instructions on **pretraining and deploying [DOFA](https://arxiv.org/abs/2403.15356)**, please refer to the [main DOFA repository](https://github.com/zhu-xlab/DOFA).

---

## Setup

To get started, you can install the required dependencies from the `pyproject.toml`:

For this navigate to the root directory of this repository and do:

```
pip install -e .
```

You will also need to for the moment install the ViT Adapter part below.

### To use [ViT Adapter](https://arxiv.org/abs/2205.08534)
```bash
cd src/foundation_models/modules/ops/
sh make.sh
```


### Model Weights
Pretrained model weights are available on [Hugging Face](https://huggingface.co/XShadow/GeoFMs).

You can set this environment variable in your terminal with:

```shell
export MODEL_WEIGHTS_DIR=/your/custom/path
```

When using any of the FMs, the init method will check whether it can find the pre-trained checkpoint of the respective FM in the above `MODEL_WEIGHTS_DIR` and download it there if not found. If you do not change the env
variable, the default will be `./fm_weights`.

Some models depend on [torch hub](https://pytorch.org/docs/stable/hub.html#where-are-my-downloaded-models-saved), which by default will load models to `~.cache/torch/hub`. If you would like to change the directory if this to
for example have a single place where all weights across the models are stored, you can also change

```shell
export TORCH_HOME=/your/custom/path
```

---

## Available Models

This repository includes the following models for evaluation:

- CROMA
- DOFA
- GFM
- PanOpticOn
- RemoteCLIP
- SatMAE
- ScaleMAE
- Skyscript
- SoftCON

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

## Running an experiment

To run the evaluation script, use the following command:

```bash
python main.py model=dinov2_cls dataset=benv2_rgb
```

The model and dataset arguments are the names of the config.yaml files specified under the src/configs directory. Additional argumentes can be passed to the command, basically, anything that you see in `src/main.py` and has `cfg.{something}` passing the argument with the command line command will overwrite the configs with the dedicated values.

For example:

```bash
python main.py model=dinov2_cls dataset=benv2_rgb output_dir=experiments/dinov2_benv2_rgb  batch_size=32 num_gpus=1 epochs=100 lr=1e-4
```

overwrites the output directory where experiments are stored, batch size, epochs and learning rate.

## Hyperparameter Tuning

There is also a script included that can optimize hyperparameters with [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) similar to the hydra setup above but with additional parameters for hparam tuning.

The python file `generate_bash_scripts_ray_tune.py` can generate bash scripts that execute the `src/hparam_ray_hdra.py` script with optimizing the learning rate and batch size. The additonal ray relevant parameters are `cfg.ray.{something}` inside that script. Some defaults are provided, but if you need more specific control over ray tune configuration, additional ray arguments can be passed to the command line or a script with the plus sign. For more information you can see how the `generate_bash_scripts_ray_tune.py` configures an experiment.

---

## Contributing

We welcome contributions! If you'd like to add new models, datasets, or evaluation scripts, please submit a pull request, and ensure that you have tested your changes.
