# Evaluation of Foundation Models for Earth Observation

This repository provides tools for evaluating various foundation models on Earth Observation tasks. For detailed instructions on **pretraining and deploying [DOFA](https://arxiv.org/abs/2403.15356)**, please refer to the [main DOFA repository](https://github.com/zhu-xlab/DOFA).

---

## Setup

To get started, you can install the required dependencies from the `pyproject.toml`:

For this navigate to the root directory of this repository and do:

```
pip install -e .
```

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
- **Spectral Earth** *(in progress)*

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
   - Define the configuration settings in [`config.py`](util/config.py).

---

## Running the Evaluation

To run the evaluation script, use the following command:

```bash
python scripts/exp_config.py
```

This script initiates training for model evaluation on GeoBench with CROMA.

---

## Contributing

We welcome contributions! If you'd like to add new models, datasets, or evaluation scripts, please submit a pull request, and ensure that you have tested your changes.
