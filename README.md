# Evaluation of Foundation Models for Earth Observation

This repository provides tools for evaluating various foundation models on Earth Observation tasks. For detailed instructions on **pretraining and deploying [DOFA](https://arxiv.org/abs/2403.15356)**, please refer to the [main DOFA repository](https://github.com/zhu-xlab/DOFA).

---

## Setup

To get started, ensure you have the following dependencies installed:

```bash
pip install torch==2.1.2
pip install torchvision==0.16.2
pip install numpy==1.26.4
pip install openmim
mim install mmengine
mim install "mmcv==2.1.0"
mim install mmsegmentation
```

### Model Weights
Pretrained model weights are available on [Hugging Face](https://huggingface.co/XShadow/GeoFMs). Download the necessary weights for your evaluation tasks.

---

## Available Models

This repository includes the following models for evaluation:

- **ScaleMAE**
- **DOFA**
- **GFM**
- **CROMA**
- **Dinov2** (variants: large, base, with or without registers)

---

## Supported Datasets

The following datasets are currently supported:

- **GeoBench**
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
sh scripts/train_linear_geobench_cls_croma.sh
```

This script initiates training for model evaluation on GeoBench with CROMA.

---

## Contributing

We welcome contributions! If you'd like to add new models, datasets, or evaluation scripts, please submit a pull request, and ensure that you have tested your changes.
