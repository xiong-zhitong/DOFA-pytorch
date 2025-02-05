import os
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.


# models = ["dinov2_seg", "gfm_seg", "satmae_seg_rgb"]
experiments = [
    # {
    #     "model": "dinov2_seg",
    #     "dataset": "flair2_rgb",
    #     "task": "segmentation",
    #     "batch_size": 16,
    #     "epochs": 30,
    #     "lr": 0.002,
    #     "warmup_epochs": 3,
    # },
    # {
    #     "model": "dinov2_seg",
    #     "dataset": "loveda_rgb",
    #     "task": "segmentation",
    #     "batch_size": 16,
    #     "epochs": 30,
    #     "lr": 0.002,
    #     "warmup_epochs": 3,
    # },
    # {
    #     "model": "dinov2_seg",
    #     "dataset": "caffe_rgb",
    #     "task": "segmentation",
    #     "batch_size": 16,
    #     "epochs": 30,
    #     "lr": 0.002,
    #     "warmup_epochs": 3,
    # },
    # # GFM RGB SEG
    # {
    #     "model": "gfm_seg",
    #     "dataset": "flair2_rgb",
    #     "task": "segmentation",
    #     "batch_size": 16,
    #     "epochs": 30,
    #     "lr": 0.002,
    #     "warmup_epochs": 3,
    # },
    # {
    #     "model": "gfm_seg",
    #     "dataset": "loveda_rgb",
    #     "task": "segmentation",
    #     "batch_size": 16,
    #     "epochs": 30,
    #     "lr": 0.002,
    #     "warmup_epochs": 3,
    # },
    # {
    #     "model": "gfm_seg",
    #     "dataset": "caffe_rgb",
    #     "task": "segmentation",
    #     "batch_size": 16,
    #     "epochs": 30,
    #     "lr": 0.002,
    #     "warmup_epochs": 3,
    # },
    # # SATMAE seg
    # {
    #     "model": "satmae_seg_rgb",
    #     "dataset": "flair2_rgb",
    #     "task": "segmentation",
    #     "batch_size": 16,
    #     "epochs": 30,
    #     "lr": 0.002,
    #     "warmup_epochs": 3,
    # },
    # {
    #     "model": "satmae_seg_rgb",
    #     "dataset": "loveda_rgb",
    #     "task": "segmentation",
    #     "batch_size": 16,
    #     "epochs": 30,
    #     "lr": 0.002,
    #     "warmup_epochs": 3,
    # },
    # {
    #     "model": "satmae_seg_rgb",
    #     "dataset": "caffe_rgb",
    #     "task": "segmentation",
    #     "batch_size": 16,
    #     "epochs": 30,
    #     "lr": 0.002,
    #     "warmup_epochs": 3,
    # },
    # # RCF
    # {
    #     "model": "rcf_seg",
    #     "dataset": "flair2_rgb",
    #     "task": "segmentation",
    #     "batch_size": 16,
    #     "epochs": 30,
    #     "lr": 0.002,
    #     "warmup_epochs": 3,
    # },
    # {
    #     "model": "rcf_seg",
    #     "dataset": "loveda_rgb",
    #     "task": "segmentation",
    #     "batch_size": 16,
    #     "epochs": 30,
    #     "lr": 0.002,
    #     "warmup_epochs": 3,
    # },
    # {
    #     "model": "rcf_seg",
    #     "dataset": "caffe_rgb",
    #     "task": "segmentation",
    #     "batch_size": 16,
    #     "epochs": 30,
    #     "lr": 0.002,
    #     "warmup_epochs": 3,
    # },
    {
        "model": "dofa_cls_lora",
        "dataset": "geobench_eurosat",
        "task": "classification",
        "epochs": 10,
        "lr": 0.002,
        "warmup_epochs": 1,
        "batch_size": 32,
    },

    {
        "model": "dofa_cls_linear_probe",
        "dataset": "geobench_eurosat",
        "task": "classification",
        "epochs": 30,
        "lr": 0.002,
        "warmup_epochs": 3,
    },

    {
        "model": "dofa_cls_linear_probe",
        "dataset": "benv2_s2",
        "task": "classification",
        "epochs": 30,
        "lr": 0.002,
        "warmup_epochs": 3,
    },
    {
        "model": "dinov2_cls_linear_probe",
        "dataset": "benv2_rgb",
        "task": "classification",
        "epochs": 30,
        "lr": 0.002,
        "warmup_epochs": 3,
    },
    {
        "model": "dinov2_cls_linear_probe",
        "dataset": "geobench_eurosat_rgb",
        "task": "classification",
        "epochs": 1,
        "lr": 0.002,
        "warmup_epochs": 5,
    },
]


REPO_PATH=os.getenv("REPO_PATH")
assert REPO_PATH is not None, "REPO_PATH environment variable must be set"

SEED=13
BATCH_SIZE=400
MODEL_SIZE='base' #can be 'base' or 'large'
NUM_WORKERS=8

# assert ODIR is not None, "Please set the ODIR environment variable in your .env file to the output directory where logs will be"

def generate_bash_scripts(experiments, out_dir="."):
    os.makedirs(out_dir, exist_ok=True)
    for exp in experiments:
        model = exp["model"]
        dataset = exp["dataset"]
        dataset_dir = os.path.join(out_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        batch_size = exp.get("batch_size", BATCH_SIZE)

        lr = exp["lr"]
        epochs = exp["epochs"]
        task = exp["task"]
        warmup_epochs = exp.get("warmup_epochs", 0)

        script_name = f"run_{model}-{MODEL_SIZE}-{dataset}.sh"
        script_path = os.path.join(dataset_dir, script_name)

        lr = exp["lr"]

        # Generate script content
        script_content = f"""#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export $(cat {REPO_PATH}/.env)
export MODEL_SIZE={MODEL_SIZE}
echo "Output Directory": $ODIR
echo "Model Size": $MODEL_SIZE

model={model}
dataset={dataset}
batch_size={batch_size}
lr={lr}
epochs={epochs}
warmup_epochs={warmup_epochs}
task={task}
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\\n' | wc -l)

python {REPO_PATH}/src/main.py \\
output_dir=${{ODIR}}/exps/${{model}}_${{dataset}} \\
model=${{model}} \\
dataset=${{dataset}} \\
lr=${{lr}} \\
task=${{task}} \\
num_gpus=${{num_gpus}} \\
num_workers={NUM_WORKERS} \\
epochs=${{epochs}} \\
warmup_epochs=${{warmup_epochs}} \\
seed={SEED} \\
"""

        with open(script_path, "w") as f:
            f.write(script_content)

        os.chmod(script_path, 0o755)


if __name__ == "__main__":
    # Generate scripts in the same directory as this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    generate_bash_scripts(experiments, out_dir=script_dir)
