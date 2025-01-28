import os

models = ["dinov2_seg", "gfm_seg", "satmae_seg_rgb"]
experiments = [
    {
        "model": "dinov2_seg",
        "dataset": "flair2_rgb",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 30,
        "lr": 0.002,
        "warmup_epochs": 3,
    },
    {
        "model": "dinov2_seg",
        "dataset": "loveda_rgb",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 30,
        "lr": 0.002,
        "warmup_epochs": 3,
    },
    {
        "model": "dinov2_seg",
        "dataset": "caffe_rgb",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 30,
        "lr": 0.002,
        "warmup_epochs": 3,
    },
    # GFM RGB SEG
    {
        "model": "gfm_seg",
        "dataset": "flair2_rgb",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 30,
        "lr": 0.002,
        "warmup_epochs": 3,
    },
    {
        "model": "gfm_seg",
        "dataset": "loveda_rgb",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 30,
        "lr": 0.002,
        "warmup_epochs": 3,
    },
    {
        "model": "gfm_seg",
        "dataset": "caffe_rgb",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 30,
        "lr": 0.002,
        "warmup_epochs": 3,
    },
    # SATMAE seg
    {
        "model": "satmae_seg_rgb",
        "dataset": "flair2_rgb",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 30,
        "lr": 0.002,
        "warmup_epochs": 3,
    },
    {
        "model": "satmae_seg_rgb",
        "dataset": "loveda_rgb",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 30,
        "lr": 0.002,
        "warmup_epochs": 3,
    },
    {
        "model": "satmae_seg_rgb",
        "dataset": "caffe_rgb",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 30,
        "lr": 0.002,
        "warmup_epochs": 3,
    },
    # RCF
    {
        "model": "rcf_seg",
        "dataset": "flair2_rgb",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 30,
        "lr": 0.002,
        "warmup_epochs": 3,
    },
    {
        "model": "rcf_seg",
        "dataset": "loveda_rgb",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 30,
        "lr": 0.002,
        "warmup_epochs": 3,
    },
    {
        "model": "rcf_seg",
        "dataset": "caffe_rgb",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 30,
        "lr": 0.002,
        "warmup_epochs": 3,
    },
    {
        "model": "rcf_cls",
        "dataset": "benv2_rgb",
        "task": "classification",
        "batch_size": 16,
        "epochs": 30,
        "lr": 0.002,
        "warmup_epochs": 3,
    },
]


def generate_bash_scripts(experiments, out_dir="."):
    os.makedirs(out_dir, exist_ok=True)
    for exp in experiments:
        model = exp["model"]
        dataset = exp["dataset"]
        dataset_dir = os.path.join(out_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        batch_size = exp["batch_size"]
        lr = exp["lr"]
        epochs = exp["epochs"]
        task = exp["task"]
        warmup_epochs = exp.get("warmup_epochs", 0)

        script_name = f"run_{model}_{dataset}.sh"
        script_path = os.path.join(dataset_dir, script_name)

        lr = exp["lr"]

        # Generate script content
        script_content = f"""#!/bin/bash
echo "Contents of the current directory:"
ls -lah

export CUDA_VISIBLE_DEVICES=0
export GEO_BENCH_DIR=/mnt/data/cc_benchmark


model="{model}"
dataset="{dataset}"
batch_size="{batch_size}"
lr="{lr}"
epochs="{epochs}"
warmup_epochs="{warmup_epochs}"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\\n' | wc -l)

/home/toolkit/.conda/envs/dofaEnv/bin/python src/main.py \\
output_dir=/mnt/results/nils/exps/${{model}}_${{dataset}} \\
model=${{model}} \\
dataset=${{dataset}} \\
lr=${{lr}} \\
task=${{task}} \\
num_gpus=${{num_gpus}} \\
num_workers=8 \\
epochs=${{epochs}} \\
warmup_epochs=${{warmup_epochs}} \\
seed=42 \\
"""

        with open(script_path, "w") as f:
            f.write(script_content)

        os.chmod(script_path, 0o755)


if __name__ == "__main__":
    # Generate scripts in the same directory as this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    generate_bash_scripts(experiments, out_dir=script_dir)
