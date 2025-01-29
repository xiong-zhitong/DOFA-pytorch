import os

models = ["dinov2_cls"]
experiments = [
    {
        "model": "dinov2_cls",
        "dataset": "benv2_rgb",
        "task": "classification",
        "epochs": 30,
        "lr_min": 0.001,
        "lr_max": 0.1,
        "batch_choices": [64, 128, 256],
        "warmup_epochs": 3,
    }
]


def generate_bash_scripts(experiments, out_dir="."):
    os.makedirs(out_dir, exist_ok=True)
    for exp in experiments:
        model = exp["model"]
        dataset = exp["dataset"]
        dataset_dir = os.path.join(out_dir, "tune", dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        epochs = exp["epochs"]
        task = exp["task"]
        batch_choices = exp["batch_choices"]
        # batch_choices_str = " ".join(str(x) for x in batch_choices)
        batch_choices_str = "[" + ",".join(str(x) for x in batch_choices) + "]"

        lr_min = exp["lr_min"]
        lr_max = exp["lr_max"]
        warmup_epochs = exp.get("warmup_epochs", 0)

        script_name = f"run_{model}_{dataset}.sh"
        script_path = os.path.join(dataset_dir, script_name)

        # Generate script content
        script_content = f"""#!/bin/bash
echo "Contents of the current directory:"
ls -lah

export CUDA_VISIBLE_DEVICES=0
export GEO_BENCH_DIR=/mnt/data/cc_benchmark


model="{model}"
dataset="{dataset}"
lr_min="{lr_min}"
lr_max="{lr_max}"
task="{task}"
batch_choices_str="{batch_choices_str}"


epochs="{epochs}"
warmup_epochs="{warmup_epochs}"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\\n' | wc -l)

/home/toolkit/.conda/envs/dofaEnv/bin/python src/hparam_ray_hydra.py \\
output_dir=/mnt/results/nils/exps/${{model}}_${{dataset}} \\
model=${{model}} \\
dataset=${{dataset}} \\
task=${{task}} \\
num_gpus=${{num_gpus}} \\
num_workers=8 \\
epochs=${{epochs}} \\
warmup_epochs=${{warmup_epochs}} \\
seed=42 \\
+ray.lr_min=${{lr_min}} \\
+ray.lr_max=${{lr_max}} \\
+ray.batch_choices=${{batch_choices_str}} \\
"""

        with open(script_path, "w") as f:
            f.write(script_content)

        os.chmod(script_path, 0o755)


if __name__ == "__main__":
    # Generate scripts in the same directory as this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    generate_bash_scripts(experiments, out_dir=script_dir)
