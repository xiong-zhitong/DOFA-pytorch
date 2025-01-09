import subprocess

# Define all the experiments
########################-DOFA-########################
experiments = [
    {
        "model": "dinov2_seg",
        "dataset": "geobench_nzcattle",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 30,
        "lr": 0.002,
        "warmup_epochs": 3,
    },
    {
        "model": "dinov2_seg",
        "dataset": "geobench_pv4ger_seg",
        "task": "segmentation",
        "batch_size": 32,
        "epochs": 20,
        "lr": 0.001,
        "warmup_epochs": 3,
    },
    {
        "model": "anysat_cls",
        "dataset": "geobench_pv4ger_cls",
        "task": "classification",
        "batch_size": 128,
        "lr": 0.0001,
        "epochs": 100,
    },
    {
        "model": "dofa_seg",
        "dataset": "geobench_NeonTree_3",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 20,
        "lr": 0.005,
        "warmup_epochs": 3,
    },
    {
        "model": "dofa_seg",
        "dataset": "geobench_chesapeake",
        "task": "segmentation",
        "batch_size": 84,
        "epochs": 20,
        "lr": 0.005,
        "warmup_epochs": 3,
    },
    {
        "model": "dofa_seg",
        "dataset": "geobench_cashew",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 20,
        "lr": 0.005,
        "warmup_epochs": 3,
    },
    {
        "model": "dofa_seg",
        "dataset": "geobench_SAcrop_9",
        "task": "segmentation",
        "batch_size": 84,
        "epochs": 20,
        "lr": 0.005,
        "warmup_epochs": 3,
    },
    {
        "model": "dofa_seg",
        "dataset": "geobench_pv4ger_seg",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 20,
        "lr": 0.001,
        "warmup_epochs": 3,
    },
    {
        "model": "dofa_seg",
        "dataset": "geobench_nzcattle",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 20,
        "lr": 0.001,
        "warmup_epochs": 3,
    },
    {
        "model": "dofa_cls",
        "dataset": "geobench_so2sat_cls",
        "task": "classification",
        "batch_size": 256,
        "lr": 0.05,
        "epochs": 50,
    },
    {
        "model": "dofa_cls",
        "dataset": "geobench_pv4ger_cls",
        "task": "classification",
        "batch_size": 256,
        "lr": 0.05,
        "epochs": 50,
    },
    {
        "model": "dofa_cls",
        "dataset": "geobench_brick_kiln_13",
        "task": "classification",
        "batch_size": 128,
        "lr": 0.05,
        "epochs": 10,
    },
    {
        "model": "dofa_cls",
        "dataset": "geobench_forestnet_9",
        "task": "classification",
        "batch_size": 128,
        "lr": 0.01,
        "epochs": 10,
    },
    {
        "model": "dofa_cls",
        "dataset": "geobench_eurosat_13",
        "task": "classification",
        "batch_size": 256,
        "lr": 0.05,
        "epochs": 50,
    },
]

# Run each experiment
for exp in experiments:
    print(f"Running experiment: {exp['model']} on {exp['dataset']}")
    # exp["epochs"] = 1  # This is for debug
    if not "warmup_epochs" in exp.keys():
        exp["warmup_epochs"] = 0
    subprocess.run(
        [
            "bash",
            "scripts/run.sh",  # Path to the template script
            exp["model"],
            exp["dataset"],
            exp["task"],
            str(exp["batch_size"]),
            str(exp["lr"]),
            str(exp["epochs"]),
            str(exp["warmup_epochs"]),
        ],
        check=True,
    )
    print(f"Completed: {exp['model']} on {exp['dataset']}")
