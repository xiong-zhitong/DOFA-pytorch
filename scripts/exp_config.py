import subprocess

# Define all the experiments
'''
    ########################-DinoV2-########################
    {
        "model": "dinov2_base_seg",
        "dataset": "geobench_chesapeake",
        "task": "segmentation",
        "batch_size": 84,
        "epochs": 20,
        "lr": 0.005,
    },
    {
        "model": "dinov2_seg",
        "dataset": "geobench_pv4ger_seg",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 20,
        "lr": 0.005,
    },
    {
        "model": "dinov2_base_seg",
        "dataset": "geobench_NeonTree_3",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 20,
        "lr": 0.005,
    },
    {
        "model": "dinov2_base_cls",
        "dataset": "geobench_pv4ger_cls",
        "task": "classification",
        "lr": 0.5,
        "batch_size": 256,
        "epochs": 50,
    },
    {
        "model": "dinov2_base_seg",
        "dataset": "geobench_nzcattle",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 20,
        "lr": 0.001,
    },

'''
experiments = [
    ########################-DOFA-########################
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
        "batch_size": 12,
        "epochs": 20,
        "lr": 0.005,
        "warmup_epochs": 3,
    },
    {
        "model": "dofa_seg",
        "dataset": "geobench_SAcrop_10",
        "task": "segmentation",
        "batch_size": 80,
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
        "lr": 0.005,
        "warmup_epochs": 3,
    },
    {
        "model": "dofa_seg",
        "dataset": "geobench_nzcattle",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 20,
        "lr": 0.005,
        "warmup_epochs": 3,
    },
    {
        "model": "dofa_cls",
        "dataset": "geobench_so2sat_cls",
        "task": "classification",
        "batch_size": 256,
        "lr": 0.5,
        "epochs": 50,
    },
    {
        "model": "dofa_cls",
        "dataset": "geobench_pv4ger_cls",
        "task": "classification",
        "batch_size": 256,
        "lr": 0.5,
        "epochs": 50,
    },
    {
        "model": "dofa_cls",
        "dataset": "geobench_brick_kiln_13",
        "task": "classification",
        "batch_size": 256,
        "lr": 0.5,
        "epochs": 50,
    },
     {
        "model": "dofa_cls",
        "dataset": "geobench_forestnet_9",
        "task": "classification",
        "batch_size": 256,
        "lr": 0.5,
        "epochs": 50,
    },
     {
        "model": "dofa_cls",
        "dataset": "geobench_eurosat_13",
        "task": "classification",
        "batch_size": 256,
        "lr": 0.5,
        "epochs": 50,
    },

    ########################-SoftCon-########################
    {
        "model": "softcon_seg",
        "dataset": "geobench_cashew_13",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 20,
        "lr": 0.01,
    },
    {
        "model": "softcon_cls",
        "dataset": "geobench_eurosat_13",
        "task": "classification",
        "batch_size": 256,
        "epochs": 50,
        "lr": 0.5,
    },
    {
        "model": "softcon_seg",
        "dataset": "geobench_SAcrop_13",
        "task": "segmentation",
        "batch_size": 84,
        "epochs": 20,
        "lr": 0.05,
    },
    {
        "model": "softcon_cls",
        "dataset": "geobench_brick_kiln_13",
        "task": "classification",
        "batch_size": 256,
        "epochs": 50,
        "lr": 0.5,
    },
    {
        "model": "softcon_cls",
        "dataset": "geobench_so2sat_13",
        "task": "classification",
        "batch_size": 256,
        "epochs": 50,
        "lr": 0.5,
    },
    ########################-ScaleMAE-########################
    {
        "model": "scalemae_seg",
        "dataset": "geobench_pv4ger_seg",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 20,
        "lr": 0.005,
    },
    {
        "model": "scalemae_seg",
        "dataset": "geobench_NeonTree",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 20,
        "lr": 0.001,
    },
    {
        "model": "scalemae_cls",
        "dataset": "geobench_pv4ger_cls",
        "task": "classification",
        "batch_size": 128,
        "lr": 1,
        "epochs": 50,
    },
    {
        "model": "scalemae_seg",
        "dataset": "geobench_chesapeake",
        "task": "segmentation",
        "batch_size": 84,
        "epochs": 20,
        "lr": 0.005,
    },
    {
        "model": "scalemae_seg",
        "dataset": "geobench_nzcattle",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 20,
        "lr": 0.005,
    },
    ########################-Panopticon-########################
    
    ########################-Croma-########################
    {
        "model": "croma_cls",
        "dataset": "geobench_eurosat_12",
        "task": "classification",
        "batch_size": 256,
        "epochs": 50,
        "lr": 0.5,
    },
    {
        "model": "croma_cls",
        "dataset": "geobench_so2sat_12",
        "task": "classification",
        "batch_size": 256,
        "epochs": 50,
        "lr": 0.5,
    },
    {
        "model": "croma_seg",
        "dataset": "geobench_SAcrop_12",
        "task": "segmentation",
        "batch_size": 84,
        "epochs": 20,
        "lr": 0.005,
    },
    {
        "model": "croma_seg",
        "dataset": "geobench_cashew_12",
        "task": "segmentation",
        "batch_size": 12,
        "epochs": 20,
        "lr": 0.001,
    },
    {
        "model": "croma_cls",
        "dataset": "geobench_brick_kiln_12",
        "task": "classification",
        "batch_size": 256,
        "epochs": 50,
        "lr": 0.5,
    },
    ########################-GFM-########################
    {
        "model": "gfm_seg",
        "dataset": "geobench_pv4ger_seg",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 20,
        "lr": 0.005,
    },
    {
        "model": "gfm_cls",
        "dataset": "geobench_pv4ger_cls",
        "task": "classification",
        "batch_size": 256,
        "lr": 0.5,
        "epochs": 50,
    },
    {
        "model": "gfm_seg",
        "dataset": "geobench_NeonTree",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 20,
        "lr": 0.001,
    },
    {
        "model": "gfm_seg",
        "dataset": "geobench_chesapeake",
        "task": "segmentation",
        "batch_size": 84,
        "epochs": 20,
        "lr": 0.005,
    },
    {
        "model": "gfm_seg",
        "dataset": "geobench_nzcattle",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 20,
        "lr": 0.005,
    },
    ########################-SatMAE-########################
    {
        "model": "satmae_cls",
        "dataset": "geobench_eurosat_10",
        "task": "classification",
        "batch_size": 256,
        "lr": 0.5,
        "epochs": 50,
    },
    {
        "model": "satmae_seg_rgb",
        "dataset": "geobench_nzcattle",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 20,
        "lr": 0.005,
    },
    {
        "model": "satmae_seg",
        "dataset": "geobench_SAcrop_10",
        "task": "segmentation",
        "batch_size": 84,
        "epochs": 20,
        "lr": 0.005,
    },
    {
        "model": "satmae_seg",
        "dataset": "geobench_cashew_10",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 20,
        "lr": 0.001,
    },
    {
        "model": "satmae_cls",
        "dataset": "geobench_brick_kiln_10",
        "task": "classification",
        "batch_size": 256,
        "lr": 0.1,
        "epochs": 50,
    },
    {
        "model": "satmae_seg_rgb",
        "dataset": "geobench_NeonTree",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 20,
        "lr": 0.001,
    },
    {
        "model": "satmae_cls",
        "dataset": "geobench_so2sat_10band",
        "task": "classification",
        "batch_size": 256,
        "lr": 0.5,
        "epochs": 50,
    },
    {
        "model": "satmae_seg_rgb",
        "dataset": "geobench_pv4ger_seg",
        "task": "segmentation",
        "batch_size": 16,
        "epochs": 20,
        "lr": 0.005,
    },
    {
        "model": "satmae_seg_rgb",
        "dataset": "geobench_chesapeake",
        "task": "segmentation",
        "batch_size": 84,
        "epochs": 20,
        "lr": 0.005,
    },
]

# Run each experiment
for exp in experiments:
    print(f"Running experiment: {exp['model']} on {exp['dataset']}")
    #exp["epochs"] = 1  # This is for debug
    subprocess.run(
        [
            "bash", "scripts/run.sh",  # Path to the template script
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

