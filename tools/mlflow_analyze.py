import os
import json
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime
import shutil
from tqdm import tqdm


def parse_mlflow_logs(base_dir: str, model_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Recursively parse MLflow logs and extract experimental statistics.
    
    Args:
        base_dir (str): Base directory containing the logs
        model_names (List[str], optional): List of specific model names to parse. If None, parse all.
    
    Returns:
        pd.DataFrame: DataFrame containing experimental statistics
    """
    all_runs = []
    
    # Get all model directories
    model_dirs = [d for d in os.listdir(base_dir) 
                 if os.path.isdir(os.path.join(base_dir, d))]
    
    if model_names:
        model_dirs = [d for d in model_dirs if d in model_names]
    
    
    for model_dir in model_dirs:
        model_path = os.path.join(base_dir, model_dir)
        
        # Find mlruns directory
        for root, dirs, _ in os.walk(model_path):
            if 'mlruns' in dirs:
                mlruns_path = os.path.join(root, 'mlruns')
                
                # Process each experiment ID directory
                for exp_id in os.listdir(mlruns_path):
                    exp_path = os.path.join(mlruns_path, exp_id)
                    if not os.path.isdir(exp_path) or exp_id == '.trash':
                        continue
                    
                    # Process each run directory
                    for run_id in os.listdir(exp_path):
                        if "trash" in run_id:
                            continue
                        run_path = os.path.join(exp_path, run_id)
                        if not os.path.isdir(run_path):
                            continue
                        
                        run_info = parse_run_directory(run_path, model_dir, exp_id, run_id)
                        if run_info:
                            all_runs.append(run_info)
    
    if not all_runs:
        return pd.DataFrame()
    
    # Convert to DataFrame and handle datetime
    df = pd.DataFrame(all_runs)
    if 'start_time' in df.columns:
        df['start_time'] = pd.to_datetime(df['start_time'], unit='ms')
    if 'end_time' in df.columns:
        df['end_time'] = pd.to_datetime(df['end_time'], unit='ms')
    
    return df

def parse_run_directory(run_path: str, model_name: str, exp_id: str, run_id: str) -> Optional[Dict]:
    """
    Parse a single run directory and extract relevant information.
    
    Args:
        run_path (str): Path to the run directory
        model_name (str): Name of the model
        exp_id (str): Experiment ID
        run_id (str): Run ID
    
    Returns:
        Dict: Dictionary containing run information
    """
    # try:
    #     import pdb
    #     pdb.set_trace()
        # Parse metrics
    metrics_path = os.path.join(run_path, 'metrics')
    metrics = {}
    if os.path.exists(metrics_path):
        for metric_file in os.listdir(metrics_path):
            metric_name = metric_file
            metric_path = os.path.join(metrics_path, metric_file)
            with open(metric_path, 'r') as f:
                # Get the last line for final metric value
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    metric_value = float(last_line.split(' ')[1])
                    metrics[metric_name] = metric_value
    
    # Parse run metadata
    meta_path = os.path.join(run_path, 'meta.yaml')
    meta_data = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta_content = f.read()
            # Parse relevant fields from meta.yaml
            for line in meta_content.split('\n'):
                if line.startswith('start_time:'):
                    meta_data['start_time'] = int(line.split(':')[1].strip())
                elif line.startswith('end_time:'):
                    out = line.split(':')[1].strip()
                    if out == 'null':
                        meta_data['end_time'] = None
                    else:
                        meta_data['end_time'] = int(line.split(':')[1].strip())
                elif line.startswith('status:'):
                    meta_data['status'] = line.split(':')[1].strip()
    
    # Parse parameters
    params_path = os.path.join(run_path, 'params')
    parameters = {}
    if os.path.exists(params_path):
        for param_file in os.listdir(params_path):
            param_name = param_file
            param_path = os.path.join(params_path, param_file)
            with open(param_path, 'r') as f:
                param_value = f.read().strip()
                parameters[param_name] = param_value
    
    # Combine all information
    run_info = {
        'model_name': model_name,
        'experiment_id': exp_id,
        'run_id': run_id,
        **meta_data,
        **parameters,
        **metrics
    }
    
    return run_info
    
    # except Exception as e:
    #     import pdb
    #     pdb.set_trace()
    #     print(f"Error parsing run directory {run_path}: {str(e)}")
    #     return None

# [Previous code for parse_mlflow_logs, safe_float_convert, safe_int_convert, and parse_run_directory remains the same]

def analyze_experiments_by_model(df: pd.DataFrame, output_dir: str) -> Dict:
    """
    Analyze the experimental results organized by model and export to CSV files.
    
    Args:
        df (pd.DataFrame): DataFrame containing experimental results
        output_dir (str): Directory to save the CSV files
    
    Returns:
        Dict: Dictionary containing summary statistics organized by model
    """
    if df.empty:
        return {"error": "No experimental data found"}
    
    # Create output directory if it doesn't exist
    # os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analysis dictionary
    analysis = {
        "overall_summary": {
            "total_experiments": len(df['experiment_id'].unique()),
            "total_runs": len(df),
            "total_models": len(df['model_name'].unique()),
        },
        "models": {}
    }
    
    # Get all metric columns (excluding metadata columns)
    metadata_cols = ['model_name', 'experiment_id', 'run_id', 'status', 'start_time', 'end_time']
    # metric_cols = [col for col in df.select_dtypes(include=['float64', 'int64']).columns 
    #               if col not in metadata_cols]

    metric_cols = {
        "classification": ['train_loss', 'val_macro_acc1', 'val_macro_acc5', 'val_micro_acc1', 'val_micro_acc5', 'test_macro_acc1', 'test_macro_acc5', 'test_micro_acc1', 'test_micro_acc5'],
        "multilabel": ['train_loss', 'val_macro_f1', 'val_micro_f1', 'test_macro_f1', 'test_micro_f1'],
        "segmentation": ['train_loss', 'val_macro_miou', 'val_macro_acc', 'test_macro_miou', 'test_macro_acc']
    }
    
    # Analyze each model separately
    # get unique comb of model_name, experimnet_id, run_id
    # experiment_iter = df.groupby(['model_name', 'experiment_id', 'run_id']).apply(lambda x: x)
    for model_name in tqdm(df['model_name'].unique()):
        model_df = df[df['model_name'] == model_name]

        # TODO need to identify multilabel datasets
        if "benv2" in model_df["dataset"].values[0]:
            run_metric_cols = metric_cols["multilabel"]
        else:
            run_metric_cols = metric_cols[model_df['task'].values[0]]

        model_analysis = {
            "summary": {
                "total_runs": len(model_df),
                "total_experiments": len(model_df['experiment_id'].unique()),
                "average_metrics": {},
                "best_metrics": {},
                "metrics_std": {},
                "metrics_min": {},
                "metrics_max": {},
            },
            "best_runs": {},
            "parameter_distribution": {}
        }
    
        # Analyze metrics
        for metric in run_metric_cols:
            if metric in model_df.columns:
                model_analysis["summary"]["average_metrics"][metric] = model_df[metric].mean()
                model_analysis["summary"]["metrics_std"][metric] = model_df[metric].std()
                model_analysis["summary"]["metrics_min"][metric] = model_df[metric].min()
                model_analysis["summary"]["metrics_max"][metric] = model_df[metric].max()
                
                # Find best run for each metric, given that they exist, otherwise
                # indicates that not finished or error
                if not model_df[metric].isna().all():
                    best_idx = model_df[metric].idxmax()
                    model_analysis["best_runs"][metric] = {
                        "experiment_id": model_df.loc[best_idx, "experiment_id"],
                        "run_id": model_df.loc[best_idx, "run_id"],
                        "value": model_df.loc[best_idx, metric],
                        "parameters": {col: model_df.loc[best_idx, col] 
                                    for col in model_df.columns 
                                    if col not in metadata_cols + run_metric_cols}
                    }
                else:
                    model_analysis["best_runs"][metric] = "N/A"
        
        # Analyze parameter distribution
        param_cols = [col for col in model_df.columns 
                     if col not in metadata_cols + run_metric_cols]
        for param in param_cols:
            model_analysis["parameter_distribution"][param] = (
                model_df[param].value_counts().to_dict()
            )
        
        analysis["models"][model_name] = model_analysis
        
        # Export model-specific results to CSV
        export_model_results_to_csv(model_df, model_name, output_dir)
 
    # Export overall summary
    export_overall_summary_to_csv(df, analysis, output_dir)
    
    return analysis

def export_model_results_to_csv(model_df: pd.DataFrame, model_name: str, output_dir: str):
    """
    Export model-specific results to CSV files.
    """
    # Clean model name for file naming
    safe_model_name = "".join(c if c.isalnum() else "_" for c in model_name)
    
    # Export all runs for the model
    model_df.to_csv(os.path.join(output_dir, f"{safe_model_name}_all_runs.csv"), index=False)
    
    # Export summary statistics
    summary_stats = model_df.describe()
    #summary_stats.to_csv(os.path.join(output_dir, f"{safe_model_name}_summary_stats.csv"))

def export_overall_summary_to_csv(df: pd.DataFrame, analysis: Dict, output_dir: str):
    """
    Export overall summary statistics to CSV files.
    """
    # Create overall metrics summary
    metrics_summary = []
    for model_name, model_data in analysis["models"].items():
        model_metrics = {"model_name": model_name}
        model_metrics.update(model_data["summary"]["average_metrics"])
        metrics_summary.append(model_metrics)
    
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df.to_csv(os.path.join(output_dir, "overall_metrics_summary.csv"), index=False)
    
    # Export best runs across all models
    best_runs = []
    for model_name, model_data in analysis["models"].items():
        for metric, run_data in model_data["best_runs"].items():
            if run_data != "N/A": # runs that have not finished or with error
                best_runs.append({
                    "model_name": model_name,
                    "metric": metric,
                    "value": run_data["value"],
                    "experiment_id": run_data["experiment_id"],
                    "run_id": run_data["run_id"]
                })
    
    best_runs_df = pd.DataFrame(best_runs)
    best_runs_df.to_csv(os.path.join(output_dir, "best_runs_summary.csv"), index=False)

# Example usage function
def analyze_mlflow_experiments(logs_dir: str, output_dir: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Parse MLflow logs and analyze experiments with CSV export.
    
    Args:
        logs_dir (str): Directory containing MLflow logs
        output_dir (str): Directory to save CSV files
    
    Returns:
        Tuple[pd.DataFrame, Dict]: Raw DataFrame and analyzed results
    """
    # Parse logs
    df = parse_mlflow_logs(logs_dir)
    
    # Analyze and export results
    analysis = analyze_experiments_by_model(df, output_dir)
    
    return df, analysis


# Example usage


logs_dir = "/mnt/rg_climate_benchmark/data/experiment_log"
output_dir = "/mnt/rg_climate_benchmark/data/experiment_log/analysis"

# Or parse specific models
# Analyze the results
df, analysis = analyze_mlflow_experiments(logs_dir, output_dir)

# zip the output directory
shutil.make_archive(output_dir, 'zip', output_dir)
