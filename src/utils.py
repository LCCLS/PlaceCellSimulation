import pandas as pd
import numpy as np
import os
import torch
import datetime


def create_experiment_directory(base_dir="experiment"):
    date_str = datetime.datetime.now().strftime("%d-%m-%Y")
    time_str = datetime.datetime.now().strftime("%H-%M-%S")  # Avoid colons in folder names
    experiment_dir = os.path.join(base_dir, date_str, time_str)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def create_directory(base_dir, *subdirs):
    directory_path = os.path.join(base_dir, *map(str, subdirs))
    os.makedirs(directory_path, exist_ok=True)
    return directory_path


def create_grid_size_directory(experiment_dir, grid_size):
    subset_dir = os.path.join(experiment_dir, f"grid_size_{grid_size}")
    os.makedirs(subset_dir, exist_ok=True)
    return subset_dir


def create_trajectory_directory(experiment_dir, trajectory_type):
    traj_dir = os.path.join(experiment_dir, f"{trajectory_type}")
    os.makedirs(traj_dir, exist_ok=True)
    return traj_dir


def create_network_architecture_directory(experiment_dir, network_architecture):
    architecture_dir = os.path.join(experiment_dir, f"{network_architecture}")
    os.makedirs(architecture_dir, exist_ok=True)
    return architecture_dir


def create_iteration_directory(subset_dir, iteration):
    iteration_dir = os.path.join(subset_dir, f"iteration_{iteration + 1}")
    os.makedirs(iteration_dir, exist_ok=True)
    return iteration_dir


def save_logger(pandas_log, iteration_directory):
    evolution_df = pandas_log.to_dataframe().reset_index()
    evolution_df.to_csv(os.path.join(iteration_directory, "metrics.csv"), index=False)


def save_model(best_individual, iteration_directory):
    model_path = os.path.join(iteration_directory, "XNES.pth")
    torch.save(best_individual, model_path)
    print(f"[✓] Saved model to: {model_path}")


def average_df_and_save_metrics(all_metrics, subset_dir):
    descriptions = all_metrics[0].iloc[:, 0]
    numeric_columns = [df.iloc[:, 1:] for df in all_metrics]

    df = (
        pd.concat(numeric_columns)
        .replace(0, np.nan)
        .reset_index()
        .groupby("index")
        .mean()
    )
    df.insert(0, "Description", descriptions)
    output_path = os.path.join(subset_dir, "average_evaluation_metrics.csv")
    df.to_csv(output_path, index=False)
    print(f"[✓] Averaged metrics saved to: {output_path}")


def calculate_trajectory_coverage(trajectory, grid_size):
    possible_moves = set()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(grid_size):
        for j in range(grid_size):
            for d in directions:
                ni, nj = i + d[0], j + d[1]
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    possible_moves.add(((i, j), (ni, nj)))

    trajectory_moves = {
        (trajectory[i], trajectory[i + 1])
        for i in range(len(trajectory) - 1)
    }

    coverage_percentage = (len(trajectory_moves) / len(possible_moves)) * 100.0
    return coverage_percentage


def flatten_metrics_df(metrics_df):
    flattened = {}
    for _, row in metrics_df.iterrows():
        category = row["Category"]
        for col in metrics_df.columns:
            if col == "Category":
                continue
            key = f"{category}_{col.replace(' ', '_')}"
            flattened[key] = row[col]
    return flattened


def log_device_status():
    if torch.cuda.is_available():
        print(f"[✓] CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("[!] CUDA not available. Using CPU.")
