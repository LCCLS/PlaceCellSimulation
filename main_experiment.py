import time
import os
import pandas as pd
import torch

from models.min_model import PlaceCellNetwork
from evolution import run_evolution  
from trajectory_testing import test_model_predictions
from utils import create_directory, save_model, flatten_metrics_df, log_device_status


def run_single_experiment(
    grid_size,
    replicate,
    trajectories,
    experiment_dir,
    evaluation_max_trajectory_length=300,
    evaluation_max_repetitions=20,
    evaluation_interval=50
):
    # === Create directory for this replicate ===
    iteration_dir = create_directory(
        experiment_dir, f'iteration_{grid_size}x{grid_size}_replicate_{replicate}'
    )

    # === Set and log device ===
    log_device_status()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # === Run evolutionary search ===
    start_time = time.time()
    evo_searcher, _ = run_evolution(
        trajectories=trajectories,
        max_generations=20000,
        patience=250,
        num_output_neurons=grid_size ** 2,
        device=device
    )
    end_time = time.time()
    total_time = end_time - start_time

    generations_run = evo_searcher.status['iter']
    time_per_generation = total_time / generations_run if generations_run > 0 else total_time

    # === Load model with best weights ===
    network = PlaceCellNetwork(2, output_size=grid_size ** 2).to(device)
    network.set_weights_flat(evo_searcher.status['best'])

    param_count = network.get_num_parameters()
    best_fitness = evo_searcher.status.get("best_eval", None)

    save_model(evo_searcher.status['best'], iteration_dir)

    # === Run evaluation ===
    metrics_df = test_model_predictions(
        network,
        grid_size,
        grid_size ** 2,
        evaluation_max_trajectory_length=evaluation_max_trajectory_length,
        evaluation_max_repetitions=evaluation_max_repetitions,
        evaluation_interval=evaluation_interval,
        device=device  # Make sure this is supported
    )
    metrics_df.to_csv(os.path.join(iteration_dir, 'evaluation_metrics.csv'), index=False)
    flattened_metrics = flatten_metrics_df(metrics_df)

    # === Save result row ===
    row = {
        "grid_size": f"{grid_size}x{grid_size}",
        "replicate": replicate + 1,
        "total_time_sec": total_time,
        "time_per_generation_sec": time_per_generation,
        "param_count": param_count,
        "best_fitness": best_fitness,
        "generations_run": generations_run,
    }
    row.update(flattened_metrics)

    iteration_result_df = pd.DataFrame([row])
    iteration_result_df.to_csv(os.path.join(iteration_dir, 'iteration_result.csv'), index=False)

    return row, metrics_df
