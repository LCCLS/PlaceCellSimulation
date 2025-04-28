import time
import os
import pandas as pd
import torch

from models.min_model import PlaceCellNetwork
from evolution import run_evolution
from trajectory_generation import generate_single_step_trajectories_tensor
from trajectory_testing import test_model_predictions
from utils import create_directory, save_model, flatten_metrics_df, log_device_status


def run_single_experiment(
    grid_size: int,
    replicate: int,
    experiment_dir: str,
    num_actors: int | None,
    evaluation_max_trajectory_length: int = 300,
    evaluation_max_repetitions: int = 20,
    evaluation_interval: int = 50,
):
    """
    Run one evolutionary search replicate (on CPU) and evaluate the resulting network (on GPU if available).

    Returns:
        row: Summary metrics for this replicate
        metrics_df: Detailed evaluation DataFrame
    """
    # === Create directory for this replicate ===
    iteration_dir = create_directory(
        experiment_dir,
        f"iteration_{grid_size}x{grid_size}_replicate_{replicate}"
    )

    # === Log device status and set devices ===
    log_device_status()
   # device_evo  = torch.device("cpu")
   # device_eval = torch.device("cpu")

    device_evo  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_eval = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # === Generate trajectories on CPU ===
    trajectories = generate_single_step_trajectories_tensor(
        grid_size,
        device=device_evo
    )

    # === Run evolutionary search on CPU ===
    start_time = time.time()
    evo_result = run_evolution(
        trajectories=trajectories,
        max_generations=1000000,
        patience=250,
        num_output_neurons=grid_size ** 2,
        device=device_evo,
        num_actors=num_actors,
    )
    total_time = time.time() - start_time

    # === Unpack evolution results ===
    searcher        = evo_result["searcher"]
    best_solution   = evo_result["best"]
    best_weights    = best_solution.values if hasattr(best_solution, 'values') else best_solution
    best_fitness    = evo_result["best_eval"]
    generations_run = evo_result.get(
        "generations_run",
        searcher.status.get("iter", 0)
    )
    time_per_generation = (
        total_time / generations_run if generations_run > 0 else total_time
    )

    # === Build network on GPU (or CPU fallback) and load best weights ===
    network = PlaceCellNetwork(
        input_size=2,
        output_size=grid_size ** 2,
        dtype=torch.float32,
        device=device_eval
    )
    # Convert best_weights to tensor on correct device/dtype
    best_weights_tensor = torch.as_tensor(
        best_weights,
        device=device_eval,
        dtype=network.dtype
    )
    network.set_weights_flat(best_weights_tensor)
    param_count = network.get_num_parameters()

    # === Save best weights ===
    save_model(best_weights_tensor, iteration_dir)

    # Free up any unused GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # === Evaluate model on GPU/CPU ===
    metrics_df = test_model_predictions(
        net=network,
        grid_size=grid_size,
        _unused_num_output_neurons=grid_size ** 2,
        evaluation_max_trajectory_length=evaluation_max_trajectory_length,
        evaluation_max_repetitions=evaluation_max_repetitions,
        evaluation_interval=evaluation_interval,
        device=device_eval,
    )
    metrics_df.to_csv(
        os.path.join(iteration_dir, "evaluation_metrics.csv"),
        index=False
    )
    flattened_metrics = flatten_metrics_df(metrics_df)

    # === Save summary row ===
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

    pd.DataFrame([row]).to_csv(
        os.path.join(iteration_dir, "iteration_result.csv"),
        index=False
    )

    return row, metrics_df

