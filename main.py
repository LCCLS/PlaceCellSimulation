import time
import os
import pandas as pd
import argparse
import torch
from joblib import Parallel, delayed

from trajectory_generation import generate_single_step_trajectories
from utils import create_experiment_directory
from main_experiment import run_single_experiment


def run_all_experiments(grid_size, num_replicates):
    experiment_dir = create_experiment_directory(f"grid_{grid_size}")
    use_gpu = torch.cuda.is_available()

    all_tasks = []
    for replicate in range(num_replicates):
        trajectories = generate_single_step_trajectories(grid_size)
        all_tasks.append((grid_size, replicate, trajectories, experiment_dir))

    print(f"[✓] Starting {num_replicates} replicates for grid size {grid_size}...")
    print(f"[✓] CUDA available: {use_gpu} → Running {'sequentially' if use_gpu else 'in parallel'}.")

    start = time.time()

    if use_gpu:
        # Run sequentially to avoid GPU contention
        results_with_metrics = [run_single_experiment(*task) for task in all_tasks]
    else:
        # Parallel CPU-based execution
        results_with_metrics = Parallel(n_jobs=num_replicates)(
            delayed(run_single_experiment)(*task) for task in all_tasks
        )

    end = time.time()

    results = [result for result, _ in results_with_metrics]
    results_df = pd.DataFrame(results)

    output_csv_path = os.path.join(experiment_dir, "results.csv")
    results_df.to_csv(output_csv_path, index=False)

    aggregated_results = results_df.groupby(["grid_size"]).mean().reset_index()
    aggregated_csv_path = os.path.join(experiment_dir, "aggregated_results.csv")
    aggregated_results.to_csv(aggregated_csv_path, index=False)

    print(f"[✓] Done in {end - start:.2f} seconds.")
    print(f"[✓] Results saved to: {output_csv_path}")
    print(f"[✓] Aggregated results saved to: {aggregated_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=3, help="Grid size (e.g., 3 for 3x3)")
    parser.add_argument("--replicates", type=int, default=10, help="Number of repetitions")
    args = parser.parse_args()

    run_all_experiments(grid_size=args.grid_size, num_replicates=args.replicates)
s