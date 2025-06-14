"""
Run ONE replicate of the experiment.
Repeat via Slurm job arrays instead of an internal Python loop.
"""

import argparse
import os
import time
import torch
import pandas as pd

from utils import create_experiment_directory
from main_experiment import run_single_experiment


def main(grid_size: int, rep_id: int, num_actors: int | None):
    # ── 1.  Prepare data & paths ────────────────────────────────────── #
    experiment_dir = create_experiment_directory(f"grid_{grid_size}")

    print(f"[✓] CUDA available: {torch.cuda.is_available()}")
    print(f"[✓] Grid {grid_size}×{grid_size}  |  Replicate {rep_id}")
    print(f"[✓] Actors per evolutionary run: {num_actors or 'all available'}")

    # ── 2.  Run a single replicate ──────────────────────────────────── #
    t0 = time.time()
    result_row, metrics_df = run_single_experiment(
        grid_size,
        rep_id,
        experiment_dir,
        num_actors=num_actors,
    )
    elapsed = time.time() - t0
    print(f"[✓] Finished in {elapsed:.2f} s")

    # ── 3.  Persist results for this replicate ──────────────────────── #
    results_csv = os.path.join(experiment_dir, "results.csv")
    pd.DataFrame([result_row]).to_csv(results_csv, index=False)
    print(f"[✓] Saved results → {results_csv}")


# ── CLI / Slurm entry point ─────────────────────────────────────────── #
if __name__ == "__main__":
    # Defaults from environment if available
    grid_default = int(os.getenv("GRID_SIZE", "5"))
    actors_env = os.getenv("SLURM_CPUS_PER_TASK")
    default_actors = int(actors_env) if actors_env and actors_env.isdigit() else None
    replicate_default = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid_size", "-g",
        type=int,
        default=grid_default,
        help="Grid size (default from env GRID_SIZE or 5×5)"
    )
    parser.add_argument(
        "--num_actors", "-a",
        type=int,
        default=default_actors,
        help="CPU actors per evolutionary search (default from SLURM_CPUS_PER_TASK or all available)",
    )
    parser.add_argument(
        "--rep_id", "-r",
        type=int,
        default=replicate_default,
        help="Replicate index (default from SLURM_ARRAY_TASK_ID or 0)",
    )
    args = parser.parse_args()

    main(args.grid_size, args.rep_id, args.num_actors)
