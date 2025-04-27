# evolution.py
# ─────────────────────────────────────────────────────────────
# Runs SNES on a single GPU (no Ray actors) to minimize memory while leveraging CUDA.

from __future__ import annotations
import torch
from evotorch import Problem
from evotorch.algorithms import SNES
from evotorch.logging import PandasLogger

from models.min_model import PlaceCellNetwork
from objective_function import ObjectiveFunction
from utils import log_device_status

def run_evolution(
    trajectories,
    max_generations: int,
    patience: int,
    num_output_neurons: int,
    device: torch.device | None = None,
    num_actors: int | None = 0,  # 0 → GPU-only evaluation
):
    """
    Run an evolutionary search (SNES) on the given trajectories,
    entirely on GPU (no CPU parallel Ray actors).

    Args:
        trajectories: dict of tensors for the objective function.
        max_generations: maximum number of generations to run.
        patience: gens with no improvement for early stopping.
        num_output_neurons: dimensionality of the model output (grid_size**2).
        device: device for the search (GPU recommended).
        num_actors: number of Ray actors for evaluation (0 for GPU-only).

    Returns:
        A dict with the searcher, pandas_logger, best solution, fitness, and stats.
    """
    # 1. Choose device and log status
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_device_status()

    # 2. Build a dummy model to count parameters
    dummy = PlaceCellNetwork(2, num_output_neurons).to(device)
    solution_len = dummy.get_num_parameters()
    print(f"[✓] Number of network parameters: {solution_len}")

    # 3. Create the objective function
    objective = ObjectiveFunction(trajectories, num_output_neurons, device)

    # 4. Define the EvoTorch problem
    problem = Problem(
        objective_sense="min",
        objective_func=objective,
        solution_length=solution_len,
        initial_bounds=(-1, 1),
        device=device,
        num_actors=num_actors,  # 0 = no Ray actors
    )

    # 5. Initialize the SNES searcher
    searcher = SNES(
        problem,
        radius_init=0.5,
    )

    # 6. Attach a pandas logger for diagnostics
    pandas_logger = PandasLogger(searcher)

    # 7. Manual global-best bookkeeping
    global_best: torch.Tensor | None = None
    global_best_fval = float("inf")
    stagnant = 0

    # 8. Main optimization loop
    for gen in range(1, max_generations + 1):
        searcher.step()

        pop_best      = searcher.status["pop_best"]
        pop_best_eval = searcher.status["pop_best_eval"]

        # update global best
        if pop_best_eval < global_best_fval:
            global_best_fval = pop_best_eval
            global_best      = pop_best.clone()
            stagnant = 0
        else:
            stagnant += 1

        # print progress every 100 generations
        if gen % 100 == 0:
            print(f"[▹] Gen {gen:5d}  best_eval = {global_best_fval:.6f}")

        # early stopping
        if stagnant >= patience:
            print(f"[✓] Early stopping at gen {gen} (no improvement in {patience} gens).")
            break
        if global_best_fval < 0.01:
            print(f"[✓] Converged at gen {gen} (fitness < 0.01).")
            break
    else:
        gen = max_generations
        print(f"[▹] Final gen {gen:5d}  best_eval = {global_best_fval:.6f}")

    return {
        "searcher":        searcher,
        "pandas_logger":   pandas_logger,
        "best":            global_best,
        "best_eval":       global_best_fval,
        "pop_best":        pop_best,
        "pop_best_eval":   pop_best_eval,
        "generations_run": gen,
    }
