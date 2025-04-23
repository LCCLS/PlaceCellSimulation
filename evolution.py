import torch
from evotorch import Problem
from evotorch.algorithms import XNES
from evotorch.logging import StdOutLogger, PandasLogger
from models.min_model import PlaceCellNetwork
from objective_function import ObjectiveFunction
from utils import log_device_status


def run_evolution(
    trajectories,
    max_generations,
    patience,
    num_output_neurons,
    device=None
):
    # === Determine device if not explicitly passed ===
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_device_status()

    # === Create model to calculate parameter count ===
    dummy_model = PlaceCellNetwork(input_size=2, output_size=num_output_neurons).to(device)
    solution_length = sum(p.numel() for p in dummy_model.parameters())
    print(f"[✓] Number of network parameters: {solution_length}")

    # === Define the objective function (remains on CPU or GPU agnostic) ===
    objective_function = ObjectiveFunction(
        trajectories,
        num_output_neurons
    )

    # === Setup the optimization problem ===
    problem = Problem(
        objective_sense="min",
        objective_func=objective_function,
        solution_length=solution_length,
        initial_bounds=(-1, 1),
        device=device,
        dtype=torch.float32,  # optional for clarity,
        num_actors=4,  # 👈 Enable parallelism
    )

    # === Create searcher (XNES optimizer) ===
    searcher = XNES(problem, radius_init=0.5)

    # === Logging ===
    stdout_logger = StdOutLogger(searcher)
    pandas_logger = PandasLogger(searcher)

    # === Optimization loop ===
    best_fitness = float("inf")
    no_improvement_count = 0

    for generation in range(max_generations):
        searcher.step()
        current_fitness = searcher.status["best_eval"]

        if current_fitness < best_fitness - 0.005:
            best_fitness = current_fitness
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f"[✓] Early stopping after {patience} stagnant generations.")
            break

        if current_fitness < 0.01:
            print(f"[✓] Converged at generation {generation} (fitness < 0.01).")
            break

    return searcher, pandas_logger
