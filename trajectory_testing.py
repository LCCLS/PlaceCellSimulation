import torch
import pandas as pd

from trajectory_generation import (
    generate_single_step_trajectories_tensor,
    generate_random_trajectory_tensor,
)
from utils import log_device_status
from models.min_model import PlaceCellNetwork


# ──────────────────────────────────────────────────────────────
#  CE helper (probabilities + one-hot)
# ──────────────────────────────────────────────────────────────
def ce_prob_onehot(p, y, eps=1e-12, reduction="mean"):
    p = torch.clamp(p, eps, 1.0)
    ce = -(y * torch.log(p)).sum(dim=-1)
    if reduction == "sum":
        return ce.sum()
    elif reduction == "mean":
        return ce.mean()
    return ce          # 'none'


# ──────────────────────────────────────────────────────────────
#  1.  Accuracy & CE on a *batch* of single-step trajectories
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_single_step_batch(net: PlaceCellNetwork, grid_size: int, device):
    batch = generate_single_step_trajectories_tensor(grid_size, device=device)
    vel, init_p, targets = batch["vel"], batch["init_probs"], batch["targets"]  # vel [T,1,2]

    probs = net.forward_steps(vel, h0=init_p)        # [T,1,N]

    # accuracy (class with max prob)
    pred_idx = probs.squeeze(1).argmax(-1)           # [T]
    true_idx = targets.squeeze(1).argmax(-1)
    acc = (pred_idx == true_idx).float().mean().item()

    # cross-entropy
    ce = ce_prob_onehot(probs, targets, reduction="mean").item()
    return acc, ce


# ──────────────────────────────────────────────────────────────
#  2.  Random long trajectories
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_random_batch(
    net: PlaceCellNetwork,
    grid_size: int,
    traj_len: int,
    reps: int,
    device,
):
    acc_sum, ce_step_sum, ce_traj_sum = 0.0, 0.0, 0.0

    for _ in range(reps):
        traj = generate_random_trajectory_tensor(grid_size, traj_len, device=device)
        vel, init_p, targets = traj["vel"].unsqueeze(0), traj["init_probs"].unsqueeze(0), traj["targets"].unsqueeze(0)

        probs = net.forward_steps(vel, h0=init_p)          # [1,S,N]
        S = probs.shape[1]

        # step-wise CE & accuracy
        ce_steps = ce_prob_onehot(probs, targets, reduction="none").squeeze(0)  # [S]
        ce_step_sum += ce_steps.mean().item()
        ce_traj_sum += ce_steps.sum().item()

        pred_idx = probs.argmax(-1)
        true_idx = targets.argmax(-1)
        acc_sum += (pred_idx == true_idx).float().mean().item()

    return (
        acc_sum / reps,
        ce_step_sum / reps,
        ce_traj_sum / reps,
    )


# ──────────────────────────────────────────────────────────────
#  3.  Public entry – produces a DataFrame identical to before
# ──────────────────────────────────────────────────────────────
def test_model_predictions(
    net: PlaceCellNetwork,
    grid_size: int,
    _unused_num_output_neurons: int,
    evaluation_max_trajectory_length: int = 300,
    evaluation_max_repetitions: int = 30,
    evaluation_interval: int = 100,
    device=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device).eval()
    log_device_status()

    # single-step exhaustive set
    single_acc, single_ce = evaluate_single_step_batch(net, grid_size, device)

    rows = [{
        "Category": "AllSingleSteps",
        "Success Rate": single_acc,
        "Weighted Success Rate": single_acc,     # identical for single-step
        "Avg Cross-Entropy per Step": single_ce,
        "Cross-Entropy Loss per Trajectory": single_ce,
        "Avg Accuracy per Trajectory": single_acc,
    }]

    # random long trajectories
    for S in range(evaluation_interval,
                   evaluation_max_trajectory_length + 1,
                   evaluation_interval):
        acc, ce_step, ce_traj = evaluate_random_batch(
            net, grid_size, S, evaluation_max_repetitions, device
        )
        rows.append({
            "Category": f"RandTraj_{S}__{evaluation_max_repetitions}Reps",
            "Success Rate": acc,
            "Weighted Success Rate": acc,
            "Avg Cross-Entropy per Step": ce_step,
            "Cross-Entropy Loss per Trajectory": ce_traj,
            "Avg Accuracy per Trajectory": acc,
        })

    return pd.DataFrame(rows)
