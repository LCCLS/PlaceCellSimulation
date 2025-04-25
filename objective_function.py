import torch
import numpy as np
from models.min_model import PlaceCellNetwork


# ---------- Probability-based CE (Torch) -------------------------------- #
def ce_prob_onehot(pred_probs: torch.Tensor,
                   true_onehot: torch.Tensor,
                   eps: float = 1e-12,
                   reduction: str = "mean") -> torch.Tensor:
    """
    Cross-entropy given probabilities + one-hot labels.
    `reduction` can be 'mean' (default), 'sum', or 'none'.
    """
    p  = torch.clamp(pred_probs, eps, 1.0)
    ce = -(true_onehot * torch.log(p)).sum(dim=-1)   # [...,]
    if reduction == "sum":
        return ce.sum()
    elif reduction == "mean":
        return ce.mean()
    return ce


# ---------- EvoTorch objective (fully batched) ------------------------- #
class ObjectiveFunction:
    """
    • Network keeps soft-maxed hidden state and returns probabilities.
    • Loss uses `ce_prob_onehot` (pure Torch).
    • Fully batched over all trajectories and steps.
    """

    def __init__(self, trajectories, num_output_neurons, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.N = num_output_neurons
        self.vel_tensor   = trajectories["vel"]        # [T,1,2]
        self.init_probs   = trajectories["init_probs"] # [T,N]
        self.targets_1hot = trajectories["targets"]    # [T,1,N]

        # Parameter count (for EvoTorch)
        self.param_len = PlaceCellNetwork(
            2, self.N, device=self.device
        ).get_num_parameters()

    # -------------------------------------------------------------------- #
    def __call__(self, flat_weights: torch.Tensor) -> float:
        # build candidate network
        net = PlaceCellNetwork(2, self.N).to(self.device)
        net.set_weights_flat(flat_weights)

        # forward once over ALL trajectories & steps (probabilities)
        probs = net.forward_steps(self.vel_tensor, h0=self.init_probs)   # [T,S,N]

        # cross-entropy (mean over T·S)
        loss = ce_prob_onehot(
            probs.flatten(0, 1),               # [(T*S), N]
            self.targets_1hot.flatten(0, 1),    # [(T*S), N]
            reduction="mean",
        )
        return loss.item()
