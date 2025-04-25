import torch
import torch.nn as nn
import torch.nn.functional as F


class PlaceCellNetwork(nn.Module):
    """
    Minimal recurrent place-cell network.
    The hidden state *and* the returned tensor are **probabilities**
    (soft-maxed), exactly like your original implementation.
    """

    def __init__(
        self,
        input_size: int = 2,
        output_size: int = 10,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.N = output_size
        self.dtype = dtype
        self.device = device or torch.device("cpu")

        self.linear = nn.Linear(input_size, output_size, dtype=dtype, device=self.device)

        k = (1.0 / output_size) ** 0.5
        self.recurrent_weights = nn.Parameter(
            torch.randn(output_size, output_size, dtype=dtype, device=self.device) * k
        )

    # ------------------------------------------------------------------ #
    def forward_steps(self, velocities: torch.Tensor, h0: torch.Tensor | None = None):
        """
        velocities : *batch × S × 2
        h0         : *batch × N   (initial probs)    – optional
        returns    : *batch × S × N   (probabilities)
        """
        *batch, S, _ = velocities.shape
        vel_flat = velocities.reshape(-1, 2)                     # [(Πbatch)*S, 2]
        base_logits = self.linear(vel_flat).reshape(*batch, S, self.N)

        if h0 is None:
            h = torch.zeros(*batch, self.N, device=self.device, dtype=self.dtype)
        else:
            h = h0.to(self.device, self.dtype)

        outs = []
        for t in range(S):
            z = base_logits[..., t, :] + h @ self.recurrent_weights   # logits
            h = F.softmax(z, dim=-1)                                  # probs
            outs.append(h)

        return torch.stack(outs, dim=-2)   # *batch × S × N (probs)

    # ------------------------------------------------------------------ #
    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor | None = None):
        """One-step wrapper kept for any legacy sequential code."""
        x_t = x_t.unsqueeze(-2)                   # *batch × 1 × 2
        out = self.forward_steps(x_t, h_prev)     # *batch × 1 × N
        probs = out.squeeze(-2)                   # *batch × N
        return probs, probs

    # ------------------------------------------------------------------ #
    def set_weights_flat(self, flat: torch.Tensor):
        flat = flat.to(self.device, dtype=self.dtype)
        i = self.linear.weight.numel()
        self.linear.weight.data = flat[:i].reshape_as(self.linear.weight)
        j = i + self.linear.bias.numel()
        self.linear.bias.data   = flat[i:j]
        self.recurrent_weights.data = flat[j:].reshape_as(self.recurrent_weights)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())
