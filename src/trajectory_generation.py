"""
trajectory_generation.py
────────────────────────────────────────────────────────────
Tensor-based generators for the new pipeline.
Each function returns ready-to-use tensors on the specified device.

Shapes
------
vel        : [T, S, 2]  or [S, 2]   (dx, dy)
init_probs : [T, N]     or [N]
targets    : [T, S, N]  or [S, N]

where T = #trajectories, S = #steps, N = grid_size²
"""

from __future__ import annotations
import torch
import itertools
import random


# --------------------------------------------------------------------- #
def _one_hot(idx: int, length: int, device) -> torch.Tensor:
    v = torch.zeros(length, device=device)
    v[idx] = 1.0
    return v


# --------------------------------------------------------------------- #
# 1. Exhaustive single-step transitions
# --------------------------------------------------------------------- #
def generate_single_step_trajectories_tensor(grid_size: int, device=None):
    device = device or torch.device("cpu")
    N = grid_size * grid_size
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    vel_list, init_list, tgt_list = [], [], []

    for r, c in itertools.product(range(grid_size), range(grid_size)):
        for dx, dy in moves:
            nr, nc = r + dx, c + dy
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                vel_list.append([[dx, dy]])                         # [1,2]
                init_list.append(_one_hot(r * grid_size + c,  N, device))
                tgt_list .append(_one_hot(nr * grid_size + nc, N, device).unsqueeze(0))

    vel        = torch.tensor(vel_list, dtype=torch.float32, device=device)  # [T,1,2]
    init_probs = torch.stack(init_list)                                       # [T,N]
    targets    = torch.stack(tgt_list)                                        # [T,1,N]
    return {"vel": vel, "init_probs": init_probs, "targets": targets}


# --------------------------------------------------------------------- #
# 2. One random trajectory of arbitrary length
# --------------------------------------------------------------------- #
def generate_random_trajectory_tensor(
    grid_size: int,
    traj_len: int,
    starting_point=None,
    device=None,
):
    """
    Generates ONE random trajectory of length ≤ traj_len.

    Returns tensors:
        vel        : [S, 2]
        init_probs : [N]
        targets    : [S, N]
    """
    device = device or torch.device("cpu")
    N = grid_size * grid_size
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # --- choose start --------------------------------------------------
    if starting_point is None:
        pos = (random.randrange(grid_size), random.randrange(grid_size))
    else:
        pos = starting_point

    positions, velocities = [pos], []

    while len(velocities) < traj_len:
        r, c = positions[-1]

        # all neighbours except stay-put, staying inside grid
        legal = [
            (r + dx, c + dy)
            for dx, dy in moves
            if 0 <= r + dx < grid_size and 0 <= c + dy < grid_size
        ]

        if not legal:          # dead end
            break

        nxt = random.choice(legal)
        velocities.append((nxt[0] - r, nxt[1] - c))
        positions .append(nxt)

    S = len(velocities)
    vel_tensor = torch.tensor(velocities, dtype=torch.float32, device=device)  # [S,2]

    init_probs = _one_hot(
        positions[0][0] * grid_size + positions[0][1], N, device
    )

    targets = torch.zeros(S, N, device=device)
    for t, (r, c) in enumerate(positions[1:]):        # skip t=0
        targets[t, r * grid_size + c] = 1.0

    return {"vel": vel_tensor, "init_probs": init_probs, "targets": targets}
