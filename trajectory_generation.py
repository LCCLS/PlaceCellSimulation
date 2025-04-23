import numpy as np
import torch
import random


def compute_velocities_from_positions(positions):
    """
    Compute velocity vectors from a list of positions.
    Returns list of (dx, dy) tuples.
    """
    return [
        (positions[i][0] - positions[i - 1][0], positions[i][1] - positions[i - 1][1])
        for i in range(1, len(positions))
    ]


def compute_place_cells_from_positions(positions, grid_size):
    """
    Generate one-hot encoded place cell activations for a list of (x, y) positions.
    Returns a list of one-hot vectors.
    """
    grid_size_squared = grid_size * grid_size
    one_hot_vectors = []

    for pos in positions:
        one_hot_vector = [0] * grid_size_squared
        index = pos[0] * grid_size + pos[1]
        one_hot_vector[index] = 1
        one_hot_vectors.append(one_hot_vector)

    return one_hot_vectors


def generate_single_step_trajectories(grid_size):
    """
    Generate all valid one-step transitions (trajectories) in a grid.
    Returns a list of dictionaries with keys: positions, velocities, place_cell_activations.
    """
    possible_moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    trajectories = []

    for row in range(grid_size):
        for col in range(grid_size):
            current_position = [row, col]
            for dx, dy in possible_moves:
                next_position = [row + dx, col + dy]

                if 0 <= next_position[0] < grid_size and 0 <= next_position[1] < grid_size:
                    trajectories.append({
                        "positions": [current_position, next_position],
                        "velocities": [(dx, dy)],
                        "place_cell_activations": compute_place_cells_from_positions(
                            [current_position, next_position], grid_size
                        ),
                    })

    return trajectories


def generate_random_trajectory(grid_size, trajectory_length, starting_point=None):
    """
    Generate a single random trajectory in a grid with given length.

    Returns:
        dict with positions, velocities, and place_cell_activations
    """
    if starting_point is None:
        starting_point = (
            random.randint(0, grid_size - 1),
            random.randint(0, grid_size - 1),
        )

    positions = [starting_point]
    velocities = []

    while len(velocities) < trajectory_length:
        current_pos = positions[-1]
        possible_moves = [
            (current_pos[0] + dx, current_pos[1] + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= current_pos[0] + dx < grid_size
            and 0 <= current_pos[1] + dy < grid_size
        ]

        possible_moves = [move for move in possible_moves if move != current_pos]
        if not possible_moves:
            break

        next_pos = random.choice(possible_moves)
        velocity = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
        velocities.append(velocity)
        positions.append(next_pos)

    return {
        "positions": positions,
        "velocities": velocities,
        "place_cell_activations": compute_place_cells_from_positions(positions, grid_size)
    }
