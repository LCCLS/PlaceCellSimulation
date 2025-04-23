import numpy as np
import torch
from models.min_model import PlaceCellNetwork
from move import individual_movements
from evaluate import evaluate_path_from_activations


class ObjectiveFunction:
    """
    Objective function wrapper for EvoTorch.

    Evaluates a solution (flat weights) by loading it into a PlaceCellNetwork
    and running it across a set of trajectories.
    """

    def __init__(self, trajectories, num_output_neurons, device=None):
        """
        Args:
            trajectories (list of dict): Each dict contains 'velocities' and 'place_cell_activations'.
            num_output_neurons (int): Number of output neurons in the PlaceCellNetwork.
            device (torch.device, optional): Device to run the evaluations on.
        """
        self.trajectories = trajectories
        self.num_output_neurons = num_output_neurons
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, solution):
        """
        Evaluate the solution (flat weight vector).

        Args:
            solution (torch.Tensor): Flat tensor of weights.

        Returns:
            float: Average fitness across all trajectories.
        """
        # Build and load weights into model
        model = PlaceCellNetwork(input_size=2, output_size=self.num_output_neurons).to(self.device)
        model.set_weights_flat(solution)

        fitness_scores = []

        for traj in self.trajectories:
            velocities = traj["velocities"]
            place_cell_activations = traj["place_cell_activations"]

            predicted_activations = individual_movements(
                model,
                velocities,
                first_activation=place_cell_activations[0]
            )

            loss = evaluate_path_from_activations(predicted_activations, place_cell_activations)
            fitness_scores.append(loss)

        return np.mean(fitness_scores)
