import pandas as pd
import numpy as np
import torch

from move import individual_movements
from evaluate import cross_entropy
from trajectory_generation import generate_single_step_trajectories, generate_random_trajectory


def evaluate_single_trajectory(velocities, true_activations, best_individual, device):
    predicted_activations = individual_movements(
        best_individual,
        velocities,
        first_activation=torch.tensor(true_activations[0], dtype=torch.float32, device=device),
    )

    correct_preds = 0
    total_preds = 0
    loss = []

    actual_activations = true_activations[1:]

    for i, pred_tensor in enumerate(predicted_activations):
        predicted_activation = pred_tensor.squeeze(0).detach().cpu().numpy()
        actual_activation = np.array(actual_activations[i])

        if np.argmax(predicted_activation) == np.argmax(actual_activation):
            correct_preds += 1

        loss.append(cross_entropy(predicted_activation, actual_activation))
        total_preds += 1

    avg_loss = np.mean(loss)
    return correct_preds, total_preds, avg_loss


def evaluate_trajectory_predictions(individual, trajectories, device):
    total_correct = 0
    total_predictions = 0
    loss_list = []

    for trajectory in trajectories:
        velocities = trajectory["velocities"]
        true_activations = trajectory["place_cell_activations"]

        correct, preds, avg_loss = evaluate_single_trajectory(
            velocities, true_activations, individual, device
        )

        total_correct += correct
        total_predictions += preds
        loss_list.append(avg_loss)

    return total_correct, total_predictions, np.mean(loss_list)


def evaluate_success_rate_and_cross_entropy(
    individual,
    num_repetitions,
    grid_size,
    trajectory_length,
    device
):
    total_successes = 0
    total_weighted_successes = 0
    total_trajectory_length = 0
    total_avg_step_cross_entropy = []
    total_final_cross_entropy_loss = []
    total_trajectory_accuracies = []

    for _ in range(num_repetitions):
        trajectory_data = generate_random_trajectory(grid_size, trajectory_length)

        velocities = trajectory_data["velocities"]
        true_activations = trajectory_data["place_cell_activations"]

        predicted_activations = individual_movements(
            individual,
            velocities,
            first_activation=torch.tensor(true_activations[0], dtype=torch.float32, device=device),
        )

        trajectory_cross_entropy = []
        correct_steps = 0

        for i, pred_tensor in enumerate(predicted_activations):
            predicted = pred_tensor.squeeze(0).detach().cpu().numpy()
            actual = np.array(true_activations[i + 1])

            trajectory_cross_entropy.append(cross_entropy(predicted, actual))

            if np.argmax(predicted) == np.argmax(actual):
                correct_steps += 1

        avg_ce_step = np.mean(trajectory_cross_entropy)
        total_final_cross_entropy_loss.append(np.sum(trajectory_cross_entropy))
        total_avg_step_cross_entropy.append(avg_ce_step)
        total_trajectory_accuracies.append(correct_steps / trajectory_length)

        pred_final = np.argmax(predicted_activations[-1].squeeze(0).detach().cpu().numpy())
        true_final = np.argmax(true_activations[-1])

        if pred_final == true_final:
            total_successes += 1
            total_weighted_successes += trajectory_length

        total_trajectory_length += trajectory_length

    return (
        total_successes / num_repetitions if num_repetitions else 0,
        total_weighted_successes / total_trajectory_length if total_trajectory_length else 0,
        np.mean(total_avg_step_cross_entropy),
        np.mean(total_final_cross_entropy_loss),
        np.mean(total_trajectory_accuracies)
    )


def test_model_predictions(
    individual,
    grid_size,
    num_output_neurons,
    evaluation_max_trajectory_length=300,
    evaluation_max_repetitions=30,
    evaluation_interval=100,
    device=None
):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    individual = individual.to(device)
    additional_metrics = []

    for traj_len in range(evaluation_interval, evaluation_max_trajectory_length + 1, evaluation_interval):
        metrics = evaluate_success_rate_and_cross_entropy(
            individual,
            num_repetitions=evaluation_max_repetitions,
            grid_size=grid_size,
            trajectory_length=traj_len,
            device=device
        )

        additional_metrics.append({
            "Category": f"RandTraj_{traj_len}__{evaluation_max_repetitions}Reps",
            "Success Rate": metrics[0],
            "Weighted Success Rate": metrics[1],
            "Avg Cross-Entropy per Step": metrics[2],
            "Cross-Entropy Loss per Trajectory": metrics[3],
            "Avg Accuracy per Trajectory": metrics[4],
        })

    return pd.DataFrame(additional_metrics)
