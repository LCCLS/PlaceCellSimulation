import numpy as np

def evaluate_path_from_activations(predicted_activations, place_cell_activations):
    """
    Compute total cross-entropy loss between predicted and actual place cell activations
    for a sequence of timesteps.

    Args:
        predicted_activations (list of torch.Tensor): Predicted activations for each timestep.
        place_cell_activations (list of np.array): Ground truth place cell activations (one-hot or soft labels).

    Returns:
        float: Total cross-entropy loss over the trajectory.
    """
    total_loss = 0.0

    # Ground truth skips first activation (due to init step)
    actual_activations = place_cell_activations[1:]

    for i, predicted_tensor in enumerate(predicted_activations):
        predicted = predicted_tensor.squeeze(0).detach().cpu().numpy()
        actual = np.array(actual_activations[i])

        total_loss += cross_entropy(predicted, actual)

    return total_loss


def cross_entropy(predicted_probs, true_labels, eps=1e-12):
    """
    Compute cross-entropy loss between two probability distributions.

    Args:
        predicted_probs (np.ndarray): Model's predicted probability distribution.
        true_labels (np.ndarray): Ground truth one-hot encoded vector or soft distribution.
        eps (float): Small value to avoid log(0).

    Returns:
        float: Cross-entropy loss.
    """
    predicted_probs = np.clip(predicted_probs, eps, 1.0 - eps)
    return -np.sum(true_labels * np.log(predicted_probs))
