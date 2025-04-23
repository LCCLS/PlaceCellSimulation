import torch


def individual_movements(
    individual, velocities, activation_fn="tanh", first_activation=None
):
    all_activations = []

    # Infer device and dtype from the model
    device = next(individual.parameters()).device
    dtype = next(individual.parameters()).dtype

    # Initialize recurrent hidden state
    if first_activation is None:
        initial_velocity = torch.tensor([0, 0], dtype=dtype, device=device).unsqueeze(0)
        output, h_out = individual.forward(initial_velocity, activation_fn=activation_fn)
    else:
        h_out = first_activation
        if not isinstance(h_out, torch.Tensor):
            h_out = torch.tensor(h_out, dtype=dtype)
        h_out = h_out.to(device=device, dtype=dtype).detach().clone().unsqueeze(0)

    # Forward pass for each velocity step
    for v in velocities:
        current_velocity = torch.tensor(v, dtype=dtype, device=device).unsqueeze(0)
        output, h_out = individual.forward(current_velocity, h_out)
        all_activations.append(output)

    return all_activations
