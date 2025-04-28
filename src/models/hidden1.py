import torch
import torch.nn as nn
import torch.nn.functional as F


class PlaceCellNetwork(nn.Module):
    def __init__(self, input_size=2, hidden_size=20, output_size=10, dtype=torch.float32, device=None):
        super(PlaceCellNetwork, self).__init__()

        self.input_size = input_size + 1
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.device = device if device is not None else torch.device("cpu")

        self.hidden_layer = nn.Linear(self.input_size, self.hidden_size, dtype=dtype, device=self.device)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size, dtype=dtype, device=self.device)
        self.feedback_weights = nn.Parameter(
            torch.randn(self.output_size, self.hidden_size, dtype=dtype, device=self.device)
            * torch.sqrt(torch.tensor(1.0 / self.output_size, dtype=dtype, device=self.device))
        )

    def forward(self, x, prev_output=None):
        x = x.to(self.device)
        batch_size = x.shape[0]

        bias = torch.ones(batch_size, 1, dtype=x.dtype, device=self.device)
        x = torch.cat((x, bias), dim=1)

        if prev_output is None:
            prev_output = torch.zeros(batch_size, self.output_size, dtype=x.dtype, device=self.device)

        hidden_in = self.hidden_layer(x) + torch.matmul(prev_output, self.feedback_weights)
        hidden = torch.tanh(hidden_in)

        output = self.output_layer(hidden)
        output = F.softmax(output, dim=1)

        return output, output  # returning output as new_prev_output

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def get_weights_flat(self):
        return torch.cat([p.flatten() for p in self.parameters()])

    def set_weights_flat(self, flat_weights):
        offset = 0
        for param in self.parameters():
            numel = param.numel()
            param.data = flat_weights[offset:offset + numel].reshape(param.shape).to(param.device)
            offset += numel
