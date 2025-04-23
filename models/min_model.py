import torch
import torch.nn as nn
import torch.nn.functional as F


class PlaceCellNetwork(nn.Module):
    def __init__(self, input_size=2, output_size=10, dtype=torch.float32, device=None):
        super(PlaceCellNetwork, self).__init__()

        self.input_size = input_size + 1  # bias term
        self.output_size = output_size
        self.dtype = dtype
        self.device = device if device is not None else torch.device("cpu")

        self.linear = nn.Linear(self.input_size, self.output_size, dtype=self.dtype, device=self.device)

        self.recurrent_weights = nn.Parameter(
            torch.randn(self.output_size, self.output_size, dtype=self.dtype, device=self.device)
            * torch.sqrt(torch.tensor(1.0 / self.output_size, dtype=self.dtype, device=self.device))
        )

    def forward(self, x, h_out=None):
        x = x.to(self.device, dtype=self.dtype)

        batch_size = x.shape[0]
        bias = torch.ones(batch_size, 1, dtype=self.dtype, device=self.device)
        x = torch.cat((x, bias), dim=1)

        if h_out is None:
            h_out = torch.zeros(batch_size, self.output_size, dtype=self.dtype, device=self.device)
        else:
            h_out = h_out.to(self.device, dtype=self.dtype)

        x = self.linear(x)
        x = x.unsqueeze(0)
        x += torch.matmul(h_out, self.recurrent_weights)
        x = F.softmax(x, dim=2)
        h_out = x

        return x, h_out

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def get_weights_flat(self):
        return torch.cat([p.flatten() for p in self.parameters()]).to(self.device, dtype=self.dtype)

    def set_weights_flat(self, flat_weights):
        flat_weights = flat_weights.to(self.device, dtype=self.dtype)
        offset = 0
        for param in self.parameters():
            numel = param.numel()
            param.data = flat_weights[offset:offset + numel].reshape(param.shape).to(self.device, dtype=self.dtype)
            offset += numel
