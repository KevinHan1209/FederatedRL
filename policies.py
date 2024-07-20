import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianMLPPolicy(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation="tanh", std_bias=0.0):
        super(GaussianMLPPolicy, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_layers[0])
        self.fc_layers = nn.ModuleList([nn.Linear(h_in, h_out) for h_in, h_out in zip(hidden_layers[:-1], hidden_layers[1:])])
        # Adjust final layer output size to match desired action space dimension (1)
        self.fc2 = nn.Linear(hidden_layers[-1], output_size)
        self.activation = nn.functional.tanh if activation == "tanh" else getattr(F, activation)
        self.std_bias = nn.Parameter(torch.tensor(std_bias))

    def forward(self, x):
        x = self.activation(self.fc1(x))
        for layer in self.fc_layers:
            x = self.activation(layer(x))
        outputs = self.fc2(x)

        # No slicing needed for action space dimension of 1
        mu = outputs
        std = F.softplus(outputs) + self.std_bias
        return mu, std

    def get_action(self, x):
        mu, std = self.forward(x)
        eps = torch.randn(mu.shape)
        return mu + eps * std, mu, std

    def get_log_prob(self, x, action):
        mu, std = self.forward(x)
        log_prob = -0.5 * (torch.sum((action-mu)**2/std**2 + 2 * torch.log(std)) + len(action) * np.log(2 * np.pi))

        # Handle potential dimension mismatch for action space
        if len(action.shape) > len(mu.shape):
            # Expand mu to match action space dimension
            mu = mu.unsqueeze(dim=1)  # Add a new dimension at index 1

        # Ensure both tensors have the same number of dimensions
        if len(action.shape) != len(mu.shape):
            raise ValueError("Action and mean have different number of dimensions.")

        # Sum over the appropriate dimensions for log probability
        log_prob = log_prob.sum(dim=list(range(1, len(action.shape))))

        return log_prob
