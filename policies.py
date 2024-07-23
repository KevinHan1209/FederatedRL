import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaussianMLPPolicy(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation="tanh", log_std_bias=0.0, log_std_weight=1.0):
        super(GaussianMLPPolicy, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_layers[0])
        self.fc_layers = nn.ModuleList([nn.Linear(h_in, h_out) for h_in, h_out in zip(hidden_layers[:-1], hidden_layers[1:])])
        self.fc2 = nn.Linear(hidden_layers[-1], output_size)

        self.activation = nn.functional.tanh if activation == "tanh" else getattr(F, activation)
        self.log_std_bias = nn.Parameter(torch.tensor(log_std_bias))
        self.log_std_weight = nn.Parameter(torch.tensor(log_std_weight))

    def forward(self, x):
        x = self.activation(self.fc1(x))
        for layer in self.fc_layers:
            x = self.activation(layer(x))
        outputs = self.fc2(x)

        mu = outputs
        log_std = self.log_std_weight * outputs + self.log_std_bias
        std = torch.exp(log_std)
        return mu, std

    def get_action(self, x):
        mu, std = self.forward(x)
        eps = torch.randn(mu.shape)
        return mu + eps * std, mu, std

    def get_log_prob(self, x, action):
        mu, std = self.forward(x)
        log_prob = -0.5 * (torch.sum((action - mu) ** 2 / std ** 2 + 2 * torch.log(std)) + len(action) * np.log(2 * np.pi))

        if len(action.shape) > len(mu.shape):
            mu = mu.unsqueeze(dim=1)

        if len(action.shape) != len(mu.shape):
            raise ValueError("Action and mean have different number of dimensions.")

        return log_prob

# Example usage
policy = GaussianMLPPolicy(input_size=27, hidden_layers=[512, 512, 256, 128], output_size=1)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

# Dummy input and target action for demonstration
x = torch.randn(1, 27)
target_action = torch.randn(1, 1)

# Training loop
for _ in range(200):
    optimizer.zero_grad()
    action, mu, std = policy.get_action(x)
    log_prob = policy.get_log_prob(x, target_action)
    loss = -log_prob.mean()  # Dummy loss for demonstration
    loss.backward()
    optimizer.step()


model_params = []
for i in policy.named_parameters():
    print(i[0] + ': ' + str(i[1].shape))

'''
log_std_bias: torch.Size([])
log_std_weight: torch.Size([])
fc1.weight: torch.Size([512, 27])
fc1.bias: torch.Size([512])
fc_layers.0.weight: torch.Size([512, 512])
fc_layers.0.bias: torch.Size([512])
fc_layers.1.weight: torch.Size([256, 512])
fc_layers.1.bias: torch.Size([256])
fc_layers.2.weight: torch.Size([128, 256])
fc_layers.2.bias: torch.Size([128])
fc2.weight: torch.Size([1, 128])
fc2.bias: torch.Size([1])
'''