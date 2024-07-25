import torch
import torch.nn as nn
import torch.nn.init as init
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
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        for layer in self.fc_layers:
            x = self.activation(layer(x))
        outputs = self.fc2(x)

        mu = self.tanh(outputs)
        log_std = self.log_std_weight * outputs + self.log_std_bias
        log_std = torch.clamp(log_std, -20, 2)  # Clamping log_std to prevent extreme values
        std = torch.exp(log_std)
        return mu, std

    def get_action(self, x):
        mu, std = self.forward(x)
        eps = torch.randn(mu.shape)
        return mu + eps * std, mu, std

    def get_log_prob(self, x, action):
        mu, std = self.forward(x)
        std = std + 1e-6  # Adding small value to std to prevent division by zero
        log_prob = -0.5 * (torch.sum((action - mu) ** 2 / std ** 2 + 2 * torch.log(std)) + len(action) * np.log(2 * np.pi))

        if len(action.shape) > len(mu.shape):
            mu = mu.unsqueeze(dim=1)

        if len(action.shape) != len(mu.shape):
            raise ValueError("Action and mean have different number of dimensions.")

        return log_prob
    
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
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

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        for layer in self.fc_layers:
            x = self.activation(layer(x))
        outputs = self.fc2(x)

        mu = outputs
        log_std = self.log_std_weight * outputs + self.log_std_bias
        log_std = torch.clamp(log_std, -20, 2)  # Clamping log_std to prevent extreme values
        std = torch.exp(log_std)
        return mu, std

    def get_action(self, x):
        mu, std = self.forward(x)
        eps = torch.randn_like(mu)
        return mu + eps * std, mu, std

    def get_log_prob(self, x, action):
        mu, std = self.forward(x)
        std = std + 1e-6  # Adding small value to std to prevent division by zero
        log_prob = -0.5 * (torch.sum((action - mu) ** 2 / std ** 2 + 2 * torch.log(std), dim=-1) + len(action) * np.log(2 * np.pi))

        if len(action.shape) > len(mu.shape):
            mu = mu.unsqueeze(dim=1)

        if len(action.shape) != len(mu.shape):
            raise ValueError("Action and mean have different number of dimensions.")

        return log_prob'''
    

class ValueFunctionNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, layer_sizes: list):
        """
        Initialize the Value Function Network with intelligent weight initialization.

        :param input_size: Size of the input features.
        :param output_size: Size of the output value.
        :param layer_sizes: List of integers representing the number of units in each hidden layer.
        """
        super(ValueFunctionNetwork, self).__init__()
        
        # List to hold layers
        layers = []
        
        # Input layer
        prev_size = input_size
        
        # Create hidden layers
        for size in layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        # Sequential container
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights and biases of the network using intelligent methods.
        """
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, x):
        """
        Forward pass through the network.

        :param x: Input tensor.
        :return: Output tensor.
        """
        return self.network(x)
    
'''
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
