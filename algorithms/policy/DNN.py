import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)


class QNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, c=1, action_std=0.5):
        super(QNet, self).__init__()
        self.c = c
        # activation
        self.activation = nn.Tanh()
        self.output_activation = nn.Sigmoid()

        # layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size * 2)

    def forward(self, input):
        x = self.input_layer(input)
        x = self.activation(x)

        x = self.hidden_layer(x)
        x = self.activation(x)

        x = self.output_layer(x)
        #action = self.output_activation(x)
        return x #action * self.c


class Policy(QNet):
    def __init__(self, input_size, output_size, config):
        super(Policy, self).__init__(input_size, output_size, config['hidden_size'], config['c'])
        self.optimizer = optim.Adam(self.parameters(), lr=config['learning_rate'])

    def predict(self, input):
        q_vals = self.forward(torch.tensor(np.array(input)))
        return q_vals