import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)
from torch.distributions import Categorical


class QNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(QNet, self).__init__()
        # activation
        self.activation = nn.Tanh()
        # self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()

        # layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        x = self.input_layer(input)
        x = self.activation(x)

        x = self.hidden_layer(x)
        x = self.activation(x)

        x = self.output_layer(x)
        return x


class Policy(QNet):
    def __init__(self, input_size, output_size, config):
        super(Policy, self).__init__(input_size, output_size, config['hidden_size'])
        self.optimizer = optim.Adam(self.parameters(), lr=config['learning_rate'])

    def predict(self, input):
        q_vals = self.forward(torch.tensor(np.array(input)))
        return q_vals


class Actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(Actor, self).__init__()
        self.activation = nn.Tanh()
        self.output_activation = nn.Softmax()

        # layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)

        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        x = self.input_layer(input)
        x = self.activation(x)

        x = self.hidden_layer(x)
        x = self.activation(x)

        # x = self.hidden_layer2(x)
        # x = self.activation(x)

        x = self.output_layer(x)
        action_prob = self.output_activation(x)
        return action_prob


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(Critic, self).__init__()
        self.activation = nn.Tanh()

        # layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer_2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, input):
        x = self.input_layer(input)
        x = self.activation(x)

        x = self.hidden_layer(x)
        x = self.activation(x)

        x = self.output_layer(x)
        return x


class ActorCriticPolicy:
    def __init__(self, input_size, output_size, hidden_size, lr):
        self.actor = Actor(input_size, output_size, hidden_size)
        self.critic = Critic(input_size, output_size)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

    def get_action(self, obs):
        obs = torch.tensor(obs)
        action_probs = self.actor(obs)
        dist = Categorical(action_probs)
        actions = dist.sample()
        values = self.critic(obs)
        return actions, values, dist.log_prob(actions)

    def get_best_action(self, obs):
        obs = torch.tensor(obs)
        action_probs = self.actor(obs)
        actions = torch.argmax(action_probs, dim=-1)
        return actions, action_probs

    def evaluate_action(self, obs, actions):
        obs = torch.tensor(obs)
        action_probs = self.actor(obs)
        dist = Categorical(action_probs)
        values = self.critic(obs)
        return actions, values, dist.log_prob(torch.tensor(actions))


class ActorPolicy:
    def __init__(self, input_size, output_size, hidden_size, lr):
        self.actor = Actor(input_size, output_size, hidden_size)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)

    def get_action(self, obs):
        obs = torch.tensor(obs)
        action_probs = self.actor(obs)
        action = torch.multinomial(action_probs, 1).squeeze(dim=-1)
        return action, F.log_softmax(action_probs, dim=-1)

    def get_best_action(self, obs):
        obs = torch.tensor(obs)
        action_probs = self.actor(obs)
        action = torch.argmax(action_probs, dim=-1)
        return action, action_probs
