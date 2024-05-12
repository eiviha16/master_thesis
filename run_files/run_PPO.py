import torch
import numpy as np
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Proximal_Policy_Optimization.PPO import PPO
from algorithms.policy.DNN import ActorCriticPolicy as Policy

config = {'env_name': 'cartpole', 'algorithm': 'PPO', 'n_steps': 512, "batch_size": 256, 'gamma': 0.99, 'lam': 0.973,
          'clip_range': 0.35000000000000003, 'epochs': 3, 'hidden_size': 32, 'learning_rate': 0.0006600000000000001,
          'test_freq': 1, 'save': True}
print(config)

env = gym.make("CartPole-v1")
# env = gym.make("Acrobot-v1")


agent = PPO(env, Policy, config)
agent.learn(nr_of_episodes=5000)

from test_policy import test_policy

file = f'./results/PPO/{agent.run_id}/best_model'
model = torch.load(file)
test_policy(f'./results/PPO/{agent.run_id}/final_test_results', model.actor, config["env_name"])
