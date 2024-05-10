import random
import torch
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from algorithms.Q_Networks.n_step_DQN import DQN
from algorithms.policy.DNN import Policy
import gymnasium as gym


config = {'env_name': 'acrobot', 'n_steps': 44, "tau": 0.001, 'algorithm': 'n_step_DQN', 'gamma': 0.997, 'buffer_size': 9_000, 'batch_size': 80, 'epsilon_init': 0.9, 'epsilon_decay': 0.002, 'epsilon_min': 0.03, 'hidden_size': 160, 'learning_rate': 0.0007, 'test_freq': 5, 'save': True}
print(config)
#env = gym.make("CartPole-v1")
env = gym.make("Acrobot-v1")


agent = DQN(env, Policy, config)
agent.learn(nr_of_episodes=2500)

from test_policy import test_policy
file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best_model'
model = torch.load(file)
test_policy(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results', model, config["env_name"])