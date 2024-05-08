import random
import torch
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from algorithms.Q_Network.n_step_DQN import DQN
from algorithms.policy.DNN import Policy
import gymnasium as gym


config = {'env_name': 'acrobot', 'n_steps': 50, 'algorithm': 'n_step_DQN', 'gamma': 0.994, 'buffer_size': 50000, 'batch_size': 16, 'epsilon_init': 1.0, 'epsilon_decay': 0.001, 'sampling_iterations': 1, 'hidden_size': 160, 'learning_rate': 0.00623, 'test_freq': 5, 'save': True}
print(config)
#env = gym.make("CartPole-v1")
env = gym.make("Acrobot-v1")


agent = DQN(env, Policy, config)
agent.learn(nr_of_episodes=2500)

from test_policy import test_policy
file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best_model'
model = torch.load(file)
test_policy(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results', model, config["env_name"])