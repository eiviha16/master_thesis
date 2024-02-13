import wandb
import numpy as np
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


wandb.login(key="74a10e58809253b0e1f243f34bb17d8f34c21e59")

def objective(config):

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Q_Network.Double_TMQN import TMQN
    from algorithms.policy.RTM import Policy

    _config = {
        'soft_update_type': 'soft_update_2', 'algorithm': 'Double_TMQN', 'nr_of_clauses': config.nr_of_clauses, 'T': int(config.nr_of_clauses * config.threshold),
        's': config.specificity, 'y_max': config.y_max, 'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.total_bits_per_feature,
        'gamma': config.gamma, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000,
        'batch_size': 64, 'epochs': 1, 'test_freq': 5, "save": False, "seed": 42,
        'number_of_state_bits_ta': config.number_of_state_bits_ta, 'update_grad': 0.05, 'update_freq': config.update_freq}


    env = gym.make("CartPole-v1")

    agent = TMQN(env, Policy, _config)
    agent.learn(nr_of_episodes=5000)
    scores = np.array(agent.total_score)
    score = np.mean(scores)
    return score

def main():
    wandb.init(project="Double-DQN")
    score = objective(wandb.config)
    wandb.log({"score": score})


sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "threshold": {"max": 1.0, "min": 0.01},
        "nr_of_clauses": {"max": 2500, "min": 100},
        "specificity": {"max": 10.0, "min": 1.0},
        "total_bits_per_feature": {"max": 20, "min": 1},
        "gamma": {"max": 1.00, "min": 0.50},
        "y_max": {"max": 100, "min": 60},
        "y_min": {"max": 40, "min": 0},
        "update_freq": {"max": 30, "min": 1},
        "number_of_state_bits_ta": {"max": 25, "min": 1},
    }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Double-DQN")
wandb.agent(sweep_id, function=main, count=10_000)