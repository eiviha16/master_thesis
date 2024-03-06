"""

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
    #from algorithms.VPG.TM_DDPG import DDPG
    #from algorithms.Q_Network.Double_TMQN import TMQN
    #from algorithms.Q_Network.n_step_Double_TMQN import TMQN
    #from algorithms.Q_Network.TMQN import TMQN
    #from algorithms.policy.RTM import Policy
    from algorithms.Proximal_policy.TM_PPO import PPO
    from algorithms.policy.RTM import ActorCriticPolicy as Policy

    _config = {
        'algorithm': 'Double_TMQN', 'soft_update_type': 'soft_update_1', 'nr_of_clauses': 1160, 'T': 359,
        's': 9.79, 'y_max': 75, 'y_min': 35, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 6,
        'gamma': 0.974, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 3000,
        'batch_size': config.batch_size, 'epochs': config.epochs, 'test_freq': 50, "save": False, "seed": 42,
        'number_of_state_bits_ta': 7, 'update_grad': config.update_grad, 'update_freq': 10000}
    _config = {
        'algorithm': 'Double_TMQN', 'soft_update_type': 'soft_update_1', 'n_steps': config.n_steps, 'nr_of_clauses': config.nr_of_clauses, 'T': int(config.t * config.nr_of_clauses),
        's': config.specificity, 'y_max': config.y_max, 'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': config.buffer_size,
        'batch_size': config.batch_size, 'epochs': config.epochs, 'test_freq': 50, "save": False, "seed": 42,
        'number_of_state_bits_ta': config.number_of_state_bits_ta, 'update_grad': config.update_grad, 'update_freq': 10000}

      actor = {'nr_of_classes': 2, 'nr_of_clauses': config.a_nr_of_clauses, 'T': int(config.a_nr_of_clauses * config.a_t),
             's': config.a_specificity, 'device': 'CPU', 'weighted_clauses': False,
             'bits_per_feature': config.a_bits_per_feature, "seed": 42, 'number_of_state_bits_ta': config.a_number_of_state_bits_ta}
    critic = {'nr_of_clauses': config.c_nr_of_clauses, 'T': int(config.c_nr_of_clauses * config.c_t), 's': config.c_specificity, 'y_max': config.c_y_max, 'y_min': config.c_y_min, 'device': 'CPU',
              'weighted_clauses': False, 'bits_per_feature': config.c_bits_per_feature, "seed": 42, 'number_of_state_bits_ta': config.c_number_of_state_bits_ta}
    _config = {'algorithm': 'TM_DDPG_2', 'soft_update_type': 'soft_update_2', 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'update_freq': config.update_freq, 'gamma': config.gamma,
               'actor': actor, 'critic': critic, 'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True}

    _config = {'algorithm': 'TM_PPO', 'gamma': config.gamma, 'lam': config.lam, "clip": config.clip, 'nr_of_clauses': config.nr_of_clauses, 'T': int(config.nr_of_clauses * config.t), 's': config.specificity,
              'y_max': config.y_max, 'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
              'batch_size': 64, 'epochs': config.epochs, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': config.number_of_state_bits_ta}
    env = gym.make("CartPole-v1")

    agent = PPO(env, Policy, _config)
    #agent = TMQN(env, Policy, _config)
    agent.learn(nr_of_episodes=1_000)
    score = np.array(agent.best_score)
    #scores = np.array(agent.total_score)
    #score = np.mean(scores)
    return score


def main():
    wandb.init(project="TM_PPO_without_norm")
    score = objective(wandb.config)
    wandb.log({"score": score})


# actor = {'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 2.0, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, "seed": 42, 'number_of_state_bits_ta': 8}
# critic = {'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 100.0, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, "seed": 42, 'number_of_state_bits_ta': 8}
# config = {'algorithm': 'TM_DDPG', 'soft_update_type': 'soft_update_2', 'update_freq': 6, 'gamma': 0.98, 'actor': actor, 'critic': critic, 'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True}
#PPO
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.94, 0.98, 0.001))},
        "lam": {"values": list(np.arange(0.94, 0.98, 0.001))},
        "t": {"values": list(np.arange(0.3, 0.9, 0.01))},
        "nr_of_clauses": {"values": list(range(900, 1200, 10))},
        "specificity": {"values": list(np.arange(1.0, 4.0, 0.01))},
        "bits_per_feature": {"values": list(range(5, 15, 1))},
        "number_of_state_bits_ta": {"values": list(range(3, 8, 1))},
        "clip": {"values": list(np.arange(0.001, 0.5001, 0.01))},
        "epochs": {"values": list(range(1, 5, 1))},
        "y_max": {"values": list(np.arange(5.5, 35, 0.5))},
        "y_min": {"values": list(np.arange(0.0, 1.0, 0.1))}
    }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="TM_PPO_without_norm")
wandb.agent(sweep_id, function=main, count=10_000)
"""

import numpy as np
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Q_Network.n_step_Double_TMQN import TMQN
from algorithms.policy.RTM import Policy

#config = {'algorithm': 'n_step_Double_TMQN', 'soft_update_type': 'soft_update_2', 'n_steps': 19, 'nr_of_clauses': 980, 'T': (980 * 0.51), 's': 6.56, 'y_max': 60, 'y_min': 25, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7, 'gamma': 0.977, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 8000, 'batch_size': 80, 'epochs': 2, 'test_freq': 1,  "save": True, 'number_of_state_bits_ta': 5, 'update_grad': 0.05, 'update_freq': 7}
#config = {'algorithm': 'n_step_Double_TMQN', 'soft_update_type': 'soft_update_1', 'n_steps': 17, 'nr_of_clauses': 980, 'T': int(980 * 0.34), 's': 8.58, 'y_max': 60, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 9, 'gamma': 0.992, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 4000, 'batch_size': 80, 'epochs': 2, 'test_freq': 1,  "save": True, 'number_of_state_bits_ta': 6, 'update_grad': 0.05, 'update_freq': 8}
config = {'algorithm': 'n_step_Double_TMQN', 'soft_update_type': 'soft_update_1', 'n_steps': 15, 'nr_of_clauses': 880, 'T': int(880 * 0.59), 's': 9.26, 'y_max': 75, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, 'gamma': 0.976, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 9000, 'batch_size': 32, 'epochs': 5, 'test_freq': 1,  "save": True, 'number_of_state_bits_ta': 8, 'update_grad': 0.859, 'update_freq': 9999}

env = gym.make("CartPole-v1")

agent = TMQN(env, Policy, config)
agent.learn(nr_of_episodes=10_000)

from test_policy import test_policy

save_file = f'results/n_step_Double_TMQN/{agent.run_id}/final_test_results'
tms = torch.load(f'results/n_step_Double_TMQN/{agent.run_id}/best')

agent.target_policy.tm1.set_params(tms[0]['ta_state'], tms[0]['clause_sign'], tms[0]['clause_output'], tms[0]['feedback_to_clauses'])
agent.target_policy.tm2.set_params(tms[1]['ta_state'], tms[1]['clause_sign'], tms[1]['clause_output'], tms[1]['feedback_to_clauses'])

test_policy(save_file, agent.target_policy)
