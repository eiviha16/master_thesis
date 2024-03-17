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
    from algorithms.VPG.TM_DDPG import DDPG
    #from algorithms.Q_Network.Double_TMQN import TMQN
    #from algorithms.Q_Network.n_step_Double_TMQN import TMQN
    #from algorithms.Q_Network.TMQN import TMQN
    #from algorithms.policy.RTM import Policy
    from algorithms.policy.CTM import ActorCriticPolicy as Policy
    #from algorithms.Proximal_policy.TM_PPO import PPO
    #from algorithms.Proximal_policy.PPO import PPO
    #from algorithms.policy.DNN import ActorCriticPolicy as Policy
    #from algorithms.policy.RTM import ActorCriticPolicy as Policy
    """
    _config = {'algorithm': 'PPO', 'gamma': config.gamma, 'lam': config.lam, 'clip_range': config.clip, 'batch_size': config.batch_size, 'epochs': config.epochs,
              'hidden_size': config.hidden_size, 'learning_rate': config.learning_rate, 'test_freq': 1, "save": True}"""

    """_config = {
        'algorithm': 'Double_TMQN', 'soft_update_type': 'soft_update_1', 'nr_of_clauses': 1160, 'T': 359,
        's': 9.79, 'y_max': 75, 'y_min': 35, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 6,
        'gamma': 0.974, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 3000,
        'batch_size': config.batch_size, 'epochs': config.epochs, 'test_freq': 50, "save": False, "seed": 42,
        'number_of_state_bits_ta': 7, 'update_grad': config.update_grad, 'update_freq': 10000}"""
    """_config = {
        'algorithm': 'Double_TMQN', 'soft_update_type': 'soft_update_1', 'n_steps': config.n_steps, 'nr_of_clauses': config.nr_of_clauses, 'T': int(config.t * config.nr_of_clauses),
        's': config.specificity, 'y_max': config.y_max, 'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': config.buffer_size,
        'batch_size': config.batch_size, 'epochs': config.epochs, 'test_freq': 50, "save": False, "seed": 42,
        'number_of_state_bits_ta': config.number_of_state_bits_ta, 'update_grad': config.update_grad, 'update_freq': 10000}
    """
    actor = {'nr_of_classes': 2, 'nr_of_clauses': config.a_nr_of_clauses, 'T': int(config.a_nr_of_clauses * config.a_t),
             's': config.a_specificity, 'device': 'CPU', 'weighted_clauses': False,
             'bits_per_feature': config.a_bits_per_feature, "seed": 42, 'number_of_state_bits_ta': config.a_number_of_state_bits_ta}
    critic = {'nr_of_clauses': config.c_nr_of_clauses, 'T': int(config.c_nr_of_clauses * config.c_t), 's': config.c_specificity, 'y_max': config.c_y_max, 'y_min': config.c_y_min, 'device': 'CPU',
              'weighted_clauses': False, 'bits_per_feature': config.c_bits_per_feature, "seed": 42, 'number_of_state_bits_ta': config.c_number_of_state_bits_ta}
    _config = {'algorithm': 'TM_DDPG_2', 'soft_update_type': 'soft_update_2', 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'update_grad': 0, "update_freq": config.update_freq, 'gamma': config.gamma,
               "buffer_size": config.buffer_size, 'actor': actor, 'critic': critic, 'batch_size': config.batch_size, 'epochs': config.epochs, 'test_freq': 1, "save": True}

    env = gym.make("CartPole-v1")
    #env = gym.make("Acrobot-v1")

    #agent = PPO(env, Policy, _config)
    #agent = TMQN(env, Policy, _config)
    agent = DDPG(env, Policy, _config)
    agent.learn(nr_of_episodes=1000)
    score = np.array(agent.best_score)
    #scores = np.array(agent.total_score)
    #score = np.mean(scores)
    return score


def main():
    wandb.init(project="TAC_b_final_2")
    score = objective(wandb.config)
    wandb.log({"score": score})


# actor = {'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 2.0, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, "seed": 42, 'number_of_state_bits_ta': 8}
# critic = {'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 100.0, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, "seed": 42, 'number_of_state_bits_ta': 8}
# config = {'algorithm': 'TM_DDPG', 'soft_update_type': 'soft_update_2', 'update_freq': 6, 'gamma': 0.98, 'actor': actor, 'critic': critic, 'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True}

sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "update_freq": {"values": list(range(1, 10, 1))},
        "batch_size": {"values": list(range(16, 128, 16))},
        "epochs": {"values": list(range(1, 8, 1))},

        "a_t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "a_nr_of_clauses": {"values": list(range(800, 1200, 20))},
        "a_specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "a_bits_per_feature": {"values": list(range(5, 15, 1))},
        "a_number_of_state_bits_ta": {"values": list(range(3, 10, 1))},

        "c_t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "c_nr_of_clauses": {"values": list(range(800, 2000, 50))},
        "c_specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "c_bits_per_feature": {"values": list(range(5, 15, 1))},
        "c_number_of_state_bits_ta": {"values": list(range(3, 10, 1))},

        "c_y_max": {"values": list(range(60, 80, 5))},
        "c_y_min": {"values": list(range(20, 40, 5))},
        "buffer_size": {"values": list(range(500, 10_000))}

    }
}


sweep_id = wandb.sweep(sweep=sweep_configuration, project="TAC_b_final_2")
wandb.agent(sweep_id, function=main, count=10_000)
