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
    #from algorithms.policy.CTM import ActorCriticPolicy as Policy
    from algorithms.Proximal_policy.TM_PPO import PPO
    #from algorithms.Proximal_policy.PPO import PPO
    #from algorithms.policy.DNN import ActorCriticPolicy as Policy
    from algorithms.policy.RTM import ActorCriticPolicy as Policy
    actor = {"max_update_p": config.a_max_update_p, "min_update_p": config.a_min_update_p, 'nr_of_clauses': config.a_nr_of_clauses, 'T': int(config.a_nr_of_clauses * config.a_t), 's': config.a_specificity,
             'y_max': 100, 'y_min': 0, 'bits_per_feature': config.a_bits_per_feature, 'number_of_state_bits_ta': config.a_number_of_state_bits_ta}
    critic = {"max_update_p": config.c_max_update_p, "min_update_p": 0.0,
             'nr_of_clauses': config.c_nr_of_clauses, 'T': int(config.c_nr_of_clauses * config.c_t),
             's': config.c_specificity,
             'y_max': config.c_y_max, 'y_min': config.c_y_min, 'bits_per_feature': config.c_bits_per_feature,
             'number_of_state_bits_ta': config.c_number_of_state_bits_ta}

    _config = {'comment': 'newest', 'algorithm': 'TM_PPO', 'gamma': config.gamma, 'lam': config.lam,  'device': 'CPU', 'weighted_clauses': False,
              "actor": actor, "critic": critic, 'batch_size': config.batch_size, 'epochs': config.epochs, 'test_freq': 1, "save": True, "seed": 42,
                'n_timesteps': config.n_timesteps, "dataset_file_name": "acrobot_obs_data"}

    #env = gym.make("CartPole-v1")
    env = gym.make("Acrobot-v1")

    #agent = PPO(env, Policy, _config)
    #agent = TMQN(env, Policy, _config)
    agent = PPO(env, Policy, _config)
    #agent.learn(nr_of_episodes=1000)
    agent.learn(nr_of_episodes=500)
    score = np.array(agent.best_score)
    #scores = np.array(agent.total_score)
    #score = np.mean(scores)
    return score


def main():
    wandb.init(project="TPPO_acrobot")
    score = objective(wandb.config)
    wandb.log({"score": score})


# actor = {'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 2.0, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, "seed": 42, 'number_of_state_bits_ta': 8}
# critic = {'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 100.0, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, "seed": 42, 'number_of_state_bits_ta': 8}
# config = {'algorithm': 'TM_DDPG', 'soft_update_type': 'soft_update_2', 'update_freq': 6, 'gamma': 0.98, 'actor': actor, 'critic': critic, 'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True}

sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.94, 0.98, 0.001))},
        "lam": {"values": list(np.arange(0.94, 0.98, 0.001))},
        "n_timesteps": {"values": list(range(8, 32, 4))},
        "batch_size": {"values": list(range(16, 512, 16))},
        "epochs": {"values": list(range(1, 5, 1))},

        "a_t": {"values": list(np.arange(0.3, 0.9, 0.01))},
        "a_nr_of_clauses": {"values": list(range(800, 1500, 10))},
        "a_specificity": {"values": list(np.arange(1.0, 4.0, 0.01))},
        "a_bits_per_feature": {"values": list(range(5, 15, 1))},
        "a_number_of_state_bits_ta": {"values": list(range(3, 8, 1))},
        "a_max_update_p": {"values": list(np.arange(0.001, 0.5000, 0.001))},
        "a_min_update_p": {"values": list(np.arange(0.0001, 0.001, 0.0001))},

        "c_t": {"values": list(np.arange(0.3, 0.9, 0.01))},
        "c_nr_of_clauses": {"values": list(range(800, 1500, 10))},
        "c_specificity": {"values": list(np.arange(1.0, 4.0, 0.01))},
        "c_bits_per_feature": {"values": list(range(5, 15, 1))},
        "c_number_of_state_bits_ta": {"values": list(range(3, 8, 1))},
        "c_max_update_p": {"values": list(np.arange(0.001, 0.200, 0.001))},
        "c_y_max": {"values": list(np.arange(-100, -1.0, 0.5))},
        "c_y_min": {"values": list(np.arange(-300.0, -100.0, 0.1))},

    }
}


sweep_id = wandb.sweep(sweep=sweep_configuration, project="TPPO_acrobot")
wandb.agent(sweep_id, function=main, count=10_000)

"""
Cartpole
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.94, 0.98, 0.001))},
        "lam": {"values": list(np.arange(0.94, 0.98, 0.001))},
        "n_timesteps": {"values": list(range(8, 32, 4))},
        "batch_size": {"values": list(range(16, 512, 16))},
        "epochs": {"values": list(range(1, 5, 1))},

        "a_t": {"values": list(np.arange(0.3, 0.9, 0.01))},
        "a_nr_of_clauses": {"values": list(range(900, 1200, 10))},
        "a_specificity": {"values": list(np.arange(1.0, 4.0, 0.01))},
        "a_bits_per_feature": {"values": list(range(5, 15, 1))},
        "a_number_of_state_bits_ta": {"values": list(range(3, 8, 1))},
        "a_max_update_p": {"values": list(np.arange(0.001, 0.2000, 0.001))},
        "a_min_update_p": {"values": list(np.arange(0.0001, 0.001, 0.0001))},

        "c_t": {"values": list(np.arange(0.3, 0.9, 0.01))},
        "c_nr_of_clauses": {"values": list(range(900, 1200, 10))},
        "c_specificity": {"values": list(np.arange(1.0, 4.0, 0.01))},
        "c_bits_per_feature": {"values": list(range(5, 15, 1))},
        "c_number_of_state_bits_ta": {"values": list(range(3, 8, 1))},
        "c_max_update_p": {"values": list(np.arange(0.001, 0.200, 0.001))},
        "c_y_max": {"values": list(np.arange(5.5, 35, 0.5))},
        "c_y_min": {"values": list(np.arange(0.0, 1.0, 0.1))},
    }
}

"""
#DDPG
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"max": 1.00, "min": 0.50},
        "update_freq": {"max": 7, "min": 2},

        "a_t": {"max": 1.0, "min": 0.01},
        "a_nr_of_clauses": {"max": 1500, "min": 800},
        "a_specificity": {"max": 8.0, "min": 1.0},
        "a_bits_per_feature": {"max": 15, "min": 5},
        "a_number_of_state_bits_ta": {"max": 10, "min": 3},

    }
}
#n-step TMQN
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

"""sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "update_freq": {"values": list(range(3, 8, 1))},

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
        "c_y_min": {"values": list(range(20, 40, 5))}
    }
}"""

#n-step Double TMQN
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "n_steps": {"values": list(range(1, 20, 1))},
        "update_freq": {"values": list(range(2, 10, 1))},
        "batch_size": {"values": list(range(16, 128, 16))},
        "buffer_size": {"values": list(range(500, 10_000, 500))},
        "epochs": {"values": list(range(1, 8, 1))},
        "t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "nr_of_clauses": {"values": list(range(800, 1200, 20))},
        "specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "bits_per_feature": {"values": list(range(5, 15, 1))},
        "number_of_state_bits_ta": {"values": list(range(3, 10, 1))},
        "y_max": {"values": list(range(60, 80, 5))},
        "y_min": {"values": list(range(20, 40, 5))}
    }
}

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
        "clip": {"values": list(np.arange(0.001, 0.5001, 0.01))}
    }
}
#double DQN
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "update_freq": {"values": list(range(2, 10, 1))},
        "batch_size": {"values": list(range(16, 128, 16))},
        "buffer_size": {"values": list(range(500, 10_000, 500))},
        "epochs": {"values": list(range(1, 8, 1))},
        "t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "nr_of_clauses": {"values": list(range(800, 1200, 20))},
        "specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "bits_per_feature": {"values": list(range(5, 15, 1))},
        "number_of_state_bits_ta": {"values": list(range(3, 10, 1))},
        "y_max": {"values": list(range(60, 80, 5))},
        "y_min": {"values": list(range(20, 40, 5))}
    }
}

sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "batch_size": {"values": list(range(16, 128, 16))},
        "buffer_size": {"values": list(range(500, 10_000, 500))},
        "epochs": {"values": list(range(1, 8, 1))},
        "t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "nr_of_clauses": {"values": list(range(800, 1200, 20))},
        "specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "bits_per_feature": {"values": list(range(5, 15, 1))},
        "number_of_state_bits_ta": {"values": list(range(3, 10, 1))},
        "y_max": {"values": list(range(60, 80, 5))},
        "y_min": {"values": list(range(20, 40, 5))}
    }
}


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
        "epochs": {"values": list(range(1, 5, 1))}
    }
}
#Double TMQN with type a update
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "n_steps": {"values": list(range(1, 20, 1))},
        "update_grad": {"values": list(np.arange(0.001, 1.0, 0.001))},
        "batch_size": {"values": list(range(16, 128, 16))},
        "buffer_size": {"values": list(range(500, 10_000, 500))},
        "epochs": {"values": list(range(1, 8, 1))},
        "t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "nr_of_clauses": {"values": list(range(800, 1200, 20))},
        "specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "bits_per_feature": {"values": list(range(5, 15, 1))},
        "number_of_state_bits_ta": {"values": list(range(3, 10, 1))},
        "y_max": {"values": list(range(60, 80, 5))},
        "y_min": {"values": list(range(20, 40, 5))}
    }
}
#type a update double TMQN n-steps
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "n_steps": {"values": list(range(1, 20, 1))},
        "update_grad": {"values": list(np.arange(0.001, 1.0, 0.001))},
        "batch_size": {"values": list(range(16, 128, 16))},
        "buffer_size": {"values": list(range(500, 10_000, 500))},
        "epochs": {"values": list(range(1, 8, 1))},
        "t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "nr_of_clauses": {"values": list(range(800, 1200, 20))},
        "specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "bits_per_feature": {"values": list(range(5, 15, 1))},
        "number_of_state_bits_ta": {"values": list(range(3, 10, 1))},
        "y_max": {"values": list(range(60, 80, 5))},
        "y_min": {"values": list(range(20, 40, 5))}
    }
}

sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "update_grad": {"values": list(np.arange(0.001, 1.0, 0.001))},
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
        "c_y_min": {"values": list(range(20, 40, 5))}
    }
}
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 0.999, 0.001))},
        "lam": {"values": list(np.arange(0.90, 0.999, 0.001))},
        "learning_rate": {"values": list(np.arange(0.0001, 0.01, 0.0001))},
        "epochs": {"values": list(range(1, 10, 1))},
        "batch_size": {"values": list(range(16, 512, 16))},
        "hidden_size": {"values": list(range(32, 512, 32))},
        "clip": {"values": list(np.arange(0.001, 0.5, 0.001))}
    }
}