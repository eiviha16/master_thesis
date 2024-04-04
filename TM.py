import torch
import numpy as np
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import DQN
import os
from tqdm import tqdm

class PPO_c:
    def __init__(self, config):
        self.nr_of_test_episodes = 100
        self.test_random_seeds = [83811, 14593, 3279, 97197, 36049, 32099, 29257, 18290, 96531, 13435, 88697, 97081,
                                  71483,
                                  11396, 77398, 55303, 4166, 3906, 12281, 28658, 30496, 66238, 78908, 3479, 73564,
                                  26063, 93851,
                                  85182, 91925, 71427, 54988, 28894, 58879, 77237, 36464, 852, 99459, 20927, 91507,
                                  55393,
                                  44598, 36422, 20380, 28222, 44119, 13397, 12157, 49798, 12677, 47053, 45083, 79132,
                                  34672,
                                  5696, 95648, 60218, 70285, 16362, 49616, 10329, 72358, 38428, 82398, 81071, 47401,
                                  75675,
                                  25204, 92350, 9117, 6007, 86674, 29872, 37931, 10459, 30513, 13239, 49824, 36435,
                                  59430,
                                  83321, 47820, 21320, 48521, 46567, 27461, 87842, 34994, 91989, 89594, 84940, 9359,
                                  79841,
                                  83228, 22432, 70011, 95569, 32088, 21418, 60590, 49736]

        self.best_score = - float("inf")
        self.config = config
        self.n_timesteps = 3500 #1000
        self.file_path = f'./results/{self.config["env_name"]}/{self.config["algorithm"]}/{self.config["run"]}'

        # Parallel environments
        if self.config["env_name"] == "cartpole":
            self.env = gym.make("CartPole-v1")
        elif self.config["env_name"] == "acrobot":
            self.env = gym.make("Acrobot-v1")


        self.model = DQN("MlpPolicy", self.env, seed=42, verbose=0, learning_rate=0.0001, buffer_size=10000)#, n_steps=16)
        #self.model = DQN("MlpPolicy", self.env, seed=42, verbose=0, learning_rate=0.001, buffer_size=10000)#, n_steps=16)
        #self.model = PPO("MlpPolicy", self.env, seed=42, verbose=0)
    def learn(self, intervals):
        for i in tqdm(range(intervals)):
            self.test_environment()
            self.model.learn(total_timesteps=self.n_timesteps)

    def test_environment(self):

        episode_rewards = np.array([0 for _ in range(self.nr_of_test_episodes)])

        for episode in range(self.nr_of_test_episodes):
            obs, _ = self.env.reset(seed=self.test_random_seeds[episode])  # episode)
            while True:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = self.env.step(action)
                episode_rewards[episode] += reward
                if done or truncated:
                    break

        mean = np.mean(episode_rewards)
        std = np.std(episode_rewards)

        self.save_results(mean, std)

        if mean > self.best_score:
            self.best_score = mean
            print(f'New best mean: {mean}!')
            self.model.save(f"{self.file_path}/best")
            # self.save_q_vals(nr_of_steps)

    def save_results(self, mean, std):
        file_name = 'test_results.csv'
        file_exists = os.path.exists(os.path.join(self.file_path, file_name))

        with open(os.path.join(self.file_path, file_name), "a") as file:
            if not file_exists:
                file.write("mean,std\n")
            file.write(f"{mean},{std}\n")


if __name__ == "__main__":
    config = {"env_name": "cartpole", "algorithm": "DQN", "run": "run_4sb"}
    ppo = PPO_c(config)
    ppo.learn(100)

    import test_policy
    model = ppo.model.load(f"{ppo.file_path}/best")
    test_policy.test_policy(ppo.file_path, model, ppo.config["env_name"], sb=True)

exit(0)

































import torch
import numpy as np
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Proximal_policy.TM_PPO import PPO
from algorithms.policy.RTM import ActorCriticPolicy as Policy

"""actor = {"max_update_p": 0.005, "min_update_p": 0.0004, 'nr_of_clauses': 1120, 'T': int(1120 * 0.74), 's': 2.35, 'y_max': 100, 'y_min': 0,  'bits_per_feature': 10, 'number_of_state_bits_ta': 3}
critic = {"max_update_p": 0.059, "min_update_p": 0.0, 'nr_of_clauses': 930, 'T': int(930 * 0.36), 's': 3.89, 'y_max': 31.5, 'y_min': 0.3,  'bits_per_feature': 8, 'number_of_state_bits_ta': 6}
config = {'algorithm': 'TM_PPO', "n_timesteps": 430, 'gamma': 0.959, 'lam': 0.979, "actor": actor, "critic": critic, 'device': 'CPU', 'weighted_clauses': False,  'batch_size': 64, 'epochs': 4, 'test_freq': 1, "save": True, "seed": 42, "dataset_file_name": "observation_data"} #"dataset_file_name": "acrobot_obs_data"}
"""#change gamma and lambda
"""actor = {"max_update_p": 0.024, "min_update_p": 0.0009, 'nr_of_clauses': 1110, 'T': int(1110 * 0.36), 's': 2.42, 'y_max': 100, 'y_min': 0,  'bits_per_feature': 5, 'number_of_state_bits_ta': 5}
critic = {"max_update_p": 0.043, "min_update_p": 0.0, 'nr_of_clauses': 1030, 'T': int(1030 * 0.56), 's': 2.03, 'y_max': 14, 'y_min': 0.5,  'bits_per_feature': 5, 'number_of_state_bits_ta': 4}
config = {'algorithm': 'TM_PPO', "n_timesteps": 430, 'gamma': 0.944, 'lam': 0.966, "actor": actor, "critic": critic, 'device': 'CPU', 'weighted_clauses': False,  'batch_size': 384, 'epochs': 3, 'test_freq': 1, "save": True, "seed": 42, "dataset_file_name": "observation_data"} #"dataset_file_name": "acrobot_obs_data"}
"""

actor = {"max_update_p": 0.068, "min_update_p": 0.0009, 'nr_of_clauses': 1140, 'T': int(1140 * 0.5), 's': 1.92, 'y_max': 100, 'y_min': 0,  'bits_per_feature': 5, 'number_of_state_bits_ta': 7}
critic = {"max_update_p": 0.056, "min_update_p": 0.0, 'nr_of_clauses': 1120, 'T': int(1120 * 0.57), 's': 3.28, 'y_max': 25.5, 'y_min': 0.8,  'bits_per_feature': 8, 'number_of_state_bits_ta': 4}
    #config = {'env_name': "cartpole", 'algorithm': 'TPPO', "n_timesteps": 430, 'gamma': 0.945, 'lam': 0.968, "actor": actor, "critic": critic, 'device': 'CPU', 'weighted_clauses': False,  'batch_size': 64, 'epochs': 2, 'test_freq': 1, "save": , "seed": 42, "dataset_file_name": "observation_data", "threshold": -1000} #"dataset_file_name": "acrobot_obs_data"}

#run_908 has been initialized! 500.0 (best)
print(config)
#run_895 - 500.0
#env = gym.make("Acrobot-v1")
env = gym.make("CartPole-v1")


agent = PPO(env, Policy, config)
agent.learn(nr_of_episodes=766)

from test_policy import test_policy

#agent.policy.actor.tms[0].set_state()
#agent.policy.actor.tms[1].set_state()
#save_file = f'../results/TM_PPO/{agent.run_id}'
save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results'

#tms = torch.load(f'../results/TM_PPO/{agent.run_id}/best')
tms = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')

for i in range(len(tms)):
    #eval_ta_state, eval_clause_sign, eval_clause_output, eval_feedback_to_clauses
    agent.policy.actor.tms[i].set_params(tms[i]['ta_state'], tms[i]['clause_sign'], tms[i]['clause_output'], tms[i]['feedback_to_clauses'])

test_policy(save_file, agent.policy.actor)

exit(0)

import gymnasium as gym
import random
from algorithms.Q_Network.DQN import DQN
from algorithms.policy.DNN import Policy
import torch
import numpy as np

#config = {'algorithm': 'DQN', 'gamma': 0.98, 'c': 30, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 20000, 'batch_size': 256, 'epochs': 4, 'hidden_size': 64, 'learning_rate': 0.001, 'test_freq': 1, 'threshold_score': 450, "save": True}
#config = {'algorithm': 'DQN', 'gamma': 0.99, 'c': 30, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 20000, 'batch_size': 256, 'epochs': 4, 'hidden_size': 64, 'learning_rate': 0.001, 'test_freq': 1, 'threshold_score': 450, "save": True}
config = {'algorithm': 'DQN', 'gamma': 0.943, 'c':  1, 'exploration_prob_init': 0.65, 'exploration_prob_decay': 0.006, 'buffer_size': 9_000, 'batch_size': 416, 'epochs': 10, 'hidden_size': 480, 'learning_rate': 0.002, 'test_freq': 1, 'threshold_score': 450, "save": True}

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

#env = gym.make("CartPole-v1")
env = gym.make("Acrobot-v1")


agent = DQN(env, Policy, config)
agent.learn(nr_of_episodes=10000)
from test_policy import test_policy

file = f'./results/DQN/{agent.run_id}/best_model'
model = torch.load(file)
test_policy(f'./results/DQN/{agent.run_id}/final_test_results', model.actor)

"""import numpy as np
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Q_Network.n_step_TMQN import TMQN
from algorithms.policy.RTM import Policy

#config = {'algorithm': 'n_step_Double_TMQN', 'soft_update_type': 'soft_update_2', 'n_steps': 17, 'nr_of_clauses': 980, 'T': (980 * 0.34), 's': 8.58, 'y_max': 60, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 9, 'gamma': 0.992, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 4000, 'batch_size': 80, 'epochs': 2, 'test_freq': 1,  "save": True, 'number_of_state_bits_ta': 6, 'update_grad': 0.05, 'update_freq': 8}

#Winner run 76 - 500.0 - 500.0 - config = {'algorithm': 'n_step_TMQN', 'n_steps': 5, 'nr_of_clauses': 1000, 'T': 100, 's': 4.9, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': True, 'bits_per_feature': 5, 'gamma': 0.95, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 1, 'threshold_score': 450, "save": True, "seed": 42, "balance_feedback": False, "min_feedback_p": 1.0, 'dynamic_memory': False, 'dynamic_memory_max_size': 10, 'number_of_state_bits_ta': 10}
#run 94 - 498.95 - 7.41 ,config = {'algorithm': 'n_step_TMQN', 'n_steps': 5, 'nr_of_clauses': 1000, 'T': 100, 's': 4.9, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, 'gamma': 0.95, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 1,  "save": True, 'dynamic_memory': False, 'number_of_state_bits_ta': 10}
#config = {'algorithm': 'n_step_TMQN', 'n_steps': 10, 'nr_of_clauses': 1000, 'T': 350, 's': 3.7, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, 'gamma': 0.95, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 1,  "save": True, 'dynamic_memory': False, 'number_of_state_bits_ta': 10}
#config = {'algorithm': 'n_step_TMQN', 'n_steps': 17, 'nr_of_clauses': 980, 'T': int(980 * 0.34), 's': 8.58, 'y_max': 60, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 9, 'gamma': 0.992, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 4000, 'batch_size': 80, 'epochs': 2, 'test_freq': 1,  "save": True, 'dynamic_memory': False, 'number_of_state_bits_ta': 6}
config = {'algorithm': 'n_step_TMQN', 'n_steps': 18, 'nr_of_clauses': 880, 'T': int(880 * 0.46), 's': 8.88, 'y_max': 70, 'y_min': 25, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 11, 'gamma': 0.98, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 9500, 'batch_size': 48, 'epochs': 4, 'test_freq': 1,  "save": True, 'dynamic_memory': False, 'number_of_state_bits_ta': 9}

env = gym.make("CartPole-v1")

agent = TMQN(env, Policy, config)
# agent.learn(nr_of_episodes=10000)
agent.learn(nr_of_episodes=10_000)

from test_policy import test_policy

#test_policy(agent.policy)
#test_policy(agent.current_policy)
save_file = f'results/n_step_TMQN/{agent.run_id}/final_test_results'

#agent.policy.tm1.set_state()
#agent.policy.tm2.set_state()

save_file = f'results/n_step_TMQN/{agent.run_id}'

tms = torch.load(f'results/n_step_TMQN/{agent.run_id}/best')

agent.policy.tm1.set_params(tms[0]['ta_state'], tms[0]['clause_sign'], tms[0]['clause_output'], tms[0]['feedback_to_clauses'])
agent.policy.tm2.set_params(tms[1]['ta_state'], tms[1]['clause_sign'], tms[0]['clause_output'], tms[0]['feedback_to_clauses'])

test_policy(save_file, agent.policy)

"""
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

"""import numpy as np
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

test_policy(save_file, agent.target_policy)"""

"""import wandb
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
    from algorithms.Q_Network.n_step_TMQN import TMQN
    #from algorithms.Q_Network.TMQN import TMQN
    from algorithms.policy.RTM import Policy
    #from algorithms.policy.CTM import ActorCriticPolicy as Policy
    #from algorithms.Proximal_policy.TM_PPO import PPO
    #from algorithms.policy.RTM import ActorCriticPolicy as Policy

    _config = {
        'algorithm': 'n_step_TMQN', 'n_steps': config.n_steps, 'nr_of_clauses': config.nr_of_clauses, 'T': int(config.t * config.nr_of_clauses),
        's': config.specificity, 'y_max': config.y_max, 'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': config.buffer_size,
        'batch_size': config.batch_size, 'epochs': config.epochs, 'test_freq': 50, "save": False, "seed": 42,
        #'number_of_state_bits_ta': config.number_of_state_bits_ta}
    #_config = {
        #'algorithm': 'Double_TMQN', 'soft_update_type': 'soft_update_1', 'n_steps': config.n_steps, 'nr_of_clauses': config.nr_of_clauses, 'T': int(config.t * config.nr_of_clauses),
        #'s': config.specificity, 'y_max': config.y_max, 'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        #'gamma': config.gamma, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': config.buffer_size,
        #'batch_size': config.batch_size, 'epochs': config.epochs, 'test_freq': 50, "save": False, "seed": 42,
        #'number_of_state_bits_ta': config.number_of_state_bits_ta, 'update_grad': config.update_grad, 'update_freq': 10000}
    actor = {'nr_of_classes': 2, 'nr_of_clauses': config.a_nr_of_clauses, 'T': int(config.a_nr_of_clauses * config.a_t),
             's': config.a_specificity, 'device': 'CPU', 'weighted_clauses': False,
             'bits_per_feature': config.a_bits_per_feature, "seed": 42, 'number_of_state_bits_ta': config.a_number_of_state_bits_ta}
    critic = {'nr_of_clauses': config.c_nr_of_clauses, 'T': int(config.c_nr_of_clauses * config.c_t), 's': config.c_specificity, 'y_max': config.c_y_max, 'y_min': config.c_y_min, 'device': 'CPU',
              'weighted_clauses': False, 'bits_per_feature': config.c_bits_per_feature, "seed": 42, 'number_of_state_bits_ta': config.c_number_of_state_bits_ta}
    _config = {'algorithm': 'TM_DDPG_2', 'soft_update_type': 'soft_update_1', 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'update_grad': config.update_grad, 'gamma': config.gamma,
               'actor': actor, 'critic': critic, 'batch_size': config.batch_size, 'epochs': config.epochs, 'test_freq': 1, "save": True}
    
    _config = {'algorithm': 'TM_PPO', 'gamma': config.gamma, 'lam': config.lam, "clip": config.clip, 'nr_of_clauses': config.nr_of_clauses, 'T': int(config.nr_of_clauses * config.t), 's': config.specificity,
         'y_max': 7.5, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
         'batch_size': 64, 'epochs': config.epochs, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': config.number_of_state_bits_ta}
    
    env = gym.make("CartPole-v1")

    #agent = PPO(env, Policy, _config)
    #agent = TMQN(env, Policy, _config)
    agent = TMQN(env, Policy, _config)
    agent.learn(nr_of_episodes=5000)
    #score = np.array(agent.best_score)
    scores = np.array(agent.total_score)
    score = np.mean(scores)
    return score

#Syncing run fresh-sweep-2
def main():
    wandb.init(project="n-step TMQN")
    score = objective(wandb.config)
    wandb.log({"score": score})

sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "n_steps": {"values": list(range(1, 20, 1))},
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
sweep_id = wandb.sweep(sweep=sweep_configuration, project="n-step TMQN")
wandb.agent(sweep_id, function=main, count=10_000)"""
# actor = {'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 2.0, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, "seed": 42, 'number_of_state_bits_ta': 8}
# critic = {'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 100.0, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, "seed": 42, 'number_of_state_bits_ta': 8}
# config = {'algorithm': 'TM_DDPG', 'soft_update_type': 'soft_update_2', 'update_freq': 6, 'gamma': 0.98, 'actor': actor, 'critic': critic, 'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True}

#wandb: Syncing run fresh-sweep-2
