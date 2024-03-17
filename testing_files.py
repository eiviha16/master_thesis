import torch
import numpy as np
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Proximal_policy.TM_PPO import PPO
from algorithms.policy.RTM import ActorCriticPolicy as Policy

#498.91 (on validation) so far run 28 - config = {'algorithm': 'TM_PPO', 'gamma': 0.99, 'lam': 0.95, 'clip_range': 0.5, 'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 2, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5,  'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, "min_feedback_p": 1.0, 'number_of_state_bits_ta': 8}
#experiment with specificity?
#shows greater stability config = {'algorithm': 'TM_PPO', 'gamma': 0.99, 'lam': 0.95, 'clip_range': 0.5, 'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 2.0, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5,  'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': 6}
#498.12 - run 46 - config = {'algorithm': 'TM_PPO', 'gamma': 0.99, 'lam': 0.95, 'clip_range': 0.5, 'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 2.0, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5,  'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': 8}
#495.24 - run 48 - config = {'algorithm': 'TM_PPO', 'gamma': 0.99, 'lam': 0.95, 'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 7.5, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5,  'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': 8}
"""wandb: 	bits_per_feature: 12
wandb: 	clip: 0.3809999999999999
wandb: 	gamma: 0.944
wandb: 	lam: 0.948
wandb: 	nr_of_clauses: 1178
wandb: 	number_of_state_bits_ta: 4
wandb: 	specificity: 1.6300000000000006
wandb: 	t: 0.7"""
"""andb: Agent Starting Run: cfdh3rcq with config:
wandb: 	bits_per_feature: 12
wandb: 	clip: 0.001
wandb: 	gamma: 0.951
wandb: 	lam: 0.958
wandb: 	nr_of_clauses: 1160
wandb: 	number_of_state_bits_ta: 3
wandb: 	specificity: 1.5000000000000004
wandb: 	t: 0.76"""
#config = {'algorithm': 'TM_PPO', 'gamma': 0.94, 'lam': 0.946, "clip": 0.431, 'nr_of_clauses': 1050, 'T': int(1050 * 0.59), 's': 1.6, 'y_max': 7.5, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 14,  'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': 6}
#config = {'algorithm': 'TM_PPO', 'gamma': 0.952, 'lam': 0.964, "clip": 0.121, 'nr_of_clauses': 1020, 'T': int(1020 * 0.38), 's': 2.22, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7,  'batch_size': 64, 'epochs': 3, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': 4}
#config = {'comment': '--Does not use normalization--', 'algorithm': 'TM_PPO', 'gamma': 0.942, 'lam': 0.947, "clip": 0.301, 'nr_of_clauses': 900, 'T': int(900 * 0.5), 's': 2.54, 'y_max': 23.5, 'y_min': 0.9, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5,  'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': 4}
#config = {'comment': '--Does not use normalization--', 'algorithm': 'TM_PPO', 'gamma': 0.961, 'lam': 0.95, "clip": 0.041, 'nr_of_clauses': 1100, 'T': 743, 's': 1.66, 'y_max': 20.0, 'y_min': 0.2, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 11,  'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': 4}
#config = {'comment': '--Does not use normalization--', 'algorithm': 'TM_PPO', 'gamma': 0.954, 'lam': 0.941, "clip": 0.451, 'nr_of_clauses': 1160, 'T': int(1160 * 0.66), 's': 1.97, 'y_max': 28.5, 'y_min': 0.2, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7,  'batch_size': 208, 'epochs': 3, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': 4}
config = {'comment': '--Does not use normalization--', 'algorithm': 'TM_PPO', 'gamma': 0.941, 'lam': 0.98, "clip": 0.031, 'nr_of_clauses': 1030, 'T': int(1030 * 0.49), 's': 1.62, 'y_max': 28, 'y_min': 0.2, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 12,  'batch_size': 96, 'epochs': 2, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': 3}
#change gamma and lambda

print(config)

env = gym.make("CartPole-v1")


agent = PPO(env, Policy, config)
agent.learn(nr_of_episodes=10_000)

from test_policy import test_policy

#agent.policy.actor.tms[0].set_state()
#agent.policy.actor.tms[1].set_state()
save_file = f'results/TM_PPO/{agent.run_id}'

tms = torch.load(f'results/TM_PPO/{agent.run_id}/best')

for i in range(len(tms)):
    #eval_ta_state, eval_clause_sign, eval_clause_output, eval_feedback_to_clauses
    agent.policy.actor.tms[i].set_params(tms[i]['ta_state'], tms[i]['clause_sign'], tms[i]['clause_output'], tms[i]['feedback_to_clauses'])

test_policy(save_file, agent.policy.actor)
"""import wandb
import numpy as np
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

wandb.login(key="74a10e58809253b0e1f243f34bb17d8f34c21e59")

"""
"""def objective(config):
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
    #from algorithms.Proximal_policy.TM_PPO import PPO
    from algorithms.Q_Network.DQN import DQN
    from algorithms.policy.DNN import Policy
    #from algorithms.policy.RTM import ActorCriticPolicy as Policy
    #_config = {'algorithm': 'PPO', 'gamma': config.gamma, 'lam': config.lam, 'clip_range': config.clip, 'batch_size': config.batch_size, 'epochs': config.epochs,
    #          'hidden_size': config.hidden_size, 'learning_rate': config.learning_rate, 'test_freq': 1, "save": True}
    _config = {'algorithm': 'DQN', 'gamma': config.gamma, 'c': 1, 'exploration_prob_init': config.exploration_prob_init, 'exploration_prob_decay': config.exploration_prob_decay,
              'buffer_size': config.buffer_size, 'batch_size': config.batch_size, 'epochs': config.epochs, 'hidden_size': config.hidden_size, 'learning_rate': config.learning_rate,
              'test_freq': 1, 'threshold_score': 450, "save": False}
    """
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
"""    actor = {'nr_of_classes': 2, 'nr_of_clauses': config.a_nr_of_clauses, 'T': int(config.a_nr_of_clauses * config.a_t),
             's': config.a_specificity, 'device': 'CPU', 'weighted_clauses': False,
             'bits_per_feature': config.a_bits_per_feature, "seed": 42, 'number_of_state_bits_ta': config.a_number_of_state_bits_ta}
    critic = {'nr_of_clauses': config.c_nr_of_clauses, 'T': int(config.c_nr_of_clauses * config.c_t), 's': config.c_specificity, 'y_max': config.c_y_max, 'y_min': config.c_y_min, 'device': 'CPU',
              'weighted_clauses': False, 'bits_per_feature': config.c_bits_per_feature, "seed": 42, 'number_of_state_bits_ta': config.c_number_of_state_bits_ta}
    _config = {'algorithm': 'TM_DDPG_2', 'soft_update_type': 'soft_update_1', 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'update_grad': config.update_grad, 'gamma': config.gamma,
               'actor': actor, 'critic': critic, 'batch_size': config.batch_size, 'epochs': config.epochs, 'test_freq': 1, "save": True}
    _config = {'algorithm': 'TM_PPO', 'gamma': config.gamma, 'lam': config.lam, "clip": config.clip, 'nr_of_clauses': config.nr_of_clauses, 'T': int(config.nr_of_clauses * config.t), 's': config.specificity,
              'y_max': 7.5, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
              'batch_size': 64, 'epochs': config.epochs, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': config.number_of_state_bits_ta}
        #env = gym.make("CartPole-v1")
    env = gym.make("Acrobot-v1")

    #agent = PPO(env, Policy, _config)
    #agent = TMQN(env, Policy, _config)
    agent = DQN(env, Policy, _config)
    agent.learn(nr_of_episodes=200)
    score = np.array(agent.best_scores['mean'])
    #scores = np.array(agent.total_score)
    #score = np.mean(scores)
    return score
"""



#def main():
#    wandb.init(project="DQN")
#    score = objective(wandb.config)
#    wandb.log({"score": score})"""


# actor = {'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 2.0, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, "seed": 42, 'number_of_state_bits_ta': 8}
# critic = {'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 100.0, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, "seed": 42, 'number_of_state_bits_ta': 8}
# config = {'algorithm': 'TM_DDPG', 'soft_update_type': 'soft_update_2', 'update_freq': 6, 'gamma': 0.98, 'actor': actor, 'critic': critic, 'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True}


"""sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 0.999, 0.001))},
        "learning_rate": {"values": list(np.arange(0.0001, 0.01, 0.0001))},
        "exploration_prob_init": {"values": list(np.arange(0.05, 1.00, 0.05))},
        "exploration_prob_decay": {"values": list(np.arange(0.001, 0.01, 0.001))},
        "epochs": {"values": list(range(1, 16, 1))},
        "batch_size": {"values": list(range(16, 512, 16))},
        "hidden_size": {"values": list(range(32, 512, 32))},
        "buffer_size": {"values": list(range(500, 100_000, 500))}
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="DQN")
wandb.agent(sweep_id, function=main, count=10_000)
"""
"""import torch
import numpy as np
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Proximal_policy.TM_PPO import PPO
from algorithms.policy.RTM import ActorCriticPolicy as Policy

#498.91 (on validation) so far run 28 - config = {'algorithm': 'TM_PPO', 'gamma': 0.99, 'lam': 0.95, 'clip_range': 0.5, 'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 2, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5,  'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, "min_feedback_p": 1.0, 'number_of_state_bits_ta': 8}
#experiment with specificity?
#shows greater stability config = {'algorithm': 'TM_PPO', 'gamma': 0.99, 'lam': 0.95, 'clip_range': 0.5, 'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 2.0, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5,  'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': 6}
#498.12 - run 46 - config = {'algorithm': 'TM_PPO', 'gamma': 0.99, 'lam': 0.95, 'clip_range': 0.5, 'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 2.0, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5,  'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': 8}
#495.24 - run 48 - config = {'algorithm': 'TM_PPO', 'gamma': 0.99, 'lam': 0.95, 'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 7.5, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5,  'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': 8}
""""""wandb: 	bits_per_feature: 12
wandb: 	clip: 0.3809999999999999
wandb: 	gamma: 0.944
wandb: 	lam: 0.948
wandb: 	nr_of_clauses: 1178
wandb: 	number_of_state_bits_ta: 4
wandb: 	specificity: 1.6300000000000006
wandb: 	t: 0.7"""
"""andb: Agent Starting Run: cfdh3rcq with config:
wandb: 	bits_per_feature: 12
wandb: 	clip: 0.001
wandb: 	gamma: 0.951
wandb: 	lam: 0.958
wandb: 	nr_of_clauses: 1160
wandb: 	number_of_state_bits_ta: 3
wandb: 	specificity: 1.5000000000000004
wandb: 	t: 0.76"""
"""#config = {'algorithm': 'TM_PPO', 'gamma': 0.94, 'lam': 0.946, "clip": 0.431, 'nr_of_clauses': 1050, 'T': int(1050 * 0.59), 's': 1.6, 'y_max': 7.5, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 14,  'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': 6}
#config = {'algorithm': 'TM_PPO', 'gamma': 0.952, 'lam': 0.964, "clip": 0.121, 'nr_of_clauses': 1020, 'T': int(1020 * 0.38), 's': 2.22, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7,  'batch_size': 64, 'epochs': 3, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': 4}
config = {'comment': '--Does not use normalization--', 'algorithm': 'TM_PPO', 'gamma': 0.961, 'lam': 0.95, "clip": 0.041, 'nr_of_clauses': 1110, 'T': int(1110 * 0.67), 's': 1.66, 'y_max': 20, 'y_min': 0.2, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 11,  'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': 4}
#change gamma and lambda

print(config)

env = gym.make("CartPole-v1")


agent = PPO(env, Policy, config)
agent.learn(nr_of_episodes=10_000)

from test_policy import test_policy

#agent.policy.actor.tms[0].set_state()
#agent.policy.actor.tms[1].set_state()
save_file = f'results/TM_PPO/{agent.run_id}'

tms = torch.load(f'results/TM_PPO/{agent.run_id}/best')

for i in range(len(tms)):
    #eval_ta_state, eval_clause_sign, eval_clause_output, eval_feedback_to_clauses
    agent.policy.actor.tms[i].set_params(tms[i]['ta_state'], tms[i]['clause_sign'], tms[i]['clause_output'], tms[i]['feedback_to_clauses'])

test_policy(save_file, agent.policy.actor)"""
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
config = {'algorithm': 'n_step_Double_TMQN', 'soft_update_type': 'soft_update_2', 'n_steps': 17, 'nr_of_clauses': 980, 'T': (980 * 0.34), 's': 8.58, 'y_max': 60, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 9, 'gamma': 0.992, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 4000, 'batch_size': 80, 'epochs': 2, 'test_freq': 1,  "save": True, 'number_of_state_bits_ta': 6, 'update_grad': 0.05, 'update_freq': 8}

env = gym.make("CartPole-v1")

agent = TMQN(env, Policy, config)
agent.learn(nr_of_episodes=10_000)

from test_policy import test_policy

save_file = f'results/n_step_Double_TMQN/{agent.run_id}/final_test_results'
tms = torch.load(f'results/n_step_Double_TMQN/{agent.run_id}/best')

agent.target_policy.tm1.set_params(tms[0]['ta_state'], tms[0]['clause_sign'], tms[0]['clause_output'], tms[0]['feedback_to_clauses'])
agent.target_policy.tm2.set_params(tms[1]['ta_state'], tms[1]['clause_sign'], tms[1]['clause_output'], tms[1]['feedback_to_clauses'])

test_policy(save_file, agent.target_policy)
"""