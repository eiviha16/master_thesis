import numpy as np
import torch
import random

################################################
################### TAC a #######################
################################################

n_episodes_1 = 1000
n_episodes_2 = 5000
test_freq_2 = 50
cartpole_threshold = 15
acrobot_threshold = -495
def cartpole_TAC_a(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.VPG.TM_DDPG import DDPG
    from algorithms.policy.CTM import ActorCriticPolicy as Policy

    actor = {'nr_of_clauses': config.a_nr_of_clauses, 'T': int(config.a_nr_of_clauses * config.a_t),
             's': config.a_specificity, 'device': 'CPU', 'weighted_clauses': False,
             'bits_per_feature': config.a_bits_per_feature, "seed": 42,
             'number_of_state_bits_ta': config.a_number_of_state_bits_ta}
    critic = {"max_update_p": config.c_max_update_p, 'nr_of_clauses': config.c_nr_of_clauses,
              'T': int(config.c_nr_of_clauses * config.c_t), 's': config.c_specificity, 'y_max': config.c_y_max,
              'y_min': config.c_y_min, 'device': 'CPU',
              'weighted_clauses': False, 'bits_per_feature': config.c_bits_per_feature, "seed": 42,
              'number_of_state_bits_ta': config.c_number_of_state_bits_ta}
    _config = {'algorithm': 'TM_DDPG_2', 'soft_update_type': 'soft_update_1',
               'exploration_prob_init': config.exploration_p_init, 'exploration_prob_decay': config.exploration_p_decay,
               'update_grad': config.update_grad, 'gamma': config.gamma,
               "buffer_size": config.buffer_size, 'actor': actor, 'critic': critic, 'batch_size': config.batch_size,
               'epochs': config.epochs, 'test_freq': 1, "save": False, "threshold": cartpole_threshold, "dataset_file_name": "observation_data"}

    env = gym.make("CartPole-v1")

    agent = DDPG(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_1)
    score = np.array(agent.best_score)
    return score


def acrobot_TAC_a(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.VPG.TM_DDPG import DDPG
    from algorithms.policy.CTM import ActorCriticPolicy as Policy

    actor = {'nr_of_clauses': config.a_nr_of_clauses, 'T': int(config.a_nr_of_clauses * config.a_t),
             's': config.a_specificity, 'device': 'CPU', 'weighted_clauses': False,
             'bits_per_feature': config.a_bits_per_feature, "seed": 42,
             'number_of_state_bits_ta': config.a_number_of_state_bits_ta}
    critic = {"max_update_p": config.c_max_update_p, 'nr_of_clauses': config.c_nr_of_clauses,
              'T': int(config.c_nr_of_clauses * config.c_t), 's': config.c_specificity, 'y_max': config.c_y_max,
              'y_min': config.c_y_min, 'device': 'CPU',
              'weighted_clauses': False, 'bits_per_feature': config.c_bits_per_feature, "seed": 42,
              'number_of_state_bits_ta': config.c_number_of_state_bits_ta}
    _config = {'algorithm': 'TM_DDPG_2', 'soft_update_type': 'soft_update_1',
               'exploration_prob_init': config.exploration_p_init, 'exploration_prob_decay': config.exploration_p_decay,
               'update_grad': config.update_grad, 'gamma': config.gamma,
               "buffer_size": config.buffer_size, 'actor': actor, 'critic': critic, 'batch_size': config.batch_size,
               'epochs': config.epochs, 'test_freq': 1, "save": False, "threshold": acrobot_threshold,
               "dataset_file_name": "acrobot_obs_data"}  # "observation_data"}

    env = gym.make("Acrobot-v1")

    agent = DDPG(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_1)
    score = np.array(agent.best_score)
    return score


################################################
################### TAC b #######################
################################################

def cartpole_TAC_b(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.VPG.TM_DDPG import DDPG
    from algorithms.policy.CTM import ActorCriticPolicy as Policy

    actor = {'nr_of_clauses': config.a_nr_of_clauses, 'T': int(config.a_nr_of_clauses * config.a_t),
             's': config.a_specificity, 'device': 'CPU', 'weighted_clauses': False,
             'bits_per_feature': config.a_bits_per_feature, "seed": 42,
             'number_of_state_bits_ta': config.a_number_of_state_bits_ta}
    critic = {"max_update_p": config.c_max_update_p, 'nr_of_clauses': config.c_nr_of_clauses,
              'T': int(config.c_nr_of_clauses * config.c_t), 's': config.c_specificity, 'y_max': config.c_y_max,
              'y_min': config.c_y_min, 'device': 'CPU',
              'weighted_clauses': False, 'bits_per_feature': config.c_bits_per_feature, "seed": 42,
              'number_of_state_bits_ta': config.c_number_of_state_bits_ta}
    _config = {'algorithm': 'TM_DDPG_2', 'soft_update_type': 'soft_update_2',
               'exploration_prob_init': config.exploration_p_init, 'exploration_prob_decay': config.exploration_p_decay,
               'update_freq': config.update_freq, 'gamma': config.gamma,
               "buffer_size": config.buffer_size, 'actor': actor, 'critic': critic, 'batch_size': config.batch_size,
               'epochs': config.epochs, 'test_freq': 1,  "threshold": cartpole_threshold, "save": False, "dataset_file_name": "observation_data"}

    env = gym.make("CartPole-v1")

    agent = DDPG(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_1)
    score = np.array(agent.best_score)
    return score


def acrobot_TAC_b(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.VPG.TM_DDPG import DDPG
    from algorithms.policy.CTM import ActorCriticPolicy as Policy

    actor = {'nr_of_clauses': config.a_nr_of_clauses, 'T': int(config.a_nr_of_clauses * config.a_t),
             's': config.a_specificity, 'device': 'CPU', 'weighted_clauses': False,
             'bits_per_feature': config.a_bits_per_feature, "seed": 42,
             'number_of_state_bits_ta': config.a_number_of_state_bits_ta}
    critic = {"max_update_p": config.c_max_update_p, 'nr_of_clauses': config.c_nr_of_clauses,
              'T': int(config.c_nr_of_clauses * config.c_t), 's': config.c_specificity, 'y_max': config.c_y_max,
              'y_min': config.c_y_min, 'device': 'CPU',
              'weighted_clauses': False, 'bits_per_feature': config.c_bits_per_feature, "seed": 42,
              'number_of_state_bits_ta': config.c_number_of_state_bits_ta}
    _config = {'algorithm': 'TM_DDPG_2', 'soft_update_type': 'soft_update_2',
               'exploration_prob_init': config.exploration_p_init, 'exploration_prob_decay': config.exploration_p_decay,
               'update_freq': config.update_freq, 'gamma': config.gamma,
               "buffer_size": config.buffer_size, 'actor': actor, 'critic': critic, 'batch_size': config.batch_size,
               'epochs': config.epochs, 'test_freq': 1, "save": False, "threshold": acrobot_threshold,
               "dataset_file_name": "acrobot_obs_data"}  # "observation_data"}

    env = gym.make("Acrobot-v1")

    agent = DDPG(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_1)
    score = np.array(agent.best_score)
    return score


################################################
################### TPPO #######################
################################################

def cartpole_TPPO(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Proximal_policy.TM_PPO import PPO
    from algorithms.policy.RTM import ActorCriticPolicy as Policy

    actor = {"max_update_p": config.a_max_update_p, "min_update_p": config.a_min_update_p,
             'nr_of_clauses': config.a_nr_of_clauses, 'T': int(config.a_nr_of_clauses * config.a_t),
             's': config.a_specificity,
             'y_max': 100, 'y_min': 0, 'bits_per_feature': config.a_bits_per_feature,
             'number_of_state_bits_ta': config.a_number_of_state_bits_ta}
    critic = {"max_update_p": config.c_max_update_p, "min_update_p": 0.0,
              'nr_of_clauses': config.c_nr_of_clauses, 'T': int(config.c_nr_of_clauses * config.c_t),
              's': config.c_specificity,
              'y_max': config.c_y_max, 'y_min': config.c_y_min, 'bits_per_feature': config.c_bits_per_feature,
              'number_of_state_bits_ta': config.c_number_of_state_bits_ta}

    _config = {'comment': 'newest', 'algorithm': 'TM_PPO', 'gamma': config.gamma, 'lam': config.lam, 'device': 'CPU',
               'weighted_clauses': False,
               "actor": actor, "critic": critic, 'batch_size': config.batch_size, 'epochs': config.epochs,
               'test_freq': 1, "save": False, "seed": 42,  "threshold": cartpole_threshold,
               'n_timesteps': config.n_timesteps, "dataset_file_name": "observation_data"}

    env = gym.make("CartPole-v1")

    agent = PPO(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_1)
    score = np.array(agent.best_score)
    return score


def acrobot_TPPO(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Proximal_policy.TM_PPO import PPO
    from algorithms.policy.RTM import ActorCriticPolicy as Policy

    actor = {"max_update_p": config.a_max_update_p, "min_update_p": config.a_min_update_p,
             'nr_of_clauses': config.a_nr_of_clauses, 'T': int(config.a_nr_of_clauses * config.a_t),
             's': config.a_specificity,
             'y_max': 100, 'y_min': 0, 'bits_per_feature': config.a_bits_per_feature,
             'number_of_state_bits_ta': config.a_number_of_state_bits_ta}
    critic = {"max_update_p": config.c_max_update_p, "min_update_p": 0.0,
              'nr_of_clauses': config.c_nr_of_clauses, 'T': int(config.c_nr_of_clauses * config.c_t),
              's': config.c_specificity,
              'y_max': config.c_y_max, 'y_min': config.c_y_min, 'bits_per_feature': config.c_bits_per_feature,
              'number_of_state_bits_ta': config.c_number_of_state_bits_ta}

    _config = {'comment': 'newest', 'algorithm': 'TM_PPO', 'gamma': config.gamma, 'lam': config.lam, 'device': 'CPU',
               'weighted_clauses': False,
               "actor": actor, "critic": critic, 'batch_size': config.batch_size, 'epochs': config.epochs,
               'test_freq': 1, "save": False, "seed": 42, "threshold": acrobot_threshold,
               'n_timesteps': config.n_timesteps, "dataset_file_name": "acrobot_obs_data"}

    # env = gym.make("CartPole-v1")
    env = gym.make("Acrobot-v1")

    agent = PPO(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_1)
    score = np.array(agent.best_score)
    return score


################################################
######### n-step Double QTM type a ############
################################################
def cartpole_n_step_DQTM_a(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Q_Network.n_step_Double_TMQN import TMQN
    from algorithms.policy.RTM import Policy

    _config = {
        'algorithm': 'n_step_Double_TMQN', 'soft_update_type': 'soft_update_1', 'n_steps': config.n_steps,
        'nr_of_clauses': config.nr_of_clauses, 'T': int(config.t * config.nr_of_clauses),
        "max_update_p": config.max_update_p, "min_update_p": 0, 's': config.specificity, 'y_max': config.y_max,
        'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma, 'exploration_prob_init':config.exploration_p_init,
        'exploration_prob_decay': config.exploration_p_decay, 'buffer_size': config.buffer_size,
        'batch_size': config.batch_size, 'epochs': config.epochs, 'test_freq': test_freq_2, "save": False, "seed": 42,  "threshold": cartpole_threshold,
        'number_of_state_bits_ta': config.number_of_state_bits_ta, 'update_grad': config.update_grad, 'update_freq': -1,
        "dataset_file_name": "observation_data"}

    env = gym.make("CartPole-v1")

    agent = TMQN(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_2)
    scores = np.array(agent.total_score)
    score = np.mean(scores)
    return score


def acrobot_n_step_DQTM_a(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Q_Network.n_step_Double_TMQN import TMQN
    from algorithms.policy.RTM import Policy

    _config = {
        'algorithm': 'n_step_Double_TMQN', 'soft_update_type': 'soft_update_1', 'n_steps': config.n_steps,
        'nr_of_clauses': config.nr_of_clauses, 'T': int(config.t * config.nr_of_clauses),
        "max_update_p": config.max_update_p, "min_update_p": 0, 's': config.specificity, 'y_max': config.y_max,
        'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma, 'exploration_prob_init': config.exploration_p_decay,
        'exploration_prob_decay': config.exploration_p_decay, 'buffer_size': config.buffer_size,
        'batch_size': config.batch_size, 'epochs': config.epochs, 'test_freq': 1, "save": False, "seed": 42, "threshold": acrobot_threshold,
        'number_of_state_bits_ta': config.number_of_state_bits_ta, 'update_grad': config.update_grad, 'update_freq': -1,
        "dataset_file_name": "acrobot_obs_data"}

    env = gym.make("Acrobot-v1")

    agent = TMQN(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_1)
    score = np.array(agent.best_scores['mean'])
    return score


################################################
######### n-step Double TMQN type b ############
################################################
def cartpole_n_step_DQTM_b(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Q_Network.n_step_Double_TMQN import TMQN
    from algorithms.policy.RTM import Policy

    _config = {
        'algorithm': 'n_step_Double_TMQN', 'soft_update_type': 'soft_update_2', 'n_steps': config.n_steps,
        'nr_of_clauses': config.nr_of_clauses, 'T': int(config.t * config.nr_of_clauses),
        "max_update_p": config.max_update_p, "min_update_p": 0, 's': config.specificity, 'y_max': config.y_max,
        'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma, 'exploration_prob_init': config.exploration_p_decay,
        'exploration_prob_decay': config.exploration_p_decay, 'buffer_size': config.buffer_size,
        'batch_size': config.batch_size, 'epochs': config.epochs, 'test_freq': test_freq_2, "save": False, "seed": 42,  "threshold": cartpole_threshold,
        'number_of_state_bits_ta': config.number_of_state_bits_ta, 'update_grad': -1, 'update_freq': config.update_freq,
        "dataset_file_name": "observation_data"}

    env = gym.make("CartPole-v1")

    agent = TMQN(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_2)
    scores = np.array(agent.total_score)
    score = np.mean(scores)
    return score


def acrobot_n_step_DQTM_b(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Q_Network.n_step_Double_TMQN import TMQN
    from algorithms.policy.RTM import Policy

    _config = {
        'algorithm': 'n_step_Double_TMQN', 'soft_update_type': 'soft_update_2', 'n_steps': config.n_steps,
        'nr_of_clauses': config.nr_of_clauses, 'T': int(config.t * config.nr_of_clauses),
        "max_update_p": config.max_update_p, "min_update_p": 0, 's': config.specificity, 'y_max': config.y_max,
        'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma, 'exploration_prob_init': config.exploration_p_decay,
        'exploration_prob_decay': config.exploration_p_decay, 'buffer_size': config.buffer_size,
        'batch_size': config.batch_size, 'epochs': config.epochs, 'test_freq': 1, "save": False, "seed": 42, "threshold": acrobot_threshold,
        'number_of_state_bits_ta': config.number_of_state_bits_ta, 'update_grad': -1, 'update_freq': config.update_freq,
        "dataset_file_name": "acrobot_obs_data"}

    env = gym.make("Acrobot-v1")

    agent = TMQN(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_2)
    score = np.array(agent.best_scores['mean'])
    return score


################################################
############ Double TMQN type a ################
################################################

def cartpole_DQTM_a(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Q_Network.Double_TMQN import TMQN
    from algorithms.policy.RTM import Policy

    _config = {
        'algorithm': 'Double_TMQN', 'soft_update_type': 'soft_update_1',
        'nr_of_clauses': config.nr_of_clauses, 'T': int(config.t * config.nr_of_clauses),
        "max_update_p": config.max_update_p, "min_update_p": 0, 's': config.specificity, 'y_max': config.y_max,
        'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma, 'exploration_prob_init': config.exploration_p_decay,
        'exploration_prob_decay': config.exploration_p_decay, 'buffer_size': config.buffer_size,
        'batch_size': config.batch_size, 'epochs': config.epochs, 'test_freq': test_freq_2, "save": False, "seed": 42,  "threshold": cartpole_threshold,
        'number_of_state_bits_ta': config.number_of_state_bits_ta, 'update_grad': config.update_grad, 'update_freq': -1,
        "dataset_file_name": "observation_data"}

    env = gym.make("CartPole-v1")

    agent = TMQN(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_2)
    scores = np.array(agent.total_score)
    score = np.mean(scores)
    return score


def acrobot_DQTM_a(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Q_Network.Double_TMQN import TMQN
    from algorithms.policy.RTM import Policy

    _config = {
        'algorithm': 'Double_TMQN', 'soft_update_type': 'soft_update_1', 'nr_of_clauses': config.nr_of_clauses,
        'T': int(config.t * config.nr_of_clauses),
        "max_update_p": config.max_update_p, "min_update_p": 0, 's': config.specificity, 'y_max': config.y_max,
        'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma, 'exploration_prob_init': config.exploration_p_decay,
        'exploration_prob_decay': config.exploration_p_decay, 'buffer_size': config.buffer_size,
        'batch_size': config.batch_size, 'epochs': config.epochs, 'test_freq': 1, "save": False, "seed": 42, "threshold": acrobot_threshold,
        'number_of_state_bits_ta': config.number_of_state_bits_ta, 'update_grad': config.update_grad, 'update_freq': -1,
        "dataset_file_name": "acrobot_obs_data"}

    env = gym.make("Acrobot-v1")

    agent = TMQN(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_2)
    score = np.array(agent.best_scores['mean'])
    return score


################################################
############ Double TMQN type b ################
################################################
def cartpole_DQTM_b(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Q_Network.Double_TMQN import TMQN
    from algorithms.policy.RTM import Policy

    _config = {
        'algorithm': 'Double_TMQN', 'soft_update_type': 'soft_update_2', 'nr_of_clauses': config.nr_of_clauses,
        'T': int(config.t * config.nr_of_clauses),
        "max_update_p": config.max_update_p, "min_update_p": 0, 's': config.specificity, 'y_max': config.y_max,
        'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma, 'exploration_prob_init': config.exploration_p_decay,
        'exploration_prob_decay': config.exploration_p_decay, 'buffer_size': config.buffer_size,
        'batch_size': config.batch_size, 'epochs': config.epochs, 'test_freq': test_freq_2, "save": False, "seed": 42,  "threshold": cartpole_threshold,
        'number_of_state_bits_ta': config.number_of_state_bits_ta, 'update_grad': -1, 'update_freq': config.update_freq,
        "dataset_file_name": "observation_data"}

    env = gym.make("CartPole-v1")

    agent = TMQN(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_2)
    scores = np.array(agent.total_score)
    score = np.mean(scores)
    return score


def acrobot_DQTM_b(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Q_Network.Double_TMQN import TMQN
    from algorithms.policy.RTM import Policy

    _config = {
        'algorithm': 'Double_TMQN', 'soft_update_type': 'soft_update_2', 'nr_of_clauses': config.nr_of_clauses,
        'T': int(config.t * config.nr_of_clauses),
        "max_update_p": config.max_update_p, "min_update_p": 0, 's': config.specificity, 'y_max': config.y_max,
        'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma, 'exploration_prob_init': config.exploration_p_decay,
        'exploration_prob_decay': config.exploration_p_decay, 'buffer_size': config.buffer_size,
        'batch_size': config.batch_size, 'epochs': config.epochs, 'test_freq': 1, "save": False, "seed": 42, "threshold": acrobot_threshold,
        'number_of_state_bits_ta': config.number_of_state_bits_ta, 'update_grad': -1, 'update_freq': config.update_freq,
        "dataset_file_name": "acrobot_obs_data"}

    env = gym.make("Acrobot-v1")

    agent = TMQN(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_2)
    score = np.array(agent.best_scores['mean'])
    return score


################################################
################# n-step TMQN  #################
################################################

def cartpole_n_step_QTM(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Q_Network.n_step_TMQN import TMQN
    from algorithms.policy.RTM import Policy

    _config = {
        'algorithm': 'n_step_TMQN', 'n_steps': config.n_steps,
        'nr_of_clauses': config.nr_of_clauses, 'T': int(config.t * config.nr_of_clauses),
        "max_update_p": config.max_update_p, "min_update_p": 0.0, 's': config.specificity, 'y_max': config.y_max,
        'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma, 'exploration_prob_init': config.exploration_p_decay,
        'exploration_prob_decay': config.exploration_p_decay, 'buffer_size': config.buffer_size,  "threshold": cartpole_threshold,
        'batch_size': config.batch_size, 'epochs': config.epochs, 'test_freq': test_freq_2, "save": False, "seed": 42,
        'number_of_state_bits_ta': config.number_of_state_bits_ta,
        "dataset_file_name": "observation_data"}

    env = gym.make("CartPole-v1")

    agent = TMQN(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_2)
    scores = np.array(agent.total_score)
    score = np.mean(scores)
    return score


def acrobot_n_step_QTM(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Q_Network.n_step_TMQN import TMQN
    from algorithms.policy.RTM import Policy

    _config = {
        'algorithm': 'n_step_TMQN', 'n_steps': config.n_steps,
        'nr_of_clauses': config.nr_of_clauses, 'T': int(config.t * config.nr_of_clauses),
        "max_update_p": config.max_update_p, "min_update_p": 0, 's': config.specificity, 'y_max': config.y_max,
        'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma, 'exploration_prob_init': config.exploration_p_decay,
        'exploration_prob_decay': config.exploration_p_decay, 'buffer_size': config.buffer_size, "threshold": acrobot_threshold,
        'batch_size': config.batch_size, 'epochs': config.epochs, 'test_freq': 1, "save": False, "seed": 42,
        'number_of_state_bits_ta': config.number_of_state_bits_ta,
        "dataset_file_name": "acrobot_obs_data"}

    env = gym.make("Acrobot-v1")

    agent = TMQN(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_2)
    score = np.array(agent.best_scores['mean'])
    return score


################################################
################### TMQN  ######################
################################################
def cartpole_QTM(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Q_Network.TMQN import TMQN
    from algorithms.policy.RTM import Policy

    _config = {
        'algorithm': 'TMQN',
        'nr_of_clauses': config.nr_of_clauses, 'T': int(config.t * config.nr_of_clauses),
        "max_update_p": 0.5, "min_update_p": 0, 's': config.specificity, 'y_max': config.y_max,
        'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma, 'exploration_prob_init': config.exploration_p_decay,
        'exploration_prob_decay': config.exploration_p_decay, 'buffer_size': config.buffer_size,  "threshold": cartpole_threshold,
        'batch_size': config.batch_size, 'epochs': config.epochs, 'test_freq': test_freq_2, "save": False, "seed": 42,
        'number_of_state_bits_ta': config.number_of_state_bits_ta,
        "dataset_file_name": "observation_data"}

    env = gym.make("CartPole-v1")

    agent = TMQN(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_2)
    scores = np.array(agent.total_score)
    score = np.mean(scores)
    return score


def acrobot_QTM(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Q_Network.TMQN import TMQN
    from algorithms.policy.RTM import Policy

    _config = {
        'algorithm': 'TMQN',
        'nr_of_clauses': config.nr_of_clauses, 'T': int(config.t * config.nr_of_clauses),
        "max_update_p": config.max_update_p, "min_update_p": 0, 's': config.specificity, 'y_max': config.y_max,
        'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma, 'exploration_prob_init': config.exploration_p_decay,
        'exploration_prob_decay': config.exploration_p_decay, 'buffer_size': config.buffer_size,  "threshold": acrobot_threshold,
        'batch_size': config.batch_size, 'epochs': config.epochs, 'test_freq': 1, "save": False, "seed": 42,
        'number_of_state_bits_ta': config.number_of_state_bits_ta,
        "dataset_file_name": "acrobot_obs_data"}

    env = gym.make("Acrobot-v1")

    agent = TMQN(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_2)
    score = np.array(agent.best_scores['mean'])
    return score
