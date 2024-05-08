import numpy as np
import torch
import random

################################################
################### TAC a #######################
################################################

n_episodes_1 = 1000
n_epsidoes_acro = 250
n_episodes_2 = 5000
test_freq_2 = 25
cartpole_threshold = 20
acrobot_threshold = -495

def cartpole_TAC_a(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Tsetlin_actor_critic.TAC import TAC
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
               'epsilon_init': config.epsilon_init, 'epsilon_decay': config.epsilon_decay,
               'clause_update_p': config.clause_update_p, 'gamma': config.gamma,
               "buffer_size": config.buffer_size, 'actor': actor, 'critic': critic, 'batch_size': config.batch_size,
               'sampling_iterations': config.sampling_iterations, 'test_freq': 1, "save": False, "threshold": cartpole_threshold, "dataset_file_name": "observation_data"}
    print(_config)

    env = gym.make("CartPole-v1")

    agent = TAC(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_1)
    score = np.array(agent.best_score)
    print(f'mean: {np.mean(np.array(agent.scores))}')
    return score


def acrobot_random_TAC_a(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Tsetlin_actor_critic.TAC_random import TAC
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
               'epsilon_init': config.epsilon_init, 'epsilon_decay': config.epsilon_decay,
               'clause_update_p': config.clause_update_p, 'gamma': config.gamma,
               "buffer_size": config.buffer_size, 'actor': actor, 'critic': critic, 'batch_size': config.batch_size,
               'sampling_iterations': config.sampling_iterations, 'test_freq': 1, "save": False, "threshold": acrobot_threshold,
               "dataset_file_name": "acrobot_obs_data"}  # "observation_data"}
    print(_config)

    env = gym.make("Acrobot-v1")

    agent = TAC(env, Policy, _config)
    agent.learn(nr_of_episodes=n_epsidoes_acro)
    score = np.array(agent.best_score)
    print(f'mean: {np.mean(np.array(agent.scores))}')

    return score

def cartpole_random_TAC_a(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Tsetlin_actor_critic.TAC_random import TAC
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
               'epsilon_init': config.epsilon_init, 'epsilon_decay': config.epsilon_decay,
               'clause_update_p': config.clause_update_p, 'gamma': config.gamma,
               "buffer_size": config.buffer_size, 'actor': actor, 'critic': critic, 'batch_size': config.batch_size,
               'sampling_iterations': config.sampling_iterations, 'test_freq': 1, "save": False, "threshold": cartpole_threshold, "dataset_file_name": "observation_data"}
    print(_config)

    env = gym.make("CartPole-v1")

    agent = TAC(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_1)
    score = np.array(agent.best_score)
    print(f'mean: {np.mean(np.array(agent.scores))}')
    return score


def acrobot_TAC_a(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Tsetlin_actor_critic.TAC import TAC
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
               'epsilon_init': config.epsilon_init, 'epsilon_decay': config.epsilon_decay,
               'clause_update_p': config.clause_update_p, 'gamma': config.gamma,
               "buffer_size": config.buffer_size, 'actor': actor, 'critic': critic, 'batch_size': config.batch_size,
               'sampling_iterations': config.sampling_iterations, 'test_freq': 1, "save": False, "threshold": acrobot_threshold,
               "dataset_file_name": "acrobot_obs_data"}  # "observation_data"}
    print(_config)

    env = gym.make("Acrobot-v1")

    agent = TAC(env, Policy, _config)
    agent.learn(nr_of_episodes=n_epsidoes_acro)
    score = np.array(agent.best_score)
    print(f'mean: {np.mean(np.array(agent.scores))}')

    return score


################################################
################### TAC b #######################
################################################

def cartpole_TAC_b(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Tsetlin_actor_critic.TAC import TAC
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
               'epsilon_init': config.epsilon_init, 'epsilon_decay': config.epsilon_decay,
               'update_freq': config.update_freq, 'gamma': config.gamma,
               "buffer_size": config.buffer_size, 'actor': actor, 'critic': critic, 'batch_size': config.batch_size,
               'sampling_iterations': config.sampling_iterations, 'test_freq': 1,  "threshold": cartpole_threshold, "save": False, "dataset_file_name": "observation_data"}
    print(_config)

    env = gym.make("CartPole-v1")

    agent = TAC(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_1)
    score = np.array(agent.best_score)
    print(f'mean: {np.mean(np.array(agent.scores))}')

    return score


def acrobot_TAC_b(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Tsetlin_actor_critic.TAC import TAC
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
               'epsilon_init': config.epsilon_init, 'epsilon_decay': config.epsilon_decay,
               'update_freq': config.update_freq, 'gamma': config.gamma,
               "buffer_size": config.buffer_size, 'actor': actor, 'critic': critic, 'batch_size': config.batch_size,
               'sampling_iterations': config.sampling_iterations, 'test_freq': 1, "save": False, "threshold": acrobot_threshold,
               "dataset_file_name": "acrobot_obs_data"}  # "observation_data"}
    print(_config)

    env = gym.make("Acrobot-v1")

    agent = TAC(env, Policy, _config)
    agent.learn(nr_of_episodes=n_epsidoes_acro)
    score = np.array(agent.best_score)
    print(f'mean: {np.mean(np.array(agent.scores))}')

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
    print(_config)

    env = gym.make("CartPole-v1")

    agent = PPO(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_1)
    score = np.array(agent.best_score)
    print(f'Mean: {np.mean(np.array(agent.total_scores))}')
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
    print(_config)

    # env = gym.make("CartPole-v1")
    env = gym.make("Acrobot-v1")

    agent = PPO(env, Policy, _config)
    agent.learn(nr_of_episodes=n_epsidoes_acro)
    score = np.array(agent.best_score)
    print(f'Mean: {np.mean(np.array(agent.total_scores))}')
    return score


################################################
######### n-step Double QTM type a ############
################################################
def cartpole_n_step_DQTM_a(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Q_Network.n_step_Double_QTM import QTM
    from algorithms.policy.RTM import Policy

    _config = {
        'algorithm': 'n_step_Double_TMQN', 'soft_update_type': 'soft_update_1', 'n_steps': config.n_steps,
        'nr_of_clauses': config.nr_of_clauses, 'T': int(config.t * config.nr_of_clauses),
        "max_update_p": config.max_update_p, "min_update_p": 0, 's': config.specificity, 'y_max': config.y_max,
        'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma,
        'epsilon_init': config.epsilon_init, 'epsilon_decay': config.epsilon_decay, 'buffer_size': config.buffer_size,
        'batch_size': config.batch_size, 'sampling_iterations': config.sampling_iterations, 'test_freq': test_freq_2, "save": False, "seed": 42,  "threshold": cartpole_threshold,
        'number_of_state_bits_ta': config.number_of_state_bits_ta, 'clause_update_p': config.clause_update_p, 'update_freq': -1,
        "dataset_file_name": "observation_data"}
    print(_config)

    env = gym.make("CartPole-v1")

    agent = QTM(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_2)
    scores = np.array(agent.total_score)
    score = np.mean(scores)
    return score


def acrobot_n_step_DQTM_a(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Q_Network.n_step_Double_QTM import QTM
    from algorithms.policy.RTM import Policy

    _config = {
        'algorithm': 'n_step_Double_TMQN', 'soft_update_type': 'soft_update_1', 'n_steps': config.n_steps,
        'nr_of_clauses': config.nr_of_clauses, 'T': int(config.t * config.nr_of_clauses),
        "max_update_p": config.max_update_p, "min_update_p": 0, 's': config.specificity, 'y_max': config.y_max,
        'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma, 'epsilon_init': config.epsilon_init, 'epsilon_decay': config.epsilon_decay, 'buffer_size': config.buffer_size,
        'batch_size': config.batch_size, 'sampling_iterations': config.sampling_iterations, 'test_freq': 1, "save": False, "seed": 42, "threshold": acrobot_threshold,
        'number_of_state_bits_ta': config.number_of_state_bits_ta, 'clause_update_p': config.clause_update_p, 'update_freq': -1,
        "dataset_file_name": "acrobot_obs_data"}
    print(_config)

    env = gym.make("Acrobot-v1")

    agent = QTM(env, Policy, _config)
    agent.learn(nr_of_episodes=n_epsidoes_acro)
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
    from algorithms.Q_Network.n_step_Double_QTM import QTM
    from algorithms.policy.RTM import Policy

    _config = {
        'algorithm': 'n_step_Double_TMQN', 'soft_update_type': 'soft_update_2', 'n_steps': config.n_steps,
        'nr_of_clauses': config.nr_of_clauses, 'T': int(config.t * config.nr_of_clauses),
        "max_update_p": config.max_update_p, "min_update_p": 0, 's': config.specificity, 'y_max': config.y_max,
        'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma,
        'epsilon_init': config.epsilon_init, 'epsilon_decay': config.epsilon_decay, 'buffer_size': config.buffer_size,
        'batch_size': config.batch_size, 'sampling_iterations': config.sampling_iterations, 'test_freq': test_freq_2, "save": False, "seed": 42,  "threshold": cartpole_threshold,
        'number_of_state_bits_ta': config.number_of_state_bits_ta, 'update_grad': -1, 'update_freq': config.update_freq,
        "dataset_file_name": "observation_data"}
    print(_config)
    env = gym.make("CartPole-v1")

    agent = QTM(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_2)
    scores = np.array(agent.total_score)
    score = np.mean(scores)
    return score


def acrobot_n_step_DQTM_b(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Q_Network.n_step_Double_QTM import QTM
    from algorithms.policy.RTM import Policy

    _config = {
        'algorithm': 'n_step_Double_TMQN', 'soft_update_type': 'soft_update_2', 'n_steps': config.n_steps,
        'nr_of_clauses': config.nr_of_clauses, 'T': int(config.t * config.nr_of_clauses),
        "max_update_p": config.max_update_p, "min_update_p": 0, 's': config.specificity, 'y_max': config.y_max,
        'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma,
        'epsilon_init': config.epsilon_init, 'epsilon_decay': config.epsilon_decay, 'buffer_size': config.buffer_size,
        'batch_size': config.batch_size, 'sampling_iterations': config.sampling_iterations, 'test_freq': 1, "save": False, "seed": 42, "threshold": acrobot_threshold,
        'number_of_state_bits_ta': config.number_of_state_bits_ta, 'update_grad': -1, 'update_freq': config.update_freq,
        "dataset_file_name": "acrobot_obs_data"}
    print(_config)

    env = gym.make("Acrobot-v1")

    agent = QTM(env, Policy, _config)
    agent.learn(nr_of_episodes=n_epsidoes_acro)
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
    from algorithms.Q_Network.Double_QTM import QTM
    from algorithms.policy.RTM import Policy

    _config = {
        'algorithm': 'Double_TMQN', 'soft_update_type': 'soft_update_1',
        'nr_of_clauses': config.nr_of_clauses, 'T': int(config.t * config.nr_of_clauses),
        "max_update_p": config.max_update_p, "min_update_p": 0, 's': config.specificity, 'y_max': config.y_max,
        'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma,
        'epsilon_init': config.epsilon_init, 'epsilon_decay': config.epsilon_decay, 'buffer_size': config.buffer_size,
        'batch_size': config.batch_size, 'sampling_iterations': config.sampling_iterations, 'test_freq': test_freq_2, "save": False, "seed": 42,  "threshold": cartpole_threshold,
        'number_of_state_bits_ta': config.number_of_state_bits_ta, 'clause_update_p': config.clause_update_p, 'update_freq': -1,
        "dataset_file_name": "observation_data"}
    print(_config)

    env = gym.make("CartPole-v1")

    agent = QTM(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_2)
    scores = np.array(agent.total_score)
    score = np.mean(scores)
    return score


def acrobot_DQTM_a(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Q_Network.Double_QTM import QTM
    from algorithms.policy.RTM import Policy

    _config = {
        'algorithm': 'Double_TMQN', 'soft_update_type': 'soft_update_1', 'nr_of_clauses': config.nr_of_clauses,
        'T': int(config.t * config.nr_of_clauses),
        "max_update_p": config.max_update_p, "min_update_p": 0, 's': config.specificity, 'y_max': config.y_max,
        'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma,
        'epsilon_init': config.epsilon_init, 'epsilon_decay': config.epsilon_decay, 'buffer_size': config.buffer_size,
        'batch_size': config.batch_size, 'sampling_iterations': config.sampling_iterations, 'test_freq': 1, "save": False, "seed": 42, "threshold": acrobot_threshold,
        'number_of_state_bits_ta': config.number_of_state_bits_ta, 'clause_update_p': config.clause_update_p, 'update_freq': -1,
        "dataset_file_name": "acrobot_obs_data"}
    print(_config)

    env = gym.make("Acrobot-v1")

    agent = QTM(env, Policy, _config)
    agent.learn(nr_of_episodes=n_epsidoes_acro)
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
    from algorithms.Q_Network.Double_QTM import QTM
    from algorithms.policy.RTM import Policy

    _config = {
        'algorithm': 'Double_TMQN', 'soft_update_type': 'soft_update_2', 'nr_of_clauses': config.nr_of_clauses,
        'T': int(config.t * config.nr_of_clauses),
        "max_update_p": config.max_update_p, "min_update_p": 0, 's': config.specificity, 'y_max': config.y_max,
        'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma,
        'epsilon_init': config.epsilon_init, 'epsilon_decay': config.epsilon_decay, 'buffer_size': config.buffer_size,
        'batch_size': config.batch_size, 'sampling_iterations': config.sampling_iterations, 'test_freq': test_freq_2, "save": False, "seed": 42,  "threshold": cartpole_threshold,
        'number_of_state_bits_ta': config.number_of_state_bits_ta, 'update_grad': -1, 'update_freq': config.update_freq,
        "dataset_file_name": "observation_data"}
    print(_config)

    env = gym.make("CartPole-v1")

    agent = QTM(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_2)
    scores = np.array(agent.total_score)
    score = np.mean(scores)
    return score


def acrobot_DQTM_b(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Q_Network.Double_QTM import QTM
    from algorithms.policy.RTM import Policy

    _config = {
        'algorithm': 'Double_TMQN', 'soft_update_type': 'soft_update_2', 'nr_of_clauses': config.nr_of_clauses,
        'T': int(config.t * config.nr_of_clauses),
        "max_update_p": config.max_update_p, "min_update_p": 0, 's': config.specificity, 'y_max': config.y_max,
        'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma,
        'epsilon_init': config.epsilon_init, 'epsilon_decay': config.epsilon_decay, 'buffer_size': config.buffer_size,
        'batch_size': config.batch_size, 'sampling_iterations': config.sampling_iterations, 'test_freq': 1, "save": False, "seed": 42, "threshold": acrobot_threshold,
        'number_of_state_bits_ta': config.number_of_state_bits_ta, 'update_grad': -1, 'update_freq': config.update_freq,
        "dataset_file_name": "acrobot_obs_data"}
    print(_config)

    env = gym.make("Acrobot-v1")

    agent = QTM(env, Policy, _config)
    agent.learn(nr_of_episodes=n_epsidoes_acro)
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
    from algorithms.Q_Network.n_step_QTM import QTM
    from algorithms.policy.RTM import Policy

    _config = {
        'algorithm': 'n_step_TMQN', 'n_steps': config.n_steps,
        'nr_of_clauses': config.nr_of_clauses, 'T': int(config.t * config.nr_of_clauses),
        "max_update_p": config.max_update_p, "min_update_p": 0.0, 's': config.specificity, 'y_max': config.y_max,
        'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma,
        'epsilon_init': config.epsilon_init, 'epsilon_decay': config.epsilon_decay, 'buffer_size': config.buffer_size,  "threshold": cartpole_threshold,
        'batch_size': config.batch_size, 'sampling_iterations': config.sampling_iterations, 'test_freq': test_freq_2, "save": False, "seed": 42,
        'number_of_state_bits_ta': config.number_of_state_bits_ta,
        "dataset_file_name": "observation_data"}
    print(_config)

    env = gym.make("CartPole-v1")

    agent = QTM(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_2)
    scores = np.array(agent.total_score)
    score = np.mean(scores)
    return score


def acrobot_n_step_QTM(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Q_Network.n_step_QTM import QTM
    from algorithms.policy.RTM import Policy

    _config = {
        'algorithm': 'n_step_TMQN', 'n_steps': config.n_steps,
        'nr_of_clauses': config.nr_of_clauses, 'T': int(config.t * config.nr_of_clauses),
        "max_update_p": config.max_update_p, "min_update_p": 0, 's': config.specificity, 'y_max': config.y_max,
        'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma,
        'epsilon_init': config.epsilon_init, 'epsilon_decay': config.epsilon_decay, 'buffer_size': config.buffer_size, "threshold": acrobot_threshold,
        'batch_size': config.batch_size, 'sampling_iterations': config.sampling_iterations, 'test_freq': 1, "save": False, "seed": 42,
        'number_of_state_bits_ta': config.number_of_state_bits_ta,
        "dataset_file_name": "acrobot_obs_data"}
    print(_config)

    env = gym.make("Acrobot-v1")

    agent = QTM(env, Policy, _config)
    agent.learn(nr_of_episodes=n_epsidoes_acro)
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
    from algorithms.Q_Network.QTM import QTM
    from algorithms.policy.RTM import Policy

    _config = {
        'algorithm': 'TMQN',
        'nr_of_clauses': config.nr_of_clauses, 'T': int(config.t * config.nr_of_clauses),
        "max_update_p": 0.5, "min_update_p": 0, 's': config.specificity, 'y_max': config.y_max,
        'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma,
        'epsilon_init': config.epsilon_init, 'epsilon_decay': config.epsilon_decay, 'buffer_size': config.buffer_size,  "threshold": cartpole_threshold,
        'batch_size': config.batch_size, 'sampling_iterations': config.sampling_iterations, 'test_freq': test_freq_2, "save": False, "seed": 42,
        'number_of_state_bits_ta': config.number_of_state_bits_ta,
        "dataset_file_name": "observation_data"}
    print(_config)
    env = gym.make("CartPole-v1")

    agent = QTM(env, Policy, _config)
    agent.learn(nr_of_episodes=n_episodes_2)
    scores = np.array(agent.total_score)
    score = np.mean(scores)
    return score


def acrobot_QTM(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Q_Network.QTM import QTM
    from algorithms.policy.RTM import Policy

    _config = {
        'algorithm': 'TMQN',
        'nr_of_clauses': config.nr_of_clauses, 'T': int(config.t * config.nr_of_clauses),
        "max_update_p": config.max_update_p, "min_update_p": 0, 's': config.specificity, 'y_max': config.y_max,
        'y_min': config.y_min, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': config.bits_per_feature,
        'gamma': config.gamma, 'epsilon_init': config.epsilon_init, 'epsilon_decay': config.epsilon_decay, 'buffer_size': config.buffer_size,  "threshold": acrobot_threshold,
        'batch_size': config.batch_size, 'sampling_iterations': config.sampling_iterations, 'test_freq': 1, "save": False, "seed": 42,
        'number_of_state_bits_ta': config.number_of_state_bits_ta,
        "dataset_file_name": "acrobot_obs_data"}
    print(_config)
    env = gym.make("Acrobot-v1")
    agent = QTM(env, Policy, _config)
    agent.learn(nr_of_episodes=n_epsidoes_acro)
    score = np.array(agent.best_scores['mean'])
    return score

def acrobot_PPO(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Proximal_policy.PPO import PPO
    from algorithms.policy.DNN import ActorCriticPolicy as Policy

    _config = {'env_name': 'acrobot', 'algorithm': 'PPO', 'n_steps': config.n_steps, 'gamma': config.gamma, 'lam': config.lam, 'clip_range': config.clip_range,
              'batch_size': config.batch_size, 'epochs': config.epochs, 'hidden_size': config.hidden_size, 'learning_rate': config.lr, 'test_freq': 50, "save": False}

    print(_config)
    env = gym.make("Acrobot-v1")
    agent = PPO(env, Policy, _config)
    agent.learn(nr_of_episodes=5000)
    score = agent.best_score
    print(f'mean: {np.mean(np.array(agent.scores))}')
    return score

def cartpole_PPO(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Proximal_policy.PPO import PPO
    from algorithms.policy.DNN import ActorCriticPolicy as Policy

    _config = {'env_name': 'cartpole', 'algorithm': 'PPO', 'n_steps': config.n_steps, 'gamma': config.gamma, 'lam': config.lam, 'clip_range': config.clip_range,
              'batch_size': config.batch_size, 'epochs': config.epochs, 'hidden_size': config.hidden_size, 'learning_rate': config.lr, 'test_freq': 50, "save": False}

    print(_config)
    env = gym.make("CartPole-v1")
    agent = PPO(env, Policy, _config)
    agent.learn(nr_of_episodes=5000)
    score = agent.best_score
    print(f'mean: {np.mean(np.array(agent.scores))}')
    return score

def acrobot_DQN(config):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import gymnasium as gym
    from algorithms.Q_Network.n_step_DQN import DQN
    from algorithms.policy.DNN import Policy as Policy

    _config = {'env_name': 'acrobot', "n_steps": config.n_steps, 'algorithm': 'DQN', "c": 0, 'gamma': config.gamma, "buffer_size": config.buffer_size,
              'batch_size': config.batch_size,
            'epsilon_init': config.epsilon_init, 'epsilon_decay': config.epsilon_decay, 'sampling_iterations': config.sampling_iterations, 'hidden_size': config.hidden_size, 'learning_rate': config.lr, 'test_freq': 50, "save": False}

    print(_config)
    env = gym.make("Acrobot-v1")
    agent = DQN(env, Policy, _config)
    agent.learn(nr_of_episodes=2500)
    score = agent.best_scores['mean']
    print(f'mean: {np.mean(np.array(agent.scores))}')
    return score