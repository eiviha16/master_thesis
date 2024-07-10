import numpy as np
import os
import yaml
from algorithms.misc.replay_buffer import ReplayBuffer
import torch
from tqdm import tqdm
import random


class TAC:
    def __init__(self, env, Policy, config):
        self.env = env
        self.action_space_size = env.action_space.n
        self.obs_space_size = env.observation_space.shape[0]
        config['action_space_size'] = self.action_space_size
        config['obs_space_size'] = self.obs_space_size
        self.gamma = config['gamma']
        self.policy = Policy(config)
        self.replay_buffer = ReplayBuffer(config['buffer_size'], config['batch_size'])
        self.init_epsilon = config['epsilon_init']
        self.epsilon = self.init_epsilon
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_min = 0

        self.sampling_iterations = config['sampling_iterations']
        self.config = config

        self.test_random_seeds = [83811, 14593, 3279, 97197, 36049, 32099, 29257, 18290, 96531, 13435, 88697, 97081,
                                  71483, 11396, 77398, 55303, 4166, 3906, 12281, 28658, 30496, 66238, 78908, 3479,
                                  73564, 26063, 93851, 85182, 91925, 71427, 54988, 28894, 58879, 77237, 36464, 852,
                                  99459, 20927, 91507, 55393, 44598, 36422, 20380, 28222, 44119, 13397, 12157, 49798,
                                  12677, 47053, 45083, 79132, 34672, 5696, 95648, 60218, 70285, 16362, 49616, 10329,
                                  72358, 38428, 82398, 81071, 47401, 75675, 25204, 92350, 9117, 6007, 86674, 29872,
                                  37931, 10459, 30513, 13239, 49824, 36435, 59430, 83321, 47820, 21320, 48521, 46567,
                                  27461, 87842, 34994, 91989, 89594, 84940, 9359, 79841, 83228, 22432, 70011, 95569,
                                  32088, 21418, 60590, 49736]

        self.test_seeds = np.random
        self.save = config['save']
        self.save_path = ''

        if self.save:
            self.run_id = 'run_' + str(
                len([i for i in os.listdir(f'../results/{config["env_name"]}/{config["algorithm"]}')]) + 1)
        else:
            print('Warning SAVING is OFF!')
            self.run_id = "unidentified_run"
        if self.save:
            self.make_run_dir()
            self.save_config(config)
        self.best_score = float('-inf')
        self.announce()
        self.cur_episode = 0
        self.total_score = []
        self.batch_size = config['batch_size']
        self.threshold = config['threshold']
        self.scores = []
        self.timesteps = 0

    def announce(self):
        print(f'{self.run_id} has been initialized!')

    def save_config(self, config):
        if self.save:
            with open(f'{self.save_path}/config.yaml', "w") as yaml_file:
                yaml.dump(config, yaml_file, default_flow_style=False)

    def rollout(self):
        cur_obs, _ = self.env.reset(seed=random.randint(1, 100))
        while True:
            action, actions = self.get_next_action(cur_obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.replay_buffer.save_experience(actions, cur_obs, next_obs, reward, terminated)
            self.timesteps += 1
            if terminated or truncated:
                break
            cur_obs = next_obs

    def get_actor_update(self, actions):
        tm = {'observations': [], 'actions': [], 'feedback': []}
        for index, action in enumerate(np.argmax(actions, axis=1)):
            random_float = np.random.rand()
            if random_float > 0.5:
                feedback = 1
            else:
                feedback = 2
            tm['feedback'].append(feedback)
            tm['actions'].append(action)
            tm['observations'].append(self.replay_buffer.sampled_cur_obs[index])
        return tm

    def get_q_val_for_action(self, actions, q_values):
        q_vals = []
        for index, action in enumerate(actions):
            q_vals.append(q_values[index][action])
        return np.array(q_vals)

    def get_next_action(self, cur_obs):
        if np.random.random() < self.epsilon:
            actions = np.array([0 for i in range(self.action_space_size)])
            actions[np.random.randint(0, self.action_space_size)] = 1
            action = np.argmax(actions)
        else:
            action, actions = self.policy.get_action(cur_obs)
        return action, actions

    def get_q_val_and_obs_for_tm(self, actions, target_q_vals):
        tms = {'observations': [], 'actions': [], 'target': []}
        tms['observations'] = self.replay_buffer.sampled_cur_obs
        tms['actions'] = actions
        tms['target'] = target_q_vals

        return tms

    def train(self):
        for _ in range(self.sampling_iterations):
            self.replay_buffer.clear_cache()
            self.replay_buffer.sample()
            actor_tm_feedback = self.get_actor_update(self.replay_buffer.sampled_actions)
            self.policy.actor.update(actor_tm_feedback)

    def update_epsilon_greedy(self):
        self.epsilon = self.epsilon_min + (self.init_epsilon - self.epsilon_min) * np.exp(
            -self.cur_episode * self.epsilon_decay)

    def learn(self, nr_of_episodes):
        for episode in tqdm(range(nr_of_episodes)):
            if episode % self.config['test_freq'] == 0:
                self.test()
            self.cur_episode = episode + 1
            self.rollout()
            if len(self.replay_buffer.cur_obs) >= self.batch_size:
                self.train()
            self.update_epsilon_greedy()
            if self.best_score < self.threshold and self.cur_episode == 100:
                break

    def test(self):
        episode_rewards = np.array([0 for _ in range(100)])
        for episode, seed in enumerate(self.test_random_seeds):
            obs, _ = self.env.reset(seed=seed)
            while True:
                action, actions = self.policy.get_best_action(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_rewards[episode] += reward
                if terminated or truncated:
                    break

        mean = np.mean(episode_rewards)
        std = np.std(episode_rewards)
        self.save_results(mean, std)
        self.total_score.append(mean)
        if mean > self.best_score:
            self.save_model(True)
            self.best_score = mean
            print(f'New best mean: {mean}!')
        self.scores.append(mean)

    def save_model(self, best_model):
        if self.save:
            if best_model:
                tms = []
                ta_state, clause_sign, clause_count = self.policy.actor.tm.get_params()
                ta_state_save = np.zeros((len(ta_state), len(ta_state[0]), len(ta_state[0][0])), dtype=np.int32)
                clause_sign_save = np.zeros((len(clause_sign), len(clause_sign[0]), len(clause_sign[0][0])),
                                            dtype=np.int32)
                clause_count_save = np.zeros((len(clause_count)), dtype=np.int32)

                for i in range(len(ta_state)):
                    for j in range(len(ta_state[i])):
                        for k in range(len(ta_state[i][j])):
                            ta_state_save[i][j][k] = int(ta_state[i][j][k])
                for i in range(len(clause_sign)):
                    for j in range(len(clause_sign[i])):
                        for k in range(len(clause_sign[i][j])):
                            clause_sign_save[i][j][k] = int(clause_sign[i][j][k])
                for i in range(len(clause_count)):
                    clause_count_save[i] = int(clause_count[i])
                tms.append(
                    {'ta_state': ta_state_save, 'clause_sign': clause_sign_save, 'clause_count': clause_count_save})
                torch.save(tms, os.path.join(self.save_path, 'best'))

    def save_results(self, mean, std):
        if self.save:
            file_name = 'test_results.csv'
            file_exists = os.path.exists(os.path.join(self.save_path, file_name))
            with open(os.path.join(self.save_path, file_name), "a") as file:
                if not file_exists:
                    file.write("mean,std,steps\n")
                file.write(f"{mean},{std},{self.timesteps}\n")

    def make_run_dir(self):
        base_dir = '../results'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        if not os.path.exists(os.path.join(base_dir, self.config['env_name'])):
            os.makedirs(os.path.join(base_dir, self.config['env_name']))
        if not os.path.exists(os.path.join(base_dir, self.config['env_name'], self.config['algorithm'])):
            os.makedirs(os.path.join(base_dir, self.config['env_name'], self.config['algorithm']))
        if not os.path.exists(os.path.join(base_dir, self.config['env_name'], self.config['algorithm'], self.run_id)):
            os.makedirs(os.path.join(base_dir, self.config['env_name'], self.config['algorithm'], self.run_id))
        self.save_path = os.path.join(base_dir, self.config['env_name'], self.config['algorithm'], self.run_id)
