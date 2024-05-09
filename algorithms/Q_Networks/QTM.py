import numpy as np
import torch
import os
import yaml
from tqdm import tqdm
import random

from algorithms.misc.replay_buffer import ReplayBuffer


class QTM:
    def __init__(self, env, Policy, config):
        self.env = env
        self.action_space_size = env.action_space.n
        self.obs_space_size = env.observation_space.shape[0]
        config['action_space_size'] = self.action_space_size
        config['obs_space_size'] = self.obs_space_size

        self.policy = Policy(config)

        self.gamma = config['gamma']
        self.init_epsilon = config['epsilon_init']
        self.epsilon = self.init_epsilon
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_min = 0


        self.sampling_iterations = config['sampling_iterations']
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']

        self.y_max = config['y_max']
        self.y_min = config['y_min']

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.test_freq = config['test_freq']
        self.nr_of_test_episodes = 100
        if config['save']:
            self.run_id = 'run_' + str(len([i for i in os.listdir(f'../results/{config["env_name"]}/{config["algorithm"]}')]) + 1)
        else:
            print('Warning SAVING is OFF!')
            self.run_id = "unidentified_run"

        self.test_random_seeds = [83811, 14593, 3279, 97197, 36049, 32099, 29257, 18290, 96531, 13435, 88697, 97081, 71483, 11396, 77398, 55303, 4166, 3906, 12281, 28658, 30496, 66238, 78908, 3479, 73564, 26063, 93851, 85182, 91925, 71427, 54988, 28894, 58879, 77237, 36464, 852, 99459, 20927, 91507, 55393, 44598, 36422, 20380, 28222, 44119, 13397, 12157, 49798, 12677, 47053, 45083, 79132, 34672, 5696, 95648, 60218, 70285, 16362, 49616, 10329, 72358, 38428, 82398, 81071, 47401, 75675, 25204, 92350, 9117, 6007, 86674, 29872, 37931, 10459, 30513, 13239, 49824, 36435, 59430, 83321, 47820, 21320, 48521, 46567, 27461, 87842, 34994, 91989, 89594, 84940, 9359, 79841, 83228, 22432, 70011, 95569, 32088, 21418, 60590, 49736]

        self.save = config['save']
        self.best_scores = {'mean': -float('inf'), 'std': float('inf')}
        self.cur_mean = 0
        self.config = config
        self.save_path = ''
        if self.save:
            self.make_run_dir()
            self.save_config()
        self.announce()
        self.q_values = {f'q{i}': [] for i in range(self.action_space_size)}
        self.nr_actions = 0
        self.cur_episode = 0
        self.abs_errors = {}
        self.total_score = []
        self.nr_of_steps = 0
        self.threshold = config['threshold']


    def announce(self):
        print(f'{self.run_id} has been initialized!')
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

    def save_config(self):
        if self.save:
            with open(f'{self.save_path}/config.yaml', "w") as yaml_file:
                yaml.dump(self.config, yaml_file, default_flow_style=False)

    def get_next_action(self, cur_obs):
        if np.random.random() < self.epsilon:
            q_vals = np.array([np.random.random() for _ in range(self.action_space_size)])
        else:
            q_vals = self.policy.predict(cur_obs)

        return np.argmax(q_vals), q_vals

    def temporal_difference(self, next_q_vals):
        return np.array(self.replay_buffer.sampled_rewards) + (
                1 - np.array(self.replay_buffer.sampled_dones)) * self.gamma * next_q_vals

    def update_epsilon_greedy(self):
        self.epsilon = self.epsilon_min + (self.init_epsilon - self.epsilon_min) * np.exp(-self.cur_episode * self.epsilon_decay)


    def get_q_val_and_obs_for_tm(self, target_q_vals):

        tm_inputs = [{'observations': [], 'target_q_vals': []} for _ in range(self.action_space_size)]
        actions = self.replay_buffer.sampled_actions
        for index, action in enumerate(actions):
            tm_inputs[action]['observations'].append(self.replay_buffer.sampled_cur_obs[index])
            tm_inputs[action]['target_q_vals'].append(target_q_vals[index])

        return tm_inputs

    def train(self):
        for _ in range(self.sampling_iterations):
            self.replay_buffer.clear_cache()
            self.replay_buffer.sample()
            next_q_vals = self.policy.predict(np.array(self.replay_buffer.sampled_next_obs))  # next_obs?

            next_q_vals = np.max(next_q_vals, axis=1)

            # calculate target q vals
            target_q_vals = self.temporal_difference((next_q_vals))

            tm_inputs = self.get_q_val_and_obs_for_tm(target_q_vals)
            abs_errors = self.policy.update(tm_inputs)

            for key in abs_errors:
                if key not in self.abs_errors:
                    self.abs_errors[key] = []
                for val in abs_errors[key]:
                    self.abs_errors[key].append(val)

            self.abs_errors = {}


    def rollout(self):
        cur_obs, _ = self.env.reset(seed=random.randint(1, 100))
        while True:
            action, _ = self.get_next_action(cur_obs)
            next_obs, reward, done, truncated, _ = self.env.step(action)

            self.replay_buffer.save_experience(action, cur_obs, next_obs, reward, int(done), self.nr_of_steps)
            cur_obs = next_obs
            self.nr_of_steps += 1

            if done or truncated:
                break

    def learn(self, nr_of_episodes):
        for episode in tqdm(range(nr_of_episodes)):
            self.cur_episode = episode
            if episode % self.test_freq == 0:
                self.test(self.nr_of_steps)
            if self.best_scores['mean'] < self.threshold and self.cur_episode == 100:
                break
            self.rollout()
            if self.nr_of_steps >= self.batch_size:
                self.train()
            self.update_epsilon_greedy()


    def test(self, nr_of_steps):

        epsilon = self.epsilon
        self.epsilon = 0
        episode_rewards = np.array([0 for _ in range(self.nr_of_test_episodes)])

        for episode in range(self.nr_of_test_episodes):
            obs, _ = self.env.reset(seed=self.test_random_seeds[episode])
            while True:
                action, q_vals_ = self.get_next_action(obs)
                obs, reward, done, truncated, _ = self.env.step(action)
                episode_rewards[episode] += reward
                if done or truncated:
                    break

        mean = np.mean(episode_rewards)
        std = np.std(episode_rewards)
        self.total_score.append(mean)
        self.cur_mean = mean
        self.save_results(mean, std, nr_of_steps)
        self.epsilon = epsilon
        if mean > self.best_scores['mean']:
            self.save_model(True)
            self.best_scores['mean'] = mean
            print(f'New best mean after {nr_of_steps} steps: {mean}!')
        self.save_model(False)

    def save_model(self, best_model):
        if self.save:
            if best_model:
                tms = self.policy.tms
                tms_save = []
                for tm in range(len(tms)):
                    ta_state, clause_sign, clause_output, feedback_to_clauses = tms[tm].get_params()
                    ta_state_save = np.zeros((len(ta_state), len(ta_state[0]), len(ta_state[0][0])), dtype=np.int32)
                    clause_sign_save = np.zeros((len(clause_sign)), dtype=np.int32)
                    clause_output_save = np.zeros((len(clause_output)), dtype=np.int32)
                    feedback_to_clauses_save = np.zeros((len(feedback_to_clauses)), dtype=np.int32)

                    for i in range(len(ta_state)):
                        for j in range(len(ta_state[i])):
                            for k in range(len(ta_state[i][j])):
                                ta_state_save[i][j][k] = int(ta_state[i][j][k])
                        clause_sign_save[i] = int(clause_sign[i])
                        clause_output_save[i] = int(clause_output[i])
                        feedback_to_clauses_save[i] = int(feedback_to_clauses[i])
                    tms_save.append({'ta_state': ta_state_save, 'clause_sign': clause_sign_save, 'clause_output': clause_output_save, 'feedback_to_clauses': feedback_to_clauses_save})
                torch.save(tms_save, os.path.join(self.save_path, 'best'))

    def save_results(self, mean, std, nr_of_steps):
        if self.save:
            file_name = 'test_results.csv'
            file_exists = os.path.exists(os.path.join(self.save_path, file_name))

            with open(os.path.join(self.save_path, file_name), "a") as file:
                if not file_exists:
                    file.write("mean,std, steps\n")
                file.write(f"{mean},{std},{nr_of_steps}\n")

    def save_q_vals(self, q_vals):
        if self.save:
            folder_name = 'q_values'
            file_name = f'{self.cur_episode}.csv'
            if not os.path.exists(os.path.join(self.save_path, folder_name)):
                os.makedirs(os.path.join(self.save_path, folder_name))
            file_exists = os.path.exists(os.path.join(self.save_path, folder_name, file_name))

            with open(os.path.join(self.save_path, folder_name, file_name), "a") as file:
                if not file_exists:
                    file.write(f"{'actor_' + str(i) for i in range(len(q_vals))}\n")
                file.write(f"{','.join(map(str, q_vals))}\n")

    def save_abs_errors(self):
        if self.save:
            for key in self.abs_errors:
                self.abs_errors[key] = np.array(self.abs_errors[key])
            folder_name = 'absolute_errors.csv'
            file_exists = os.path.exists(os.path.join(self.save_path, folder_name))

            with open(os.path.join(self.save_path, folder_name), "a") as file:
                if not file_exists:
                    file.write('actor1_mean,actor1_std,actor2_mean,actor2_std\n')
                file.write(f"{np.mean(self.abs_errors['actor1'])},{np.std(self.abs_errors['actor1'])},{np.mean(self.abs_errors['actor2'])},{np.std(self.abs_errors['actor2'])}\n")

    def save_feedback_data(self, feedback_1, feedback_2, nr_of_steps):
        if self.save:

            file_name = 'feedback.csv'
            file_exists = os.path.exists(os.path.join(self.save_path, file_name))

            with open(os.path.join(self.save_path, file_name), "a") as file:
                if not file_exists:
                    file.write("1_typeI, 1_typeII, 2_typeI, 2_typeII, steps\n")
                file.write(f"{feedback_1[0]},{feedback_1[1]},{feedback_2[0]},{feedback_2[1]},{nr_of_steps}\n")
