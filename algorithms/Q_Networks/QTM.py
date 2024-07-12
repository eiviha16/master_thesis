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
        self.online_policy = Policy(config)

        self.epsilon = config['epsilon_init']
        self.replay_buffer = ReplayBuffer(config["buffer_size"], config["sample_size"], n=config["n_steps"])
        self.config = config
        self.save_path = ''

        if config['save']:
            self.run_id = 'run_' + str(
                len([i for i in os.listdir(f'../results/{config["env_name"]}/{config["algorithm"]}')]) + 1)
            self.make_run_dir()
            self.save_config()
        else:
            print('Warning SAVING is OFF!')
            self.run_id = "unidentified_run"

        self.nr_of_test_episodes = 100
        self.test_random_seeds = [83811, 14593, 3279, 97197, 36049, 32099, 29257, 18290, 96531, 13435, 88697, 97081,
                                  71483, 11396, 77398, 55303, 4166, 3906, 12281, 28658, 30496, 66238, 78908, 3479,
                                  73564, 26063, 93851, 85182, 91925, 71427, 54988, 28894, 58879, 77237, 36464, 852,
                                  99459, 20927, 91507, 55393, 44598, 36422, 20380, 28222, 44119, 13397, 12157, 49798,
                                  12677, 47053, 45083, 79132, 34672, 5696, 95648, 60218, 70285, 16362, 49616, 10329,
                                  72358, 38428, 82398, 81071, 47401, 75675, 25204, 92350, 9117, 6007, 86674, 29872,
                                  37931, 10459, 30513, 13239, 49824, 36435, 59430, 83321, 47820, 21320, 48521, 46567,
                                  27461, 87842, 34994, 91989, 89594, 84940, 9359, 79841, 83228, 22432, 70011, 95569,
                                  32088, 21418, 60590, 49736]

        self.nr_of_steps = 0
        self.cur_episode = 0
        self.cur_mean = 0
        self.total_score = []
        self.best_score = -float('inf')
        self.announce()

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
        if self.config["save"]:
            with open(f'{self.save_path}/config.yaml', "w") as yaml_file:
                yaml.dump(self.config, yaml_file, default_flow_style=False)

    def get_next_action(self, cur_obs):
        if np.random.random() < self.epsilon:
            q_vals = np.array([np.random.random() for _ in range(self.action_space_size)])
        else:
            q_vals = self.online_policy.predict(cur_obs)

        return np.argmax(q_vals)

    def temporal_difference(self, i, next_q_vals):
        return self.replay_buffer.sampled_rewards[i] + (
                1 - self.replay_buffer.sampled_terminated[i]) * self.config["gamma"] * next_q_vals

    def n_step_temporal_difference(self, i, next_q_vals):
        target_q_vals = []
        target_q_val = 0
        for j in range(len(self.replay_buffer.sampled_rewards[i])):
            target_q_val += (self.config["gamma"] ** j) * self.replay_buffer.sampled_rewards[i][j]
            if self.replay_buffer.sampled_terminated[i][j] or self.replay_buffer.sampled_trunc[i][j]:
                break
            target_q_val += (1 - self.replay_buffer.sampled_terminated[i][j]) * (self.config["gamma"] ** j) * \
                            next_q_vals[0]
        target_q_vals.append(target_q_val)
        return target_q_vals

    def update_epsilon_greedy(self):
        self.epsilon = self.config["epsilon_min"] + (self.config["epsilon_init"] - self.config["epsilon_min"]) * np.exp(
            -self.cur_episode * self.config["epsilon_decay"])

    def get_q_val_and_obs_for_tm(self, action, target_q_vals, cur_obs):
        tm_inputs = [{'observations': [], 'target_q_vals': []} for _ in range(self.action_space_size)]
        tm_inputs[action]['observations'].append(cur_obs)
        tm_inputs[action]['target_q_vals'].append(target_q_vals)

        return tm_inputs

    def train_n_step(self):
        pass

    def train(self):
        pass

    def rollout(self):
        cur_obs, _ = self.env.reset(seed=random.randint(1, 100))
        while True:
            action = self.get_next_action(cur_obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.replay_buffer.save_experience(action, cur_obs, next_obs, reward, int(terminated), truncated)
            cur_obs = next_obs
            self.nr_of_steps += 1

            if self.nr_of_steps >= self.config["sample_size"] + self.config["n_steps"] and self.nr_of_steps % self.config["train_freq"] == 0:
                if self.config['n_steps'] > 1:
                    self.train_n_step()
                else:
                    self.train()

            if terminated or truncated:
                break

    def learn(self, nr_of_episodes):
        for episode in tqdm(range(nr_of_episodes)):
            self.cur_episode = episode + 1
            if episode % self.config["test_freq"] == 0:
                self.test(self.nr_of_steps)
            # if self.best_score < self.config["threshold"] and self.cur_episode == 100:
            #    break
            self.rollout()
            self.update_epsilon_greedy()

    def test(self, nr_of_steps):
        episode_rewards = np.array([0 for _ in range(self.nr_of_test_episodes)])
        for episode in range(self.nr_of_test_episodes):
            obs, _ = self.env.reset(seed=self.test_random_seeds[episode])
            while True:
                q_vals = self.online_policy.predict(obs)
                action = np.argmax(q_vals)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_rewards[episode] += reward
                if terminated or truncated:
                    break

        mean = np.mean(episode_rewards)
        std = np.std(episode_rewards)
        self.total_score.append(mean)
        self.cur_mean = mean

        # save
        self.save_results(mean, std, nr_of_steps)
        if mean > self.best_score:
            self.save_model()
            self.best_score = mean
            print(f'New best mean after {nr_of_steps} steps: {mean}!')

    def save_model(self):
        if self.config["save"]:
            tms = self.online_policy.tms
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
                tms_save.append(
                    {'ta_state': ta_state_save, 'clause_sign': clause_sign_save, 'clause_output': clause_output_save,
                     'feedback_to_clauses': feedback_to_clauses_save})
            torch.save(tms_save, os.path.join(self.save_path, 'best'))

    def save_results(self, mean, std, nr_of_steps):
        if self.config["save"]:
            file_name = 'test_results.csv'
            file_exists = os.path.exists(os.path.join(self.save_path, file_name))

            with open(os.path.join(self.save_path, file_name), "a") as file:
                if not file_exists:
                    file.write("mean,std, steps\n")
                file.write(f"{mean},{std},{nr_of_steps}\n")


class SingleQTM(QTM):
    def __init__(self, env, Policy, config):
        super().__init__(env, Policy, config)
        self.online_policy = Policy(config)

    def train_n_step(self):
        self.replay_buffer.sample_n_seq()
        for i in range(self.config["sample_size"]):
            sampled_next_obs = np.array(self.replay_buffer.sampled_next_obs[i])

            next_q_vals = self.online_policy.predict(sampled_next_obs[-1, :])  # next_obs?
            next_q_vals = np.max(next_q_vals, axis=1)

            target_q_vals = self.n_step_temporal_difference(i, next_q_vals)
            tm_inputs = self.get_q_val_and_obs_for_tm(self.replay_buffer.sampled_actions[i][0], target_q_vals[0],
                                                      self.replay_buffer.sampled_cur_obs[i][0])
            _ = self.online_policy.update(tm_inputs)

    def train(self):
        self.replay_buffer.sample()
        for i in range(self.config["sample_size"]):
            next_q_vals = self.online_policy.predict(np.array(self.replay_buffer.sampled_next_obs[i]))
            next_q_vals = np.max(next_q_vals, axis=1)
            target_q_vals = self.temporal_difference(i, next_q_vals[0])

            tm_inputs = self.get_q_val_and_obs_for_tm(self.replay_buffer.sampled_actions[i], target_q_vals,
                                                      self.replay_buffer.sampled_cur_obs[i])
            _ = self.online_policy.update(tm_inputs)


class DoubleQTM(QTM):
    def __init__(self, env, Policy, config):
        super().__init__(env, Policy, config)
        self.online_policy = Policy(config)
        self.target_policy = Policy(config)

    def train_n_step(self):
        self.replay_buffer.sample_n_seq()
        for i in range(self.config["sample_size"]):
            sampled_next_obs = np.array(self.replay_buffer.sampled_next_obs[i])

            action = np.argmax(self.online_policy.predict(np.array(sampled_next_obs[-1, :])), axis=1)  # next_obs?
            next_q_vals = self.target_policy.predict(sampled_next_obs[-1, :])
            next_q_vals = next_q_vals[0][action]

            target_q_vals = self.n_step_temporal_difference(i, next_q_vals)
            tm_inputs = self.get_q_val_and_obs_for_tm(self.replay_buffer.sampled_actions[i][0], target_q_vals[0],
                                                      self.replay_buffer.sampled_cur_obs[i][0])
            _ = self.online_policy.update(tm_inputs)
        self.soft_update()

    def train(self):
        self.replay_buffer.sample()
        for i in range(self.config["sample_size"]):
            action = np.argmax(self.online_policy.predict(np.array(self.replay_buffer.sampled_next_obs[i])), axis=1)

            next_q_vals = self.target_policy.predict(np.array(self.replay_buffer.sampled_next_obs[i]))
            next_q_vals = next_q_vals[0][action]
            target_q_vals = self.temporal_difference(i, next_q_vals[0])

            tm_inputs = self.get_q_val_and_obs_for_tm(self.replay_buffer.sampled_actions[i], target_q_vals,
                                                      self.replay_buffer.sampled_cur_obs[i])
            _ = self.online_policy.update(tm_inputs)
        self.soft_update()

    def soft_update(self):
        if self.config['soft_update_type'] == 'soft_update_a':
            for i in range(len(self.online_policy.tms)):
                self.soft_update_a(self.online_policy.tms[i], self.target_policy.tms[i])
        elif self.config['soft_update_type'] == 'soft_update_b':
            for i in range(len(self.online_policy.tms)):
                self.soft_update_b(self.online_policy.tms[i], self.target_policy.tms[i])

    def soft_update_a(self, online_tm, target_tm):
        online_ta_state, online_clause_sign, online_clause_output, online_feedback_to_clauses = online_tm.get_params()
        target_ta_state, target_clause_sign, target_clause_output, target_feedback_to_clauses = target_tm.get_params()
        nr_of_clauses = len(list(online_clause_sign))
        clauses_to_update = random.sample(range(nr_of_clauses), int(nr_of_clauses * self.config['clause_update_p']))
        for clause in clauses_to_update:
            target_clause_sign[clause] = online_clause_sign[clause]
            target_clause_output[clause] = online_clause_output[clause]
            target_feedback_to_clauses[clause] = online_feedback_to_clauses[clause]
            for i in range(len(online_ta_state[clause])):
                for j in range(len(online_ta_state[clause][i])):
                    target_ta_state[clause][i][j] = online_ta_state[clause][i][j]

        target_tm.set_params(target_ta_state, target_clause_sign, target_clause_output, target_feedback_to_clauses)

    def soft_update_b(self, online_tm, target_tm):
        if self.cur_episode % self.config['update_freq'] == 0:
            online_ta_state, _, _, _ = online_tm.get_params()
            target_ta_state, target_clause_sign, target_clause_output, target_feedback_to_clauses = target_tm.get_params()
            for i in range(len(online_ta_state)):
                for j in range(len(online_ta_state[i])):
                    for k in range(len(online_ta_state[i][j])):
                        if online_ta_state[i][j][k] > target_ta_state[i][j][k]:
                            target_ta_state[i][j][k] += 1
                        if online_ta_state[i][j][k] < target_ta_state[i][j][k]:
                            target_ta_state[i][j][k] -= 1
            target_tm.set_params(target_ta_state, target_clause_sign, target_clause_output, target_feedback_to_clauses)
