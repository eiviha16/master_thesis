import numpy as np
import os
import yaml
from algorithms.misc.batch_buffer import Batch_TM_VPG as Batch
import torch
from tqdm import tqdm
import random


class VPG:
    def __init__(self, env, Policy, config):
        self.env = env
        self.action_space_size = env.action_space.n.size
        self.obs_space_size = env.observation_space.shape[0]
        self.gamma = config['gamma']
        self.policy = Policy(config)
        self.batch = Batch(config['batch_size'])
        # self.exploration_prob = config['exploration_prob_init']
        # self.exploration_prob_decay = config['exploration_prob_decay']

        self.epochs = config['epochs']
        self.config = config

        # self.test_random_seeds = [random.randint(1, 100000) for _ in range(100)]
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
        self.run_id = 'run_' + str(len([i for i in os.listdir(f"./results/{config['algorithm']}")]) + 1)
        self.make_run_dir(config['algorithm'])

        self.best_score = float('-inf')
        self.save_config(config)
        self.announce()
        self.cur_episode = 0

    def announce(self):
        print(f'{self.run_id} has been initialized!')

    def save_config(self, config):
        with open(f'{self.save_path}/config.yaml', "w") as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)

    def normalize_discounted_rewards(self):
        dr = np.array(self.batch.discounted_rewards)
        norm_dr = (dr - dr.mean()) / (dr.std() + 1e-10)
        self.batch.discounted_rewards = norm_dr

    def calculate_discounted_rewards(self):
        discounted_reward = 0
        for i in reversed(range(len(self.batch.actions))):
            discounted_reward = self.batch.rewards[i] + self.gamma * discounted_reward
            self.batch.discounted_rewards.insert(0, discounted_reward)

    def rollout(self):
        cur_obs, _ = self.env.reset(seed=42)
        while True:
            cur_action, prob, q_vals = self.policy.get_action(cur_obs)
            next_obs, reward, done, truncated, _ = self.env.step(cur_action)
            self.batch.save_experience(cur_action, prob, q_vals, cur_obs, next_obs, reward, done)
            if done or truncated:
                break
            cur_obs = next_obs
    #u
    def temporal_difference(self, next_q_vals):
        return np.array(self.batch.sampled_rewards) + (
                1 - np.array(self.batch.sampled_dones)) * self.gamma * next_q_vals

    def get_actor_update(self, actions, target_q_vals):

        tm = [{'observations': [], 'feedback': [], 'feedback_probs': []},
              {'observations': [], 'feedback': [], 'feedback_probs': []}]
        q_vals = self.policy.target_critic.predict_q_vals(np.array(self.batch.sampled_cur_obs))

        for index, action in enumerate(actions):
            #predict Q-vals by obs-sample
            feedback = 1 if q_vals[index][action] > target_q_vals[index] else 2
            feedback_prob = 1.0 * (abs(q_vals[index][action] - target_q_vals[index]) / (self.config['critic']['y_max'] - self.config['critic']['y_min']))
            tm[action]['feedback'].append(feedback)
            tm[action]['feedback_probs'].append(feedback_prob)
            tm[action]['observations'].append(self.batch.sampled_cur_obs[index])
        return tm

    def get_q_val_for_action(self, actions, q_values):
        q_vals = []
        for index, action in enumerate(actions):
            q_vals.append(q_values[index][action])
        return np.array(q_vals)

    def get_q_val_and_obs_for_tm(self, actions, target_q_vals):

        tms = [{'observations': [], 'target': []}, {'observations': [], 'target': []}]
        # actions = self.replay_buffer.sampled_actions
        for index, action in enumerate(actions):
            tms[action]['observations'].append(self.batch.sampled_cur_obs[index])
            tms[action]['target'].append(target_q_vals[index])

        return tms

    def train(self):
        for epoch in range(self.epochs):
            self.batch.sample()
            next_q_vals = self.policy.evaluation_critic.predict_q_vals(
                np.array(self.batch.sampled_next_obs))  # next_obs?
            next_q_vals = self.get_q_val_for_action(self.batch.sampled_actions, next_q_vals)

            # calculate target q vals
            target_q_vals = self.temporal_difference(next_q_vals)
            critic_update = self.get_q_val_and_obs_for_tm(self.batch.sampled_actions, target_q_vals)

            #self.policy.target_critic.update_q_vals([{'observations': self.batch.sampled_cur_obs,
            #                                          'actions': self.batch.sampled_actions, 'target': target_q_vals}])

            actor_tm_feedback = self.get_actor_update(self.batch.sampled_actions, target_q_vals)
            self.policy.actor.update_3(actor_tm_feedback)
            self.policy.target_critic.update(critic_update)

        if self.config['soft_update_type'] == 'soft_update_1':
            self.soft_update_1(self.policy.target_critic.tms[0], self.policy.evaluation_critic.tms[0])
            self.soft_update_1(self.policy.target_critic.tms[1], self.policy.evaluation_critic.tms[1])
        else:
            self.soft_update_2(self.policy.target_critic.tms[0], self.policy.evaluation_critic.tms[0])
            self.soft_update_2(self.policy.target_critic.tms[1], self.policy.evaluation_critic.tms[1])

    def learn(self, nr_of_episodes):
        for episode in tqdm(range(nr_of_episodes)):
            self.cur_episode = episode
            self.rollout()
            # self.calculate_discounted_rewards()
            # self.normalize_discounted_rewards()
            self.batch.convert_to_numpy()

            self.train()
            self.test()

            self.batch.clear()

    def test(self):
        # remember to remove exploration when doing this
        episode_rewards = np.array([0 for _ in range(100)])
        for episode, seed in enumerate(self.test_random_seeds):
            obs, _ = self.env.reset(seed=seed)
            while True:
                action, probs = self.policy.get_best_action(obs)
                obs, reward, done, truncated, _ = self.env.step(action[0])
                episode_rewards[episode] += reward
                if done or truncated:
                    break
                if episode == 1:
                    self.save_probs(probs)

        mean = np.mean(episode_rewards)
        std = np.std(episode_rewards)
        self.save_results(mean, std)

        if mean > self.best_score:
            self.save_model('best_model')
            self.best_score = mean
            #print(f'New best mean: {mean}!')
        self.save_model('last_model')

    def save_model(self, best_model):
        if best_model:
            self.policy.actor.tms[0].save_state()
            self.policy.actor.tms[1].save_state()
            tms = []
            for tm in range(len(self.policy.actor.tms)):
                ta_state, clause_sign, clause_output, feedback_to_clauses = self.policy.actor.tms[tm].get_params()
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
                tms.append({'ta_state': ta_state_save, 'clause_sign': clause_sign_save, 'clause_output': clause_output_save, 'feedback_to_clauses': feedback_to_clauses_save})
            torch.save(tms, os.path.join(self.save_path, 'best'))

        else:
            pass

    def save_results(self, mean, std):
        file_name = 'test_results.csv'
        file_exists = os.path.exists(os.path.join(self.save_path, file_name))

        with open(os.path.join(self.save_path, file_name), "a") as file:
            if not file_exists:
                file.write("mean,std\n")
            file.write(f"{mean},{std}\n")

    def soft_update_2(self, target_tm, evaluation_tm):
        if self.cur_episode % self.config['update_freq'] == 0:
            target_ta_state, target_clause_sign, target_clause_output, target_feedback_to_clauses = target_tm.get_params()
            eval_ta_state, eval_clause_sign, eval_clause_output, eval_feedback_to_clauses = evaluation_tm.get_params()
            for i in range(len(target_ta_state)):
                for j in range(len(target_ta_state[i])):
                    for k in range(len(target_ta_state[i][j])):
                        if target_ta_state[i][j][k] > eval_ta_state[i][j][k]:
                            eval_ta_state[i][j][k] += 1
                        if target_ta_state[i][j][k] < eval_ta_state[i][j][k]:
                            eval_ta_state[i][j][k] -= 1
            evaluation_tm.set_params(eval_ta_state, eval_clause_sign, eval_clause_output, eval_feedback_to_clauses)

    def soft_update_1(self, target_tm, evaluation_tm):
        target_ta_state, target_clause_sign, target_clause_output, target_feedback_to_clauses = target_tm.get_params()
        eval_ta_state, eval_clause_sign, eval_clause_output, eval_feedback_to_clauses = evaluation_tm.get_params()
        nr_of_clauses = len(list(target_clause_sign))
        clauses_to_update = random.sample(range(nr_of_clauses), int(nr_of_clauses * self.update_grad))
        for clause in clauses_to_update:
            eval_clause_sign[clause] = target_clause_sign[clause]
            eval_clause_output[clause] = target_clause_output[clause]
            eval_feedback_to_clauses[clause] = target_feedback_to_clauses[clause]
            for i in range(len(target_ta_state[clause])):
                for j in range(len(target_ta_state[clause][i])):
                    eval_ta_state[clause][i][j] = target_ta_state[clause][i][j]

        evaluation_tm.set_params(eval_ta_state, eval_clause_sign, eval_clause_output, eval_feedback_to_clauses)

    def make_run_dir(self, algorithm):
        base_dir = './results'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        if not os.path.exists(os.path.join(base_dir, algorithm)):
            os.makedirs(os.path.join(base_dir, algorithm))
        if not os.path.exists(os.path.join(base_dir, algorithm, self.run_id)):
            os.makedirs(os.path.join(base_dir, algorithm, self.run_id))
        self.save_path = os.path.join(base_dir, algorithm, self.run_id)
    def save_probs(self, probs):
        folder_name = 'action_probabilities'
        file_name = f'{self.cur_episode}.csv'
        if not os.path.exists(os.path.join(self.save_path, folder_name)):
            os.makedirs(os.path.join(self.save_path, folder_name))
        file_exists = os.path.exists(os.path.join(self.save_path, folder_name, file_name))

        with open(os.path.join(self.save_path, folder_name, file_name), "a") as file:
            if not file_exists:
                file.write("actor_1,actor_2\n")
            file.write(f"{probs[0][0]}, {probs[0][1]}\n")