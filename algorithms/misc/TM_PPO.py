import numpy as np
import os
import yaml
import random
from algorithms.misc.batch_buffer import Batch
import torch
import torch.nn.functional as F
from tqdm import tqdm


class PPO:
    def __init__(self, env, Policy, config):
        self.env = env
        self.action_space_size = env.action_space.n.size
        self.obs_space_size = env.observation_space.shape[0]

        self.policy = Policy(config)
        self.batch = Batch()

        self.gamma = config['gamma']
        self.lam = config['lam']
        self.clip = config["clip"]
        self.epochs = config['epochs']

        #self.test_random_seeds = [random.randint(1, 100000) for _ in range(100)]
        self.test_random_seeds = [83811, 14593, 3279, 97197, 36049, 32099, 29257, 18290, 96531, 13435, 88697, 97081, 71483, 11396, 77398, 55303, 4166, 3906, 12281, 28658, 30496, 66238, 78908, 3479, 73564, 26063, 93851, 85182, 91925, 71427, 54988, 28894, 58879, 77237, 36464, 852, 99459, 20927, 91507, 55393, 44598, 36422, 20380, 28222, 44119, 13397, 12157, 49798, 12677, 47053, 45083, 79132, 34672, 5696, 95648, 60218, 70285, 16362, 49616, 10329, 72358, 38428, 82398, 81071, 47401, 75675, 25204, 92350, 9117, 6007, 86674, 29872, 37931, 10459, 30513, 13239, 49824, 36435, 59430, 83321, 47820, 21320, 48521, 46567, 27461, 87842, 34994, 91989, 89594, 84940, 9359, 79841, 83228, 22432, 70011, 95569, 32088, 21418, 60590, 49736]
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

    def calculate_advantage(self):
        advantage = 0
        next_value = 0
        for i in reversed(range(len(self.batch.actions))):
            dt = self.batch.rewards[i] + self.gamma * next_value - self.batch.values[i][0][0]
            advantage = dt + self.gamma * self.lam * advantage * int(not self.batch.dones[i])
            next_value = self.batch.values[i][0][0]

            self.batch.advantages.insert(0, advantage)

    def normalize_advantages(self):
        advantages = np.array(self.batch.advantages)
        norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        self.batch.advantages = norm_advantages

    def rollout(self):
        obs, _ = self.env.reset(seed=42)
        while True:
            action, value, log_prob, entropy = self.policy.get_action(obs)
            obs, reward, done, truncated, _ = self.env.step(action[0])
            self.batch.save_experience(action[0], log_prob[0], value, obs, reward, done, entropy)
            if done or truncated:
                break
        self.batch.convert_to_numpy()

    def evaluate_actions(self):
        actions, values, log_probs = self.policy.get_action(self.batch.obs)
        return actions, values, log_probs


    def get_update_data_actor(self):
        tm = [{'observations': [], 'target': [], 'advantages': [], 'entropy': []},
              {'observations': [], 'target': [], 'advantages': [], 'entropy': []}]
        for i in range(len(self.batch.actions)):
            idx = self.batch.actions[i]
            tm[idx]['observations'].append(self.batch.obs[i])
            tm[idx]['target'].append(self.batch.action_log_prob[i][idx])
            tm[idx]['advantages'].append(self.batch.advantages[i])
            tm[idx]['entropy'].append(self.batch.entropies[i][idx])
            """            if idx == 1:
                tm[0]['observations'].append(self.batch.obs[i])
                tm[0]['target'].append(self.batch.action_log_prob[i][idx])
                tm[0]['advantages'].append(-self.batch.advantages[i])
                tm[0]['entropy'].append(self.batch.entropies[i][idx])
            else:
                tm[1]['observations'].append(self.batch.obs[i])
                tm[1]['target'].append(self.batch.action_log_prob[i][idx])
                tm[1]['advantages'].append(-self.batch.advantages[i])
                tm[1]['entropy'].append(self.batch.entropies[i][idx])"""
        return tm

    def get_update_data_critic(self):
        tm = [{'observations': [], 'target': []}, {'observations': [], 'target': []}]
        for i in range(len(self.batch.actions)):
            idx = self.batch.actions[i]
            tm[idx]['observations'].append(self.batch.obs[i])
            tm[idx]['target'].append(self.batch.advantages[i] + self.batch.values[i, 0, 0])
            #tm[idx]['target'].append(self.batch.rewards[i])

        #print(self.batch.advantages[0] + self.batch.values[0, 0, 0])
            #tm[idx]['target'].append(self.batch.rewards[i])
            # return F.mse_loss(torch.from_numpy(self.batch.sampled_advantages - self.batch.sampled_values).to(dtype=torch.float32), values)

        return tm

    def train(self):
        for _ in range(self.epochs):
            actor_update = self.get_update_data_actor()
            self.policy.actor.update_2(actor_update, self.clip)

            critic_update = self.get_update_data_critic()
            self.policy.critic.update(critic_update)
    def learn(self, nr_of_episodes):
        for episode in tqdm(range(nr_of_episodes)):
            if self.best_score < 12 and episode > 50:
                break
            self.cur_episode = episode
            self.rollout()
            self.calculate_advantage()
            self.normalize_advantages()

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
            self.save_model(True)
            self.best_score = mean
            print(f'New best mean: {mean}!')
        self.save_model(False)

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