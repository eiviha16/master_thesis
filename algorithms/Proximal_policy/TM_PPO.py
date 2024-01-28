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
        self.clip_range = config['clip_range']
        self.epochs = config['epochs']

        self.test_random_seeds = [random.randint(1, 100000) for _ in range(100)]

        self.test_seeds = np.random
        self.save = config['save']
        self.save_path = ''
        self.run_id = 'run_' + str(len([i for i in os.listdir(f"./results/{config['algorithm']}")]) + 1)
        self.make_run_dir(config['algorithm'])

        self.best_score = float('-inf')
        self.save_config(config)
        self.announce()

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
            action, value, log_prob = self.policy.get_action(obs)
            obs, reward, done, truncated, _ = self.env.step(action[0])
            self.batch.save_experience(action[0], log_prob[0], value, obs, reward, done)
            if done or truncated:
                break
        self.batch.convert_to_numpy()

    def evaluate_actions(self):
        actions, values, log_probs = self.policy.get_action(self.batch.obs)
        return actions, values, log_probs

    def calculate_actor_loss(self, log_prob):
        #im finding the difference between new log_prob and old log_prob, advantage is used to determine if the action was better than the other (is it negative and positive?
        #I have to find a way to remove log_prob from the equation and use that later, maybe I add a function in the TM that updates based on different values than it normally predicts
        # Advantage is used to get the magnitude of how good an action is.
        #if the advantage is high 1.8 it will likely have a higher update
        #what i want to do is push the predicted probability in the direction determined by the advantage.
        #if it was a poor action i want the probability to go down, if it was a good action i want the probability to go up.
        #multiply the predicted prob with the advantage, and thats it?
        actor_loss = torch.from_numpy(self.batch.advantages) * self.batch.action_log_prob
        actor_loss = - actor_loss.mean()
        return actor_loss

    def calculate_critic_loss(self, values):
        return F.mse_loss(torch.from_numpy(self.batch.rewards).to(dtype=torch.float32), values.squeeze(-1))
    def get_update_data_actor(self):
        tm = [{'observations': [], 'target': [], 'advantages': []}, {'observations': [], 'target': [], 'advantages': []}]
        for i in range(len(self.batch.actions)):
            idx = self.batch.actions[i]
            tm[idx]['observations'].append(self.batch.obs[i])
            tm[idx]['target'].append(self.batch.action_log_prob[i][idx])
            tm[idx]['advantages'].append(self.batch.advantages[i])
        return tm
    def get_update_data_critic(self):
        tm = [{'observations': [], 'target': []}, {'observations': [], 'target': []}]
        for i in range(len(self.batch.actions)):
            idx = self.batch.actions[i]
            tm[idx]['observations'].append(self.batch.obs[i])
            tm[idx]['target'].append(self.batch.rewards[i])
        return tm
    def train(self):
        for _ in range(self.epochs):

            actor_update = self.get_update_data_actor()
            self.policy.actor.update_2(actor_update)
            #self.policy.actor.tm1.fit_2(actor_update[0]['x'], actor_update[0]['y'], actor_update[0]['advantages'])
            #self.policy.actor.tm2.fit_2(actor_update[1]['x'], actor_update[1]['y'], actor_update[1]['advantages'])

            critic_update = self.get_update_data_critic()
            self.policy.critic.update(critic_update)
            #self.policy.critic.tm1.fit(critic_update[0]['x'], critic_update[0]['y'])
            #self.policy.critic.tm2.fit(critic_update[1]['x'], critic_update[1]['y'])

    def learn(self, nr_of_episodes):
        for episode in tqdm(range(nr_of_episodes)):
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
                action = self.policy.get_best_action(obs)
                obs, reward, done, truncated, _ = self.env.step(action[0])
                episode_rewards[episode] += reward
                if done or truncated:
                    break

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
