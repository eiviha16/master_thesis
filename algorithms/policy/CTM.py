from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
import numpy as np
import pyximport;

pyximport.install(setup_args={
    "include_dirs": np.get_include()},
    reload_support=True)

import TM_lib.rtm as RTM
import TM_lib.mtm as MTM

import numpy as np
import random
import torch

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


class MTMS:
    def __init__(self, config):
        self.tms = []
        self.tm = MTM.MultiClassTsetlinMachine(number_of_classes=config['action_space_size'],
                                               number_of_clauses=config['actor']['nr_of_clauses'],
                                               number_of_features=config['actor']['bits_per_feature'] * config[
                                                   'obs_space_size'],
                                               s=config['actor']['s'],
                                               number_of_states=config['actor']['number_of_state_bits_ta'],
                                               threshold=config['actor']['T'])

        self.vals = np.loadtxt(f'../algorithms/misc/{config["dataset_file_name"]}.txt', delimiter=',').astype(
            dtype=np.float32)
        self.config = config

        self.binarizer = StandardBinarizer(max_bits_per_feature=config['actor']['bits_per_feature'])
        self.init_binarizer()

    def init_binarizer(self):
        self.binarizer.fit(self.vals)

    def update(self, tm_input):
        if len(tm_input['observations']) > 0:
            tm_input['observations'] = self.binarizer.transform(np.array(tm_input['observations']))
            self.tm.fit(tm_input['observations'].astype(dtype=np.int32),
                        np.array(tm_input['actions']).astype(dtype=np.int32),
                        np.array(tm_input['feedback']).astype(dtype=np.int32))

    def predict(self, obs):
        # binarize input
        if obs.ndim == 1:
            obs = obs.reshape(1, self.config['obs_space_size'])
        b_obs = self.binarizer.transform(obs)
        # pass it through each tm
        b_obs = b_obs.astype(dtype=np.int32)
        result = []
        for obs in b_obs:
            actions = np.array([0 for i in range(self.config['action_space_size'])])
            tm_vals = self.tm.predict(obs)
            actions[tm_vals] = 1
            result.append(actions)
        return np.array(result)


class RTMS:
    def __init__(self, config):
        self.tm = RTM.TsetlinMachine(number_of_clauses=config['critic']['nr_of_clauses'],
                                     number_of_features=config['critic']['bits_per_feature'] * config['obs_space_size'],
                                     s=config['critic']['s'],
                                     number_of_states=config['critic']['number_of_state_bits_ta'],
                                     threshold=config['critic']['T'],
                                     max_target=config['critic']['y_max'], min_target=config['critic']['y_min'],
                                     max_update_p=config['critic']['max_update_p'])

        self.vals = np.loadtxt(f'../algorithms/misc/{config["dataset_file_name"]}.txt', delimiter=',').astype(
            dtype=np.float32)
        self.config = config

        self.binarizer = StandardBinarizer(max_bits_per_feature=config['critic']['bits_per_feature'])
        self.init_binarizer()
        self.init_TMs()

    def init_binarizer(self):
        self.binarizer.fit(self.vals)

    def init_TMs(self):
        vals = self.binarizer.transform(self.vals)
        vals = vals.astype(dtype=np.int32)
        self.tm.fit(vals, np.array(
            [random.randint(int(self.config['critic']['y_min']), int(self.config['critic']['y_max'])) for _ in
             range(len(vals[:20]))]).astype(dtype=np.float32))

    def update(self, tm_input):
        # take a list for each tm that is being updated.
        if len(tm_input['observations']) > 0:
            tm_input['observations'] = self.binarizer.transform(np.array(tm_input['observations']))
            Xs = np.concatenate((tm_input['observations'], np.expand_dims(tm_input['actions'], axis=1)), axis=1)
            self.tm.fit(Xs.astype(dtype=np.int32), np.array(tm_input['target']).astype(dtype=np.float32))

    def predict(self, obs, actions):
        # binarize input
        if obs.ndim == 1:
            obs = obs.reshape(1, self.config['obs_space_size'])
        b_obs = self.binarizer.transform(obs)
        # pass it through each tm
        b_obs = np.concatenate((b_obs, actions), axis=1)
        b_obs = b_obs.astype(dtype=np.int32)
        result = []
        for obs in b_obs:
            tm_vals = self.tm.predict(obs)
            result.append(tm_vals)

        return np.array(result)


class ActorCriticPolicy:
    def __init__(self, config):
        self.online_critic = RTMS(config)
        self.target_critic = RTMS(config)
        self.actor = MTMS(config)

    def get_action(self, obs):
        actions = self.actor.predict(obs)
        action = np.argmax(actions, axis=1)
        return action[0], actions[0]

    def get_best_action(self, obs):
        actions = self.actor.predict(obs)
        action = np.argmax(actions, axis=1)
        return action[0], actions
