from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
from tmu.models.regression.vanilla_regressor import TMRegressor
import torch.nn.functional as F
import numpy as np
import pyximport;

pyximport.install(setup_args={
    "include_dirs": np.get_include()},
    reload_support=True)

# import RTM.RegressionTsetlinMachine as RTM
import TM_lib.rtm as RTM
import TM_lib.mtm as MTM
# import RTM.rtm_custom_continious as RTM
# import RTM.rtm_custom as RTM
import numpy as np
import random
import torch

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
#https://github.com/cair/TsetlinMachine/blob/master/MultiClassTsetlinMachine.pyx


class MTMS:
    def __init__(self, config):
        self.tms = []
        self.tm = MTM.MultiClassTsetlinMachine(number_of_classes=config['nr_of_classes'],
                                    number_of_clauses=config['nr_of_clauses'],
                                    number_of_features=config['bits_per_feature'] * 4,
                                    s=config['s'],
                                    number_of_states=config['number_of_state_bits_ta'],
                                    threshold=config['T'])

        self.vals = np.loadtxt('./algorithms/misc/observation_data.txt', delimiter=',').astype(dtype=np.float32)

        self.binarizer = StandardBinarizer(max_bits_per_feature=config['bits_per_feature'])
        self.init_binarizer()
        #self.init_TMs()

    def init_binarizer(self):
        # create a list of lists of values?
        self.binarizer.fit(self.vals)

    def init_TMs(self):
        vals = self.binarizer.transform(self.vals)
        vals = vals.astype(dtype=np.int32)
        self.tm.fit(vals, np.array([random.randint(0, 2000) / 1000 for _ in range(len(vals[:10]))]).astype(dtype=np.float32))

    def update(self, tm_input):
        # take a list for each tm that is being updated.
        if len(tm_input['observations']) > 0:
            tm_input['observations'] = self.binarizer.transform(np.array(tm_input['observations']))
            self.tm.fit(tm_input['observations'].astype(dtype=np.int32), np.array(tm_input['actions']).astype(dtype=np.int32), np.array(tm_input['feedback']).astype(dtype=np.int32))

    def predict(self, obs):
        # binarize input
        if obs.ndim == 1:
            obs = obs.reshape(1, 4)
        b_obs = self.binarizer.transform(obs)
        # pass it through each tm
        b_obs = b_obs.astype(dtype=np.int32)
        result = []
        for obs in b_obs:
            actions = np.array([0 for i in range(2)])
            tm_vals = self.tm.predict(obs)
            actions[tm_vals] = 1
            result.append(actions)
        return np.array(result)

class RTMS:
    def __init__(self,config):
        self.tm = RTM.TsetlinMachine(number_of_clauses=config['nr_of_clauses'],
                                    number_of_features=config['bits_per_feature'] * 4,
                                    s=config['s'],
                                    number_of_states=config['number_of_state_bits_ta'],
                                    threshold=config['T'],
                                    max_target=config['y_max'], min_target=config['y_min'])

        self.vals = np.loadtxt('./algorithms/misc/observation_data.txt', delimiter=',').astype(dtype=np.float32)
        self.config = config

        self.binarizer = StandardBinarizer(max_bits_per_feature=config['bits_per_feature'])
        self.init_binarizer()
        self.init_TMs()

    def init_binarizer(self):
        # create a list of lists of values?
        self.binarizer.fit(self.vals)

    def init_TMs(self):
        vals = self.binarizer.transform(self.vals)
        vals = vals.astype(dtype=np.int32)
        #self.tm.fit(vals, np.array([random.randint(int(self.config['y_min']), int(self.config['y_max'])) / (0.5 * self.config['y_max']) for _ in range(len(vals[:10]))]).astype(dtype=np.float32))
        self.tm.fit(vals, np.array([random.randint(int(self.config['y_min']), int(self.config['y_max'])) for _ in range(len(vals[:10]))]).astype(dtype=np.float32))
        #self.tm.fit(vals, np.array([random.randint(0, 2000) / 1000 for _ in range(len(vals[:10]))]).astype(dtype=np.float32))

    def update(self, tm_input):
        # take a list for each tm that is being updated.
        if len(tm_input['observations']) > 0:
            tm_input['observations'] = self.binarizer.transform(np.array(tm_input['observations']))
            Xs = np.concatenate((tm_input['observations'], np.expand_dims(tm_input['actions'], axis=1)), axis=1)
            self.tm.fit(Xs.astype(dtype=np.int32), np.array(tm_input['target']).astype(dtype=np.float32))

    def predict(self, obs, actions):
        # binarize input
        if obs.ndim == 1:
            obs = obs.reshape(1, 4)
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
        self.target_critic = RTMS(config['critic'])
        self.evaluation_critic = RTMS(config['critic'])
        self.actor = MTMS(config['actor'])


    def get_action(self, obs):
        actions = self.actor.predict(obs)
        action = np.argmax(actions, axis=1)
        #critic_input = np.concatenate((obs, actions))
        #values = self.critic.predict(critic_input)
        return action[0], actions[0]#, values

    def get_best_action(self, obs):
        actions = self.actor.predict(obs)
        action = np.argmax(actions, axis=1)
        return action[0], actions


