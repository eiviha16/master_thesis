from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
from tmu.models.regression.vanilla_regressor import TMRegressor
import torch.nn.functional as F
import numpy as np
import pyximport;

pyximport.install(setup_args={
    "include_dirs": np.get_include()},
    reload_support=True)

# import RTM.RegressionTsetlinMachine as RTM
# import RTM.rtm_custom2 as RTM
#import TM_lib.rtm as RTM
#import TM_lib_2.rtm as RTM
import TM_lib_3.rtm as RTM
# import RTM.rtm_custom_continious as RTM
# import RTM.rtm_custom as RTM
import numpy as np
import random
import torch

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


class Policy():
    def __init__(self, config):
        # initialize each tm
        self.tms = [RTM.TsetlinMachine(number_of_clauses=config['nr_of_clauses'],
                                      number_of_features=config['bits_per_feature'] * config["obs_space_size"],
                                      s=config['s'],
                                      number_of_states=config['number_of_state_bits_ta'],
                                      threshold=config['T'],
                                      max_target=config['y_max'], min_target=config['y_min'],
                                      max_update_p=config['max_update_p'],
                                      min_update_p=config['min_update_p'])

                                      for _ in range(config['action_space_size'])]


        self.vals = np.loadtxt(f'../algorithms/misc/{config["dataset_file_name"]}.txt', delimiter=',').astype(dtype=np.float32)
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
        for tm in self.tms:
            tm.fit(vals,
                   np.array([random.randint(self.config['y_min'], self.config['y_max']) for _ in range(len(vals[:20]))]).astype(dtype=np.float32))



    #def update(self, tm_input, tm_2_input):
    def update(self, tms_input):
        # take a list for each tm that is being updated.
        abs_errors = {f'actor{i}': [] for i in range(len(self.tms))}
        for idx, input in enumerate(tms_input):
            if len(tms_input[idx]['observations']) > 0:
                tms_input[idx]['observations'] = self.binarizer.transform(np.array(tms_input[idx]['observations']))
                abs_errors['actor1'] = self.tms[idx].fit(tms_input[idx]['observations'].astype(dtype=np.int32),
                                                    np.array(tms_input[idx]['target_q_vals']).astype(dtype=np.float32))

        return abs_errors
    def predict(self, obs):
        # binarize input
        if obs.ndim == 1:
            obs = obs.reshape(1, self.config['obs_space_size'])
        b_obs = self.binarizer.transform(obs)

        # pass it through each tm
        b_obs = b_obs.astype(dtype=np.int32)
        result = []
        for obs in b_obs:
            tm_vals = [0 for _ in range(len(self.tms))]
            for i, tm in enumerate(self.tms):
                tm_vals[i] = tm.predict(obs)
            result.append(tm_vals)

        return np.array(result)



class TMS:
    def __init__(self, nr_of_tms, config, obs_space_size, file_name):
        self.tms = []
        for _ in range(nr_of_tms):
            tm = RTM.TsetlinMachine(number_of_clauses=config['nr_of_clauses'],
                                    number_of_features=config['bits_per_feature'] * obs_space_size,
                                    s=config['s'],
                                    number_of_states=config['number_of_state_bits_ta'],
                                    threshold=config['T'],
                                    max_target=config['y_max'], min_target=config['y_min'],
                                    max_update_p = config['max_update_p'],
                                    min_update_p = config['min_update_p'])

            self.tms.append(tm)

        self.vals = np.loadtxt(f'../algorithms/misc/{file_name}.txt', delimiter=',').astype(dtype=np.float32)
        self.config = config
        self.obs_space_size = obs_space_size
        self.binarizer = StandardBinarizer(max_bits_per_feature=config['bits_per_feature'])
        self.init_binarizer()
        self.init_TMs()

    def init_binarizer(self):
        # create a list of lists of values?
        self.binarizer.fit(self.vals)

    def init_TMs(self):
        vals = self.binarizer.transform(self.vals)
        vals = vals.astype(dtype=np.int32)
        for tm in self.tms:
            tm.fit(vals,
                   np.array([random.randint(int(self.config['y_min']), int(self.config['y_max'])) for _ in range(len(vals[:20]))]).astype(dtype=np.float32))

    def update(self, tm_input):
        keys = ['critic']
        abs_errors = {}
        # take a list for each tm that is being updated.
        for i, tm in enumerate(self.tms):
            if len(tm_input[i]['observations']) > 0:
                tm_input[i]['observations'] = self.binarizer.transform(np.array(tm_input[i]['observations']))
                abs_error = tm.fit(tm_input[i]['observations'].astype(dtype=np.int32),
                       np.array(tm_input[i]['target']).astype(dtype=np.float32))
                abs_errors[keys[i]] = abs_error
        return abs_errors
    def update_2(self, tm_input):#, clip):
        # take a list for each tm that is being updated.
        for i, tm in enumerate(self.tms):
            if len(tm_input[i]['observations']) > 0:
                tm_input[i]['observations'] = self.binarizer.transform(np.array(tm_input[i]['observations']))
                tm.fit_2(
                    tm_input[i]['observations'].astype(dtype=np.int32),
                    np.array(tm_input[i]['target']).astype(dtype=np.float32),
                    np.array(tm_input[i]['advantages']).astype(dtype=np.float32),
                    np.array(tm_input[i]['entropy']).astype(dtype=np.float32)
                    #clip
                )

    def predict(self, obs):
        # binarize input
        if obs.ndim == 1:
            obs = obs.reshape(1, self.obs_space_size)
        b_obs = self.binarizer.transform(obs)
        b_obs = b_obs.astype(dtype=np.int32)
        result = []
        for obs in b_obs:
            tm_vals = [0 for _ in range(len(self.tms))]
            for i, tm in enumerate(self.tms):
                tm_vals[i] = tm.predict(obs)
            result.append(tm_vals)

        return np.array(result)


class ActorCriticPolicy:
    def __init__(self, config):
        self.critic = TMS(1, config['critic'], config['obs_space_size'], config["dataset_file_name"])
        self.actor = TMS(config['action_space_size'], config['actor'], config['obs_space_size'], config["dataset_file_name"])
        self.config = config

    def get_action(self, obs):
        action_probs = self.actor.predict(obs) + 1e-10
        normalized_action_prob = action_probs / np.sum(action_probs, axis=-1, keepdims=True)
        actions = np.apply_along_axis(lambda x: np.random.choice(range(self.config['action_space_size']), p=x), axis=-1, arr=normalized_action_prob)
        entropy = [-(p * np.log2(p) + (1 - p) * np.log2(1 - p)) for p in normalized_action_prob][0]
        values = self.critic.predict(obs)
        return actions, values, action_probs, entropy  # done away with log softmax

    def get_best_action(self, obs):
        action_probs = self.actor.predict(obs)
        actions = np.argmax(action_probs, axis=-1)
        return actions, action_probs
