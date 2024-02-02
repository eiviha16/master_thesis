from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
from tmu.models.regression.vanilla_regressor import TMRegressor
import torch.nn.functional as F
import numpy as np
import pyximport;

pyximport.install(setup_args={
    "include_dirs": np.get_include()},
    reload_support=True)

# import RTM.RegressionTsetlinMachine as RTM
import RTM.rtm_custom2 as RTM
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

        self.tm1 = RTM.TsetlinMachine(number_of_clauses=config['nr_of_clauses'],
                                      number_of_features=config['bits_per_feature'] * 4,
                                      s=config['s'],
                                      number_of_states=config['number_of_state_bits_ta'],
                                      threshold=config['T'],
                                      max_target=config['y_max'], min_target=config[
                'y_min'])
        self.tm2 = RTM.TsetlinMachine(number_of_clauses=config['nr_of_clauses'],
                                      number_of_features=config['bits_per_feature'] * 4,
                                      s=config['s'],
                                      number_of_states=config['number_of_state_bits_ta'],
                                      threshold=config['T'],
                                      max_target=config['y_max'], min_target=config[
                'y_min'])
        self.vals = np.loadtxt('./algorithms/misc/observation_data.txt', delimiter=',').astype(dtype=np.float32)

        self.binarizer = StandardBinarizer(max_bits_per_feature=config['bits_per_feature'])
        self.init_binarizer()
        self.init_TMs()

    def init_binarizer(self):
        # create a list of lists of values?
        self.binarizer.fit(self.vals)

    def init_TMs(self):
        vals = self.binarizer.transform(self.vals)
        vals = vals.astype(dtype=np.int32)
        vals = vals[:20]
        _ = self.tm1.fit(vals,
                         np.array([random.randint(0, 60) for _ in range(len(vals[:100]))]).astype(dtype=np.float32))
        _ = self.tm2.fit(vals,
                         np.array([random.randint(0, 60) for _ in range(len(vals[:100]))]).astype(dtype=np.float32))

    def update(self, tm_1_input, tm_2_input):
        # take a list for each tm that is being updated.
        if len(tm_1_input['observations']) > 0:
            tm_1_input['observations'] = self.binarizer.transform(np.array(tm_1_input['observations']))
            self.tm1.fit(tm_1_input['observations'].astype(dtype=np.int32),
                         np.array(tm_1_input['target_q_vals']).astype(dtype=np.float32))

        if len(tm_2_input['observations']) > 0:
            tm_2_input['observations'] = self.binarizer.transform(np.array(tm_2_input['observations']))
            self.tm2.fit(np.array(tm_2_input['observations']).astype(dtype=np.int32),
                         np.array(tm_2_input['target_q_vals']).astype(dtype=np.float32))

    def predict(self, obs):
        # binarize input
        if obs.ndim == 1:
            obs = obs.reshape(1, 4)
        b_obs = self.binarizer.transform(obs)
        # pass it through each tm
        b_obs = b_obs.astype(dtype=np.int32)
        result = []
        for obs in b_obs:
            tm1_q_val = self.tm1.predict(obs)
            tm2_q_val = self.tm2.predict(obs)
            result.append([tm1_q_val, tm2_q_val])

        return np.array(result)


class TMS:
    def __init__(self, nr_of_tms, config):
        self.tms = []
        for _ in range(nr_of_tms):
            tm = RTM.TsetlinMachine(number_of_clauses=config['nr_of_clauses'],
                                          number_of_features=config['bits_per_feature'] * 4,
                                          s=config['s'],
                                          number_of_states=config['number_of_state_bits_ta'],
                                          threshold=config['T'],
                                          max_target=config['y_max'], min_target=config['y_min'])
            self.tms.append(tm)

        self.vals = np.loadtxt('./algorithms/misc/observation_data.txt', delimiter=',').astype(dtype=np.float32)

        self.binarizer = StandardBinarizer(max_bits_per_feature=config['bits_per_feature'])
        self.init_binarizer()
        self.init_TMs()

    def init_binarizer(self):
        # create a list of lists of values?
        self.binarizer.fit(self.vals)

    def init_TMs(self):
        vals = self.binarizer.transform(self.vals)
        vals = vals.astype(dtype=np.int32)
        vals = vals[:20]
        for tm in self.tms:
            tm.fit(vals, np.array([random.randint(0, 2000) / 1000 for _ in range(len(vals[:10]))]).astype(dtype=np.float32))

    def update(self, tm_input):
        # take a list for each tm that is being updated.
        for i, tm in enumerate(self.tms):
            if len(tm_input[i]['observations']) > 0:
                tm_input[i]['observations'] = self.binarizer.transform(np.array(tm_input[i]['observations']))
                tm.fit(tm_input[i]['observations'].astype(dtype=np.int32),
                             np.array(tm_input[i]['target']).astype(dtype=np.float32))
    def update_2(self, tm_input):
        # take a list for each tm that is being updated.
        for i, tm in enumerate(self.tms):
            if len(tm_input[i]['observations']) > 0:
                tm_input[i]['observations'] = self.binarizer.transform(np.array(tm_input[i]['observations']))
                tm.fit_2(
                        tm_input[i]['observations'].astype(dtype=np.int32),
                        np.array(tm_input[i]['target']).astype(dtype=np.float32),
                        np.array(tm_input[i]['advantages']).astype(dtype=np.float32),
                        np.array(tm_input[i]['entropy']).astype(dtype=np.float32)
                )

    def predict(self, obs):
        # binarize input
        if obs.ndim == 1:
            obs = obs.reshape(1, 4)
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


class ActorCriticPolicy:
    def __init__(self, config):
        self.actor = TMS(2, config)
        self.critic = TMS(1, config)

    def get_action(self, obs):
        action_probs = self.actor.predict(obs) + 1e-10
        normalized_action_prob = action_probs / np.sum(action_probs, axis=-1, keepdims=True)
        actions = np.apply_along_axis(lambda x: np.random.choice([0, 1], p=x), axis=-1, arr=normalized_action_prob)
        entropy = [-(p * np.log2(p) + (1 - p) * np.log2(1 - p)) for p in normalized_action_prob][0]
        values = self.critic.predict(obs)
        return actions, values, action_probs, entropy #done away with log softmax

    def get_best_action(self, obs):
        action_probs = self.actor.predict(obs)
        actions = np.argmax(action_probs, axis=-1)
        return actions
#policy gradient
#a2c a3c

"""
from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
from tmu.models.regression.vanilla_regressor import TMRegressor
import numpy as np
import pyximport;

# pyximport.install(setup_args={
#    "include_dirs": np.get_include()},
#    reload_support=True)

# import RTM.RegressionTsetlinMachine as RTM
# import RTM.rtm_custom2 as RTM
# import RTM.rtm_custom as RTM
from tmu.models.regression.vanilla_regressor import TMRegressor
import numpy as np
import random
import torch

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


class Policy():
    def __init__(self, config):
        # initialize each tm

        self.tm1 = TMRegressor(number_of_clauses=config['nr_of_clauses'],
                               s=config['s'],
                               platform=config['device'],
                               number_of_state_bits_ta=config['number_of_state_bits_ta'],
                               T=config['T'],
                               seed=42,
                               weighted_clauses=config['weighted_clauses'],
                               max_y=config['y_max'], min_y=config['y_min'])
        self.tm2 = TMRegressor(number_of_clauses=config['nr_of_clauses'],
                               s=config['s'],
                               platform=config['device'],
                               number_of_state_bits_ta=config['number_of_state_bits_ta'],
                               T=config['T'],
                               seed=42,
                               weighted_clauses=config['weighted_clauses'],
                               max_y=config['y_max'], min_y=config['y_min'])

        self.vals = np.loadtxt('./algorithms/misc/observation_data.txt', delimiter=',').astype(dtype=np.float32)

        self.binarizer = StandardBinarizer(max_bits_per_feature=config['bits_per_feature'])
        self.init_binarizer()
        self.init_TMs()

    def init_binarizer(self):
        # create a list of lists of values?
        self.binarizer.fit(self.vals)

    def init_TMs(self):
        vals = self.binarizer.transform(self.vals)
        vals = vals.astype(dtype=np.int32)
        vals = vals[:20]
        _ = self.tm1.fit(vals,
                         np.array([random.randint(0, 60) for _ in range(len(vals[:100]))]))
        _ = self.tm2.fit(vals,
                         np.array([random.randint(0, 60) for _ in range(len(vals[:100]))]))

    def update(self, tm_1_input, tm_2_input):
        # take a list for each tm that is being updated.
        if len(tm_1_input['observations']) > 0:
            tm_1_input['observations'] = self.binarizer.transform(np.array(tm_1_input['observations']))
            self.tm1.fit(tm_1_input['observations'],
                         np.array(tm_1_input['target_q_vals']))

        if len(tm_2_input['observations']) > 0:
            tm_2_input['observations'] = self.binarizer.transform(np.array(tm_2_input['observations']))
            self.tm2.fit(np.array(tm_2_input['observations']),
                         np.array(tm_2_input['target_q_vals']))

    def predict(self, obs):
        # binarize input
        if obs.ndim == 1:
            obs = obs.reshape(1, 4)
        b_obs = self.binarizer.transform(obs)
        # pass it through each tm
        #b_obs = b_obs.astype(dtype=np.int32)
        tm1_q_val = self.tm1.predict(obs)
        tm2_q_val = self.tm2.predict(obs)
        return np.transpose(np.array([tm1_q_val, tm2_q_val]))
       # return np.array(result)


"""
