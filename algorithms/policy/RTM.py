from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
from tmu.models.regression.vanilla_regressor import TMRegressor
import numpy as np
import pyximport;

pyximport.install(setup_args={
    "include_dirs": np.get_include()},
    reload_support=True)

#import RTM.RegressionTsetlinMachine as RTM
import RTM.rtm_custom2 as RTM
#import RTM.rtm_custom as RTM
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

        tm_1_input['observations'] = self.binarizer.transform(np.array(tm_1_input['observations']))
        tm_2_input['observations'] = self.binarizer.transform(np.array(tm_2_input['observations']))

        self.tm1.fit(tm_1_input['observations'].astype(dtype=np.int32),
                     np.array(tm_1_input['target_q_vals']).astype(dtype=np.float32))
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
