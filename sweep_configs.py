import numpy as np

################################################
################### TAC a #######################
################################################
config_cartpole_TAC_a = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "update_grad": {"values": list(np.arange(0.001, 1.0, 0.001))},
        "batch_size": {"values": list(range(16, 128, 16))},
        "epochs": {"values": list(range(1, 8, 1))},
        "a_t": {"values": list(np.arange(0.01, 1.0, 0.01))},
        "a_nr_of_clauses": {"values": list(range(800, 1200, 20))},
        "a_specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "a_bits_per_feature": {"values": list(range(5, 15, 1))},
        "a_number_of_state_bits_ta": {"values": list(range(3, 10, 1))},

        "c_t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "c_nr_of_clauses": {"values": list(range(800, 2000, 50))},
        "c_specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "c_bits_per_feature": {"values": list(range(5, 15, 1))},
        "c_number_of_state_bits_ta": {"values": list(range(3, 10, 1))},
        "c_max_update_p": {"values": list(np.arange(0.001, 0.1, 0.001))},

        "c_y_max": {"values": list(range(60, 80, 5))},
        "c_y_min": {"values": list(range(20, 40, 5))},
        "buffer_size": {"values": list(range(500, 10_000, 500))},
        "exploration_p_decay": {"values": list(np.arange(0.001, 0.01, 0.001))},
        "exploration_p_init": {"values": list(np.arange(0.8, 1.00, 0.1))},
    }
}

config_acrobot_TAC_a = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "update_grad": {"values": list(np.arange(0.001, 1.0, 0.001))},
        "batch_size": {"values": list(range(16, 128, 16))},
        "epochs": {"values": list(range(1, 8, 1))},

        "a_t": {"values": list(np.arange(0.01, 1.0, 0.01))},
        "a_nr_of_clauses": {"values": list(range(1000, 2000, 20))},
        "a_specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "a_bits_per_feature": {"values": list(range(5, 15, 1))},
        "a_number_of_state_bits_ta": {"values": list(range(3, 10, 1))},

        "c_t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "c_nr_of_clauses": {"values": list(range(1000, 2000, 50))},
        "c_specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "c_bits_per_feature": {"values": list(range(5, 15, 1))},
        "c_number_of_state_bits_ta": {"values": list(range(3, 10, 1))},
        "c_max_update_p": {"values": list(np.arange(0.001, 0.1, 0.001))},

        "c_y_max": {"values": list(range(-10, 0, 5))},
        "c_y_min": {"values": list(range(-80, -45, 5))},
        "buffer_size": {"values": list(range(500, 10_000, 500))},
        "exploration_p_decay": {"values": list(np.arange(0.001, 0.01, 0.001))},
        "exploration_p_init": {"values": list(np.arange(0.5, 1.00, 0.1))},
    }
}

################################################
################### TAC b #######################
################################################
config_cartpole_TAC_b = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "update_freq": {"values": list(range(1, 10, 1))},
        "batch_size": {"values": list(range(16, 128, 16))},
        "epochs": {"values": list(range(1, 8, 1))},

        "a_t": {"values": list(np.arange(0.01, 1.0, 0.01))},
        "a_nr_of_clauses": {"values": list(range(800, 1200, 20))},
        "a_specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "a_bits_per_feature": {"values": list(range(5, 15, 1))},
        "a_number_of_state_bits_ta": {"values": list(range(3, 10, 1))},

        "c_t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "c_nr_of_clauses": {"values": list(range(800, 2000, 50))},
        "c_specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "c_bits_per_feature": {"values": list(range(5, 15, 1))},
        "c_number_of_state_bits_ta": {"values": list(range(3, 10, 1))},
        "c_max_update_p": {"values": list(np.arange(0.001, 0.1, 0.001))},

        "c_y_max": {"values": list(range(60, 80, 5))},
        "c_y_min": {"values": list(range(20, 40, 5))},
        "buffer_size": {"values": list(range(500, 10_000, 500))},
        "exploration_p_decay": {"values": list(np.arange(0.001, 0.01, 0.001))},
        "exploration_p_init": {"values": list(np.arange(0.8, 1.00, 0.1))},
    }
}



config_acrobot_TAC_b = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "update_freq": {"values": list(range(1, 10, 1))},
        "batch_size": {"values": list(range(16, 128, 16))},
        "epochs": {"values": list(range(1, 8, 1))},

        "a_t": {"values": list(np.arange(0.01, 1.0, 0.01))},
        "a_nr_of_clauses": {"values": list(range(1000, 2000, 20))},
        "a_specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "a_bits_per_feature": {"values": list(range(5, 15, 1))},
        "a_number_of_state_bits_ta": {"values": list(range(3, 10, 1))},

        "c_t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "c_nr_of_clauses": {"values": list(range(1000, 2000, 50))},
        "c_specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "c_bits_per_feature": {"values": list(range(5, 15, 1))},
        "c_number_of_state_bits_ta": {"values": list(range(3, 10, 1))},
        "c_max_update_p": {"values": list(np.arange(0.001, 0.1, 0.001))},

        "c_y_max": {"values": list(range(-10, 0, 5))},
        "c_y_min": {"values": list(range(-80, -45, 5))},
        "buffer_size": {"values": list(range(500, 10_000, 500))},
        "exploration_p_decay": {"values": list(np.arange(0.001, 0.01, 0.001))},
        "exploration_p_init": {"values": list(np.arange(0.5, 1.00, 0.1))},
    }
}


################################################
################### TPPO #######################
################################################


config_cartpole_TPPO = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.94, 0.98, 0.001))},
        "lam": {"values": list(np.arange(0.94, 0.98, 0.001))},
        "n_timesteps": {"values": list(range(4, 512, 4))},
        "batch_size": {"values": list(range(16, 512, 16))},
        "epochs": {"values": list(range(1, 5, 1))},

        "a_t": {"values": list(np.arange(0.3, 0.9, 0.01))},
        "a_nr_of_clauses": {"values": list(range(800, 1500, 10))},
        "a_specificity": {"values": list(np.arange(1.0, 4.0, 0.01))},
        "a_bits_per_feature": {"values": list(range(5, 15, 1))},
        "a_number_of_state_bits_ta": {"values": list(range(3, 8, 1))},
        "a_max_update_p": {"values": list(np.arange(0.001, 0.1, 0.001))},
        "a_min_update_p": {"values": list(np.arange(0.0001, 0.001, 0.0001))},

        "c_t": {"values": list(np.arange(0.3, 0.9, 0.01))},
        "c_nr_of_clauses": {"values": list(range(800, 1500, 10))},
        "c_specificity": {"values": list(np.arange(1.0, 4.0, 0.01))},
        "c_bits_per_feature": {"values": list(range(5, 15, 1))},
        "c_number_of_state_bits_ta": {"values": list(range(3, 8, 1))},
        "c_max_update_p": {"values": list(np.arange(0.001, 0.1, 0.001))},
        "c_y_max": {"values": list(np.arange(5.5, 35, 0.5))},
        "c_y_min": {"values": list(np.arange(0.0, 1.0, 0.1))},
    }
}
config_acrobot_TPPO = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.94, 0.98, 0.001))},
        "lam": {"values": list(np.arange(0.94, 0.98, 0.001))},
        "n_timesteps": {"values": list(range(4, 512, 4))},
        "batch_size": {"values": list(range(16, 512, 16))},
        "epochs": {"values": list(range(1, 5, 1))},

        "a_t": {"values": list(np.arange(0.3, 0.9, 0.01))},
        "a_nr_of_clauses": {"values": list(range(1000, 2000, 10))},
        "a_specificity": {"values": list(np.arange(1.0, 4.0, 0.01))},
        "a_bits_per_feature": {"values": list(range(5, 15, 1))},
        "a_number_of_state_bits_ta": {"values": list(range(3, 8, 1))},
        "a_max_update_p": {"values": list(np.arange(0.001, 0.1, 0.001))},
        "a_min_update_p": {"values": list(np.arange(0.0001, 0.001, 0.0001))},

        "c_t": {"values": list(np.arange(0.3, 0.9, 0.01))},
        "c_nr_of_clauses": {"values": list(range(1000, 2000, 10))},
        "c_specificity": {"values": list(np.arange(1.0, 4.0, 0.01))},
        "c_bits_per_feature": {"values": list(range(5, 15, 1))},
        "c_number_of_state_bits_ta": {"values": list(range(3, 8, 1))},
        "c_max_update_p": {"values": list(np.arange(0.001, 0.1, 0.001))},
        "c_y_max": {"values": list(np.arange(-1.0, 0.0, 0.1))},
        "c_y_min": {"values": list(np.arange(-35.0, -5.5, 0.1))},
    }
}

################################################
######### n-step Double QTM type a ############
################################################

config_cartpole_n_step_DQTM_a = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "update_grad": {"values": list(np.arange(0.001, 1.0, 0.001))},
        "n_steps": {"values":list(range(5, 50, 1))},
        "batch_size": {"values": list(range(16, 128, 16))},
        "buffer_size": {"values": list(range(500, 10_000, 500))},
        "epochs": {"values": list(range(1, 8, 1))},
        "t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "nr_of_clauses": {"values": list(range(800, 1200, 20))},
        "specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "bits_per_feature": {"values": list(range(5, 15, 1))},
        "number_of_state_bits_ta": {"values": list(range(3, 10, 1))},
        "y_max": {"values": list(range(60, 80, 5))},
        "y_min": {"values": list(range(20, 40, 5))},
        "exploration_p_decay": {"values": list(np.arange(0.001, 0.01, 0.001))},
        "exploration_p_init": {"values": list(np.arange(0.8, 1.00, 0.1))},
        "max_update_p": {"values": list(np.arange(0.001, 0.2, 0.001))},

    }
}

config_acrobot_n_step_DQTM_a = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "update_grad": {"values": list(np.arange(0.001, 1.0, 0.001))},
        "n_steps": {"values": list(range(5, 50, 1))},
        "batch_size": {"values": list(range(16, 128, 16))},
        "buffer_size": {"values": list(range(500, 10_000, 500))},
        "epochs": {"values": list(range(1, 8, 1))},
        "t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "nr_of_clauses": {"values": list(range(1000, 2000, 20))},
        "specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "bits_per_feature": {"values": list(range(5, 15, 1))},
        "number_of_state_bits_ta": {"values": list(range(3, 10, 1))},
        "y_max": {"values": list(range(-10, 0, 5))},
        "y_min": {"values": list(range(-80, -45, 5))},
        "exploration_p_decay": {"values": list(np.arange(0.001, 0.01, 0.001))},
        "exploration_p_init": {"values": list(np.arange(0.5, 1.00, 0.1))},
        "max_update_p": {"values": list(np.arange(0.001, 0.2, 0.001))},

    }
}


################################################
######### n-step Double TMQN type b ############
################################################
config_cartpole_n_step_DQTM_b = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "update_freq": {"values": list(range(1, 10, 1))},
        "n_steps": {"values": list(range(5, 50, 1))},
        "batch_size": {"values": list(range(16, 128, 16))},
        "buffer_size": {"values": list(range(500, 10_000, 500))},
        "epochs": {"values": list(range(1, 8, 1))},
        "t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "nr_of_clauses": {"values": list(range(800, 1200, 20))},
        "specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "bits_per_feature": {"values": list(range(5, 15, 1))},
        "number_of_state_bits_ta": {"values": list(range(3, 10, 1))},
        "y_max": {"values": list(range(60, 80, 5))},
        "y_min": {"values": list(range(20, 40, 5))},
        "exploration_p_decay": {"values": list(np.arange(0.001, 0.01, 0.001))},
        "exploration_p_init": {"values": list(np.arange(0.8, 1.00, 0.1))},
        "max_update_p": {"values": list(np.arange(0.001, 0.2, 0.001))},

    }
}

config_acrobot_n_step_DQTM_b = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "update_freq": {"values": list(range(1, 10, 1))},
        "n_steps": {"values": list(range(5, 50, 1))},
        "batch_size": {"values": list(range(16, 128, 16))},
        "buffer_size": {"values": list(range(500, 10_000, 500))},
        "epochs": {"values": list(range(1, 8, 1))},
        "t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "nr_of_clauses": {"values": list(range(1000, 2000, 20))},
        "specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "bits_per_feature": {"values": list(range(5, 15, 1))},
        "number_of_state_bits_ta": {"values": list(range(3, 10, 1))},
        "y_max": {"values": list(range(-10, 0, 5))},
        "y_min": {"values": list(range(-80, -45, 5))},
        "exploration_p_decay": {"values": list(np.arange(0.001, 0.01, 0.001))},
        "exploration_p_init": {"values": list(np.arange(0.5, 1.00, 0.1))},
        "max_update_p": {"values": list(np.arange(0.001, 0.2, 0.001))},

    }
}

################################################
############ Double TMQN type a ################
################################################
config_cartpole_DQTM_a = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "update_grad": {"values": list(np.arange(0.001, 1.0, 0.001))},
        "batch_size": {"values": list(range(16, 128, 16))},
        "buffer_size": {"values": list(range(500, 10_000, 500))},
        "epochs": {"values": list(range(1, 8, 1))},
        "t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "nr_of_clauses": {"values": list(range(800, 1200, 20))},
        "specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "bits_per_feature": {"values": list(range(5, 15, 1))},
        "number_of_state_bits_ta": {"values": list(range(3, 10, 1))},
        "y_max": {"values": list(range(60, 80, 5))},
        "y_min": {"values": list(range(20, 40, 5))},
        "exploration_p_decay": {"values": list(np.arange(0.001, 0.01, 0.001))},
        "exploration_p_init": {"values": list(np.arange(0.8, 1.00, 0.1))},
        "max_update_p": {"values": list(np.arange(0.001, 0.2, 0.001))},

    }
}

config_acrobot_DQTM_a = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "update_grad": {"values": list(np.arange(0.001, 1.0, 0.001))},
        "batch_size": {"values": list(range(16, 128, 16))},
        "buffer_size": {"values": list(range(500, 10_000, 500))},
        "epochs": {"values": list(range(1, 8, 1))},
        "t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "nr_of_clauses": {"values": list(range(1000, 2000, 20))},
        "specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "bits_per_feature": {"values": list(range(5, 15, 1))},
        "number_of_state_bits_ta": {"values": list(range(3, 10, 1))},
        "y_max": {"values": list(range(-10, 0, 5))},
        "y_min": {"values": list(range(-80, -45, 5))},
        "exploration_p_decay": {"values": list(np.arange(0.001, 0.01, 0.001))},
        "exploration_p_init": {"values": list(np.arange(0.5, 1.00, 0.1))},
        "max_update_p": {"values": list(np.arange(0.001, 0.2, 0.001))},

    }
}
################################################
############ Double TMQN type b ################
################################################
config_cartpole_DQTM_b = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "update_freq": {"values": list(range(1, 10, 1))},
        "batch_size": {"values": list(range(16, 128, 16))},
        "buffer_size": {"values": list(range(500, 10_000, 500))},
        "epochs": {"values": list(range(1, 8, 1))},
        "t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "nr_of_clauses": {"values": list(range(800, 1200, 20))},
        "specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "bits_per_feature": {"values": list(range(5, 15, 1))},
        "number_of_state_bits_ta": {"values": list(range(3, 10, 1))},
        "y_max": {"values": list(range(60, 80, 5))},
        "y_min": {"values": list(range(20, 40, 5))},
        "exploration_p_decay": {"values": list(np.arange(0.001, 0.01, 0.001))},
        "exploration_p_init": {"values": list(np.arange(0.8, 1.00, 0.1))},
        "max_update_p": {"values": list(np.arange(0.001, 0.2, 0.001))},

    }
}

config_acrobot_DQTM_b = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "update_freq": {"values": list(range(1, 10, 1))},
        "batch_size": {"values": list(range(16, 128, 16))},
        "buffer_size": {"values": list(range(500, 10_000, 500))},
        "epochs": {"values": list(range(1, 8, 1))},
        "t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "nr_of_clauses": {"values": list(range(1000, 2000, 20))},
        "specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "bits_per_feature": {"values": list(range(5, 15, 1))},
        "number_of_state_bits_ta": {"values": list(range(3, 10, 1))},
        "y_max": {"values": list(range(-10, 0, 5))},
        "y_min": {"values": list(range(-80, -45, 5))},
        "exploration_p_decay": {"values": list(np.arange(0.001, 0.01, 0.001))},
        "exploration_p_init": {"values": list(np.arange(0.5, 1.00, 0.1))},
        "max_update_p": {"values": list(np.arange(0.001, 0.2, 0.001))},

    }
}

################################################
################# n-step TMQN  #################
################################################
config_cartpole_n_step_QTM = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "batch_size": {"values": list(range(16, 128, 16))},
        "n_steps": {"values": list(range(5, 50, 1))},
        "buffer_size": {"values": list(range(500, 10_000, 500))},
        "epochs": {"values": list(range(1, 8, 1))},
        "t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "nr_of_clauses": {"values": list(range(800, 1200, 20))},
        "specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "bits_per_feature": {"values": list(range(5, 15, 1))},
        "number_of_state_bits_ta": {"values": list(range(3, 10, 1))},
        "y_max": {"values": list(range(60, 80, 5))},
        "y_min": {"values": list(range(20, 40, 5))},
        "exploration_p_decay": {"values": list(np.arange(0.001, 0.01, 0.001))},
        "exploration_p_init": {"values": list(np.arange(0.8, 1.00, 0.1))},
        "max_update_p": {"values": list(np.arange(0.001, 0.2, 0.001))},

    }
}

config_acrobot_n_step_QTM = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "batch_size": {"values": list(range(16, 128, 16))},
        "n_steps": {"values": list(range(5, 50, 1))},
        "buffer_size": {"values": list(range(500, 10_000, 500))},
        "epochs": {"values": list(range(1, 8, 1))},
        "t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "nr_of_clauses": {"values": list(range(1000, 2000, 20))},
        "specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "bits_per_feature": {"values": list(range(5, 15, 1))},
        "number_of_state_bits_ta": {"values": list(range(3, 10, 1))},
        "y_max": {"values": list(range(-10, 0, 5))},
        "y_min": {"values": list(range(-80, -45, 5))},
        "exploration_p_decay": {"values": list(np.arange(0.001, 0.01, 0.001))},
        "exploration_p_init": {"values": list(np.arange(0.5, 1.00, 0.1))},
        "max_update_p": {"values": list(np.arange(0.001, 0.2, 0.001))},

    }
}

################################################
################### TMQN  ######################
################################################

config_cartpole_QTM = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "batch_size": {"values": list(range(16, 128, 16))},
        "buffer_size": {"values": list(range(500, 10_000, 500))},
        "epochs": {"values": list(range(1, 8, 1))},
        "t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "nr_of_clauses": {"values": list(range(800, 1200, 20))},
        "specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "bits_per_feature": {"values": list(range(5, 15, 1))},
        "number_of_state_bits_ta": {"values": list(range(3, 10, 1))},
        "y_max": {"values": list(range(60, 80, 5))},
        "y_min": {"values": list(range(20, 40, 5))},
        "exploration_p_decay": {"values": list(np.arange(0.001, 0.01, 0.001))},
        "exploration_p_init": {"values": list(np.arange(0.8, 1.00, 0.1))},
        "max_update_p": {"values": list(np.arange(0.001, 0.2, 0.001))},

    }
}

config_acrobot_QTM = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "batch_size": {"values": list(range(16, 128, 16))},
        "buffer_size": {"values": list(range(500, 10_000, 500))},
        "epochs": {"values": list(range(1, 8, 1))},
        "t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "nr_of_clauses": {"values": list(range(1000, 2000, 20))},
        "specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "bits_per_feature": {"values": list(range(5, 15, 1))},
        "number_of_state_bits_ta": {"values": list(range(3, 10, 1))},
        "y_max": {"values": list(range(-10, 0, 5))},
        "y_min": {"values": list(range(-80, -45, 5))},
        "exploration_p_decay": {"values": list(np.arange(0.001, 0.01, 0.001))},
        "exploration_p_init": {"values": list(np.arange(0.5, 1.00, 0.1))},
        "max_update_p": {"values": list(np.arange(0.001, 0.2, 0.001))},

    }
}