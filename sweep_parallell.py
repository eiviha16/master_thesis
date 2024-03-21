from sweep_configs import *
from sweep_functions import *

import multiprocessing
import os

def run_wandb_sweep(config):
    # Code to run a single Weights & Biases sweep with given config
    # Replace the following line with your actual command to run a sweep
    os.system(f"wandb sweep {config}")

if __name__ == "__main__":
    # List of configurations for each sweep
    sweep_configs = [
        acrobot_QTM, acrobot_n_step_QTM, acrobot_TPPO, acrobot_n_step_DQTM_a, acrobot_n_step_DQTM_b, acrobot_DQTM_a, acrobot_DQTM_b, acrobot_TAC_a, acrobot_TAC_b,
        cartpole_QTM, cartpole_n_step_QTM, cartpole_TPPO, cartpole_n_step_DQTM_a, cartpole_n_step_DQTM_b, cartpole_DQTM_a, cartpole_DQTM_b, cartpole_TAC_a, cartpole_TAC_b
    ]

    # Create a process for each sweep
    processes = []
    for config in sweep_configs:
        process = multiprocessing.Process(target=run_wandb_sweep, args=(config,))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()