import wandb
from sweep_configs import *
from sweep_functions import *

wandb.login(key="74a10e58809253b0e1f243f34bb17d8f34c21e59")

def main_c_random_TAC_a():
    wandb.init(project="cartpole-TAC_a-random")
    score = cartpole_random_TAC_a(wandb.config)
    wandb.log({"score": score})
def main_a_random_TAC_a():
    wandb.init(project="acrobot-TAC_a-random")
    score = acrobot_random_TAC_a(wandb.config)
    wandb.log({"score": score})

def start_c_random_TAC_a():
    import wandb
    sweep_id = wandb.sweep(sweep=config_cartpole_random_TAC_a, project="cartpole-TAC_a-random")
    wandb.agent(sweep_id, function=main_c_random_TAC_a, count=10_000)

def start_a_random_TAC_a():
    import wandb
    sweep_id = wandb.sweep(sweep=config_acrobot_random_TAC_a, project="acrobot-TAC_a-random")
    wandb.agent(sweep_id, function=main_a_random_TAC_a, count=10_000)


################################################
################### TAC a #######################
################################################

def main_c_TAC_a():
    wandb.init(project="cartpole-TAC_a-ff")
    score = cartpole_TAC_a(wandb.config)
    wandb.log({"score": score})
def main_a_TAC_a():
    wandb.init(project="acrobot-TAC_a-ff")
    score = acrobot_TAC_a(wandb.config)
    wandb.log({"score": score})

def start_c_TAC_a():
    import wandb
    sweep_id = wandb.sweep(sweep=config_cartpole_TAC_a, project="cartpole-TAC_a-ff")
    wandb.agent(sweep_id, function=main_c_TAC_a, count=10_000)

def start_a_TAC_a():
    import wandb
    sweep_id = wandb.sweep(sweep=config_acrobot_TAC_a, project="acrobot-TAC_a-ff")
    wandb.agent(sweep_id, function=main_a_TAC_a, count=10_000)

################################################
################### TAC b #######################
################################################
def main_c_TAC_b():
    wandb.init(project="cartpole-TAC_b-ff")
    score = cartpole_TAC_b(wandb.config)
    wandb.log({"score": score})
def main_a_TAC_b():
    wandb.init(project="acrobot-TAC_b-ff")
    score = acrobot_TAC_b(wandb.config)
    wandb.log({"score": score})


def start_c_TAC_b():
    import wandb
    sweep_id = wandb.sweep(sweep=config_cartpole_TAC_b, project="cartpole-TAC_b-ff")
    wandb.agent(sweep_id, function=main_c_TAC_b, count=10_000)


def start_a_TAC_b():
    import wandb
    sweep_id = wandb.sweep(sweep=config_acrobot_TAC_b, project="acrobot-TAC_b-ff")
    wandb.agent(sweep_id, function=main_a_TAC_b, count=10_000)
################################################
################### TPPO #######################
################################################
def main_c_TPPO():
    wandb.init(project="cartpole-TPPO-f")
    score = cartpole_TPPO(wandb.config)
    wandb.log({"score": score})
def main_a_TPPO():
    wandb.init(project="acrobot-TPPO-f")
    score = acrobot_TPPO(wandb.config)
    wandb.log({"score": score})

def start_c_TPPO():
    import wandb
    sweep_id = wandb.sweep(sweep=config_cartpole_TPPO, project="cartpole-TPPO-f")
    wandb.agent(sweep_id, function=main_c_TPPO, count=10_000)

def start_a_TPPO():
    import wandb
    sweep_id = wandb.sweep(sweep=config_acrobot_TPPO, project="acrobot-TPPO-f")
    wandb.agent(sweep_id, function=main_a_TPPO, count=10_000)

################################################
######### n-step Double QTM type a ############
################################################
def main_c_n_step_DQTM_a():
    wandb.init(project="cartpole-n_step_DQTM_a-f")
    score = cartpole_n_step_DQTM_a(wandb.config)
    wandb.log({"score": score})
def main_a_n_step_DQTM_a():
    wandb.init(project="acrobot-n_step_DQTM_a-f")
    score = acrobot_n_step_DQTM_a(wandb.config)
    wandb.log({"score": score})

def start_c_n_step_DQTM_a():
    import wandb
    sweep_id = wandb.sweep(sweep=config_cartpole_n_step_DQTM_a, project="cartpole-n_step_DQTM_a-f")
    wandb.agent(sweep_id, function=main_c_n_step_DQTM_a, count=10_000)

def start_a_n_step_DQTM_a():
    import wandb
    sweep_id = wandb.sweep(sweep=config_acrobot_n_step_DQTM_a, project="acrobot-n_step_DQTM_a-f")
    wandb.agent(sweep_id, function=main_a_n_step_DQTM_a, count=10_000)
################################################
######### n-step Double TMQN type b ############
################################################

def main_c_n_step_DQTM_b():
    wandb.init(project="cartpole-n_step_DQTM_b-f")
    score = cartpole_n_step_DQTM_b(wandb.config)
    wandb.log({"score": score})
def main_a_n_step_DQTM_b():
    wandb.init(project="acrobot-n_step_DQTM_b-f")
    score = acrobot_n_step_DQTM_b(wandb.config)
    wandb.log({"score": score})

def start_c_n_step_DQTM_b():
    import wandb
    sweep_id = wandb.sweep(sweep=config_cartpole_n_step_DQTM_b, project="cartpole-n_step_DQTM_b-f")
    wandb.agent(sweep_id, function=main_c_n_step_DQTM_b, count=10_000)

def start_a_n_step_DQTM_b():
    import wandb
    sweep_id = wandb.sweep(sweep=config_acrobot_n_step_DQTM_b, project="acrobot-n_step_DQTM_b-f")
    wandb.agent(sweep_id, function=main_a_n_step_DQTM_b, count=10_000)
################################################
############ Double TMQN type a ################
################################################
def main_c_DQTM_a():
    wandb.init(project="cartpole-DQTM_a-f")
    score = cartpole_DQTM_a(wandb.config)
    wandb.log({"score": score})
def main_a_DQTM_a():
    wandb.init(project="acrobot-DQTM_a-f")
    score = acrobot_DQTM_a(wandb.config)
    wandb.log({"score": score})
def start_c_DQTM_a():
    import wandb
    sweep_id = wandb.sweep(sweep=config_cartpole_DQTM_a, project="cartpole-DQTM_a-f")
    wandb.agent(sweep_id, function=main_c_DQTM_a, count=10_000)

def start_a_DQTM_a():
    import wandb
    sweep_id = wandb.sweep(sweep=config_acrobot_DQTM_a, project="acrobot-DQTM_a-f")
    wandb.agent(sweep_id, function=main_a_DQTM_a, count=10_000)
################################################
############ Double TMQN type b ################
################################################
def main_c_DQTM_b():
    wandb.init(project="cartpole-DQTM_b-f")
    score = cartpole_DQTM_b(wandb.config)
    wandb.log({"score": score})
def main_a_DQTM_b():
    wandb.init(project="acrobot-DQTM_b-f")
    score = acrobot_DQTM_b(wandb.config)
    wandb.log({"score": score})
def start_c_DQTM_b():
    import wandb
    sweep_id = wandb.sweep(sweep=config_cartpole_DQTM_b, project="cartpole-DQTM_b-f")
    wandb.agent(sweep_id, function=main_c_DQTM_b, count=10_000)

def start_a_DQTM_b():
    import wandb
    sweep_id = wandb.sweep(sweep=config_acrobot_DQTM_b, project="acrobot-DQTM_b-f")
    wandb.agent(sweep_id, function=main_a_DQTM_b, count=10_000)
################################################
################# n-step TMQN  #################
################################################
def main_c_n_step_QTM():
    wandb.init(project="cartpole-n_step_QTM-f")
    score = cartpole_n_step_QTM(wandb.config)
    wandb.log({"score": score})
def main_a_n_step_QTM():
    wandb.init(project="acrobot-n_step_QTM-f")
    score = acrobot_n_step_QTM(wandb.config)
    wandb.log({"score": score})
def start_c_n_step_QTM():
    import wandb
    sweep_id = wandb.sweep(sweep=config_cartpole_n_step_QTM, project="cartpole-n_step_QTM-f")
    wandb.agent(sweep_id, function=main_c_n_step_QTM, count=10_000)

def start_a_n_step_QTM():
    import wandb
    sweep_id = wandb.sweep(sweep=config_acrobot_n_step_QTM, project="acrobot-n_step_QTM-f")
    wandb.agent(sweep_id, function=main_a_n_step_QTM, count=10_000)
################################################
################### TMQN  ######################
################################################
def main_c_QTM():
    wandb.init(project="cartpole-QTM-f")
    score = cartpole_QTM(wandb.config)
    wandb.log({"score": score})
def main_a_QTM():
    wandb.init(project="acrobot-QTM-f")
    score = acrobot_QTM(wandb.config)
    wandb.log({"score": score})
def start_c_QTM():
    import wandb
    sweep_id = wandb.sweep(sweep=config_cartpole_QTM, project="cartpole-QTM-f")
    wandb.agent(sweep_id, function=main_c_QTM, count=10_000)

def start_a_QTM():
    import wandb
    sweep_id = wandb.sweep(sweep=config_acrobot_QTM, project="acrobot-QTM-f")
    wandb.agent(sweep_id, function=main_a_QTM, count=10_000)