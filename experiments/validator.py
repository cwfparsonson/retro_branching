from retro_branching.utils import get_most_recent_checkpoint_foldername, gen_co_name
from retro_branching.networks import BipartiteGCN
from retro_branching.agents import Agent, REINFORCEAgent, PseudocostBranchingAgent, StrongBranchingAgent, RandomAgent, DQNAgent, DoubleDQNAgent
from retro_branching.environments import EcoleBranching, EcoleConfiguring
from retro_branching.validators import ReinforcementLearningValidator

import torch
import ecole
import numpy as np
import os
import shutil
import glob
import time
import gzip
import pickle
import copy

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import shutil
hydra.HYDRA_FULL_ERROR = 1


@hydra.main(config_path='configs', config_name='config.yaml')
def run(cfg: DictConfig):
     
    # initialise the agent
    agents = {}
    if cfg.experiment.agent_name not in set(['pseudocost_branching', 'strong_branching', 'scip_branching']):
        # is an ML agent
        path = cfg.experiment.path_to_load_agent + f'/{gen_co_name(cfg.instances.co_class, cfg.instances.co_class_kwargs)}/{cfg.experiment.agent_name}/'
        config = path + 'config.json'
        agent = Agent(device=cfg.experiment.device, config=config, name=cfg.experiment.agent_name)
        for network_name, network in agent.get_networks().items():
            if network is not None:
                try:
                    # see if network saved under same var as 'network_name'
                    agent.__dict__[network_name].load_state_dict(torch.load(path+f'/{network_name}_params.pkl', map_location=cfg.experiment.device))
                except KeyError:
                    # network saved under generic 'network' var (as in Agent class)
                    agent.__dict__['network'].load_state_dict(torch.load(path+f'/{network_name}_params.pkl', map_location=cfg.experiment.device))
            else:
                print(f'{network_name} is None.')
        agent.eval() # put in test mode
    else:
        # is a standard heuristic
        cfg.experiment.device = 'cpu'
        agent = cfg.experiment.agent_name
    path_to_save_baseline = cfg.experiment.path_to_save + f'/{gen_co_name(cfg.instances.co_class, cfg.instances.co_class_kwargs)}/{cfg.experiment.agent_name}/'
    agents[path_to_save_baseline] = agent
    print(f'Initialised agent and agent-to-path dict: {agents}')

    # run the agent on the validation instances
    start = time.time()
    for path in agents.keys():
        run_rl_validator(path, 
                          agents, 
                          cfg.experiment.device,
                          cfg.instances.co_class,
                          cfg.instances.co_class_kwargs,
                          cfg.environment.observation_function, 
                          cfg.environment.scip_params,
                          cfg.validator.threshold_difficulty, 
                          cfg.validator.max_steps, 
                          cfg.validator.max_steps_agent, 
                          cfg.validator.overwrite,
                          cfg.experiment.path_to_load_instances)
    end = time.time()

    print(f'Finished validating agent {cfg.experiment.agent_name} in {end-start:.3f} s.')



def run_rl_validator(path,
                     agents,
                     device,
                     co_class,
                     co_class_kwargs,
                     observation_function='default',
                     scip_params='gasse_2019',
                     threshold_difficulty=None, 
                     max_steps=int(1e12), 
                     max_steps_agent=None,
                     overwrite=False,
                     instances_path=None):
    '''
    Cannot pickle ecole objects, so if agent is e.g. 'strong_branching' or 'pseudocost_branching', need to give agent as str so
    can initialise inside this ray remote function.
    '''
    start = time.time()
    agent = agents[path]
    
    if type(agent) == str:
        if agent == 'pseudocost_branching':
            agent = PseudocostBranchingAgent(name='pseudocost_branching')
        elif agent == 'strong_branching':
            agent = StrongBranchingAgent(name='strong_branching')
        elif agent == 'scip_branching':
            class SCIPBranchingAgent:
                def __init__(self):
                    self.name = 'scip_branching'
            agent = SCIPBranchingAgent()
        else:
            raise Exception(f'Unrecognised agent str {agent}, cannot initialise.')
    
    if overwrite:
        # clear all old rl_validator/ folders even if would not be overwritten with current config to prevent testing inconsistencies
        paths = sorted(glob.glob(path+'rl_validator*'))
        for p in paths:
            print('Removing old {}'.format(p))
            shutil.rmtree(p)

    # instances
    if instances_path is None:
        instances_path = f'/scratch/datasets/retro_branching/instances/'
    instances_path += f'/{co_class}'
    for key, val in co_class_kwargs.items():
        instances_path += f'_{key}_{val}'
    files = glob.glob(instances_path+f'/*.mps')
    instances = iter([ecole.scip.Model.from_file(f) for f in files])
    print(instances)
    print(f'Loaded {len(files)} instances from path {instances_path}')

    # env
    if agent.name == 'scip_branching':
        env = EcoleConfiguring(observation_function=observation_function,
                               information_function='default',
                               scip_params=scip_params)
    else:
        env = EcoleBranching(observation_function=observation_function,
                             information_function='default',
                             reward_function='default',
                             scip_params=scip_params)
    env.seed(0)
    print('Initialised env.')

    # metrics
    metrics = ['num_nodes', 'solving_time', 'lp_iterations']
    print(f'Initialised metrics: {metrics}')

    # validator
    validator = ReinforcementLearningValidator(agents={agent.name: agent},
                                               envs={agent.name: env},
                                               instances=instances,
                                               metrics=metrics,
                                               calibration_config_path=None,
#                                                calibration_config_path='/home/zciccwf/phd_project/projects/retro_branching/scripts/',
                                               seed=0,
                                               max_steps=max_steps, # int(1e12), 10, 5, 3
                                               max_steps_agent=max_steps_agent,
                                               turn_off_heuristics=False,
                                               min_threshold_difficulty=None,
                                               max_threshold_difficulty=None, # None 250
                                               threshold_agent=None,
                                               threshold_env=None,
                                               episode_log_frequency=1,
                                               path_to_save=path,
                                               overwrite=overwrite,
                                               checkpoint_frequency=10)
    print(f'Initialised validator. Will save to: {validator.path_to_save}')

    # run validation tests
    validator.test(len(files))
    end = time.time()
    print(f'Finished path {path} validator in {end-start:.3f} s')

if __name__ == '__main__':
    run()
