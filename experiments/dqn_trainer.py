from retro_branching.utils import check_if_network_params_equal, seed_stochastic_modules_globally
from retro_branching.networks import BipartiteGCN
from retro_branching.agents import DQNAgent
from retro_branching.environments import EcoleBranching
from retro_branching.learners import DQNLearner

import ecole
import torch 

import random

import os
import argparse

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import shutil
hydra.HYDRA_FULL_ERROR = 1


@hydra.main(config_path='configs', config_name='config.yaml')
def run(cfg: DictConfig):
    # seeding
    if 'seed' not in cfg.experiment:
        cfg.experiment['seed'] = random.randint(0, 10000)
    seed_stochastic_modules_globally(cfg.experiment.seed)

    # print info
    print('\n\n\n')
    print(f'~'*80)
    print(f'Config:\n{OmegaConf.to_yaml(cfg)}')
    print(f'~'*80)

    # initialise DQN value network
    value_network = BipartiteGCN(device=cfg.experiment.device, **cfg.network)
    if cfg.network.init_value_network_path is not None:
        value_network.load_state_dict(torch.load(cfg.network.init_value_network_path, map_location=value_network.device))
    if cfg.network.reinitialise_heads:
        value_network.init_model_parameters(init_gnn_params=False, init_heads_params=True)
    print(f'Initialised DQN value network.')

    # initialise DQN agent in train mode
    agent = DQNAgent(device=cfg.experiment.device, value_network=value_network, **cfg.agent)
    agent.train()
    print(f'Initialised DQN agent.')
    
    # initialise instance generator
    if 'path_to_instances' in cfg.instances:
        instances = ecole.instance.FileGenerator(cfg.instances.path_to_instances, sampling_mode=cfg.instances.sampling_mode)
    else:
        if cfg.instances.co_class == 'set_covering':
            instances = ecole.instance.SetCoverGenerator(**cfg.instances.co_class_kwargs)
        elif cfg.instances.co_class == 'combinatorial_auction':
            instances = ecole.instance.CombinatorialAuctionGenerator(**cfg.instances.co_class_kwargs)
        elif cfg.instances.co_class == 'capacitated_facility_location':
            instances = ecole.instance.CapacitatedFacilityLocationGenerator(**cfg.instances.co_class_kwargs)
        elif cfg.instances.co_class == 'maximum_independent_set':
            instances = ecole.instance.IndependentSetGenerator(**cfg.instances.co_class_kwargs)
        else:
            raise Exception(f'Unrecognised co_class {cfg.instances.co_class}')
    print(f'Initialised instance generator.')
        
    # initialise branch-and-bound environment
    env = EcoleBranching(observation_function=cfg.environment.observation_function,
                         information_function=cfg.environment.information_function,
                         reward_function=cfg.environment.reward_function,
                         scip_params=cfg.environment.scip_params)
    print(f'Initialised environment.')

    # initialise DQN learner
    learner = DQNLearner(agent=agent,
                         env=env,
                         instances=instances,
                         ecole_seed=cfg.experiment.seed,
                         **cfg.learner)
    print(f'Initialised learner.')

    # train the DQN agent
    print('Training DQN agent...')
    learner.train(cfg.experiment.num_epochs)


if __name__ == '__main__':
    run()
