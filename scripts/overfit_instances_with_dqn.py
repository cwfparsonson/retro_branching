from retro_branching.learners import DQNLearner
# from retro_branching.ray_learners import REINFORCELearner
# from retro_branching.multiprocessing_learners import REINFORCELearner
from retro_branching.agents import DoubleDQNAgent, StrongBranchingAgent
from retro_branching.networks import BipartiteGCN
from retro_branching.environments import EcoleBranching

import ecole
import pyscipopt

import torch

import matplotlib.pyplot as plt
import numpy as np
import pickle
import gzip
import os
import glob

import cProfile
import pstats


if __name__ == '__main__': 
    device = 'cuda:4'

    # init single instance for overfitting
    instances_path = 'instances_to_overfit/nrows_100_ncols_100/'
    instance_paths = sorted(glob.glob(f'{instances_path}/*.mps'))

    instances = pyscipopt.Model()
    instances.readProblem(instance_paths[9])
    instances = ecole.scip.Model.from_pyscipopt(instances)

    # value networks(s)
    init_value_network_path = None
    # init_value_network_path = '/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_265/checkpoint_305/trained_params.pkl'
    # init_value_network_path = '/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_266/checkpoint_235/trained_params.pkl'
    init_value_network_path = '/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_296/checkpoint_102/network_params.pkl'
    value_network_1 = BipartiteGCN(device=device,
                                emb_size=128,
                                num_rounds=2,
                                cons_nfeats=5,
                                edge_nfeats=1,
                                var_nfeats=19,
                                aggregator='add')
    value_network_2 = BipartiteGCN(device=device,
                                emb_size=128,
                                num_rounds=2,
                                cons_nfeats=5,
                                edge_nfeats=1,
                                var_nfeats=19,
                                aggregator='add')
    if init_value_network_path is not None:
        value_network_1.load_state_dict(retro_branching.load(init_value_network_path))
        value_network_2.load_state_dict(retro_branching.load(init_value_network_path))

    # exploration networks
    exploration_network = None

    # init agent
    agent = DoubleDQNAgent(device=device,
                          value_network_1=value_network_1,
                          value_network_2=value_network_2,
                          exploration_network=exploration_network,
                          sample_exploration_network_stochastically=True, # False True
                          name='dqn_gnn')
    agent.train()

    # init env
    env = EcoleBranching(observation_function='default', # 'default' 'label_solution_values'
                         information_function='default',
                         reward_function='default',
                         scip_params='default') # 'default' 'ml4co_item_placement' 'ml4co_load_balancing' 'ml4co_anonymous'

    # init learner
    learner = DQNLearner(agent=agent,
                        env=env,
                        instances=instances,
                        max_steps=int(1e12), # dont infinite loop while training
                        buffer_capacity=20000, # 20000
                        buffer_min_length=10000, # 500 10000
                        update_target_frequency=50, # 50
                        steps_per_update=1, # 25
                        prob_add_to_buffer=1, # 1.0 0.1
                        seed=0,
                        batch_size=64,
                        agent_reward='dual_bound_frac', # 'num_nodes' 'dual_bound_frac'
                        lr=1e-4,
                        gamma=0.99, # 0.99 0.9
                        initial_epsilon=1,
                        final_epsilon=0.0,
                        final_epsilon_epoch=10000, # 5000 10000
                        threshold_difficulty=None, # None 100
                        threshold_agent=None,
                        threshold_env=None,
                        episode_log_frequency=1,
                        checkpoint_frequency=1000,
                        # path_to_save=None, # '/scratch/datasets/retro_branching' None
                        # path_to_save='/scratch/datasets/retro_branching', # '/scratch/datasets/retro_branching' None
                        path_to_save=instances_path,
                        name='dqn_learner')
    print('Initialised learner with params {}. Will save to {}'.format(learner.episodes_log, learner.path_to_save))

    # train agent
    print('Training agent...')
    learner.train(int(100e3))
