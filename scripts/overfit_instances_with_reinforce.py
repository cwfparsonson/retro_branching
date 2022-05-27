from retro_branching.learners import REINFORCELearner
# from retro_branching.ray_learners import REINFORCELearner
# from retro_branching.multiprocessing_learners import REINFORCELearner
from retro_branching.agents import REINFORCEAgent, StrongBranchingAgent
from retro_branching.networks import BipartiteGCN
from retro_branching.environments import EcoleBranching

import pyscipopt
import torch
import ecole
import matplotlib.pyplot as plt
import numpy as np
import pickle
import gzip
import os
import glob

import cProfile
import pstats


if __name__ == '__main__':
    # init single instance for overfitting
    instances_path = 'instances_to_overfit/nrows_500_ncols_1000/'
    instance_paths = sorted(glob.glob(f'{instances_path}/*.mps'))

    # for instance_path in instance_paths:

    instances = pyscipopt.Model()
    instances.readProblem(instance_paths[9])
    instances = ecole.scip.Model.from_pyscipopt(instances)
    print('Initialised instance.')

    RLGNN_DEVICE = 'cuda:5'

    # policy networks
    policy_network = BipartiteGCN(RLGNN_DEVICE,
                               emb_size=128,
                               num_rounds=2,
                               cons_nfeats=5,
                               edge_nfeats=1,
                               var_nfeats=19, # 19 20
                               aggregator='add')
    policy_network.load_state_dict(
        retro_branching.load('/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_266/checkpoint_235/trained_params.pkl'))

    # init agent
    rlgnn_agent = REINFORCEAgent(policy_network=policy_network, 
                                 device=RLGNN_DEVICE, 
                                 temperature=1.0,
                                 name='rl_gnn')
    rlgnn_agent.train() # turn on train mode

    # init env
    env = EcoleBranching(observation_function='default', # 'default' 'label_solution_values'
                         information_function='default',
                         reward_function='default',
                         scip_params='default') # 'default' 'ml4co_item_placement' 'ml4co_load_balancing' 'ml4co_anonymous'

    # init learner
    learner = REINFORCELearner(agent=rlgnn_agent,
                               env=env,
                               instances=instances,
                               seed=0,
                               max_steps=int(1e12), # 5000 10 5 3
                               max_steps_agent=None,
                               batch_size=512, # 512 32 16 8 1
                               baseline='mean', # None 'sb' 'mean' 'pc' 'gr' 'sr'
                               agent_reward='num_nodes', # 'num_nodes' 'primal_dual_integral' 'dual_integral' 'dual_bound' 'primal_dual_gap' 'primal_dual_gap_frac' 'dual_bound_frac'
                               lr=1e-3,
                               gamma=0.99,
                               turn_off_heuristics=False,
                               threshold_difficulty=None, # None 250 100 50 75 30
                               threshold_agent=None,
                               threshold_env=None,
                               action_filter_agent=None, # None StrongBranchingAgent()
                               action_filter_percentile=10, # 90
                               validation_frequency=None,
                               episode_log_frequency=1,
                               # path_to_save='/scratch/datasets/retro_branching',
                               path_to_save=instances_path,
                               checkpoint_frequency=100,
                               )
    print('Initialised learner with params {}. Will save to {}'.format(learner.episodes_log, learner.path_to_save))

    # train agent
    print('Training agent...')
    learner.train(100e3)
