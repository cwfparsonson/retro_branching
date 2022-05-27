import retro_branching
from retro_branching.rewards import DualBound, PrimalBound, PrimalDualGap, PrimalDualGapFrac, DualBoundFrac, PrimalDualBoundFracSum, PrimalBoundFrac, PrimalBoundGapFrac, DualBoundGapFrac, NormalisedLPGain, OptimalRetroTrajectoryNormalisedLPGain, RetroBranching, BinarySolved, RetroBranchingMCTSBackprop, BinaryFathomed
from retro_branching.observations import NodeBipariteWithSolutionLabels, NodeBipariteWith28VariableFeatures, NodeBipariteWith45VariableFeatures, NodeBipariteWith40VariableFeatures, NodeBipariteWith43VariableFeatures, NodeBipariteWith24VariableFeatures, NodeBipariteWith29VariableFeatures, NodeBipariteWith37VariableFeatures, NodeBipariteWithCustomFeatures
from retro_branching.scip_params import default_scip_params, gasse_2019_scip_params, bfs_scip_params, dfs_scip_params, uct_scip_params, ml4co_anonymous_scip_params, ml4co_item_placement_scip_params, ml4co_load_balancing_scip_params
# from retro_branching.environments import EcoleConfiguring

import networkx as nx
import gym
import copy
import math
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
import time
import pandas as pd
from IPython.display import display
import ecole

class EcoleBranching(ecole.environment.Branching):
    def __init__(
        self,
        observation_function='default',
        information_function='default',
        reward_function='default',
        scip_params='default',
        pseudo_candidates=False,
    ):
        '''
        Args:
            observation_function (dict, str): Custom dict or one of: 'default',
                'label_solution_values', '28_var_features', '45_var_features',
                '40_var_features'
            scip_params (dict, str): Custom dictionary or one of: 'default', 
                'ml4co_item_placement', 'ml4co_load_balancing', 'ml4co_anonymous',
                'gasse_2019', 'dfs', 'bfs', 'uct'

        More info on node selection in SCIP: https://www.mdpi.com/2673-2688/2/2/10
        '''
        # save string names so easy to initialise new environments
        if type(observation_function) == str:
            self.str_observation_function = observation_function
        else:
            self.str_observation_function = None
        if type(information_function) == str:
            self.str_information_function = information_function
        else:
            self.str_information_function = None
        if type(reward_function) == str:
            self.str_reward_function = reward_function
        else:
            self.str_reward_function = None
        if type(scip_params) == str:
            self.str_scip_params = scip_params
        else:
            self.str_scip_params = None

        self.pseudo_candidates = pseudo_candidates

        # init default rewards
        _reward_function = {
                           'num_nodes': -ecole.reward.NNodes(),
                           'lp_iterations': -ecole.reward.LpIterations(),
                           'solving_time': -ecole.reward.SolvingTime(),
                           } 
        if reward_function == 'default':
            pass
        elif reward_function == 'primal_integral':
            _reward_function['primal_integral'] = -ecole.reward.PrimalIntegral()
        elif reward_function == 'dual_integral':
            _reward_function['dual_integral'] = -ecole.reward.DualIntegral()
        elif reward_function == 'primal_dual_integral':
            _reward_function['primal_dual_integral'] = -ecole.reward.PrimalDualIntegral()
        elif reward_function == 'primal_dual_integral':
            _reward_function['primal_dual_integral'] = -ecole.reward.PrimalDualIntegral()
        elif reward_function == 'primal_bound':
             _reward_function['primal_bound'] = PrimalBound(sense=-1)
        elif reward_function == 'dual_bound':
            _reward_function['dual_bound'] = DualBound(sense=-1)
        elif reward_function == 'primal_dual_gap':
             _reward_function['primal_dual_gap'] = PrimalDualGap()
        elif reward_function == 'dual_bound_frac':
             _reward_function['dual_bound_frac'] = DualBoundFrac(sense=-1)
        elif reward_function == 'primal_dual_frac':
             _reward_function['primal_bound_frac'] = PrimalBoundFrac(sense=-1)
        elif reward_function == 'primal_dual_gap_frac':
             _reward_function['primal_dual_gap_frac'] = PrimalDualGapFrac()
        elif reward_function == 'primal_dual_bound_frac_sum':
             _reward_function['primal_dual_bound_frac_sum'] = PrimalDualBoundFracSum()
        elif reward_function == 'primal_dual_gap_frac':
             _reward_function['primal_bound_gap_frac'] = PrimalBoundGapFrac(sense=-1)
        elif reward_function == 'dual_bound_gap_frac':
             _reward_function['dual_bound_gap_frac'] = DualBoundGapFrac(sense=-1)
        elif reward_function == 'normalised_lp_gain':
             _reward_function['normalised_lp_gain'] = NormalisedLPGain(normaliser='init_primal_bound', transform_with_log=False)
        elif reward_function == 'optimal_retro_trajectory_normalised_lp_gain':
             _reward_function['optimal_retro_trajectory_normalised_lp_gain'] = OptimalRetroTrajectoryNormalisedLPGain(normaliser='init_primal_bound')
        elif reward_function == 'retro_binary_fathomed':
             _reward_function['retro_binary_fathomed'] = RetroBranching(force_include_optimal_trajectory=False,
                                                                         force_include_last_visited_trajectory=False,
                                                                         force_include_branching_fathomed_transitions=True,
                                                                         only_return_optimal_trajectory=False,
                                                                         only_use_leaves_closed_by_brancher_as_terminal_nodes=False,
                                                                         set_score_as_subtree_size=False, # False
                                                                         set_terminal_node_score_as_retrospective_subtree_size=False, # False
                                                                         use_binary_sparse_rewards=True, # True
                                                                         normaliser='init_primal_bound', 
                                                                         min_subtree_depth=1,
                                                                         retro_trajectory_construction='max_leaf_lp_gain', # 'max_leaf_lp_gain' 'visitation_order'
                                                                         remove_nonoptimal_fathomed_leaves=False,
                                                                         use_mean_return_rooted_at_node=False,
                                                                         use_sum_return_rooted_at_node=False,
                                                                         use_retro_trajectories=True, # True
                                                                         multiarmed_bandit=False, # False
                                                                         transform_with_log=False)
        elif reward_function == 'retro_fmsts':
             _reward_function['retro_fmsts'] = RetroBranching(force_include_optimal_trajectory=False,
                                                             force_include_last_visited_trajectory=False,
                                                             force_include_branching_fathomed_transitions=True,
                                                             only_return_optimal_trajectory=False,
                                                             only_use_leaves_closed_by_brancher_as_terminal_nodes=False,
                                                             set_score_as_subtree_size=False, # False
                                                             set_terminal_node_score_as_retrospective_subtree_size=True, # False
                                                             use_binary_sparse_rewards=True, # True
                                                             normaliser='init_primal_bound', 
                                                             min_subtree_depth=1,
                                                             retro_trajectory_construction='visitation_order', # 'max_leaf_lp_gain' 'visitation_order'
                                                             remove_nonoptimal_fathomed_leaves=False,
                                                             use_mean_return_rooted_at_node=False,
                                                             use_sum_return_rooted_at_node=False,
                                                             use_retro_trajectories=True, # True
                                                             multiarmed_bandit=False, # False
                                                             transform_with_log=False)
        elif reward_function == 'retro_branching_mcts_backprop':
            _reward_function['retro_branching_mcts_backprop'] = RetroBranchingMCTSBackprop(use_binary_sparse_rewards=True, 
                                                                                           normaliser='init_primal_bound', 
                                                                                           transform_with_log=False, 
                                                                                           use_retro_trajectories=False, 
                                                                                           leaf_type='all'),
        elif reward_function == 'binary_solved':
             _reward_function['binary_solved'] = BinarySolved(solved=0, not_solved=-1)
        elif reward_function == 'binary_fathomed':
             _reward_function['binary_fathomed'] = BinaryFathomed(fathomed=0, not_fathomed=-1)
        else:
            raise Exception(f'Unrecognised reward_function {reward_function}')
        reward_function = _reward_function

        if information_function == 'default':
            information_function=({
                     'num_nodes': ecole.reward.NNodes().cumsum(),
                     'lp_iterations': ecole.reward.LpIterations().cumsum(),
                     'solving_time': ecole.reward.SolvingTime().cumsum(),
                     # 'primal_integral': ecole.reward.PrimalIntegral().cumsum(),
                     # 'dual_integral': ecole.reward.DualIntegral().cumsum(),
                     # 'primal_dual_integral': ecole.reward.PrimalDualIntegral(),
                 })
        else:
            raise Exception(f'Unrecognised information_function {information_function}')

        if observation_function == 'default':    
            observation_function = (ecole.observation.NodeBipartite())
        elif observation_function == 'label_solution_values':
            presolve_env = retro_branching.src.environments.ecole_configuring.EcoleConfiguring()
            observation_function = (NodeBipariteWithSolutionLabels(presolve_env=presolve_env))
        elif observation_function == '24_var_features':
            observation_function = (NodeBipariteWith24VariableFeatures())
        elif observation_function == '28_var_features':
            observation_function = (NodeBipariteWith28VariableFeatures())
        elif observation_function == '45_var_features':
            observation_function = (NodeBipariteWith45VariableFeatures())
        elif observation_function == '40_var_features':
            observation_function = (NodeBipariteWith40VariableFeatures())
        elif observation_function == '43_var_features':
            observation_function = (NodeBipariteWith43VariableFeatures())
        elif observation_function == '29_var_features':
            observation_function = (NodeBipariteWith29VariableFeatures())
        elif observation_function == '37_var_features':
            observation_function = (NodeBipariteWith37VariableFeatures())
        elif observation_function == 'custom_var_features':
            observation_function = (NodeBipariteWithCustomFeatures())
        else:
            raise Exception(f'Unrecognised observation_function {observation_function}')

        if scip_params == 'default':
            scip_params = default_scip_params
        elif scip_params == 'ml4co_item_placement':
            scip_params = ml4co_item_placement_scip_params
        elif scip_params == 'ml4co_load_balancing':
            scip_params = ml4co_load_balancing_scip_params 
        elif scip_params == 'ml4co_anonymous':
            scip_params = ml4co_anonymous_scip_params 
        elif scip_params == 'gasse_2019':
            scip_params = gasse_2019_scip_params
        elif scip_params == 'dfs':
            scip_params = dfs_scip_params
        elif scip_params == 'bfs':
            scip_params = bfs_scip_params
        elif scip_params == 'uct':
            scip_params = uct_scip_params
        else:
            raise Exception(f'Unrecognised scip_params {scip_params}')
        
        super(EcoleBranching, self).__init__(
            observation_function=observation_function,
            information_function=information_function,
            reward_function=reward_function,
            scip_params=scip_params,
            pseudo_candidates=pseudo_candidates,
        )
