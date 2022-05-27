from retro_branching.rewards import DualBound, PrimalBound, PrimalDualGap, PrimalDualGapFrac, DualBoundFrac, PrimalDualBoundFracSum, PrimalBoundFrac, PrimalBoundGapFrac, DualBoundGapFrac, NormalisedLPGain, OptimalRetroTrajectoryNormalisedLPGain, RetroBranching, BinarySolved, RetroBranchingMCTSBackprop, BinaryFathomed
from retro_branching.observations import NodeBipariteWithSolutionLabels, NodeBipariteWith28VariableFeatures, NodeBipariteWith45VariableFeatures, NodeBipariteWith40VariableFeatures, NodeBipariteWith43VariableFeatures, NodeBipariteWith24VariableFeatures, NodeBipariteWith29VariableFeatures, NodeBipariteWith37VariableFeatures, NodeBipariteWithCustomFeatures
from retro_branching.scip_params import default_scip_params, gasse_2019_scip_params, bfs_scip_params, dfs_scip_params, uct_scip_params, ml4co_anonymous_scip_params, ml4co_item_placement_scip_params, ml4co_load_balancing_scip_params

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

class EcoleConfiguring(ecole.environment.Configuring):
    def __init__(self,
                 observation_function='default',
                 information_function='default',
                 scip_params='default'):
        '''
        Args:
            scip_params (dict, str): Custom dictionary or one of: 'default', 
                'ml4co_item_placement', 'ml4co_load_balancing', 'ml4co_anonymous'.
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
        if type(scip_params) == str:
            self.str_scip_params = scip_params
        else:
            self.str_scip_params = None

        # init functions from strings if needed
        if information_function == 'default':
            information_function=({
                     'num_nodes': ecole.reward.NNodes().cumsum(),
                     'lp_iterations': ecole.reward.LpIterations().cumsum(),
                     'solving_time': ecole.reward.SolvingTime().cumsum(),
                 })
        if observation_function == 'default':    
            observation_function=None
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
        else:
            raise Exception(f'Unrecognised scip_params {scip_params}')

        super(EcoleConfiguring, self).__init__(observation_function=observation_function,
                                               information_function=information_function,
                                               scip_params=scip_params)