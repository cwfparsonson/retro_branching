import ecole
import torch 

import retro_branching
from retro_branching.utils import SearchTree
from retro_branching.networks import BipartiteGCN
from retro_branching.observations import NodeBipariteWith43VariableFeatures
from retro_branching.loss_functions import MeanSquaredError
# from ecole.typing import RewardFunction

import copy
import math
import random
import numpy as np
from collections import deque, defaultdict

from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.traversal.depth_first_search import dfs_tree

class OptimalRetroTrajectoryNormalisedLPGain:
    def __init__(self, normaliser='init_primal_bound'):
        '''
        Waits until end of episode to calculate rewards for each step, then retrospectively
        goes back through each step in the episode and calculates reward for that step.
        I.e. reward returned will be None until the end of the episode, at which
        point a dict mapping episode_step_idx for optimal path nodes to reward will be returned.

        Here, optimal path nodes are any node in the path from the root node
        to the optimal node at the end of the episode.
        
        Args:
            normaliser ('init_primal_bound', 'curr_primal_bound'): What to normalise
                with respect to in the numerator and denominator to calculate
                the per-step normalsed LP gain reward.
        '''
        self.normalised_lp_gain = retro_branching.src.rewards.normalised_lp_gain.NormalisedLPGain(normaliser=normaliser) # normalised lp gain reward tracker

    def before_reset(self, model):
        self.started = False
        self.normalised_lp_gain.before_reset(model)

    def extract(self, model, done):
        # update normalised LP gain tracker
        _ = self.normalised_lp_gain.extract(model, done)

        # m = model.as_pyscipopt()
        # curr_node = m.getCurrentNode()
        # if not self.started:
            # if curr_node is not None:
                # self.started = True
            # return None
        
        # if curr_node is not None:
            # # instance not yet finished
            # return None
        if not done:
            return None
        else:
            # instance finished, retrospectively compute rewards at each step
            if self.normalised_lp_gain.tree.tree.graph['root_node'] is None:
                # instance was pre-solved
                return [{0: 0}]

            # get root and optimal nodes
            root_node = list(self.normalised_lp_gain.tree.tree.graph['root_node'].keys())[0]
            optimal_node = self.normalised_lp_gain.tree.tree.graph['visited_node_ids'][-1]
            if 'score' not in self.normalised_lp_gain.tree.tree.nodes[optimal_node]:
                # hack: SCIP sometimes returns large int rather than None node_id when episode finished, which causes it to be recorded as visited by search tree
                # since never visited this node (since no score assigned), do not count this node as having been visited when calculating paths below
                optimal_node = self.normalised_lp_gain.tree.tree.graph['visited_node_ids'][-2]

            # get path from root to optimal node
            optimal_path = shortest_path(self.normalised_lp_gain.tree.tree, source=root_node, target=optimal_node)

            # gather rewards for each transition in optimal path up to optimal node
            rewards = [self.normalised_lp_gain.tree.tree.nodes[node]['score'] for node in optimal_path]

            # map which nodes were visited at which step in episode
            visited_nodes = [list(node.keys())[0] for node in self.normalised_lp_gain.tree.tree.graph['visited_nodes']]
            visited_nodes_to_step_idx = {node: idx for idx, node in enumerate(visited_nodes)}

            # get episode step indices at which each optimal path node was visited
            optimal_path_to_step_idx = {node: visited_nodes_to_step_idx[node] for node in optimal_path}

            # map each optimal path node episode step idx to its corresponding retrospectively calculated reward
            optimal_step_idx_to_reward = {step_idx: r for step_idx, r in zip(list(optimal_path_to_step_idx.values()), rewards)}

            return [optimal_step_idx_to_reward]