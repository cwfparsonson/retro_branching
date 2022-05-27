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


class RetroBranchingMCTSBackprop:
    def __init__(self, 
                 use_binary_sparse_rewards=False,
                 normaliser='init_primal_bound', 
                 transform_with_log=False,
                 use_retro_trajectories=True,
                 leaf_type='brancher_closed',
                 debug_mode=False):
        '''
        Args:
            use_retro_trajectories (bool): If False, will return whole B&B tree as one
                big trajectory. If True, will divide into multiple trajectories.
            leaf_type ('brancher_closed', 'all'): If 'brancher_closed', will only
                consider trajectories whose leaf nodes were explicitly closed by 
                branching decision rather than activity elsewhere in the tree.
                If 'all', will consider all leaf nodes.
        '''
        self.normalised_lp_gain = retro_branching.src.rewards.normalised_lp_gain.NormalisedLPGain(use_binary_sparse_rewards=use_binary_sparse_rewards, 
                                                                                                  normaliser=normaliser) # normalised lp gain reward tracker
        self.transform_with_log = transform_with_log
        self.use_retro_trajectories = use_retro_trajectories 
        self.leaf_type = leaf_type

        self.debug_mode = debug_mode

    def before_reset(self, model):
        self.normalised_lp_gain.before_reset(model)

    def extract(self, model, done):
        # update normalised LP gain tracker
        _ = self.normalised_lp_gain.extract(model, done)

        if not done:
            return None
        else:
            # instance finished, retrospectively create subtree episode paths

            if self.normalised_lp_gain.tree.tree.graph['root_node'] is None:
                # instance was pre-solved
                return [{0: 0}]

            # collect sub-tree episodes
            subtrees_step_idx_to_reward = []

            # keep track of which nodes have been added to a sub-tree
            self.nodes_added = set()

            if self.debug_mode:
                print('\nB&B tree:')
                print(f'All nodes saved: {self.normalised_lp_gain.tree.tree.nodes()}')
                print(f'Visited nodes: {self.normalised_lp_gain.tree.tree.graph["visited_node_ids"]}')
                self.normalised_lp_gain.tree.render()

            # remove nodes which were never visited by the brancher and therefore do not have a score or next state
            nodes = [node for node in self.normalised_lp_gain.tree.tree.nodes]
            for node in nodes:
                if 'score' not in self.normalised_lp_gain.tree.tree.nodes[node]:
                    # node never visited by brancher -> do not consider
                    self.normalised_lp_gain.tree.tree.remove_node(node)
                    if node in self.normalised_lp_gain.tree.tree.graph['visited_node_ids']:
                        # hack: SCIP sometimes returns large int rather than None node_id when episode finished
                        # since never visited this node (since no score assigned), do not count this node as having been visited when calculating paths below
                        if self.debug_mode:
                            print(f'Removing node {node} since was never visited by brancher.')
                        self.normalised_lp_gain.tree.tree.graph['visited_node_ids'].remove(node)

            # map which nodes were visited at which step in episode
            self.visited_nodes_to_step_idx = {node: idx for idx, node in enumerate(self.normalised_lp_gain.tree.tree.graph['visited_node_ids'])}

            # get root node
            root_node = list(self.normalised_lp_gain.tree.tree.graph['root_node'].keys())[0]

            # get sub-trees from root node to leaf nodes (out degree == 0) which were closed by agent (score == 0)
            if self.leaf_type == 'brancher_closed':
                # only use leaf nodes which were explicitly closed by brancher
                leaf_nodes = [node for node in self.normalised_lp_gain.tree.tree.nodes() if self.normalised_lp_gain.tree.tree.out_degree(node) == 0 and self.normalised_lp_gain.tree.tree.nodes[node]['score'] == 0]
            elif self.leaf_type == 'all':
                # use all leaf nodes
                leaf_nodes = [node for node in self.normalised_lp_gain.tree.tree.nodes() if self.normalised_lp_gain.tree.tree.out_degree(node) == 0]
            else:
                raise Exception(f'Unrecognised leaf_type {self.leaf_type}.')
            terminal_paths = [shortest_path(self.normalised_lp_gain.tree.tree, source=root_node, target=leaf_node) for leaf_node in leaf_nodes]
            
            # keep track of backpropagated values for each node in path
            node_to_backprop_value = {node: 0 for node in [node for path in terminal_paths for node in path]}
            node_to_num_visits = defaultdict(lambda: 0)
            for path in terminal_paths:
                for node in path:
                    node_to_num_visits[node] += 1

            # # # NEW #2: backpropagate values with summing from current node down method
            # for node in node_to_backprop_value.keys():
                # # get subtree beneath this node
                # subtree_rooted_at_node = [n for n in dfs_tree(self.normalised_lp_gain.tree.tree, node) if n in node_to_backprop_value.keys()]

                # # get total return from this node across all subsequent nodes
                # R = sum([self.normalised_lp_gain.tree.tree.nodes[n]['score'] for n in subtree_rooted_at_node])

                # # use mean return as backprop value 
                # node_to_backprop_value = R / node_to_num_visits[node]

            # # NEW #1: backpropagate values with MCTS method
            for path in terminal_paths:
                # get total return of trajectory
                R = sum([self.normalised_lp_gain.tree.tree.nodes[n]['score'] for n in path])

                # update backprop value of each node in path
                for node in path:
                    node_to_backprop_value[node] += R
                    node_to_num_visits[node] += 1

            # # OLD: backpropagate values
            # for path in terminal_paths:
                # for node in path:
                    # # sum values beneath node to get backprop value at this node
                    # backprop_value = sum([self.normalised_lp_gain.tree.tree.nodes[n]['score'] for n in path])

                    # # update backprop value of node
                    # node_to_backprop_value[node] += backprop_value
                    # node_to_num_visits[node] += 1

            for node in node_to_backprop_value.keys():
                # calculate mean backprop value of node
                node_to_backprop_value[node] /= node_to_num_visits[node]
                if self.transform_with_log:
                    sign = math.copysign(1, node_to_backprop_value[node])
                    node_to_backprop_value[node] = sign * math.log(1 + abs(node_to_backprop_value[node]), 10)

            if self.use_retro_trajectories:
                # create subtrees with backpropagated values
                for path in terminal_paths:
                    # get back propagated value at each step in sub-tree episode
                    path_node_rewards = [node_to_backprop_value[node] for node in path]

                    # get episode step indices at which each node in sub-tree was visited
                    path_to_step_idx = {node: self.visited_nodes_to_step_idx[node] for node in path}

                    # map each path node episode step idx to its corresponding backpropagated value
                    step_idx_to_reward = {step_idx: r for step_idx, r in zip(list(path_to_step_idx.values()), path_node_rewards)}
                    subtrees_step_idx_to_reward.append(step_idx_to_reward)
            else:
                # treat whole B&B tree as one big tree with backpropagated values
                # step_idx_to_node = {idx: node for node, idx in self.visited_nodes_to_step_idx.items()}
                step_idx_to_reward = {}
                for node, step_idx in self.visited_nodes_to_step_idx.items():
                    step_idx_to_reward[step_idx] = node_to_backprop_value[node]
                subtrees_step_idx_to_reward.append(step_idx_to_reward)

            if self.debug_mode:
                print(f'visited_nodes_to_step_idx: {self.visited_nodes_to_step_idx}')
                print(f'node_to_num_visits: {node_to_num_visits}')
                print(f'node_to_backprop_value: {node_to_backprop_value}')
                step_idx_to_visited_nodes = {val: key for key, val in self.visited_nodes_to_step_idx.items()}
                for i, ep in enumerate(subtrees_step_idx_to_reward):
                    print(f'>>> sub-tree episode {i+1}: {ep}')
                    ep_path = [step_idx_to_visited_nodes[idx] for idx in ep.keys()]
                    print(f'path: {ep_path}')

            if len(subtrees_step_idx_to_reward) == 0:
                # solved at root so path length < min path length so was never added to subtrees
                return [{0: 0}]
            else:
                return subtrees_step_idx_to_reward