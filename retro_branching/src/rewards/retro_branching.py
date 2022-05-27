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

class RetroBranching:
    def __init__(self, 
                 force_include_optimal_trajectory=True,
                 force_include_last_visited_trajectory=False,
                 force_include_branching_fathomed_transitions=False,
                 only_return_optimal_trajectory=False,
                 only_use_leaves_closed_by_brancher_as_terminal_nodes=False,
                 set_score_as_subtree_size=False,
                 set_terminal_node_score_as_retrospective_subtree_size=False,
                 use_binary_sparse_rewards=False,
                 normaliser='init_primal_bound', 
                 min_subtree_depth=1, 
                 retro_trajectory_construction='deepest',
                 remove_nonoptimal_fathomed_leaves=False,
                 use_mean_return_rooted_at_node=False,
                 use_sum_return_rooted_at_node=False,
                 multiarmed_bandit=False,
                 transform_with_log=False,
                 use_retro_trajectories=True,
                 debug_mode=False):
        '''
        Waits until end of episode to calculate rewards for each step, then retrospectively
        goes back through each step in the episode and calculates reward for that step.
        I.e. reward returned will be None until the end of the episode, at which
        point a dict mapping episode_step_idx for optimal path nodes to reward will be returned.

        Args:
            force_include_branching_fathomed_transitions (bool): If True, will include
                a sub-tree episode if the final transition was a sub-tree closed by
                the brancher even if they total episode length is < min_subtree_depth.
            use_binary_sparse_rewards (bool): If True, rather than LP-gain, will simply
                return a 1 on the step that solves the instance and a 0 on all
                other steps. Implemented this here to save writing another class
                which tracks sub-trees.
            normaliser ('init_primal_bound', 'curr_primal_bound'): What to normalise
                with respect to in the numerator and denominator to calculate
                the per-step normalsed LP gain reward.
            min_subtree_depth (int): Minimum depth of sub-tree (i.e. minimum length of sub-tree episode).
            retro_trajectory_construction ('random', 'deepest', 'shortest', 'max_lp_gain', 'min_lp_gain', 'max_leaf_lp_gain', 
                'reverse_visitation_order', 'visitation_order'): Which policy to use when choosing a leaf node as the final 
                node to construct a sub-tree.
            remove_nonoptimal_fathomed_leaves (bool): If True, at end of episode, will remove
                all leaves in tree which were fathomed (had score == 0) except optimal path leaf 
                so that no non-optimal sub-tree will have been fathomed and contain a 
                node/experience with score == 0.
            use_mean_return_rooted_at_node (bool): If True, for each step in a given
                sub-tree episode, rather than just using the original step reward,
                will:
                    1) get the sub-tree rooted at the current node (step)
                    2) get the paths from the sub-tree root to each of its leaves
                    3) sum the reward across each of these paths to get a backprop'd return from the current node
                    4) divide this by the number of these paths to get the mean backprop'd return from the current node
            set_score_as_subtree_size: If True, will set score at each step as being negative size of total sub-tree
                rooted at current node. I.e. if in DFS setting, sub-tree can contain multiple DFS-sub-trees.
            set_terminal_node_score_as_retrospective_subtree_size: If True, will set score at each terminal as being negative size
                of total *retrospectively defined* sub-tree rooted at current node. All non-terminal steps will have a score of 0. E.g. if in DFS setting,
                terminal node score will be negative size of DFS-sub-tree. This is similar to the reward of Etheve et al. 2020.
            use_retro_trajectories (bool): If False, will return dict mapping before forming sub-tree episodes.
            multiarmed_bandit (bool): If True, will make each subtree episode a 1-step MDP.
                N.B. use_retro_trajectories must be True and use_binary_sparse_rewards False if multiarmed_bandit is True.

        '''
        if use_mean_return_rooted_at_node and use_sum_return_rooted_at_node:
            raise Exception('use_mean_return_rooted_at_node and use_sum_return_rooted_at_node cannot both be True.')
        if force_include_optimal_trajectory and force_include_last_visited_trajectory:
            raise Exception('Cannot have both force_include_optimal_trajectory and force_include_last_visited_trajectory as True.')
        if multiarmed_bandit:
            if not use_retro_trajectories:
                raise Exception(f'Must set use_retro_trajectories=True to use multi-armed bandit.')
            if use_binary_sparse_rewards:
                raise Exception(f'Must set use_binary_sparse_rewards=False to use multi-armed bandit.')
            if min_subtree_depth != 1:
                raise Exception(f'Must set min_subtree_depth=1 to use multi-armed bandit.')

        self.force_include_optimal_trajectory = force_include_optimal_trajectory
        self.force_include_last_visited_trajectory = force_include_last_visited_trajectory
        self.force_include_branching_fathomed_transitions = force_include_branching_fathomed_transitions
        self.only_return_optimal_trajectory = only_return_optimal_trajectory
        self.only_use_leaves_closed_by_brancher_as_terminal_nodes = only_use_leaves_closed_by_brancher_as_terminal_nodes
        if not force_include_optimal_trajectory and only_return_optimal_trajectory:
            raise Exception('Must force inclusion of optimal sub-tree if only want to return optimal sub-tree.')
        self.set_score_as_subtree_size = set_score_as_subtree_size
        self.set_terminal_node_score_as_retrospective_subtree_size = set_terminal_node_score_as_retrospective_subtree_size
        self.use_binary_sparse_rewards = use_binary_sparse_rewards
        self.min_subtree_depth = min_subtree_depth
        self.retro_trajectory_construction = retro_trajectory_construction
        self.normalised_lp_gain = retro_branching.src.rewards.normalised_lp_gain.NormalisedLPGain(use_binary_sparse_rewards=use_binary_sparse_rewards, 
                                                                                                  normaliser=normaliser) # normalised lp gain reward tracker
        self.remove_nonoptimal_fathomed_leaves = remove_nonoptimal_fathomed_leaves
        self.use_mean_return_rooted_at_node = use_mean_return_rooted_at_node
        self.use_sum_return_rooted_at_node = use_sum_return_rooted_at_node
        self.transform_with_log = transform_with_log
        self.use_retro_trajectories = use_retro_trajectories
        self.multiarmed_bandit = multiarmed_bandit
        self.debug_mode = debug_mode

    def before_reset(self, model):
        self.started = False
        self.normalised_lp_gain.before_reset(model)
        
    def get_path_node_scores(self, tree, path):
        if self.use_mean_return_rooted_at_node or self.use_sum_return_rooted_at_node:
            scores = []
            # use mean return from node across all paths as score for each node
            for root_node in path:
                # get sub-tree rooted from this node
                subtree = dfs_tree(tree, root_node)

                # get leaf nodes of sub-tree
                leaf_nodes = [n for n in subtree.nodes() if subtree.out_degree(n) == 0]

                # get paths in sub-tree
                subtree_paths = [shortest_path(subtree, source=root_node, target=leaf_node) for leaf_node in leaf_nodes]

                # get total reward across all paths
                R = 0
                for p in subtree_paths:
                    for n in p:
                        R += tree.nodes[n]['score']

                if self.use_mean_return_rooted_at_node:
                    # get mean return across all paths from current node -> use as node score
                    R /= len(subtree_paths)
                elif self.use_sum_return_rooted_at_node:
                    # already summed
                    pass
                else:
                    raise Exception('Not implemented.')

                scores.append(R)

        else:
            # use original node score as score for each node
            scores = [tree.nodes[node]['score'] for node in path]

        if self.transform_with_log:
            for idx, score in enumerate(scores):
                sign = math.copysign(1, score)
                score = sign * math.log(1 + abs(score), 10)
                scores[idx] = score

        return scores


    def conv_path_to_step_idx_reward_map(self, path, check_depth=True):        
        # register which nodes have been directly included in the sub-tree
        for node in path:
            self.nodes_added.add(node)
            
        if check_depth:
            if len(path) < self.min_subtree_depth:
                # subtree not deep enough, do not use episode (but count all nodes as having been added)
                if self.force_include_branching_fathomed_transitions:
                    if self.normalised_lp_gain.tree.tree.nodes[path[-1]]['score'] == 0:
                        # brancher fathomed sub-tree, should include sub-tree even though is less than min_subtree_depth
                        pass
                    else:
                        # brancher did not fathom sub-tree and is less than min_subtree_depth, do not use
                        return None
                else:
                    # sub-tree is less than min_subtree_depth, do not use
                    return None
        
        # get rewards at each step in sub-tree episode
        path_node_rewards = self.get_path_node_scores(self.normalised_lp_gain.tree.tree, path)

        # get episode step indices at which each node in sub-tree was visited
        path_to_step_idx = {node: self.visited_nodes_to_step_idx[node] for node in path}

        # map each path node episode step idx to its corresponding reward
        step_idx_to_reward = {step_idx: r for step_idx, r in zip(list(path_to_step_idx.values()), path_node_rewards)}
        
        return step_idx_to_reward

    def _select_path_in_subtree(self, subtree):
        for root_node in subtree.nodes:
            if subtree.in_degree(root_node) == 0:
                # node is root
                break

        # use a construction method to select a sub-tree episode path through the sub-tree
        if self.retro_trajectory_construction == 'max_lp_gain' or self.retro_trajectory_construction == 'min_lp_gain':
            if self.only_use_leaves_closed_by_brancher_as_terminal_nodes:
                raise Exception('Have not implemented only_use_leaves_closed_by_brancher_as_terminal_nodes for this subtree construction method.')
            # iteratively decide next node in path at each step
            curr_node, path = root_node, [root_node]
            while True:
                # get potential next node(s)
                children = [child for child in subtree.successors(curr_node)]
                if len(children) == 0:
                    # curr node is final leaf node, path complete
                    break
                else:
                    # select next node
                    if self.retro_trajectory_construction == 'max_lp_gain':
                        idx = np.argmax([subtree.nodes[child]['lower_bound'] for child in children])
                    elif self.retro_trajectory_construction == 'min_lp_gain':
                        idx = np.argmin([subtree.nodes[child]['lower_bound'] for child in children])
                    else:
                        raise Exception(f'Unrecognised retro_trajectory_construction {self.retro_trajectory_construction}')
                    curr_node = children[idx]
                    path.append(curr_node)
                
        else:
            # first get leaf nodes and then use construction method to select leaf target for shortest path
            if self.only_use_leaves_closed_by_brancher_as_terminal_nodes:
                leaf_nodes = [node for node in subtree.nodes() if (subtree.out_degree(node) == 0 and subtree.nodes[node]['score'] == 0)]
            else:
                leaf_nodes = [node for node in subtree.nodes() if subtree.out_degree(node) == 0]
            
            if len(leaf_nodes) == 0:
                # could not find any valid path through sub-tree
                return []

            if self.retro_trajectory_construction == 'random':
                # randomly choose leaf node as final node
                final_node = leaf_nodes[random.choice(range(len(leaf_nodes)))]
            elif self.retro_trajectory_construction == 'deepest':
                # choose leaf node which would lead to deepest subtree as final node
                depths = [len(shortest_path(subtree, source=root_node, target=leaf_node)) for leaf_node in leaf_nodes]
                final_node = leaf_nodes[depths.index(max(depths))]
            elif self.retro_trajectory_construction == 'shortest':
                # choose leaf node which would lead to shortest subtree as final node
                depths = [len(shortest_path(subtree, source=root_node, target=leaf_node)) for leaf_node in leaf_nodes]
                final_node = leaf_nodes[depths.index(min(depths))]
            elif self.retro_trajectory_construction == 'max_leaf_lp_gain':
                # choose leaf node which has greatest LP gain as final node
                lp_gains = [subtree.nodes[leaf_node]['lower_bound'] for leaf_node in leaf_nodes]
                final_node = leaf_nodes[lp_gains.index(max(lp_gains))]
            elif self.retro_trajectory_construction == 'reverse_visitation_order':
                step_node_visited = [self.normalised_lp_gain.tree.tree.nodes[leaf_node]['step_visited'] for leaf_node in leaf_nodes]
                final_node = leaf_nodes[step_node_visited.index(max(step_node_visited))]
            elif self.retro_trajectory_construction == 'visitation_order':
                step_node_visited = [self.normalised_lp_gain.tree.tree.nodes[leaf_node]['step_visited'] for leaf_node in leaf_nodes]
                final_node = leaf_nodes[step_node_visited.index(min(step_node_visited))]
            else:
                raise Exception(f'Unrecognised retro_trajectory_construction {self.retro_trajectory_construction}')
            path = shortest_path(self.normalised_lp_gain.tree.tree, source=root_node, target=final_node)

        return path

    def extract(self, model, done):
        # update normalised LP gain tracker
        _ = self.normalised_lp_gain.extract(model, done)

        if not done:
            return None
        else:
            # instance finished, retrospectively create subtree episode paths

            if self.normalised_lp_gain.tree.tree.graph['root_node'] is None:
                # instance was pre-solved
                # if self.use_binary_sparse_rewards:
                    # return [{0: -1}]
                # else:
                    # return [{0: 0}]
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
                    self.normalised_lp_gain.tree.tree.remove_node(node)
                    if node in self.normalised_lp_gain.tree.tree.graph['visited_node_ids']:
                        # hack: SCIP sometimes returns large int rather than None node_id when episode finished
                        # since never visited this node (since no score assigned), do not count this node as having been visited when calculating paths below
                        if self.debug_mode:
                            print(f'Removing node {node} since was never visited by brancher.')
                        self.normalised_lp_gain.tree.tree.graph['visited_node_ids'].remove(node)
                        
            if self.only_use_leaves_closed_by_brancher_as_terminal_nodes:
                # remove leaf nodes which were never fathomed by brancher
                nodes = [node for node in self.normalised_lp_gain.tree.tree.nodes]
                for node in nodes:
                    if self.normalised_lp_gain.tree.tree.out_degree(node) == 0 and self.normalised_lp_gain.tree.tree.nodes[node]['score'] != 0:
                        self.normalised_lp_gain.tree.tree.remove_node(node)
                        if node in self.normalised_lp_gain.tree.tree.graph['visited_node_ids']:
                            if self.debug_mode:
                                print(f'Removing leaf node {node} since was never fathomed by brancher.')
                            self.normalised_lp_gain.tree.tree.graph['visited_node_ids'].remove(node)

            # map which nodes were visited at which step in episode
            self.visited_nodes_to_step_idx = {node: idx for idx, node in enumerate(self.normalised_lp_gain.tree.tree.graph['visited_node_ids'])}

            if self.set_score_as_subtree_size:
                for node in self.normalised_lp_gain.tree.tree.nodes():
                    # get sub-tree rooted at node
                    subtree = dfs_tree(self.normalised_lp_gain.tree.tree, node)
                    
                    # set node score as negative length of sub-tree beneath it (not including root node)
                    self.normalised_lp_gain.tree.tree.nodes[node]['score'] = -(len(subtree) - 1)

            if not self.use_retro_trajectories:
                # do not use any sub-tree episodes, just return whole B&B tree episode
                # subtree_step_idx_to_reward.append(self.conv_path_to_step_idx_reward_map(list(self.normalised_lp_gain.tree.tree.nodes[node]), check_depth=False))
                # return subtree_step_idx_to_reward
                step_idx_to_reward = {}
                for node, step_idx in self.visited_nodes_to_step_idx.items():
                    step_idx_to_reward[step_idx] = self.normalised_lp_gain.tree.tree.nodes[node]['score']
                subtrees_step_idx_to_reward.append(step_idx_to_reward)
                return subtrees_step_idx_to_reward 

            if self.multiarmed_bandit:
                # each step in the original B&B episode is a multi-armed bandit episode
                for node, step_idx in self.visited_nodes_to_step_idx.items():
                    subtrees_step_idx_to_reward.append({step_idx: self.normalised_lp_gain.tree.tree.nodes[node]['score']})
                return subtrees_step_idx_to_reward
            
            root_node = list(self.normalised_lp_gain.tree.tree.graph['root_node'].keys())[0]
            if self.force_include_optimal_trajectory:
                # get optimal path
                final_node = self.normalised_lp_gain.tree.tree.graph['optimum_node_id']
                path = shortest_path(self.normalised_lp_gain.tree.tree, source=root_node, target=final_node)
                subtrees_step_idx_to_reward.append(self.conv_path_to_step_idx_reward_map(path, check_depth=False))

                if self.only_return_optimal_trajectory:
                    return subtrees_step_idx_to_reward

                if self.remove_nonoptimal_fathomed_leaves:
                    for node in nodes:
                        if node in self.normalised_lp_gain.tree.tree.nodes.keys():
                            if self.normalised_lp_gain.tree.tree.nodes[node]['score'] == 0 and node != final_node:
                                # node fathomed and not in optimal path, remove
                                self.normalised_lp_gain.tree.tree.remove_node(node)
                                if self.debug_mode:
                                    print(f'Removed non-optimal fathomed leaf node {node}')

            elif self.force_include_last_visited_trajectory:
                # get path to last visited node
                final_node = self.normalised_lp_gain.tree.tree.graph['visited_node_ids'][-1]
                path = shortest_path(self.normalised_lp_gain.tree.tree, source=root_node, target=final_node)
                subtrees_step_idx_to_reward.append(self.conv_path_to_step_idx_reward_map(path, check_depth=False))

            # create sub-tree episodes from remaining B&B nodes visited by agent
            while True:
                # create depth first search sub-trees from nodes still leftover
                nx_subtrees = []
                
                # construct sub-trees containing prospective sub-tree episode(s) from remaining nodes
                if len(self.nodes_added) > 0:
                    for node in self.nodes_added:
                        children = [child for child in self.normalised_lp_gain.tree.tree.successors(node)]
                        for child in children:
                            if child not in self.nodes_added:
                                nx_subtrees.append(dfs_tree(self.normalised_lp_gain.tree.tree, child))
                else:
                    # not yet added any nodes to a sub-tree, whole B&B tree is first 'sub-tree'
                    nx_subtrees.append(dfs_tree(self.normalised_lp_gain.tree.tree, root_node))
                            
                for i, subtree in enumerate(nx_subtrees):
                    # init node attributes for nodes in subtree (since these are not transferred into new subtree by networkx)
                    for node in subtree.nodes:
                        subtree.nodes[node]['score'] = self.normalised_lp_gain.tree.tree.nodes[node]['score']
                        subtree.nodes[node]['lower_bound'] = self.normalised_lp_gain.tree.tree.nodes[node]['lower_bound']

                    # choose episode path through sub-tree
                    path = self._select_path_in_subtree(subtree)
                    
                    if len(path) > 0:
                        # gather rewards in sub-tree
                        subtree_step_idx_to_reward = self.conv_path_to_step_idx_reward_map(path, check_depth=True)
                        if subtree_step_idx_to_reward is not None:
                            if self.set_terminal_node_score_as_retrospective_subtree_size:
                                for counter, step_idx in enumerate(list(subtree_step_idx_to_reward.keys())):
                                    if counter == len(subtree_step_idx_to_reward) - 1:
                                        # terminal step
                                        subtree_step_idx_to_reward[step_idx] = -1 * len(subtree_step_idx_to_reward)
                                    else:
                                        subtree_step_idx_to_reward[step_idx] = 0
                            subtrees_step_idx_to_reward.append(subtree_step_idx_to_reward)
                        else:
                            # subtree was not deep enough to be added
                            pass
                    else:
                        # cannot establish valid path through sub-tree, do not consider nodes in this sub-tree again
                        for node in subtree.nodes():
                            self.nodes_added.add(node)

                if len(nx_subtrees) == 0:
                    # all sub-trees added
                    break
                    
            if self.debug_mode:
                print(f'visited_nodes_to_step_idx: {self.visited_nodes_to_step_idx}')
                step_idx_to_visited_nodes = {val: key for key, val in self.visited_nodes_to_step_idx.items()}
                print(f'step_idx_to_visited_nodes: {step_idx_to_visited_nodes}')
                for i, ep in enumerate(subtrees_step_idx_to_reward):
                    print(f'>>> sub-tree episode {i+1}: {ep}')
                    ep_path = [step_idx_to_visited_nodes[idx] for idx in ep.keys()]
                    print(f'path: {ep_path}')
                    ep_dual_bounds = [self.normalised_lp_gain.tree.tree.nodes[node]['lower_bound'] for node in ep_path]
                    print(f'ep_dual_bounds: {ep_dual_bounds}')
            
            if len(subtrees_step_idx_to_reward) == 0:
                # solved at root so path length < min path length so was never added to subtrees
                return [{0: 0}]
            else:
                return subtrees_step_idx_to_reward