# disable annoying tensorboard numpy warning
from re import L
import warnings
warnings.filterwarnings(action='ignore',
                        category=FutureWarning,
                        module='tensorboard')  # noqa

from retro_branching.src.learners.learner import Learner
from retro_branching.environments import EcoleBranching
from retro_branching.src.agents.pseudocost_branching_agent import PseudocostBranchingAgent
from retro_branching.loss_functions import MeanSquaredError
from retro_branching.rewards import NovelD

import ecole

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Batch


import numpy as np
import copy
import time
from collections import defaultdict, deque, namedtuple
from tqdm import tqdm
import random
import pickle
import gzip
import snappy
import threading

import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

# import ray
# import psutil
# NUM_CPUS = psutil.cpu_count(logical=False)
# try:
    # ray.init(num_cpus=NUM_CPUS)
# except RuntimeError:
    # # already initialised ray in script calling dcn sim, no need to init again
    # pass




class BacktrackRollout:
    def __init__(self,
                 expert,
                 observation_function,
                 information_function,
                 reward_function,
                 scip_params,
                 ecole_seed,
                 debug_mode=False):
        self.expert = expert 
        self.observation_function = observation_function
        self.information_function = information_function
        self.reward_function = reward_function
        self.scip_params = scip_params
        self.ecole_seed = ecole_seed
        self.debug_mode = debug_mode

    def get_iterable_episodes(self, episode_experiences, reward, agent_reward, done, episode_stats):
        # do any post-episode processing
        if self.debug_mode:
            print('>>> processing rollout expert episode(s) <<<')
        if 'retro' in agent_reward:
            # retrospectively retrieve sub-tree episodes and rewards and allocate correct next_state, done, etc.
            episodes = process_episodes_into_subtree_episodes(episode_experiences=episode_experiences,
                                                               reward=reward,
                                                               agent_reward=agent_reward,
                                                               intrinsic_reward=None,
                                                               intrinsic_extrinsic_combiner=None,
                                                               done=done,
                                                               filling_buffer=False,
                                                               episode_stats=episode_stats,
                                                               debug_mode=self.debug_mode)
        else:
            # no post-processing needed, but put episodes in list so is iterable
            episodes = [self.episode_experiences] 
        return episodes

    def solve_instance(self, instance, agent_reward):
        # reset expert
        self.expert.before_reset(instance)
    
        # save instance
        self.instance_before_reset = instance.copy_orig()

        # reset env
        env, obs, action_set, reward, done, info, self.instance_before_reset = _reset_env(instance=instance.copy_orig(),
                                                                                           observation_function=self.observation_function,
                                                                                           information_function=self.information_function,
                                                                                           reward_function=self.reward_function,
                                                                                           scip_params=self.scip_params,
                                                                                           ecole_seed=self.ecole_seed,
                                                                                           reproducible_episodes=True)
        env.seed(self.ecole_seed)

        # track actions taken at each step in trajectory and corresponding step reward
        self.expert_trajectory = {}

        # solve instance
        t = 0
        episode_experiences = [] # for constructing sub-trees later
        while not done:
            prev_obs = copy.deepcopy(obs)
            action_set = action_set.astype(int)
            action, action_idx = self.expert.action_select(action_set=action_set, 
                                                         obs=obs, 
                                                         munchausen_tau=0, 
                                                         epsilon=0, 
                                                         agent_idx=0, 
                                                         model=env.model, 
                                                         done=done)
            obs, action_set, reward, done, info = env.step(action)
            _reward = reward[agent_reward]

            # update trackers
            self.expert_trajectory[t] = action
            episode_experiences.append(
                    {'prev_state': prev_obs, 
                     'action': action, 
                     'reward': _reward, 
                     'done': done, 
                     'state': obs, 
                     'n_step_return': None,
                     'n_step_state': None,
                     'n': None,
                     'n_step_done': None,
                     }
                    )
            t += 1
        
        # gather episode experiences
        episode_stats = defaultdict(lambda: 0)
        episodes = self.get_iterable_episodes(episode_experiences=episode_experiences,
                                              reward=reward,
                                              agent_reward=agent_reward,
                                              done=done,
                                              episode_stats=episode_stats)

        # calc total return achieved by expert
        if 'extrinsic_R' in episode_stats:
            self.expert_return = episode_stats['extrinsic_R']
        else:
            self.expert_return = 0
            for episode in episodes:
                for experience in episode:
                    self.expert_return += experience['reward']

        if self.debug_mode:
            print(f'expert return: {self.expert_return} | expert_trajectory: {self.expert_trajectory}')

        return self.expert_trajectory, self.expert_return

    def rollout_env(self, t_to_rollout_up_to):
        '''Rollout up to step t'''
        # reset env
        env, obs, action_set, reward, done, info, self.instance_before_reset = _reset_env(instance=self.instance_before_reset.copy_orig(),
                                                                                           observation_function=self.observation_function,
                                                                                           information_function=self.information_function,
                                                                                           reward_function=self.reward_function,
                                                                                           scip_params=self.scip_params,
                                                                                           ecole_seed=self.ecole_seed,
                                                                                           reproducible_episodes=True)
        for t in range(t_to_rollout_up_to):
            obs, action_set, reward, done, info = env.step(self.expert_trajectory[t])
        return env, obs, action_set, reward, done, info, self.instance_before_reset

def extract_state_tensors_from_ecole_obs(obs, action_set):
    return (obs.row_features.astype(np.float32), 
            obs.edge_features.indices.astype(np.int16),
            obs.edge_features.values.astype(np.float32),
            obs.column_features.astype(np.float32),
            action_set.astype(np.int16))

def process_episodes_into_subtree_episodes(episode_experiences,
                                           reward, 
                                           agent_reward, 
                                           intrinsic_reward,
                                           intrinsic_extrinsic_combiner,
                                           done, 
                                           backtrack_step=-1,
                                           filling_buffer=False,
                                           episode_stats=None,
                                           debug_mode=False):
    '''
    Set backtrack_step > 0 to indicate that any steps < should NOT be considered
    for training the predictor of the intrinsic reward since they are rollout steps,
    and that should NOT include any steps < in returned episode experiences. Set
    backtrack_step = -1 to include all steps.

    N.B. For debugging, hardcode debug_mode=True in this function's arguments.
    '''
    if not done:
        raise Exception(f'Can only calc subtree reward {agent_reward} when episode is done.')

    # retrospectively construct episodes from sub-tree(s)
    if debug_mode:
        print(f'\nTotal episode experiences: {len(episode_experiences)} | Num sub-tree episodes reconstructed: {len(reward[agent_reward])}')
    subtrees_step_idx_to_reward = reward[agent_reward]
    episodes = []
    if backtrack_step > 0:
        if debug_mode:
            print(f'Backtrack step={backtrack_step} -> will ignore experiences before t={backtrack_step}')
        # fill out missing experiences so can index properly below
        for _ in range(backtrack_step):
            episode_experiences.insert(0, None)
    for counter, subtree_episode in enumerate(subtrees_step_idx_to_reward):
        # collect transitions for this sub-tree
        # experiences = [episode_experiences[i] for i in subtree_episode.keys() if i >= backtrack_step]
        if debug_mode:
            print(f'Initial sub-tree episode: {subtree_episode}')
        experiences = []
        step_indices = list(subtree_episode.keys())
        for i in step_indices:
            if episode_stats is not None:
                # track total reward across episode, even for steps which will not be considered by learner
                episode_stats['extrinsic_R'] += subtree_episode[i]
            if i >= backtrack_step:
                experiences.append(episode_experiences[i])
            else:
                del subtree_episode[i]
        if debug_mode:
            print(f'Filtered sub-tree episode: {subtree_episode}')

        # set reward and next_state of each transition in sub-tree
        for idx, r in enumerate(subtree_episode.values()):
            experiences[idx]['reward'] = r
            if idx < len(experiences)-1:
                # set next state
                experiences[idx]['state'] = experiences[idx+1]['prev_state'] # CHANGE
        if len(experiences) > 0:
            experiences[-1]['done'] = True # CHANGE: final sub-tree step is terminal

        if intrinsic_reward is not None:
            intrinsic_reward.before_reset(model=None)
            for idx in range(len(subtree_episode.values())):
                if filling_buffer:
                    train_predictor = False
                else:
                    train_predictor = True
                obs = experiences[idx]['prev_state'][:-1] # do not include action_set in obs
                done = experiences[idx]['done']
                r_i = intrinsic_reward.extract(model=None, done=done, obs=obs, train_predictor=train_predictor)
                if intrinsic_extrinsic_combiner == 'add':
                    experiences[idx]['reward'] += r_i 
                elif intrinsic_extrinsic_combiner == 'list':
                    if not isinstance(experiences[idx]['reward'], list):
                        experiences[idx]['reward'] = [experiences[idx]['reward']]
                    experiences[idx]['reward'].append(r_i)
                else:
                    raise Exception(f'Unrecognised intrinsic_extrinsic_combiner {intrinsic_extrinsic_combiner}')
                if episode_stats is not None:
                    episode_stats['intrinsic_R'] += r_i 

        # # TEMPORARY
        # if self.agent_reward == 'retro_binary_fathomed':
            # # try decreasing reward sparsity by setting terminal state of sub-trees to reward=1
            # experiences[-1]['reward'] = 1

        # ORIGINAL: add all transitions to buffer
        # episodes.append(experiences)

        # # CHANGE #1: Don't include final transition in non-optimal sub-trees
        # if counter == 0:
            # # optimal sub-tree -> add all experiences to buffer
            # episodes.append(experiences)
        # else:
            # # non-optimal sub-tree -> don't add final experience to buffer
            # episodes.append(experiences[:-1])

        # CHANGE #2: Only add final transition to buffer if final step's score != scores of other steps (normalised lp gain reward)
        if len(experiences) > 0:
            if isinstance(experiences[-1]['reward'], list):
                if len(experiences) > 1:
                    per_step_reward = experiences[0]['reward'][0]
                else:
                    # this step is terminal
                    per_step_reward = None
                if experiences[-1]['reward'][0] != per_step_reward:
                    add_all_transitions = True
                else:
                    add_all_transitions = False
            else:
                if len(experiences) > 1:
                    per_step_reward = experiences[0]['reward']
                else:
                    # this step is terminal
                    per_step_reward = None
                if experiences[-1]['reward'] != per_step_reward:
                    add_all_transitions = True
                else:
                    add_all_transitions = False
            if add_all_transitions:
                # add all experiences to buffer
                episodes.append(experiences)
            else:
                # don't add final experience to buffer
                episodes.append(experiences[:-1])
        
        # # CHANGE #3: Add all transitions
        # if len(experiences) > 0:
            # episodes.append(experiences)

        if debug_mode:
            print(f'Sub-tree episode step idx to reward: {subtree_episode}')

    return episodes
    









class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite` 
    observation function in a format understood by the pytorch geometric data handlers.
    """
    def __init__(self, 
                 constraint_features=None, 
                 edge_index=None, 
                 edge_features=None, 
                 variable_features=None, 
                 candidates=None):
    # def __init__(self, constraint_features, edge_index, variable_features, candidates):
        super().__init__()
        if constraint_features is not None:
            self.constraint_features = torch.FloatTensor(constraint_features)
        if edge_index is not None:
            # self.edge_index = torch.LongTensor(edge_features.indices.astype(np.int32))
            self.edge_index = torch.LongTensor(edge_index)
        if edge_features is not None:
            self.edge_attr = torch.from_numpy(edge_features).unsqueeze(1)
        if variable_features is not None:
            self.variable_features = torch.FloatTensor(variable_features)
            self.num_variables = self.variable_features.size(0)
            self.num_nodes = self.constraint_features.size(0) + self.variable_features.size(0)
        if candidates is not None:
            self.candidates = torch.from_numpy(candidates).long()
            self.raw_candidates = torch.from_numpy(candidates).long()
            self.num_candidates = len(candidates)

    # def __inc__(self, key, value):
    def __inc__(self, key, value, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs 
        for those entries (edge index, candidates) for which this is not obvious. This
        enables batching.
        """
        if key == 'edge_index':
            # constraint nodes connected via edge to variable nodes
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'candidates':
            # actions are variable nodes
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value)


# class BipartiteNodeData(torch_geometric.data.Data):
    # """
    # This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite` 
    # observation function in a format understood by the pytorch geometric data handlers.
    # """
    # def __init__(self, obs, candidates):
        # super().__init__()
        # self.obs = obs
        # self.constraint_features = torch.FloatTensor(obs.row_features)
        # self.edge_index = torch.LongTensor(obs.edge_features.indices.astype(np.int32))
        # # self.edge_attr = torch.FloatTensor(obs.edge_features.values).unsqueeze(1)
        # self.variable_features = torch.FloatTensor(obs.column_features)
        # self.candidates = torch.from_numpy(candidates.astype(np.int32)).long()
        # self.raw_candidates = torch.from_numpy(candidates.astype(np.int32)).long()
        
        # self.num_candidates = len(candidates)
        # self.num_variables = self.variable_features.size(0)
        # self.num_nodes = self.constraint_features.size(0) + self.variable_features.size(0)

    # def __inc__(self, key, value):
        # """
        # We overload the pytorch geometric method that tells how to increment indices when concatenating graphs 
        # for those entries (edge index, candidates) for which this is not obvious. This
        # enables batching.
        # """
        # if key == 'edge_index':
            # # constraint nodes connected via edge to variable nodes
            # return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        # elif key == 'candidates':
            # # actions are variable nodes
            # return self.variable_features.size(0)
        # else:
            # return super().__inc__(key, value)


# state is a BipartiteNodeData object containing obs and action_set
Transition = namedtuple('Transition', field_names=['state', 
                                                   'action', 
                                                   'reward', 
                                                   'done',  
                                                   'next_state', 
                                                   'n_step_return',
                                                   'n_step_state',
                                                   'n',
                                                   'n_step_done'])



class ReplayBuffer:
    def __init__(self, 
                 capacity, 
                 use_per=False,
                 use_cer=False,
                 per_alpha=0.6,
                 min_agent_per_priority=0.001,
                 demonstrator_buffer_capacity=5000,
                 min_demonstrator_per_priority=1,
                 compress_transitions=False,
                 profile_time=False,
                 debug_mode=False):
        '''
        Args:
            capacity (int): Maximum capacity of replay buffer.
            use_per (bool): Whether or not use use prioritised experience replay
                (https://arxiv.org/pdf/1511.05952.pdf).
            compress_transitions (bool): If True, will compress tensors before adding
                to buffer where possible.
            use_cer (bool): Whether or not to use combined experience replay 
                (https://arxiv.org/pdf/1712.01275.pdf). If True, will add most
                recent transition to sampled transition when callinf ReplayBuffer.sample()
        '''
        self.debug_mode = debug_mode
        self.profile_time = profile_time
        
        self.compress_transitions = compress_transitions

        # init experience replay buffer
        self.agent_buffer_capacity = capacity
        # self.buffer = deque(maxlen=capacity)
        self.buffer = []
        for _ in range(capacity):
            self.buffer.append(None)
        self.curr_agent_write_idx, self.available_agent_samples = 0, 0

        self.demonstrator_buffer_capacity = demonstrator_buffer_capacity
        self.min_demonstrator_per_priority = min_demonstrator_per_priority
        if demonstrator_buffer_capacity > 0:
            for _ in range(demonstrator_buffer_capacity):
                self.buffer.append(None)
            self.curr_demonstrator_write_idx, self.available_demonstrator_samples = copy.deepcopy(capacity), 0
        else:
            self.curr_demonstrator_write_idx, self.demonstrator_buffer_capacity, self.available_demonstrator_samples = None, 0, 0

        # init prioritised experience replay params (if applicable)
        self.use_per = use_per
        if self.use_per:
            self.per_alpha = per_alpha
            self.min_agent_per_priority = min_agent_per_priority
            self.sum_tree = SumTree(leaf_values=[_id for _id in range(len(self.buffer))],
                                    leaf_weights=[0 for _ in range(len(self.buffer))])
            # all experiences initially have the same priority before training starts, since no td errors yet calculated
            self.sum_tree.tree.graph['max_priority'] = self.min_agent_per_priority
            if self.debug_mode:
                print(f'Sum tree leaf ids: {self.sum_tree.tree.graph["leaf_ids"]}')
        self.use_cer = use_cer

    def append(self, transition, is_demonstrator=False):
        # for idx, trans in enumerate(transition):
            # print(f'idx {idx} transition element: {type(trans)} {trans}')
            # if type(trans) == tuple:
                # for i, el in enumerate(trans):
                    # print(f'tuple idx {i}: {type(el)} {type(el[0])} {el.shape}')
        # raise Exception()

        compress_transitions_start = time.time()
        if self.compress_transitions:
            transition = [snappy.compress(pickle.dumps(el)) for el in transition]
            if self.profile_time:
                compress_transitions_t = time.time() - compress_transitions_start
                print(f'compress_transitions_t: {compress_transitions_t*1e3:.3f} ms')

        if is_demonstrator:
            write_idx = self.curr_demonstrator_write_idx
        else:
            write_idx = self.curr_agent_write_idx
        self.buffer[write_idx] = transition
        if self.debug_mode:
            print(f'append | transition: {transition} | is_demonstrator: {is_demonstrator} | write_idx: {write_idx}')

        if self.use_per:
            update_priority_start = time.time()
            # set priority of experience to be the maximum priority so that it will be sampled at least once
            priority = self.sum_tree.tree.graph['max_priority']

            # update priority of experience in sum tree
            self.update_priority(write_idx, priority, is_demonstrator)
            if self.profile_time:
                update_priority_t = time.time() - update_priority_start 
                print(f'update_priority_t: {update_priority_t*1e3:.3f} ms')

        # update write idx
        if is_demonstrator:
            self.curr_demonstrator_write_idx += 1
            if self.curr_demonstrator_write_idx >= self.agent_buffer_capacity + self.demonstrator_buffer_capacity:
                # reset to start overwriting old experiences
                self.curr_demonstrator_write_idx = self.agent_buffer_capacity
            # max out the available samples at the memory buffer size
            if self.available_demonstrator_samples < self.demonstrator_buffer_capacity:
                self.available_demonstrator_samples += 1
            else:
                self.available_demonstrator_samples = self.demonstrator_buffer_capacity
        else:
            self.curr_agent_write_idx += 1
            if self.curr_agent_write_idx >= self.agent_buffer_capacity:
                # reset to start overwriting old experiences
                self.curr_agent_write_idx = 0
            # max out the available samples at the memory buffer size
            if self.available_agent_samples < self.agent_buffer_capacity:
                self.available_agent_samples += 1
            else:
                self.available_agent_samples = self.agent_buffer_capacity


    def update_priority(self, idx, priority, is_demonstrator):
        if priority > self.sum_tree.tree.graph['max_priority']:
            self.sum_tree.tree.graph['max_priority'] = priority
        self.sum_tree.update(self.sum_tree.tree.graph['leaf_ids'][idx], 
                             self.adjust_priority(priority, is_demonstrator))

    def adjust_priority(self, priority, is_demonstrator):
        if is_demonstrator:
            return np.power(priority + self.min_demonstrator_per_priority, self.per_alpha)
        else:
            return np.power(priority + self.min_agent_per_priority, self.per_alpha)
    
    def __len__(self):
        return self.available_agent_samples + self.available_demonstrator_samples

    def sample(self, batch_size, use_cer, per_beta=None):
        if not self.use_per:
            # standard experience replay with random uniform sampling
            indices = np.random.choice(len(self.buffer[:self.available_agent_samples+self.available_demonstrator_samples]), batch_size, replace=False)
            importance_sampling_weights = None
        else:
            # prioritised experience replay with importance sampling
            buffer_sample_start = time.time()
            if per_beta is None:
                raise Exception('Must provide per_beta to sample() method if using prioritised experience replay.')
            if self.debug_mode:
                print(f'Sum tree leaf ids: {self.sum_tree.tree.graph["leaf_ids"]}')
                print(f'available_agent_samples: {self.available_agent_samples} | available_demonstrator_samples: {self.available_demonstrator_samples}')
            indices, indices_set = [], set()
            importance_sampling_weights = []
            root_id = self.sum_tree.tree.graph['root_id']
            if use_cer:
                # will add most recent transition to batch
                num_experiences_to_sample = batch_size - 1
            else:
                num_experiences_to_sample = batch_size
            while len(indices) < num_experiences_to_sample:
                total_tree_sum = self.sum_tree.tree.nodes[root_id]['weight']

                # randomly sample a weight
                random_weight = np.random.uniform(0, total_tree_sum)

                # retrieve this weight's leaf node (transition id) from the sum tree
                node_id = self.sum_tree.retrieve(random_weight, root_id)

                # check the transition has i) been filled and is therefore valid and ii) not already been sampled
                if (node_id < self.available_agent_samples or self.agent_buffer_capacity <= node_id < self.agent_buffer_capacity+self.available_demonstrator_samples) and node_id not in indices_set:
                    if self.debug_mode:
                        print(f'node id {node_id} okay!')
                    indices.append(node_id)
                    indices_set.add(node_id)

                    # calculate this node's corresponding transition priority
                    priority = self.sum_tree.tree.nodes[node_id]['weight'] / self.sum_tree.tree.nodes[root_id]['weight']

                    # calculate the corresponding importance sampling weight
                    importance_sampling_weights.append((self.available_agent_samples+self.available_demonstrator_samples) * priority)
            if self.profile_time:
                buffer_sample_t = time.time() - buffer_sample_start
                print(f'buffer_sample_t: {buffer_sample_t*1e3:.3f} ms')


        if use_cer:
            # ensure most recently added agent transition is in sampled transitions
            prev_write_idx = self.curr_agent_write_idx - 1
            if prev_write_idx < 0:
                # write idx was reset since reached buffer capacity
                prev_write_idx = self.agent_buffer_capacity - 1
            indices.append(prev_write_idx)

            # calculate this node's corresponding transition priority
            priority = self.sum_tree.tree.nodes[prev_write_idx]['weight'] / self.sum_tree.tree.nodes[root_id]['weight']

            # calculate the corresponding importance sampling weight
            importance_sampling_weights.append((self.available_agent_samples+self.available_demonstrator_samples) * priority)

        if self.use_per:
            importance_sampling_weights_start = time.time()
            # apply the beta factor and normalise the importance sampling weights
            importance_sampling_weights = torch.tensor(importance_sampling_weights)
            importance_sampling_weights = torch.pow(importance_sampling_weights, -per_beta) / torch.max(importance_sampling_weights)
            if self.profile_time:
                importance_sampling_weights_t = time.time() - importance_sampling_weights_start
                print(f'importance_sampling_weights_t: {importance_sampling_weights_t*1e3:.3f} ms')

        if self.debug_mode:
            print(f'Buffer indices sampled: {indices}')
                
        # collect the sampled transitions 
        zip_sampled_transitions_start = time.time()
        state, action, reward, done, next_state, n_step_return, n_step_state, n, n_step_done = zip(*[self.buffer[idx] for idx in indices])
        if self.profile_time:
            zip_sampled_transitions_t = time.time() - zip_sampled_transitions_start 
            print(f'zip_sampled_transitions_t: {zip_sampled_transitions_t*1e3:.3f} ms')

        if self.compress_transitions:
            state = [pickle.loads(snappy.uncompress(_state)) for _state in state]
            action = [pickle.loads(snappy.uncompress(_action)) for _action in action]
            reward = [pickle.loads(snappy.uncompress(_reward)) for _reward in reward]
            done = [pickle.loads(snappy.uncompress(_done)) for _done in done]
            next_state = [pickle.loads(snappy.uncompress(_next_state)) for _next_state in next_state]
            n_step_return = [pickle.loads(snappy.uncompress(_n_step_return)) for _n_step_return in n_step_return]
            n_step_state = [pickle.loads(snappy.uncompress(_n_step_state)) for _n_step_state in n_step_state]
            n = [pickle.loads(snappy.uncompress(_n)) for _n in n]
            n_step_done = [pickle.loads(snappy.uncompress(_n_step_done)) for _n_step_done in n_step_done]

        # conv states to state object
        state = [BipartiteNodeData(*s) for s in state]
        next_state = [BipartiteNodeData(*s) for s in next_state]
        n_step_state = [BipartiteNodeData(*s) for s in n_step_state]

        return (Batch.from_data_list(state),
                torch.tensor(action),
                torch.tensor(reward),
                torch.tensor(done).float(),
                Batch.from_data_list(next_state),
                torch.tensor(n_step_return),
                Batch.from_data_list(n_step_state),
                n,
                torch.tensor(n_step_done).float(),
                torch.tensor(indices),
                importance_sampling_weights)



class SumTree:
    '''
    Implementation of the Sample Tree weighted sampling algorithm 
    (https://adventuresinmachinelearning.com/sumtree-introduction-python/).

    Used for prioritised experience replay (https://arxiv.org/pdf/1511.05952.pdf).
    '''
    def __init__(self, leaf_values, leaf_weights, debug_mode=False):
        self.debug_mode = debug_mode
        self.init_sum_tree(leaf_values, leaf_weights)
                
    def init_sum_tree(self, leaf_values, leaf_weights):
        self.tree = nx.DiGraph()
        
        # add leaf nodes
        leaf_ids = range(len(leaf_values))
        for _id, v, w in zip(leaf_ids, leaf_values, leaf_weights):
            self.tree.add_node(_id, value=v, weight=w, is_leaf=True)
            
        # iteratively add preceding tree layers all way up to root node
        child_ids = copy.deepcopy(leaf_ids)
        last_idx = child_ids[-1]
        while len(child_ids) > 1:
            inodes = iter(child_ids)
            if len(child_ids) % 2 != 0:
                # last node will be left over
                left_over_node = child_ids[-1]
            else:
                left_over_node = None
            child_ids = [] # track child ids to add parents for in next level of tree
            for pair in zip(inodes, inodes):
                parent_id = last_idx + 1
                child_ids.append(parent_id) # unless parent is root, will need to add parent for it in next tree layer
                self.tree.add_node(parent_id,
                                   value=None,
                                   weight=self.tree.nodes[pair[0]]['weight']+self.tree.nodes[pair[1]]['weight'],
                                   is_leaf=False)
                self.tree.add_edge(parent_id, pair[0])
                self.tree.add_edge(parent_id, pair[1])
                last_idx += 1
            if left_over_node is not None:
                child_ids.append(left_over_node)

        self.tree.graph['root_id'] = child_ids[-1]
        self.tree.graph['leaf_ids'] = list(leaf_ids)
            
    def retrieve(self, weight, node_id):
        '''Recursive traversal from node_id to a leaf node.'''
        if self.tree.nodes[node_id]['is_leaf']:
            # reached leaf node
            if self.debug_mode:
                print(f'reached leaf node {node_id}')
            return node_id
        else:
            # not yet reached leaf node, keep traversing
            if self.debug_mode:
                print(f'curr node: {node_id} | weight to consider: {weight}')
        
            # get children of current node
            children = list(self.tree.successors(node_id))
            if self.debug_mode:
                print(f'children: {children}')

            # choose if traverse to left or right child
            if self.tree.nodes[children[0]]['weight'] >= weight:
                # keep weight same, traverse to left-hand child
                if self.debug_mode:
                    print(f'traverse to LHS child w/ weight to consider: {weight}')
                return self.retrieve(weight, children[0])
            else:
                # subtract left-hand child's weight from weight, traverse to right-hand child
                if self.debug_mode:
                    print(f'traverse to RHS child w/ updated weight to consider: {weight - self.tree.nodes[children[0]]["weight"]}')
                return self.retrieve(weight - self.tree.nodes[children[0]]['weight'], children[1])
            
    def update(self, node_id, new_weight):
        change = new_weight - self.tree.nodes[node_id]['weight']
        self.tree.nodes[node_id]['weight'] = new_weight
        self.propagate_changes(change, list(self.tree.predecessors(node_id))[0])
        
    def propagate_changes(self, change, node_id):
        self.tree.nodes[node_id]['weight'] += change
        predecessor = list(self.tree.predecessors(node_id))
        if len(predecessor) > 0:
            self.propagate_changes(change, predecessor[0])
        else:
            # node_id is root node, no further change to propagate
            pass
            
    def sample(self):
        # get total tree weight
        total_tree_sum = self.tree.nodes[self.tree.graph['root_id']]['weight']
        if self.debug_mode:
            print(f'tree sum: {total_tree_sum}')
        
        # choose random weight to sample
        random_weight = np.random.uniform(0, total_tree_sum)
        if self.debug_mode:
            print(f'randomly sampled weight: {random_weight}')
        
        # traverse tree from root node to retrieve node id of this randomly sampled weight in O(log(n))
        return self.retrieve(random_weight, self.tree.graph['root_id'])
        
    def render(self):
        pos = graphviz_layout(self.tree, prog='dot')
        node_labels = {node: node for node in self.tree.nodes}
        nx.draw_networkx_nodes(self.tree,
                               pos,
                               label=node_labels)
        nx.draw_networkx_edges(self.tree,
                               pos)
        
        nx.draw_networkx_labels(self.tree, pos, labels=node_labels)
        
        plt.show()




class DQNLearner(Learner):
    def __init__(self,
                 agent,
                 env,
                 instances,
                 reset_envs_batch=1,
                 max_steps=int(1e12),
                 max_steps_agent=None,
                 buffer_capacity=1000,
                 buffer_min_length=100,
                 use_per=False,
                 use_cer=False,
                 initial_per_beta=0.4,
                 final_per_beta=1.0,
                 final_per_beta_epoch=1000,
                 per_alpha=0.6,
                 min_agent_per_priority=0.01,
                 hard_update_target_frequency=50,
                 soft_update_target_tau=None,
                 gradient_clipping_max_norm=None,
                 gradient_clipping_clip_value=None,
                 ecole_seed=0,
                 reproducible_episodes=True,
                 steps_per_update=25,
                 prob_add_to_buffer=1,
                 batch_size=1,
                 accumulate_gradient_factor=1,
                 save_gradients=False,
                 agent_reward='num_nodes',
                 intrinsic_reward=None,
                 intrinsic_extrinsic_combiner='add',
                 n_step_return=1,
                 use_n_step_dqn=False,
                 lr=1e-4,
                 gamma=0.99,
                 loss_function=None,
                 optimizer_name='adam',
                 munchausen_tau=0,
                 munchausen_lo=-1,
                 munchausen_alpha=0.9,
                 initial_epsilon=1,
                 final_epsilon=0.05,
                 final_epsilon_epoch=1000,
                 double_dqn_clipping=False,
                 demonstrator_agent=None,
                 save_demonstrator_buffer=True,
                 num_pretraining_epochs=0,
                 demonstrator_buffer_capacity=0,
                 min_demonstrator_per_priority=1,
                 demonstrator_n_step_return_loss_weight=1,
                 demonstrator_margin_loss_weight=1,
                 demonstrator_margin=0.8,
                 # demonstrator_l2_regularization_weight=1,
                 weight_decay=1e-5,
                 backtrack_rollout_expert=None,
                 max_attempts_to_reach_expert=10,
                 threshold_difficulty=None,
                 episode_log_frequency=1,
                 checkpoint_frequency=1,
                 path_to_save=None,
                 use_sqlite_database=True,
                 name='dqn_learner',
                 debug_mode=False,
                 profile_time=False,
                 profile_memory=False,
                 **kwargs):
        '''
        Args:
            reset_envs_batch (bool): If >1, rather than resetting one env
                at a time until find valid instance, will reset reset_envs_batch envs at a time 
                in parallel and store them in an iterable container before passing
                to agent. This can signficiantly improve training times, since 
                env.reset() is a time-intensive method, especially for small instances
                which are often pre-solved and therefore env.reset() must be 
                called many times until a non-presolved instance is found.
            batch_size (int): Number of experiences to pass through NN in a single batch.
            accumulate_gradient_factor (int, None): If not None, will pass through batch_size
                experiences accumulate_gradient_factor times to accumulate batch_size*accumulate_gradient_factor
                gradients before updating networks.
            save_gradients (bool): If True, will save mean, std, max, and min of
                gradients. N.B. For large instances, this can have significant
                computation time overheads.
            max_steps_agent (obj, None): Agent to use for remaining steps in epsiode after max_steps.
                If None, will terminate episode after max_steps.
            hard_update_target_frequency (int): Number of epochs (value networks updates)
                after which to update the target networks.
            soft_update_target_tau (float): If not None, rather than hard updates, will
                perform soft updates of target networks.
            steps_per_update (int): Steps to take in env before updating the networks.
            checkpoint_frequency (int): Number of epochs after
                which to save progress so far.
            tau (int, float): Temperature term for scaling entropy when using Munchausen
                DQN. Setting tau=0 recovers vanilla DQN. 0 <= tau.
            agent_reward (str, list): Either str reward or list of str rewards 
                (if want reward to be a vector for e.g. multi-head DQN).
            intrinsic_reward (None, 'noveld'): intrinsic reward to guide agent
                in addition to its extrinsic reward above.
            intrinsic_extrinsic_combiner ('list', 'add'): How to combine the intrinsic
                and extrinsic rewards. If 'list', will keep each reward as separate
                element in a list for multi-headed DQN. If 'add', will sum the
                intrinsic and extrinsic rewards to get a single reward.
            n_step_return (int): Will store n-step discounted 
                return in replay buffer for each experience.
            use_n_step_dqn (bool): If True, will use n-step DQN rather than 1-step
                DQN.
            optimizer_name ('adam', 'sgd')
            profile_time (bool): If True, will make calls to 
                torch.cuda.synchronize() to enable timing.
            use_sqlite_database (bool): If True, at each checkpoint, will write
                epoch and episode data to SQLite database and reset the in-memory
                logs to save saving time and RAM usage.
            reproducible_episodes (bool): If True, will set env.seed(ecole_seed)
                before each instance reset in order to make episode reproducible.
        '''

        self.profile_memory = profile_memory
        if self.profile_memory:
            pass
            # print('Snapshotting memory...')
            # memory_profile_start = time.time()
            # self.tracker = tracker.SummaryTracker()
            # self.tracker.print_diff()
            # print(f'Snapshotted memory in {time.time() - memory_profile_start:.3f} s')

        super(DQNLearner, self).__init__(agent, path_to_save, name)

        if max_steps < 1e12 and (agent_reward == 'optimal_retro_trajectory_normalised_lp_gain' or agent_reward == 'retro_branching'):
            print(f'WARNING: Have set max_steps to {max_steps} but agent reward relies on retrospective episode reconstruction. Consider increasing max_steps to e.g. int(1e12) or do not do episode reconstruction.')

        self.agent = agent
        self.agent.train()
        self.env = env
        self.env_ready = False
        self.ecole_seed = ecole_seed
        self.reproducible_episodes = reproducible_episodes
        self.steps_per_update = steps_per_update
        self.prob_add_to_buffer = prob_add_to_buffer
        if self.reproducible_episodes:
            self.env.seed(self.ecole_seed)
        else:
            self.env.seed(random.randint(ecole.RandomEngine.min_seed, ecole.RandomEngine.max_seed))
        self.reset_envs_batch, self.pre_reset_envs = reset_envs_batch, iter([])
        self.max_steps = max_steps
        self.max_steps_agent = max_steps_agent

        if 'default_agent_idx' in self.agent.__dict__:
            self.use_double_dqn = True
        else:
            self.use_double_dqn = False

        self.instances = instances
        if type(self.instances) == ecole.core.scip.Model:
            print('Have been given one instance, will overfit to this instance.')
            self.overfit_instance = True
            self.curr_instance = instances.copy_orig()
        else:
            self.overfit_instance = False
            self.curr_instance = None

        self.demonstrator_agent = demonstrator_agent
        if demonstrator_agent is not None:
            if not use_per:
                raise Exception('Have not implemented DQN from Demonstration without prioritised experience replay. Set demonstrator_agent=None or use_per=True.')
            self.num_pretraining_epochs = num_pretraining_epochs
            self.demonstrator_buffer_capacity = demonstrator_buffer_capacity
            self.min_demonstrator_per_priority = min_demonstrator_per_priority
            self.demonstrator_n_step_return_loss_weight = demonstrator_n_step_return_loss_weight
            self.demonstrator_margin_loss_weight = demonstrator_margin_loss_weight
            self.demonstrator_margin = demonstrator_margin
            # self.weight_decay = weight_decay
            # self.demonstrator_l2_regularization_weight = demonstrator_l2_regularization_weight
        else:
            self.num_pretraining_epochs = 0
            self.demonstrator_buffer_capacity = 0
            self.min_demonstrator_per_priority = 0
            self.demonstrator_n_step_return_loss_weight = 0
            self.demonstrator_margin_loss_weight = 0
            self.demonstrator_margin = 0
            # self.weight_decay = 0
            # self.demonstrator_l2_regularization_weight = 0
        self.weight_decay = weight_decay
        self.save_demonstrator_buffer = save_demonstrator_buffer

        self.backtrack_rollout_expert = backtrack_rollout_expert
        self.max_attempts_to_reach_expert = max_attempts_to_reach_expert
        if self.backtrack_rollout_expert is not None:
            if not reproducible_episodes:
                raise Exception('Must have reproducible_episodes if using backtrack_rollout_expert.')
            self.backtracker = BacktrackRollout(expert=self.backtrack_rollout_expert,
                    observation_function=self.env.str_observation_function,
                                                information_function=self.env.str_information_function,
                                                reward_function=self.env.str_reward_function,
                                                scip_params=self.env.str_scip_params,
                                                ecole_seed=self.ecole_seed,
                                                debug_mode=debug_mode)
            self.reached_rollout_expert = True # init
            self.curr_backtrack_step = None
            self.num_backtrack_expert_instances_solved = 0
            self.rollout_attempt_counter = 0
        else:
            self.backtracker = None
            self.reached_rollout_expert = None
            self.curr_backtrack_step = -1
            self.num_backtrack_expert_instances_solved = None
            self.rollout_attempt_counter = None

        self.use_per = use_per
        self.use_cer = use_cer
        self.initial_per_beta = initial_per_beta
        self.final_per_beta = final_per_beta
        self.final_per_beta_epoch = final_per_beta_epoch
        self.per_alpha = per_alpha
        self.min_agent_per_priority = min_agent_per_priority
        self.buffer = ReplayBuffer(buffer_capacity,
                                   use_per=use_per,
                                   use_cer=use_cer,
                                   per_alpha=per_alpha,
                                   min_agent_per_priority=min_agent_per_priority,
                                   demonstrator_buffer_capacity=self.demonstrator_buffer_capacity,
                                   min_demonstrator_per_priority=self.min_demonstrator_per_priority,
                                   profile_time=profile_time)
        self.buffer_capacity = buffer_capacity
        self.buffer_min_length = buffer_min_length
        if buffer_min_length < batch_size:
            raise Exception('batch_size must be > buffer_min_length')


        self.hard_update_target_frequency = hard_update_target_frequency 
        self.soft_update_target_tau = soft_update_target_tau
        if gradient_clipping_clip_value is not None and gradient_clipping_max_norm is not None:
            raise Exception('gradient_clipping_max_norm and gradient_clipping_clip_value cannot both not be None.')
        self.gradient_clipping_max_norm = gradient_clipping_max_norm
        self.gradient_clipping_clip_value = gradient_clipping_clip_value
        self.save_gradients = save_gradients

        self.batch_size = batch_size
        self.accumulate_gradient_factor = accumulate_gradient_factor 

        self.agent_reward = agent_reward
        if intrinsic_reward is None:
            self.intrinsic_reward = None
        elif intrinsic_reward == 'noveld':
            self.intrinsic_reward = NovelD(device=self.agent.device,
                                           observation_function=self.env.str_observation_function)
        else:
            raise Exception(f'Unrecognised intrinsic reward {intrinsic_reward}')
        self.intrinsic_extrinsic_combiner = intrinsic_extrinsic_combiner
        if isinstance(self.agent_reward, list) or (self.intrinsic_reward is not None and self.intrinsic_extrinsic_combiner == 'list'):
            self.multiple_rewards = True
        else:
            self.multiple_rewards = False

        self.n_step_return = n_step_return
        self.use_n_step_dqn = use_n_step_dqn
        if use_n_step_dqn and demonstrator_agent is not None:
            raise Exception('If using demonstrator_agent, should set use_n_step_dqn=False since demonstrator also uses 1-step DQN. To ensure DQfD also uses n-step returns, just ensure demonstrator_n_step_return_loss_weight != 0')
        self.lr = lr
        if loss_function is None:
            self.loss_function = MeanSquaredError()
        else:
            if not isinstance(loss_function, str):
                self.loss_function = loss_function
            elif loss_function == 'mean_squared_error':
                self.loss_function = MeanSquaredError()
            else:
                raise Exception(f'Not implemented handling loss function {loss_function} inside learner.')
        self.optimizer_name = optimizer_name

        self.optimizer = self.reset_optimizer(agent=self.agent, lr=self.lr)
        self.gamma = gamma

        self.munchausen_tau = munchausen_tau
        self.munchausen_alpha = munchausen_alpha
        self.munchausen_lo = munchausen_lo

        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.final_epsilon_epoch = final_epsilon_epoch

        self.double_dqn_clipping = double_dqn_clipping

        self.threshold_difficulty = threshold_difficulty
        if self.threshold_difficulty is not None:
            # init threshold env for evaluating difficulty when generating instance
            if 'threshold_env' not in kwargs:
                self.threshold_env = EcoleBranching(observation_function=self.env.str_observation_function,
                                                    information_function=self.env.str_information_function,
                                                    reward_function=self.env.str_reward_function,
                                                    scip_params=self.env.str_scip_params)
                # self.threshold_env = EcoleBranching()
            else:
                self.threshold_env = kwargs['threshold_env']
            if 'threshold_agent' not in kwargs:
                self.threshold_agent = PseudocostBranchingAgent()
            else:
                self.threshold_agent = kwargs['threshold_agent']
            if self.reproducible_episodes:
                self.threshold_env.seed(self.ecole_seed)
            else:
                self.threshold_env.seed(random.randint(ecole.RandomEngine.min_seed, ecole.RandomEngine.max_seed))
        else:
            self.threshold_env = None
            self.threshold_agent = None

        self.episode_log_frequency = episode_log_frequency
        self.checkpoint_frequency = checkpoint_frequency
        self.name = name
        self.episodes_log = self.reset_episodes_log()
        self.epochs_log = self.reset_epochs_log()
        self.kwargs = kwargs

        self.path_to_save = path_to_save
        if path_to_save is not None:
            self.path_to_save = self.init_save_dir(path=path_to_save, use_sqlite_database=use_sqlite_database)
        self.save_thread = None
        self.use_sqlite_database = use_sqlite_database

        self.debug_mode = debug_mode
        if debug_mode:
            torch.set_printoptions(precision=10)

        self.profile_time = profile_time
        self.agent.profile_time = profile_time


    def reset_optimizer(self, agent, lr):
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(agent.parameters(), lr=lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'sgd':
            return torch.optim.SGD(agent.parameters(), lr=lr, momentum=0.9, weight_decay=self.weight_decay)
        else:
            raise Exception(f'Unrecognised optimizer_name {self.optimizer_name}')
    

    def reset_env(self, env, min_num_envs_to_reset=1, paralellelisation_factor=1, max_attempts=500):
        if self.threshold_env is not None:
            threshold_observation_function = self.threshold_env.str_observation_function
            threshold_information_function = self.threshold_env.str_information_function
            threshold_reward_function = self.threshold_env.str_reward_function
            threshold_scip_params = self.threshold_env.str_scip_params
        else:
            threshold_observation_function = None
            threshold_information_function = None
            threshold_reward_function = None
            threshold_scip_params = None

        num_attempts = 0
        if min_num_envs_to_reset == 1:
            # do not reset in parallel, just reset one env at a time
            obs = None
            while obs is None:
                if self.overfit_instance:
                    instance = self.curr_instance.copy_orig()
                else:
                    instance = next(self.instances)

                if self.intrinsic_reward is not None:
                    if 'retro' in self.agent_reward:
                        # can only calc intrinsic reward when retrospectively know sub-tree episode -> do not reset yet
                        intrinsic_reward = None
                    else:
                        # calc intrinsic reward at each step in episode -> reset now
                        intrinsic_reward = self.intrinsic_reward
                else:
                    intrinsic_reward = None

                env, obs, action_set, reward, done, info, instance_before_reset = _reset_env(instance=instance,
                                                                                               observation_function=env.str_observation_function,
                                                                                               information_function=env.str_information_function,
                                                                                               reward_function=env.str_reward_function,
                                                                                               scip_params=env.str_scip_params,
                                                                                               ecole_seed=self.ecole_seed,
                                                                                               reproducible_episodes=self.reproducible_episodes,
                                                                                               threshold_difficulty=self.threshold_difficulty,
                                                                                               threshold_agent=self.threshold_agent,
                                                                                               threshold_observation_function=threshold_observation_function,
                                                                                               threshold_information_function=threshold_information_function,
                                                                                               threshold_reward_function=threshold_reward_function,
                                                                                               threshold_scip_params=threshold_scip_params,
                                                                                               profile_time=self.profile_time)
                num_attempts += 1
                if num_attempts >= max_attempts:
                    raise Exception(f'Unable to find instance which is not pre-solved after {num_attempts} attempts.')
            self.env_ready = True
            return env, obs, action_set, reward, done, info, instance_before_reset

        else:
            raise Exception('Not implemented multi-parallel resetting.')

    def update_episode_log(self, episode_stats):
        for key, val in episode_stats.items():
            self.episodes_log[key].append(val)

    def get_episode_log_str(self):
        '''Returns string logging end of episode.'''
        log_str = 'Epoch {} episode {}'.format(self.epoch_counter, self.episode_counter)
        log_str += ' | Eps: {}'.format(round(self.episodes_log['epsilon'][-1], 3))
        log_str += ' | Run Time: {} s'.format(round(self.episodes_log['episode_run_time'][-1], 3))
        log_str += ' | Nodes: {}'.format(self.episodes_log['num_nodes'][-1])
        # log_str += ' | LP iters: {}'.format(self.episodes_log['lp_iterations'][-1])
        log_str += ' | Steps: {}'.format(self.episodes_log['num_steps'][-1])
        # if self.agent_reward == 'optimal_retro_trajectory_normalised_lp_gain' or self.agent_reward == 'retro_branching':
            # log_str += f' ({len(self.episode_experiences)} in optimal path)'
        # log_str += ' | Solve time: {} s'.format(round(self.episodes_log['solving_time'][-1], 3))
        # if type(self.agent_reward) == list:
        if isinstance(self.episodes_log['R'][-1], dict):
            returns = {k: round(v, 3) for k, v in self.episodes_log['R'][-1].items()}
        else:
            returns = {self.agent_reward: round(self.episodes_log['R'][-1], 3)}
        log_str += ' | Return(s): {}'.format(returns)
        if self.backtrack_rollout_expert is not None:
            log_str += f' | Expert R_e: {self.expert_return} | Agent R_e: {self.episodes_log["extrinsic_R"][-1]} | Backtrack step: {self.curr_backtrack_step} | Step attempt counter: {self.rollout_attempt_counter} | Expert instances solved: {self.num_backtrack_expert_instances_solved}'
        return log_str

    def action_to_batch_idxs(self, action, state):
        '''Converts action from indexing action in each batch to indexing actions across all batches.'''
        return torch.cat([action[[0]], action[1:] + state.num_variables[:-1].cumsum(0)])

    def conv_q_vals_to_greedy_action(self, q_values, state):
        '''
        Takes tensor of q values and corresponding state batch and converts to greedy actions.
        '''
        action_set = torch.as_tensor(state.candidates)
        if type(q_values) == list:
            # Q-heads DQN, need to aggregate to get values for each action
            _q_values = [q_values[head][action_set] for head in range(len(q_values))]
            if self.agent.head_aggregator == 'add':
                _q_values = torch.stack(_q_values, dim=0).sum(dim=0)
            else:
                raise Exception(f'Unrecognised head_aggregator {self.agent.head_aggregator}')
        else:
            _q_values = q_values[action_set]
        _q_values = _q_values.split_with_sizes(tuple(state.num_candidates))
        action_set = state.raw_candidates.split_with_sizes(tuple(state.num_candidates))
        action_idx = torch.stack([q.argmax() for q in _q_values])
        action = torch.stack([_action_set[idx] for _action_set, idx in zip(action_set, action_idx)])

        return action




    def accumulate_gradients(self, use_cer):
        stats = {}

        sample_buffer_start = time.time()
        state, action, reward, done, next_state, n_step_return, n_step_state, n, n_step_done, indices, importance_sampling_weights = self.buffer.sample(batch_size=self.batch_size, use_cer=use_cer, per_beta=self.get_per_beta())
        is_demonstrator = torch.where(indices >= self.buffer.agent_buffer_capacity, 1, 0)
        if self.profile_time:
            print(state.to('cpu'))
            sample_buffer_t = time.time() - sample_buffer_start
            print(f'sample_buffer_t: {sample_buffer_t*1e3:.3f} ms')

        action_to_batch_idxs_start = time.time()
        action = self.action_to_batch_idxs(action, state).type(torch.LongTensor)
        if self.profile_time:
            print(action[0])
            action_to_batch_idxs_t = time.time() - action_to_batch_idxs_start
            print(f'action_to_batch_idxs_t: {action_to_batch_idxs_t*1e3:.3f} ms')

        if self.debug_mode:
            print(f'replay state:\n candidates: {state.candidates.shape} {state.candidates} | \n action: {action.shape} {action} | \n reward: {reward.shape} {reward} | \n n_step_return: {n_step_return.shape} {n_step_return} | \n done: {done.shape} {done}')
            if self.buffer.use_per:
                print(f'importance_sampling_weights from per_beta={self.get_per_beta()}: {importance_sampling_weights.shape} {importance_sampling_weights}')

        # prepare for nn forward pass
        to_start = time.time()
        state = state.to(self.agent.device)
        obs = (state.constraint_features, state.edge_index, state.edge_attr, state.variable_features)
        action = action.to(self.agent.device)
        if not self.use_n_step_dqn or self.demonstrator_agent is not None:
            # need to use 1-step ahead data
            next_state = next_state.to(self.agent.device)
            next_obs = (next_state.constraint_features, next_state.edge_index, next_state.edge_attr, next_state.variable_features)
            reward = reward.to(self.agent.device)
            done = done.to(self.agent.device)
        if self.use_n_step_dqn or self.demonstrator_agent is not None:
            # need to use n-step ahead data
            n_step_state = n_step_state.to(self.agent.device)
            n_step_gamma = torch.tensor([self.gamma ** _n for _n in n]).to(self.agent.device)
            n_step_return = n_step_return.to(self.agent.device)
            # if type(self.agent_reward) != list:
            if not self.multiple_rewards:
                # refactor tensor so can index correctly below
                n_step_return = n_step_return.unsqueeze(dim=1)
            n_step_state = n_step_state.to(self.agent.device)
            n_step_obs = (n_step_state.constraint_features, n_step_state.edge_index, n_step_state.edge_attr, n_step_state.variable_features)
            n_step_done = n_step_done.to(self.agent.device)
        is_demonstrator = is_demonstrator.to(self.agent.device)
        indices = indices.to(self.agent.device)
        if self.profile_time:
            print(state.to('cpu'))
            to_t = time.time() - to_start
            print(f'to_t: {to_t*1e3:.3f} ms')

        # get q value for each batch
        get_q_val_start = time.time()
        if self.use_double_dqn:
            # double dqn, compute q values for each agent
            q_values_1 = self.agent.calc_Q_values(obs, use_target_network=False, agent_idx=0)
            q_value_1 = [q_values_1[head][action].squeeze() for head in range(len(q_values_1))]
            q_values_2 = self.agent.calc_Q_values(obs, use_target_network=False, agent_idx=1)
            q_value_2 = [q_values_2[head][action].squeeze() for head in range(len(q_values_2))]
            if self.debug_mode:
                print(f'q_values_1: {q_values_1}\n action q_value_1: {q_value_1}\n q_values_2: {q_values_2}\n action q_value_2: {q_value_2}')
        else:
            q_values = self.agent.calc_Q_values(obs, use_target_network=False)
            q_value = [q_values[head][action].squeeze() for head in range(len(q_values))]
            if self.munchausen_tau != 0:
                # record entropy of agent's policy
                q_values_reshaped = [torch.reshape(q_values[head], (self.batch_size, int(q_values[head].shape[0] / self.batch_size))) for head in range(len(q_values))]
                agent_pi = [F.softmax(q_values_reshaped[head]/self.munchausen_tau, dim=1) for head in range(len(q_values_reshaped))]
                # use eps=1e-4 to avoid numerical instability when taking log
                # agent_pi_entropy = [-torch.sum(agent_pi[head] * (torch.log(agent_pi[head]+1e-4)/torch.log(state.num_variables[0]))) for head in range(len(agent_pi))]
                agent_pi_entropy = [-torch.sum(agent_pi[head] * (torch.log(agent_pi[head]+1e-4))) for head in range(len(agent_pi))]
                # agent_pi_entropy = [agent_pi_entropy[head] / torch.log(state.num_variables[0]) for head in range(len(agent_pi))]
            if self.debug_mode:
                print(f'q_values: {q_values[0].shape}\n action q value: {q_value}')
        if self.profile_time:
            print(q_value[0][0])
            q_val_t = time.time() - get_q_val_start
            print(f'q_val_t: {q_val_t*1e3:.3f} ms')

        # get q target for each batch
        with torch.no_grad():
            if self.debug_mode:
                print(f'replay next_state:\n candidates: {next_state.candidates.shape} {next_state.candidates}')

            # get agent greedy action
            next_action_start = time.time()
            if self.use_double_dqn:
                if self.use_n_step_dqn:
                    raise Exception('Not implemented n-step DQN for double DQN.')
                next_action_1, _ = self.agent.action_select(state=next_state, epsilon=0, agent_idx=0)
                next_action_1 = self.action_to_batch_idxs(next_action_1, next_state).type(torch.LongTensor)
                next_action_2, _ = self.agent.action_select(state=next_state, epsilon=0, agent_idx=1)
                next_action_2 = self.action_to_batch_idxs(next_action_2, next_state).type(torch.LongTensor)
                if self.debug_mode:
                    print(f'next_action_1: {next_action_1.shape} {next_action_1}\n next_action_2: {next_action_2.shape} {next_action_2}')
            else:
                if not self.use_n_step_dqn or self.demonstrator_agent is not None:
                    next_action, _ = self.agent.action_select(state=next_state, epsilon=0, agent_idx=0)
                    next_action = self.action_to_batch_idxs(next_action, next_state).type(torch.LongTensor)
                    if self.debug_mode:
                        print(f'next_action: {next_action.shape} {next_action}')
                if self.use_n_step_dqn or self.demonstrator_agent is not None:
                    n_step_action, _ = self.agent.action_select(state=n_step_state, epsilon=0, agent_idx=0)
                    n_step_action = self.action_to_batch_idxs(n_step_action, n_step_state).type(torch.LongTensor)
                    if self.debug_mode:
                        print(f'n_step_action: {n_step_action.shape} {n_step_action}')
            if self.profile_time:
                try:
                    print(next_action[0])
                except UnboundLocalError:
                    print(n_step_action[0])
                next_action_t = time.time() - next_action_start
                print(f'next_action_t: {next_action_t*1e3:.3f} ms')

            # get q targets of all actions
            q_targs_start = time.time()
            if self.use_double_dqn and not self.double_dqn_clipping:
                q_targets_next_1 = self.agent.calc_Q_values(next_obs, use_target_network=True, agent_idx=0)
                q_targets_next_2 = self.agent.calc_Q_values(next_obs, use_target_network=True, agent_idx=1)
                if self.debug_mode:
                    print(f'q_targets_next_1: {q_targets_next_1.shape} {q_targets_next_1} \n q_targets_next_2: {q_targets_next_2.shape} {q_targets_next_2}')
            else:
                if not self.use_n_step_dqn or self.demonstrator_agent is not None:
                    q_targets_next = self.agent.calc_Q_values(next_obs, use_target_network=True, agent_idx=None)
                    if self.debug_mode:
                        print(f'q_targets_next: {q_targets_next}')
                if self.use_n_step_dqn or self.demonstrator_agent is not None:
                    n_step_q_targets = self.agent.calc_Q_values(n_step_obs, use_target_network=True, agent_idx=None)
                    if self.debug_mode:
                        print(f'n_step_q_targets: {n_step_q_targets[0].shape}')
            if self.profile_time:
                try:
                    print(q_targets_next[0][0])
                except UnboundLocalError:
                    print(n_step_q_targets[0][0])
                q_targs_t = time.time() - q_targs_start
                print(f'q_targs_t: {q_targs_t*1e3:.3f} ms')

            if self.munchausen_tau > 0:
                if self.use_double_dqn and not self.double_dqn_clipping:
                    raise Exception('Not implemented for munchausen.')

                # copy reward to ensure we don't overwrite buffer
                reward = copy.deepcopy(reward)

                # reshape tensors
                if not self.use_n_step_dqn or self.demonstrator_agent is not None:
                    q_targets_next = [torch.reshape(q_targets_next[head], (self.batch_size, int(q_targets_next[head].shape[0] / self.batch_size))) for head in range(len(q_targets_next))]
                    done = torch.reshape(done, (self.batch_size, 1))
                if self.use_n_step_dqn or self.demonstrator_agent is not None:
                    n_step_q_targets = [torch.reshape(n_step_q_targets[head], (self.batch_size, int(n_step_q_targets[head].shape[0] / self.batch_size))) for head in range(len(n_step_q_targets))]
                    n_step_done = torch.reshape(n_step_done, (self.batch_size, 1))

                # calculate entropy term with logsum
                if not self.use_n_step_dqn or self.demonstrator_agent is not None:
                    logsum = [torch.logsumexp((q_targets_next[head] - q_targets_next[head].max(1)[0].unsqueeze(-1)) / self.munchausen_tau, 1) for head in range(len(q_targets_next))]
                    tau_log_pi_next = [q_targets_next[head] - q_targets_next[head].max(1)[0].unsqueeze(-1) - self.munchausen_tau*logsum[head].unsqueeze(-1) for head in range(len(q_targets_next))]
                    # print(f'logsum: {logsum.shape} {logsum}')
                    # print(f'tau log pi next: {tau_log_pi_next.shape} {tau_log_pi_next}')
                if self.use_n_step_dqn or self.demonstrator_agent is not None:
                    n_step_logsum = [torch.logsumexp((n_step_q_targets[head] - n_step_q_targets[head].max(1)[0].unsqueeze(-1)) / self.munchausen_tau, 1) for head in range(len(n_step_q_targets))]
                    n_step_tau_log_pi = [n_step_q_targets[head] - n_step_q_targets[head].max(1)[0].unsqueeze(-1) - self.munchausen_tau*n_step_logsum[head].unsqueeze(-1) for head in range(len(n_step_q_targets))]

                # get the target networks's softmax policy
                if not self.use_n_step_dqn or self.demonstrator_agent is not None:
                    pi_target = [F.softmax(q_targets_next[head]/self.munchausen_tau, dim=1) for head in range(len(q_targets_next))]
                    # print(f'pi target: {pi_target.shape} {pi_target}')
                if self.use_n_step_dqn or self.demonstrator_agent is not None:
                    n_step_pi_target = [F.softmax(n_step_q_targets[head]/self.munchausen_tau, dim=1) for head in range(len(n_step_q_targets))]

                # use the target networks's softmax policy to get the q target
                if not self.use_n_step_dqn or self.demonstrator_agent is not None:
                    q_target = [(self.gamma * (pi_target[head] * (q_targets_next[head] - tau_log_pi_next[head]) * (1-done)).sum(1)) for head in range(len(q_targets_next))]
                    # print(f'q target: {q_target.shape} {q_target}')
                if self.use_n_step_dqn or self.demonstrator_agent is not None:
                    # n_step_q_target = [(self.gamma * (n_step_pi_target[head] * (n_step_q_targets[head] - n_step_tau_log_pi[head]) * (1-n_step_done)).sum(1)) for head in range(len(n_step_q_targets))]
                    n_step_q_target = [(n_step_gamma * (n_step_pi_target[head] * (n_step_q_targets[head] - n_step_tau_log_pi[head]) * (1-n_step_done)).sum(1)) for head in range(len(n_step_q_targets))]

                # get the munchausen addon (log policy) term using the logsum trick
                if not self.use_n_step_dqn or self.demonstrator_agent is not None:
                    q_k_targets = self.agent.calc_Q_values(obs, use_target_network=True, agent_idx=None)
                    q_k_targets = [torch.reshape(q_k_targets[head], (self.batch_size, int(q_k_targets[head].shape[0] / self.batch_size))) for head in range(len(q_k_targets))]
                    v_k_target = [q_k_targets[head].max(1)[0] for head in range(len(q_k_targets))]
                    logsum = [torch.logsumexp((q_k_targets[head] - v_k_target[head].unsqueeze(-1)) / self.munchausen_tau, 1) for head in range(len(q_k_targets))]
                    log_pi = [q_k_targets[head] - v_k_target[head].unsqueeze(-1) - self.munchausen_tau * logsum[head].unsqueeze(-1) for head in range(len(q_k_targets))]
                    log_pi_flat = [log_pi[head].flatten() for head in range(len(log_pi))]
                    # if type(self.agent_reward) != list:
                    if not self.multiple_rewards:
                        # refactor tensor so can index correctly below
                        reward = reward.unsqueeze(dim=1)
                    reward = torch.stack([(reward[:, head] + torch.FloatTensor([self.munchausen_alpha]).to(self.agent.device) * torch.clamp(log_pi_flat[head][next_action], min=self.munchausen_lo, max=0)) for head in range(len(log_pi_flat))], dim=1).to(self.agent.device)
                    if self.debug_mode:
                        print(f'q_k_targets: {q_k_targets[0].shape}')
                        print(f'v_k_target: {v_k_target[0].shape}')
                        print(f'logsum: {logsum[0].shape}')
                        print(f'log_pi_flat: {log_pi_flat[0].shape}')
                        print(f'm-reward: {reward.shape} {reward}')
                    # print(f'munchausen reward: {reward.shape} {reward}')
                if self.use_n_step_dqn or self.demonstrator_agent is not None:
                    n_step_q_k_targets = self.agent.calc_Q_values(n_step_obs, use_target_network=True, agent_idx=None)
                    n_step_q_k_targets = [torch.reshape(n_step_q_k_targets[head], (self.batch_size, int(n_step_q_k_targets[head].shape[0] / self.batch_size))) for head in range(len(n_step_q_k_targets))]
                    n_step_v_k_target = [n_step_q_k_targets[head].max(1)[0] for head in range(len(n_step_q_k_targets))]
                    n_step_logsum = [torch.logsumexp((n_step_q_k_targets[head] - n_step_v_k_target[head].unsqueeze(-1)) / self.munchausen_tau, 1) for head in range(len(n_step_q_k_targets))]
                    n_step_log_pi = [n_step_q_k_targets[head] - n_step_v_k_target[head].unsqueeze(-1) - self.munchausen_tau * n_step_logsum[head].unsqueeze(-1) for head in range(len(n_step_q_k_targets))]
                    n_step_log_pi_flat = [n_step_log_pi[head].flatten() for head in range(len(n_step_log_pi))]
                    n_step_return = torch.stack([(n_step_return[:, head] + torch.FloatTensor([self.munchausen_alpha]).to(self.agent.device) * torch.clamp(n_step_log_pi_flat[head][n_step_action], min=self.munchausen_lo, max=0)) for head in range(len(n_step_log_pi_flat))], dim=1).to(self.agent.device)
                    if self.debug_mode:
                        print(f'n_step_q_k_targets: {n_step_q_k_targets[0].shape}')
                        print(f'n_step_v_k_target: {n_step_v_k_target[0].shape}')
                        print(f'n_step_logsum: {n_step_logsum[0].shape}')
                        print(f'n_step_log_pi_flat: {n_step_log_pi_flat[0].shape}')
                        print(f'n-step m-return: {n_step_return.shape} {n_step_return}')


            else:
                q_targ_start = time.time()
                if self.use_double_dqn:
                    if self.double_dqn_clipping:
                        q_target_1 = [(1-done) * self.gamma * q_targets_next[head][next_action_1].squeeze() for head in range(len(q_targets_next))]
                        q_target_2 = [(1-done) * self.gamma * q_targets_next[head][next_action_2].squeeze() for head in range(len(q_targets_next))]
                    else:
                        q_target_1 = [(1-done) * self.gamma * q_targets_next_2[head][next_action_1].squeeze() for head in range(len(q_targets_next_2))]
                        q_target_2 = [(1-done) * self.gamma * q_targets_next_1[head][next_action_2].squeeze() for head in range(len(q_targets_next_1))]
                    if self.debug_mode:
                        print(f'q_target_1: {q_target_1}')
                        print(f'q_target_2: {q_target_2}')
                else:
                    if not self.use_n_step_dqn or self.demonstrator_agent is not None:
                        q_target = [(1-done) * self.gamma * q_targets_next[head][next_action].squeeze() for head in range(len(q_targets_next))]
                        if self.debug_mode:
                            print(f'q_target: {q_target}')
                    if self.use_n_step_dqn or self.demonstrator_agent is not None:
                        n_step_q_target = [(1-n_step_done) * n_step_gamma * n_step_q_targets[head][n_step_action].squeeze() for head in range(len(n_step_q_targets))]
                        if self.debug_mode:
                            print(f'n_step_q_target: {n_step_q_target}')
                if self.profile_time:
                    try:
                        print(q_target[0][0])
                    except UnboundLocalError:
                        print(n_step_q_target[0][0])
                    q_target_t = time.time() - q_targ_start
                    print(f'q_target_t: {q_target_t*1e3:.3f} ms')

            # get td target
            td_targ_start = time.time()
            # if type(self.agent_reward) != list and (self.use_n_step_dqn or self.munchausen_tau == 0):
            if not self.multiple_rewards and (self.use_n_step_dqn or self.munchausen_tau == 0):
                # refactor tensor so can index correctly below
                reward = reward.unsqueeze(dim=1)
            if self.use_double_dqn:
                td_target_1 = [reward[:, head].squeeze() + q_target_1[head].squeeze() for head in range(len(q_target_1))]
                td_target_2 = [reward[:, head].squeeze() + q_target_2[head].squeeze() for head in range(len(q_target_2))]
                if self.debug_mode:
                    print(f'reward: {reward} \n td_target_1: {td_target_1} \n td_target_2: {td_target_2}')
            else:
                if not self.use_n_step_dqn or self.demonstrator_agent is not None:
                    td_target = [reward[:, head].squeeze() + q_target[head].squeeze() for head in range(len(q_target))]
                    if self.debug_mode:
                        print(f'td_target: {td_target}')
                if self.use_n_step_dqn or self.demonstrator_agent is not None:
                    n_step_td_target = [n_step_return[:, head].squeeze() + n_step_q_target[head].squeeze() for head in range(len(n_step_q_target))]
                    if self.debug_mode:
                        print(f'n_step_td_target: {n_step_td_target}')
            if self.profile_time:
                try:
                    print(td_target[0][0])
                except UnboundLocalError:
                    print(n_step_td_target[0][0])
                td_targ_t = time.time() - td_targ_start 
                print(f'td_targ_t: {td_targ_t*1e3:.3f} ms')

        # get td error (loss)
        td_error_start = time.time()
        loss, num_heads = [], 0
        # if self.demonstrator_l2_regularization_weight > 0:
            # l2_loss = self.calc_l2_regularization_loss()
            # if self.debug_mode:
                # print(f'l2 loss: {l2_loss.shape} {l2_loss}')
        # else:
            # l2_loss = 0
        if self.use_double_dqn:
            for head in range(len(q_value_1)):
                _loss = self.loss_function.extract(q_value_1[head], td_target_1[head], reduction='none') + self.loss_function.extract(q_value_2[head], td_target_2[head], reduction='none')
                if self.demonstrator_agent is not None:
                    raise Exception('Not implemented n-step double-DQN')
                loss.append(_loss)
                num_heads += 1
        else:
            for head in range(len(q_value)):
                # calc td loss
                if not self.use_n_step_dqn or self.demonstrator_agent is not None:
                    # use 1-step td loss
                    _loss = self.loss_function.extract(q_value[head], td_target[head], reduction='none')
                else:
                    # use n-step td loss
                    _loss = self.loss_function.extract(q_value[head], n_step_td_target[head], reduction='none') 
                if self.debug_mode:
                    print(f'head {head} td loss: {_loss.shape} {_loss}')
                if self.demonstrator_agent is not None:
                    # 1. calc n-step return loss
                    n_step_loss = self.demonstrator_n_step_return_loss_weight * self.loss_function.extract(q_value[head], n_step_td_target[head], reduction='none') 
                    _loss += n_step_loss
                    if self.debug_mode:
                        print(f'head {head} n-step td loss: {n_step_loss.shape} {n_step_loss}')

                    # 2. calc SL loss (for experiences which are demonstrator)
                    sl_loss_factor = is_demonstrator * self.demonstrator_margin_loss_weight
                    agent_action = self.conv_q_vals_to_greedy_action(q_values, state)
                    batch_agent_action = self.action_to_batch_idxs(agent_action, state).type(torch.LongTensor).to(self.agent.device)
                    margin_function = torch.where(batch_agent_action == action, 0, 1) * self.demonstrator_margin
                    sl_loss = ((q_values[head][batch_agent_action] + margin_function) - q_values[head][action]) * sl_loss_factor
                    _loss += sl_loss
                    if self.debug_mode:
                        print(f'head {head} sl loss: {sl_loss.shape} {sl_loss}')

                    # # 3. add l2 regularization loss
                    # _loss += l2_loss

                    
                loss.append(_loss)
                num_heads += 1

        if self.debug_mode:
            print(f'loss before reducing per-sample losses: {loss}')
        if self.profile_time:
            print(loss[0][0])
            td_error_t = time.time() - td_error_start
            print(f'td_error_t: {td_error_t*1e3:.3f} ms')

        per_start = time.time()
        if self.use_per:
            # use td error (loss) of each sample to update the sample's priority
            for i, replay_idx in enumerate(indices):
                priority = np.sum([loss[h][i].detach().cpu().numpy() for h in range(num_heads)]) # use total loss across heads as priority
                self.buffer.update_priority(replay_idx, priority, is_demonstrator=is_demonstrator[i]==1)

            if importance_sampling_weights is not None:
                # apply importance sampling weights to loss before updating networks
                importance_sampling_weights = importance_sampling_weights.to(self.agent.device)
                for h in range(num_heads):
                    loss[h] *= importance_sampling_weights
        if self.profile_time:
            print(priority)
            per_t = time.time() - per_start
            print(f'per_t: {per_t*1e3:.3f} ms')
        # if self.profile_time:
            # torch.cuda.synchronize(device=self.agent.device)

        # reduce per-sample losses
        reduce_loss_start = time.time()
        if self.loss_function.reduction == 'mean':
            loss = [torch.mean(loss[h]) for h in range(num_heads)]
        else:
            raise Exception(f'Unrecognised per-sample loss reduction {self.loss_function.reduction}')
        if self.debug_mode:
            print(f'loss before reducing head losses: {loss}')
        if self.profile_time:
            print(loss[0])
            reduce_loss_t = time.time() - reduce_loss_start
            print(f'reduce_loss_t: {reduce_loss_t*1e3:.3f} ms')

        # record stats for each head
        record_stats_start = time.time()
        stats['reward'] = [np.mean(reward[:, h].detach().cpu().tolist()) for h in range(num_heads)]
        if self.use_double_dqn:
            stats['td_target_1'] = [np.mean(td_target_1[h].detach().cpu().tolist()) for h in range(num_heads)]
            stats['td_target_2'] = [np.mean(td_target_2[h].detach().cpu().tolist()) for h in range(num_heads)]
        else:
            if not self.use_n_step_dqn or self.demonstrator_agent is not None:
                stats['td_target'] = [np.mean(td_target[h].detach().cpu().tolist()) for h in range(num_heads)]
            if self.use_n_step_dqn or self.demonstrator_agent is not None:
                stats['n_step_td_target'] = [np.mean(n_step_td_target[h].detach().cpu().tolist()) for h in range(num_heads)]
        if self.use_double_dqn:
            stats['q_value_1'] = [np.mean(q_value_1[h].detach().cpu().tolist()) for h in range(num_heads)]
            stats['q_value_2'] = [np.mean(q_value_2[h].detach().cpu().tolist()) for h in range(num_heads)]
        else:
            stats['q_value'] = [np.mean(q_value[h].detach().cpu().tolist()) for h in range(num_heads)]
            if self.munchausen_tau > 0:
                stats['agent_pi_entropy'] = [np.mean(agent_pi_entropy[h].detach().cpu().cpu().tolist()) for h in range(num_heads)]
        stats['loss'] = [np.mean(loss[h].detach().cpu().tolist()) for h in range(num_heads)]
        if self.profile_time:
            record_stats_t = time.time() - record_stats_start
            print(f'record_stats_t: {record_stats_t*1e3:.3f} ms')

        # sum losses across all heads
        sum_loss_start = time.time()
        loss = torch.sum(torch.stack(loss))
        if self.debug_mode:
            print(f'loss after reducing head losses: {type(loss)} {loss.shape} {loss.requires_grad} {loss}')
        if self.profile_time:
            print(loss)
            sum_loss_t = time.time() - sum_loss_start
            print(f'sum_loss_t: {sum_loss_t*1e3:.3f} ms')

        return loss, stats



    def step_optimizer(self, use_cer):
        optimizer_start = time.time()
        if self.debug_mode:
            print('\n\nStepping optimizer...')

        self.epoch_stats = {}

        # accumulate gradients
        self.optimizer.zero_grad()
        for counter in range(self.accumulate_gradient_factor):
            if self.debug_mode:
                print(f'Gradient accumulation counter: {counter}')

            # loss_start = time.time()
            loss, stats = self.accumulate_gradients(use_cer)
            # if self.profile_time:
                # print(loss.detach().cpu())
                # loss_t = time.time() - loss_start
                # print(f'loss_t: {loss_t*1e3:.3f} ms')

            # compute grads
            backward_start = time.time()
            loss.backward()
            if self.profile_time:
                print(loss.detach().cpu())
                backward_t = time.time() - backward_start
                print(f'backward_t: {backward_t*1e3:.3f} ms')
            
            # update epoch stat trackers
            for key, val in stats.items():
                if key not in self.epoch_stats:
                    self.epoch_stats[key] = val
                else:
                    # take mean
                    self.epoch_stats[key] = (np.array(self.epoch_stats[key]) + np.array(val)) / 2

        if self.gradient_clipping_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=self.gradient_clipping_max_norm)
        elif self.gradient_clipping_clip_value is not None:
            torch.nn.utils.clip_grad_value_(self.agent.parameters(), clip_value=self.gradient_clipping_clip_value)
        else:
            pass

        if self.save_gradients:
            # save gradients
            params = list(self.agent.parameters())
            gradients = np.concatenate(np.array([params[i].grad.detach().cpu().numpy().flatten() for i in range(len(params))]))
            self.epoch_stats['gradients'] = [np.mean(gradients), np.min(gradients), np.max(gradients), np.std(gradients)]

        self.epoch_stats['num_epochs'] = self.epoch_counter
        self.epoch_stats['num_episodes'] = self.episode_counter
        self.epoch_stats['num_actor_steps'] = self.actor_step_counter

        # update networks parameters
        self.optimizer.step()

        if self.profile_time:
            print(self.optimizer)
            optimizer_t = time.time() - optimizer_start
            print(f'>>>>>>>>>>>>>> optimizer_t: {optimizer_t*1e3:.3f} ms <<<<<<<<<<<<<<<')

        self.update_epoch_log(self.epoch_stats)

        # check if should update target networks
        if self.epoch_counter % self.hard_update_target_frequency == 0 or self.soft_update_target_tau is not None:
            self.agent.update_target_network(tau=self.soft_update_target_tau)

        # raise Exception()

    # def calc_l2_regularization_loss(self):
        # l2_parameters = torch.cat([param.view(-1) for param in self.agent.parameters()])
        # return self.demonstrator_l2_regularization_weight * torch.square(l2_parameters).sum()

    def update_epoch_log(self, epoch_stats):
        for key, val in epoch_stats.items():
            self.epochs_log[key].append(val)
        self.epochs_log['episode_counter'] = self.episode_counter
        self.epochs_log['epoch_counter'] = self.epoch_counter

    def get_epsilon(self):
        return self.initial_epsilon - (self.initial_epsilon-self.final_epsilon)*min(1,(self.epoch_counter-self.num_pretraining_epochs)/self.final_epsilon_epoch)

    def get_per_beta(self):
        return self.initial_per_beta + (self.final_per_beta-self.initial_per_beta)*min(1, (self.epoch_counter-self.num_pretraining_epochs)/self.final_per_beta_epoch)



    def get_iterable_episodes(self, reward, filling_buffer, backtrack_step):
        if backtrack_step is None:
            _backtrack_step = -1 # use all experiences as not doing backtrack rollouts
        else:
            _backtrack_step = backtrack_step

        # do any post-episode processing
        if 'retro' in self.agent_reward:
            # retrospectively retrieve sub-tree episodes and rewards and allocate correct next_state, done, etc.
            episodes = process_episodes_into_subtree_episodes(episode_experiences=self.episode_experiences,
                                                               reward=reward,
                                                               agent_reward=self.agent_reward,
                                                               intrinsic_reward=self.intrinsic_reward,
                                                               intrinsic_extrinsic_combiner=self.intrinsic_extrinsic_combiner,
                                                               done=self.done,
                                                               backtrack_step=_backtrack_step,
                                                               filling_buffer=filling_buffer,
                                                               episode_stats=self.episode_stats,
                                                               debug_mode=self.debug_mode)
        else:
            # no post-processing needed, but put episodes in list so is iterable
            episodes = [self.episode_experiences] 
        return episodes


    @torch.no_grad()
    def act(self, num_experiences_to_add, agent, filling_buffer=False, add_after_episode=False):
        '''
        Args:
            num_experiences_to_add (int): Number of experiences to capture and add to buffer.
            add_after_episode (bool): If True, will only add experiences to buffer after
                an episode has been completed.
        '''
        # self.agent.to('cpu')
        self.num_experiences_added = 0 # track number of experiences added to buffer
        if self.debug_mode:
            print('\nActing...')
        while self.num_experiences_added < num_experiences_to_add:
            if not self.env_ready:
                # reset env with new instance
                if self.debug_mode:
                    print('\nNew episode. Fetching instance and resetting env...')
                self.prev_obs, self.prev_action_set, self.prev_state = None, None, None
                reset_start = time.time()

                if self.backtrack_rollout_expert is None or filling_buffer:
                    # reset env normally
                    self.env, self.obs, self.action_set, reward, self.done, info, instance_before_reset = self.reset_env(env=self.env, min_num_envs_to_reset=self.reset_envs_batch)
                else:
                    if self.reached_rollout_expert or self.rollout_attempt_counter >= self.max_attempts_to_reach_expert:
                        if self.curr_backtrack_step is not None:
                            # backtrack a step
                            self.curr_backtrack_step -= 1
                            self.rollout_attempt_counter = 0
                        if self.curr_backtrack_step == -1 or self.curr_backtrack_step is None:
                            # move to next instance and solve with expert
                            if self.debug_mode:
                                if self.curr_backtrack_step == -1:
                                    print('Have matched or beaten backtrack rollout expert on all steps. Moving to next instance.')
                                print('Solving new instance with backtrack rollout expert...')
                            self.env, self.obs, self.action_set, reward, self.done, info, instance_before_reset = self.reset_env(env=self.env, min_num_envs_to_reset=self.reset_envs_batch)
                            self.expert_trajectory, self.expert_return = self.backtracker.solve_instance(instance_before_reset.copy_orig(), self.agent_reward)
                            self.curr_backtrack_step = list(self.expert_trajectory.keys())[-1]
                            self.num_backtrack_expert_instances_solved += 1
                            if self.debug_mode:
                                print(f'Solved env with rollout expert. Rollout expert return: {self.expert_return} | Total # instances solved with expert: {self.num_backtrack_expert_instances_solved}')
                            perform_rollout = False
                        else:
                            # not yet reached expert across all steps in episode, rollout to backtrack step
                            perform_rollout = True
                    else:
                        # not yet matched or beaten rollout expert for current backtrack step, re-try
                        perform_rollout = True
                    self.rollout_attempt_counter += 1
                    if perform_rollout:
                        if self.debug_mode:
                            print(f'Rolling out env up to curr_backtrack_step={self.curr_backtrack_step} with expert...')
                        self.env, self.obs, self.action_set, reward, self.done, info, instance_before_reset = self.backtracker.rollout_env(self.curr_backtrack_step)
                        self.env_ready = True
                        if self.debug_mode:
                            print(f'Rolled out env to step {self.curr_backtrack_step}. Solving rest of instance with agent...')

                if self.profile_time:
                    reset_t = time.time() - reset_start
                    print(f'total reset_t: {reset_t*1e3:.3f} ms')
                if self.debug_mode and self.backtrack_rollout_expert is None:
                    print('Fetched instance and reset env. Solving...')

                if self.max_steps_agent is not None:
                    self.max_steps_agent.before_reset(instance_before_reset)
                agent.before_reset(instance_before_reset)
                if self.intrinsic_reward is not None:
                    if 'retro' in self.agent_reward:
                        # can only calc intrinsic reward when retrospectively know sub-tree episode -> do not reset yet
                        pass
                    else:
                        # calc intrinsic reward at each step in episode -> reset now
                        self.intrinsic_reward.before_reset(instance_before_reset)

                self.action_set = self.action_set.astype(int) # ensure action set is int so gets correctly converted to torch.LongTensor later
                self.state = extract_state_tensors_from_ecole_obs(self.obs, self.action_set)
                self.episode_stats = defaultdict(lambda: 0)
                # if type(self.agent_reward) == list:
                if self.multiple_rewards:
                    if isinstance(self.agent_reward, list):
                        self.episode_stats['R'] = {r: 0 for r in self.agent_reward}
                    elif self.intrinsic_reward is not None:
                        if self.intrinsic_extrinsic_combiner == 'list':
                            self.episode_stats['R'] = {self.agent_reward: 0, self.intrinsic_reward.name: 0}
                        elif self.intrinsic_extrinsic_combiner == 'add':
                            # will combine intrinsic extrinsic rewards into single reward, do not need dict to store separately
                            pass
                        else:
                            raise Exception(f'Not sure how to handle agent_reward={self.agent_reward} intrinsic_reward={self.intrinsic_reward} multiple_rewards={self.multiple_rewards}')
                    else:
                        raise Exception(f'Not sure how to handle agent_reward={self.agent_reward} intrinsic_reward={self.intrinsic_reward} multiple_rewards={self.multiple_rewards}')

                if add_after_episode:
                    # track all experiences in episode for n-step return
                    self.episode_experiences = [] 
                else:
                    # only track last n-steps of experience in episode for n-step return
                    self.episode_experiences = deque(maxlen=self.n_step_return)
                self.episode_stats['num_steps'] = 0 # track number of steps taken in env
                self.ep_start = time.time()

            # ensure action set is int so gets correctly converted to torch.LongTensor later
            self.action_set = self.action_set.astype(int)

            self.prev_obs = copy.deepcopy(self.obs)
            self.prev_action_set = copy.deepcopy(self.action_set)
            self.prev_state = copy.deepcopy(self.state)

            # get action
            if self.episode_stats['num_steps'] >= self.max_steps and self.max_steps_agent is not None:
                # use max_steps_agent
                action, action_idx = self.max_steps_agent.action_select(action_set=self.action_set, obs=self.obs, munchausen_tau=self.munchausen_tau, epsilon=self.get_epsilon(), agent_idx=0, model=self.env.model, done=self.done)
                action = action.tolist()
            else:
                # use agent
                action, action_idx = agent.action_select(action_set=self.action_set, obs=self.obs, munchausen_tau=self.munchausen_tau, epsilon=self.get_epsilon(), agent_idx=0, model=self.env.model, done=self.done)

            # take step in environments
            self.obs, self.action_set, reward, self.done, info = self.env.step(action)
            self.episode_stats['num_steps'] += 1
            self.actor_step_counter += 1
            
            # gather reward
            # if type(self.agent_reward) == list:
            if self.multiple_rewards:
                _reward = []
                if isinstance(self.agent_reward, list):
                    for r in self.agent_reward:
                        _reward.append(reward[r])
                else:
                    _reward.append(reward[self.agent_reward])
            else:
                _reward = reward[self.agent_reward]

            if self.intrinsic_reward is not None:
                if 'retro' not in self.agent_reward:
                    if filling_buffer:
                        train_predictor = False
                    else:
                        train_predictor = True
                    intrinsic_reward = self.intrinsic_reward.extract(self.env.model, self.done, train_predictor=train_predictor)
                    if self.intrinsic_extrinsic_combiner == 'add':
                        _reward += intrinsic_reward
                    elif self.intrinsic_extrinsic_combiner == 'list':
                        if not isinstance(_reward, list):
                            _reward = [_reward]
                        _reward.append(intrinsic_reward)
                    else:
                        raise Exception(f'Unrecognised intrinsic_extrinsic_combiner {self.intrinsic_extrinsic_combiner}')
                    self.episode_stats['intrinsic_R'] += intrinsic_reward
                else:
                    # can only handle intrinsic rewards when know sub-tree
                    pass

            if self.done:
                if self.debug_mode:
                    print('Finished episode.')
                # hack so dont have None values in buffer
                self.obs = copy.deepcopy(self.prev_obs)
                self.action_set = copy.deepcopy(self.prev_action_set)

            # save experience
            self.state = extract_state_tensors_from_ecole_obs(self.obs, self.action_set)
            self.episode_experiences.append(
                    {'prev_state': self.prev_state, 
                     'action': action, 
                     'reward': _reward, 
                     'done': self.done, 
                     'state': self.state, 
                     'n_step_return': None,
                     'n_step_state': None,
                     'n': None,
                     'n_step_done': None,
                     }
                    )

            if (not add_after_episode and len(self.episode_experiences) == self.n_step_return) or self.done:
                if filling_buffer:
                    self.curr_backtrack_step = None # not doing any rollout, use all experiences
                episodes = self.get_iterable_episodes(reward, filling_buffer, self.curr_backtrack_step)

                if self.backtrack_rollout_expert is not None and not filling_buffer:
                    if self.debug_mode:
                        print(f'backtrack_step: {self.curr_backtrack_step} | # attempts at this step: {self.rollout_attempt_counter} | expert_return: {self.expert_return} | agent extrinsic return: {self.episode_stats["extrinsic_R"]}')
                    if self.episode_stats['extrinsic_R'] >= self.expert_return:
                        self.reached_rollout_expert = True
                    else:
                        self.reached_rollout_expert = False

                # if self.curr_backtrack_step is not None:
                    # if self.curr_backtrack_step < 0:
                        # raise Exception()

                # update buffer
                if self.prev_obs is not None:
                    for ep_idx, episode_experiences in enumerate(episodes):
                        if len(episode_experiences) > 0:
                            if self.debug_mode:
                                print(f'\nEpisode {ep_idx+1} of {len(episodes)} | Number of episode_experiences from which to consider adding to buffer: {len(episode_experiences)}')
                            for episode_step in range(len(episode_experiences)):
                                # look n-steps ahead or up to terminal step if < n-steps away
                                # lookahead_step = min(self.n_step_return, len(episode_experiences)-episode_step-1)
                                lookahead_step = min(self.n_step_return-1, len(episode_experiences)-episode_step-1)

                                lookahead_experiences = [episode_experiences[episode_step+i] for i in range(lookahead_step+1)]

                                # OLD
                                # # discount return from n-steps ahead back to current step
                                # R = 0
                                # for exp in reversed(lookahead_experiences):
                                    # R = exp['reward'] + (self.gamma * R)
                                # episode_experiences[episode_step]['n_step_return'] = copy.deepcopy(R)

                                # NEW
                                # calc n-step return up to gamma^{n-1} using forward-view approach https://arxiv.org/pdf/1710.02298.pdf
                                # (gamma^{n} discount is applied later when calculating target for network)
                                n = max(lookahead_step+1, 1)
                                # n = max(lookahead_step, 1)
                                n_step_return = copy.deepcopy(lookahead_experiences[0]['reward'])
                                for k in range(1, n):
                                    if isinstance(n_step_return, list):
                                        # multiple rewards, calc discounted return for each
                                        for r_idx in range(len(n_step_return)):
                                            n_step_return[r_idx] += ( (self.gamma**k) * lookahead_experiences[k]['reward'][r_idx] )
                                    else:
                                        n_step_return += ( (self.gamma**k) * lookahead_experiences[k]['reward'] )

                                # update n-step lookahead data for this episode step
                                episode_experiences[episode_step]['n_step_return'] = n_step_return
                                episode_experiences[episode_step]['n_step_state'] = episode_experiences[episode_step+lookahead_step]['state'] # CHANGE
                                episode_experiences[episode_step]['n_step_done'] = episode_experiences[episode_step+lookahead_step]['done']
                                episode_experiences[episode_step]['n'] = n
                                # if lookahead_step == 0:
                                    # # no more future transitions available, use curr state's next_state as nth state
                                    # episode_experiences[episode_step]['n_step_state'] = episode_experiences[episode_step]['state'] # CHANGE
                                # else:
                                    # # use nth step's state
                                    # episode_experiences[episode_step]['n_step_state'] = episode_experiences[episode_step+lookahead_step]['prev_state'] # CHANGE

                                # # episode_experiences[episode_step]['n_step_state'] = episode_experiences[episode_step+lookahead_step]['state']
                                # if lookahead_step == 0:
                                    # # no more future transitions available, use curr state's next_state as nth state
                                    # episode_experiences[episode_step]['n_step_state'] = episode_experiences[episode_step]['state'] # CHANGE
                                    # episode_experiences[episode_step]['n_step_done'] = episode_experiences[episode_step]['done']
                                # else:
                                    # # use nth step's state
                                    # episode_experiences[episode_step]['n_step_state'] = episode_experiences[episode_step+lookahead_step]['prev_state'] # CHANGE
                                    # episode_experiences[episode_step]['n_step_done'] = episode_experiences[episode_step+lookahead_step-1]['done']

                                # episode_experiences[episode_step]['n_step_state'] = episode_experiences[episode_step+lookahead_step]['state'] # CHANGE

                                # episode_experiences[episode_step]['n'] = lookahead_step+1

                                # episode_experiences[episode_step]['n_step_done'] = episode_experiences[episode_step+lookahead_step]['done']
                                if self.debug_mode:
                                    print(f'step {episode_step}: lookahead_step={lookahead_step} n={n} gamma={self.gamma} n_step_return={n_step_return} set at episode step {episode_step}')

                            # if len(list(self.episode_stats['R'].keys())) > 1:
                            if isinstance(self.episode_stats['R'], dict):
                                # multiple rewards
                                for r_idx, r in enumerate(self.episode_stats['R'].keys()):
                                    self.episode_stats['R'][r] += episode_experiences[0]['reward'][r_idx]
                            else:
                                self.episode_stats['R'] += episode_experiences[0]['reward']
                            if random.random() <= self.prob_add_to_buffer:
                                # add experience from n steps ago to buffer
                                is_demonstrator = agent == self.demonstrator_agent
                                self.buffer.append(Transition(*episode_experiences[0].values()), is_demonstrator=is_demonstrator)
                                self.num_experiences_added += 1
                                if self.debug_mode:
                                    print(f'\nAdded episode experience 0 to buffer. {self.num_experiences_added} of {num_experiences_to_add} experiences added.')
                                    for key, val in episode_experiences[0].items():
                                        if type(val) == tuple:
                                            for i, el in enumerate(val):
                                                print(f'{key} tuple idx {i} shape: {el.shape}')
                                        elif type(val) == int or type(val) == float or type(val) == bool:
                                            print(f'{key}: {val}')
                                        else:
                                            try:
                                                print(f'{key}: {val.shape}')
                                            except AttributeError:
                                                print(f'{key}: {val}')


                            # if episode_experiences[-1]['done'] and len(episode_experiences) > 1:
                            if self.done and len(episode_experiences) > 1: # CHANGE
                                # add remaining experiences to buffer
                                for episode_step in range(1, len(episode_experiences)):
                                    # if len(list(self.episode_stats['R'].keys())) > 1:
                                    if isinstance(self.episode_stats['R'], dict):
                                        # multiple rewards
                                        for r_idx, r in enumerate(self.episode_stats['R'].keys()):
                                            self.episode_stats['R'][r] += episode_experiences[episode_step]['reward'][r_idx]
                                    else:
                                        self.episode_stats['R'] += episode_experiences[episode_step]['reward']
                                    if random.random() <= self.prob_add_to_buffer:
                                        # add experience to buffer
                                        exp = episode_experiences[episode_step]
                                        is_demonstrator = agent == self.demonstrator_agent
                                        self.buffer.append(Transition(*exp.values()), is_demonstrator=is_demonstrator)
                                        self.num_experiences_added += 1
                                        if self.debug_mode:
                                            print(f'\nAdded episode experience {episode_step} to buffer. {self.num_experiences_added} of {num_experiences_to_add} experiences added.')
                                            for key, val in exp.items():
                                                if type(val) == tuple:
                                                    for i, el in enumerate(val):
                                                        print(f'{key} tuple idx {i} shape: {el.shape}')
                                                elif type(val) == int or type(val) == float or type(val) == bool:
                                                    print(f'{key}: {val}')
                                                else:
                                                    try:
                                                        print(f'{key}: {val.shape}')
                                                    except AttributeError:
                                                        print(f'{key}: {val}')

                    if not self.done:
                        # not yet finished epsiode, update self.episode_experiences with above calculations
                        self.episode_experiences = episode_experiences

                else:
                    raise Exception('prev_obs should not be None if adding to buffer.')

            # update trackers
            if 'primal_integral' in reward.keys():
                self.episode_stats['primal_integral'] += abs(reward['primal_integral'])
            if 'dual_integral' in reward.keys():
                self.episode_stats['dual_integral'] += abs(reward['dual_integral'])
            if 'primal_dual_integral' in reward.keys():
                self.episode_stats['primal_dual_integral'] = self.episode_stats['primal_integral'] - self.episode_stats['dual_integral']

            if self.done or (self.episode_stats['num_steps'] >= self.max_steps and self.max_steps_agent is None):
                # finished episode
                self.env_ready = False
                self.ep_end = time.time()
                self.episode_stats['episode_run_time'] = self.ep_end - self.ep_start
                self.episode_stats['num_nodes'] = info['num_nodes']
                self.episode_stats['solving_time'] = info['solving_time']
                self.episode_stats['lp_iterations'] = info['lp_iterations']
                self.episode_stats['epsilon'] = self.get_epsilon()
                self.episode_stats['per_beta'] = self.get_per_beta()
                self.episode_stats['elapsed_training_time'] = time.time() - self.train_start
                self.episode_stats['num_epochs'] = self.epoch_counter
                self.episode_stats['num_episodes'] = self.episode_counter
                self.episode_stats['num_actor_steps'] = self.actor_step_counter
                if self.backtrack_rollout_expert is not None:
                    self.episode_stats['num_backtrack_expert_instances_solved'] = self.num_backtrack_expert_instances_solved

                if not filling_buffer:
                    self.update_episode_log(self.episode_stats)
                    if self.episode_counter % self.episode_log_frequency == 0 and self.episode_log_frequency != float('inf'):
                        print(self.get_episode_log_str())
                    self.episode_counter += 1

        return self.num_experiences_added
                    
    def train(self, num_epochs):
        self.num_epochs = int(num_epochs)

        # init trackers
        self.train_start = time.time()
        self.episode_counter, self.epoch_counter, self.actor_step_counter = 0, 0, 0

        # if self.agent_reward == 'optimal_retro_trajectory_normalised_lp_gain' or self.agent_reward == 'retro_branching':
        if 'retro' in self.agent_reward:
            # can only add experiences to buffer after episode has finished, since need to retrospectively calculate reward
            add_after_episode = True
        else:
            # use n-step deque or no need to calc n-step returns, no need to wait for episode to finish
            add_after_episode = False

        if self.demonstrator_agent is not None:
            # doing DQN from demonstration, must pre-train networks
            # 1. fill demonstrator buffer
            pbar = tqdm(total=self.buffer.demonstrator_buffer_capacity, desc='Filling demonstrator buffer')
            while self.buffer.available_demonstrator_samples < self.buffer.demonstrator_buffer_capacity:
                num_experiences_added = self.act(num_experiences_to_add=1, 
                                                 agent=self.demonstrator_agent, 
                                                 filling_buffer=True,
                                                 add_after_episode=add_after_episode)
                pbar.update(num_experiences_added)
            pbar.refresh()
            time.sleep(1)

            if self.path_to_save is not None:
                if self.save_demonstrator_buffer:
                    save_buffer_start = time.time()
                    filename = self.path_to_save + '/database/demonstrator_buffer.pkl'
                    print(f'Saving demonstrator buffer to {filename}...')
                    with gzip.open(filename, 'wb') as f:
                        pickle.dump(self.buffer, f)
                    print(f'Saved demonstrator buffer data to {filename} in {time.time()-save_buffer_start:.3f} s')

            # 2. pre-train networks on demonstrator's experiences
            pbar = tqdm(total=self.num_pretraining_epochs, desc='Pre-training agent')
            self.epoch_counter = 0
            while self.epoch_counter < self.num_pretraining_epochs:
                self.step_optimizer(use_cer=False)
                if self.epoch_counter % self.checkpoint_frequency == 0 and self.path_to_save is not None:
                    # self.save_checkpoint({'episodes_log': self.episodes_log, 'epochs_log': self.epochs_log}, use_sqlite_database=self.use_sqlite_database)
                    if self.save_thread is not None:
                        self.save_thread.join()
                    self.save_thread = threading.Thread(target=self.save_checkpoint, 
                                              args=({'episodes_log': copy.deepcopy(self.episodes_log), 'epochs_log': copy.deepcopy(self.epochs_log)},
                                                    self.use_sqlite_database,))
                    self.save_thread.start()
                    if self.use_sqlite_database:
                        # reset in-memory logs
                        self.episodes_log = self.reset_episodes_log()
                        self.epochs_log = self.reset_epochs_log()
                self.epoch_counter += 1

                pbar.update(1)
            pbar.refresh()
            time.sleep(1)

            # ensure env and last n experiences are reset before generating agent experiences
            self.env_ready = False


        # fill agent buffer
        pbar = tqdm(total=self.buffer_min_length, desc='Filling agent buffer')
        while self.buffer.available_agent_samples < self.buffer_min_length:
            num_experiences_added = self.act(num_experiences_to_add=1, 
                                             agent=self.agent, 
                                             filling_buffer=True,
                                             add_after_episode=add_after_episode)
            pbar.update(num_experiences_added)
        pbar.refresh()
        time.sleep(1)

        # # # DEBUG
        # # save buffer
        # filename = '/scratch/datasets/torch/debug/observation_45_var_features_replay_buffer.pkl'
        # with gzip.open(filename, 'wb') as f:
            # pickle.dump(self.buffer, f)
            # print(f'Saved replay buffer data to {filename}')
        # raise Exception()

        # train agent
        while self.epoch_counter < self.num_epochs+self.num_pretraining_epochs:
            act_start_t = time.time()
            num_experiences_added = self.act(self.steps_per_update, 
                                             agent=self.agent, 
                                             filling_buffer=False,
                                             add_after_episode=add_after_episode)
            if self.profile_time:
                print(num_experiences_added)
                act_t = time.time() - act_start_t
                print(f'>>>>>>>>>> act_t: {act_t*1e3:.3f} ms')
            for _ in range(int(num_experiences_added/self.steps_per_update)):
                self.step_optimizer(use_cer=self.use_cer)
                if self.epoch_counter % self.checkpoint_frequency == 0 and self.path_to_save is not None:
                    # self.save_checkpoint({'episodes_log': self.episodes_log, 'epochs_log': self.epochs_log}, use_sqlite_database=self.use_sqlite_database)
                    if self.save_thread is not None:
                        self.save_thread.join()
                    self.save_thread = threading.Thread(target=self.save_checkpoint, 
                            args=({'episodes_log': copy.deepcopy(self.episodes_log), 'epochs_log': copy.deepcopy(self.epochs_log)},
                                                    self.use_sqlite_database,))
                    self.save_thread.start()
                    if self.use_sqlite_database:
                        # reset in-memory logs
                        self.episodes_log = self.reset_episodes_log()
                        self.epochs_log = self.reset_epochs_log()
                self.epoch_counter += 1
        self.train_end = time.time()
        if self.path_to_save is not None:
            # self.save_checkpoint({'episodes_log': self.episodes_log, 'epochs_log': self.epochs_log}, use_sqlite_database=self.use_sqlite_database)
            if self.save_thread is not None:
                self.save_thread.join()
            self.save_thread = threading.Thread(target=self.save_checkpoint, 
                                      args=({'episodes_log': copy.deepcopy(self.episodes_log), 'epochs_log': copy.deepcopy(self.epochs_log)},
                                            self.use_sqlite_database,))
            self.save_thread.start()
        print('Trained agent {} on {} episodes ({} epochs) in {} s.'.format(self.agent.name, self.episode_counter, self.epoch_counter, round(self.train_end-self.train_start, 3)))
            

    def reset_episodes_log(self):
        # lists for logging stats at end of each episode
        episodes_log = defaultdict(list)

        # other learner params
        episodes_log['agent_name'] = self.agent.name
        episodes_log['agent_reward'] = self.agent_reward
        episodes_log['agent_device'] = self.agent.device
        episodes_log['batch_size'] = self.batch_size
        episodes_log['lr'] = self.lr
        episodes_log['gamma'] = self.gamma
        episodes_log['threshold_difficulty'] = self.threshold_difficulty
        episodes_log['learner_name'] = self.name

        return episodes_log

    def reset_epochs_log(self):
        epochs_log = defaultdict(list)

        # other learner params
        epochs_log['agent_name'] = self.agent.name
        epochs_log['agent_reward'] = self.agent_reward
        epochs_log['agent_device'] = self.agent.device
        epochs_log['batch_size'] = self.batch_size
        epochs_log['lr'] = self.lr
        epochs_log['gamma'] = self.gamma
        epochs_log['threshold_difficulty'] = self.threshold_difficulty
        epochs_log['learner_name'] = self.name

        return epochs_log






# @ray.remote
# def _remote_reset_env(*args, **kwargs):
    # return _reset_env(*args, **kwargs)

# def _reset_env(*args, **kwargs):
    # pass

# @ray.remote
def _reset_env(instance,
               observation_function,
               information_function,
               reward_function,
               scip_params,
               ecole_seed,
               reproducible_episodes,
               threshold_difficulty=None,
               threshold_agent=None,
               threshold_observation_function=None,
               threshold_information_function=None,
               threshold_reward_function=None,
               threshold_scip_params=None,
               profile_time=False):
    if type(instance) == str:
        # load instance from path
        instance_from_file_start = time.time()
        instance = ecole.scip.Model.from_file(instance)
        if profile_time:
            instance_from_file_t = time.time() - instance_from_file_start
            print(f'instance_from_file_t: {instance_from_file_t*1e3:.3f} ms')

    env = EcoleBranching(observation_function=observation_function,
                         information_function=information_function,
                         reward_function=reward_function,
                         scip_params=scip_params)
    if reproducible_episodes:
        env.seed(ecole_seed)
    else:
        env.seed(random.randint(ecole.RandomEngine.min_seed, ecole.RandomEngine.max_seed))
    instance_before_reset = instance.copy_orig()
    env_reset_start = time.time()
    obs, action_set, reward, done, info = env.reset(instance)
    if profile_time:
        env_reset_t = time.time() - env_reset_start
        print(f'env_reset_t: {env_reset_t*1e3:.3f} ms')

    if obs is not None:
        if threshold_difficulty is not None:
            # check difficulty using threshold agent
            threshold_agent.before_reset(instance_before_reset)
            threshold_env = EcoleBranching(observation_function=threshold_observation_function,
                                             information_function=threshold_information_function,
                                             reward_function=threshold_reward_function,
                                             scip_params=threshold_scip_params)
            if reproducible_episodes:
                threshold_env.seed(ecole_seed)
            else:
                threshold_env.seed(random.randint(ecole.RandomEngine.min_seed, ecole.RandomEngine.max_seed))
            _obs, _action_set, _reward, _done, _info = threshold_env.reset(instance_before_reset.copy_orig())
            while not _done:
                _action, _action_idx = threshold_agent.action_select(action_set=_action_set, obs=_obs, agent_idx=0, model=env.model, done=done)
                _obs, _action_set, _reward, _done, _info = threshold_env.step(_action)
                if _info['num_nodes'] > threshold_difficulty:
                    # already exceeded threshold difficulty, do not pass to agent
                    obs, action_set, reward = None, None, None
                    return env, obs, action_set, reward, done, info, instance_before_reset
            if _info['num_nodes'] <= threshold_difficulty:
                # can give instance to agent to learn on
                return env, obs, action_set, reward, done, info, instance_before_reset
                # return None
        else:
            # can give instance to agent to learn on
            return env, obs, action_set, reward, done, info, instance_before_reset
            # return None

    else:
        # instance was pre-solved, cannot act in environments
        return env, obs, action_set, reward, done, info, instance_before_reset
        # return None
