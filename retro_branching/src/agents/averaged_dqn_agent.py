from retro_branching.agents import DQNAgent, REINFORCEAgent, StrongBranchingAgent, PseudocostBranchingAgent, RandomAgent
from retro_branching.networks import BipartiteGCN 

import torch
import torch.nn.functional as F

import itertools
import copy
import random
import ml_collections
import collections.abc
from collections import deque
import numpy as np
import json




class AveragedDQNAgent:
    def __init__(self,
                 value_network=None,
                 exploration_network=None,
                 averaged_dqn_k=10,
                 averaged_dqn_k_freq=1,
                 sample_exploration_network_stochastically=False,
                 config=None,
                 device=None,
                 head_aggregator='add',
                 default_epsilon=0,
                 name='dqn_gnn',
                 **kwargs):
        '''
        Implementation of Averaged DQN https://arxiv.org/pdf/1611.01929.pdf

        Every time agent.update_target_network() is called, will add a new
        agent to the previous k agents (i.e. store last k networks).

        Args:
            averaged_dqn_k (int): Number of previous DQN agents to store in
                memory and use for averaging during learning.
            exploration_agent (torch networks, str, None): Agent to use for exploration. If None,
                will explore with random action. For str, can be one of: 'strong_branching_agent',
                'pseudocost_branching_agent'.
            sample_exploration_network_stochastically (bool): If True, will sample
                agent policy stochastically.
            head_aggregator ('add'): If using multiple heads, use this aggregator
                to reduce the multiple Q-heads to a single scalar value in order
                to evaluate actions for action selection. E.g. If head_aggregator='add',
                will sum output across heads and e.g. choose highest sum action (if exploiting).
        '''
        if device is None:
            self.device = retro_branching.device('cuda' if retro_branching.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if config is not None:
            self.init_from_config(config)
        else:
            if value_network is None:
                raise Exception('Must provide value_network.')
            self.value_network = value_network.to(self.device)
            self.exploration_network = exploration_network
            self.averaged_dqn_k = averaged_dqn_k
            self.averaged_dqn_k_freq = averaged_dqn_k_freq
            self.update_target_network_counter = 0
            self.default_epsilon = default_epsilon
            self.sample_exploration_network_stochastically = sample_exploration_network_stochastically 
            self.head_aggregator = head_aggregator

            # agent to train
            self.agent = DQNAgent(device=self.device,
                                  value_network=copy.deepcopy(self.value_network),
                                  exploration_network=copy.deepcopy(self.exploration_network),
                                  sample_stochastically=self.sample_exploration_network_stochastically,
                                  head_aggregator=self.head_aggregator)

            # prev k agents
            self.prev_k_agents = deque(maxlen=self.averaged_dqn_k)
            for _ in range(self.averaged_dqn_k):
                self.update_prev_k_agents(self.agent)

            self.name = name
            self.kwargs = kwargs

        self.update_target_network()

    def init_from_config(self, config):
        if type(config) == str:
            # load from json
            with open(config, 'r') as f:
                json_config = json.load(f)
                config = ml_collections.ConfigDict(json.loads(json_config))

        NET = BipartiteGCN

        self.value_network = NET(device=self.device,
                                 config=config.value_network)
        self.value_network.to(self.device)

        if 'exploration_network' in config.keys():
            if config.exploration_network is not None and type(config.exploration_network) != str:
                self.exploration_network = NET(device=self.device,
                                               config=config.exploration_network)
                self.exploration_network.to(self.device)
            else:
                self.exploration_network = None
            if 'sample_exploration_network_stochastically' not in config.keys():
                config['sample_exploration_network_stochastically'] = False
        else:
            self.exploration_network = None
            config['sample_exploration_network_stochastically'] = False

        self.agent = DQNAgent(device=self.device,
                              value_network=copy.deepcopy(self.value_network),
                              exploration_network=copy.deepcopy(self.exploration_network),
                              sample_stochastically=config.agent.sample_exploration_network_stochastically,
                              head_aggregator=config.agent.head_aggregator)
        self.agent.eval()
        self.prev_k_agents = deque(maxlen=config.agent.averaged_dqn_k)
        for _ in range(config.agent.averaged_dqn_k):
            self.update_prev_k_agents(self.agent)

        for key, val in config.agent.items():
            self.__dict__[key] = val

    def get_networks(self):
        return {'value_network': self.agent.value_network,
                'exploration_network': self.agent.exploration_network}

    def create_config(self):
        '''Returns config dict so that can re-initialise easily.'''
        # create agent dict of self.<attribute> key-value pairs
        agent_dict = {}
        for key, val in self.__dict__.items():
            # remove NET() networks to avoid circular references and no need to save torch tensors
            if type(val) != retro_branching.Tensor and key not in list(self.get_networks().keys())+['agent', 'prev_k_agents']:
                agent_dict[key] = val

        # DEBUG
        # for key, val in agent_dict.items():
            # print(key, val)

        # create config dict
        config = {'agent': ml_collections.ConfigDict(agent_dict),
                  'value_network': self.agent.value_network.create_config(),
                  'target_network': self.agent.target_network.create_config()}
        if self.agent.exploration_network is not None:
            if type(self.agent.exploration_network) != str:
                config['exploration_network'] = self.agent.exploration_network.create_config()
            else:
                config['exploration_network'] = self.agent.exploration_network
        else:
            config['exploration_network'] = None

        config = ml_collections.ConfigDict(config)

        return config

    def update_prev_k_agents(self, _agent):
        '''Appends frozen agent to prev_k_agents.'''
        # set exploration_network to None so can deep copy 
        _agent.exploration_agent = None

        # create copy to store in prev_k_agents
        agent = copy.deepcopy(_agent)

        # restore exploration agent for original agent
        _agent.init_exploration_agent()
        
        # freeze agent params
        for param in agent.parameters():
            param.requires_grad = False

        # add new agent
        self.prev_k_agents.append(agent)

    def before_reset(self, model):
        self.agent.before_reset(model)

    def update_target_network(self, tau=None):
        self.agent.update_target_network(tau=tau)
        self.update_target_network_counter += 1
        if self.update_target_network_counter % self.averaged_dqn_k_freq == 0:
            self.update_prev_k_agents(self.agent)

    def calc_Q_values(self, obs, use_target_network=False, head_aggregator=None, **kwargs):
        if use_target_network:
            # get logits of previous k agents
            logits = [] # store logits of prev k agents
            for agent in self.prev_k_agents:
                _logits = agent.calc_Q_values(obs, use_target_network=True, **kwargs)
                if type(_logits) == list:
                    # stack list of head outputs into tensor
                    _logits = retro_branching.stack(_logits, dim=0)
                    if head_aggregator is not None:
                        # aggregate heads
                        if head_aggregator == 'add':
                            _logits = _logits.sum(dim=0)
                        else:
                            raise Exception(f'Unrecognised head_aggregator {head_aggregator}')
                    else:
                        # do not aggregate heads
                        pass
                logits.append(_logits)
            # stack prev k agents' logits
            logits = retro_branching.stack(logits, dim=0)
            # average prev k agents' logits
            logits = retro_branching.mean(logits, dim=0)

        else:
            # get logits of current agent
            logits = self.agent.calc_Q_values(obs, use_target_network=False, **kwargs)
            if type(logits) == list:
                # stack list of head outputs into tensor
                logits = retro_branching.stack(logits, dim=0)
                if head_aggregator is not None:
                    # aggregate heads
                    if head_aggregator == 'add':
                        logits = logits.sum(dim=0)
                    else:
                        raise Exception(f'Unrecognised head_aggregator {head_aggregator}')
                else:
                    # do not aggregate heads
                    pass


        return logits

    def action_select(self, **kwargs):
        '''
        state must be either action_set and obs, or a BipartiteNode object these
        attributes.

        kwargs:
            state
            action_set
            obs
            munchausen_tau
            epsilon
            model
            done
        '''
        if 'state' not in kwargs:
            if 'action_set' not in kwargs and 'obs' not in kwargs:
                raise Exception('Must provide either state or action_set and obs as kwargs.')
        if 'epsilon' not in kwargs:
            kwargs['epsilon'] = self.default_epsilon

        if 'state' in kwargs:
            self.obs = (kwargs['state'].constraint_features, kwargs['state'].edge_index, kwargs['state'].edge_attr, kwargs['state'].variable_features)
            self.action_set = retro_branching.LongTensor(kwargs['state'].candidates)
        else:
            # unpack
            action_set, obs = kwargs['action_set'], kwargs['obs']
            self.action_set = copy.deepcopy(action_set)
            self.obs = obs
            if type(self.action_set) != list:
                self.action_set = retro_branching.LongTensor(self.action_set.tolist())
            else:
                self.action_set = retro_branching.LongTensor(self.action_set)

        # need to aggregate heads inside q values to take action
        self.logits = self.calc_Q_values(self.obs, use_target_network=False, head_aggregator=self.head_aggregator)

        # get predictions for valid actions
        self.preds = self.logits[self.action_set]

        if 'state' in kwargs:
            # print('> batch of observations')
            # batch of observations
            self.preds = self.preds.split_with_sizes(tuple(kwargs['state'].num_candidates))
            self.action_set = kwargs['state'].raw_candidates.split_with_sizes(tuple(kwargs['state'].num_candidates))
            # print(f'split action set in action select: {self.action_set}')

            if kwargs['epsilon'] > 0:
                # explore
                raise Exception('Have not yet implemented exploration for batched scenarios.')
            else:
                # exploit
                self.action_idx = retro_branching.stack([q.argmax() for q in self.preds])
                self.action = retro_branching.stack([_action_set[idx] for _action_set, idx in zip(self.action_set, self.action_idx)])

        else:
            # single observation
            if random.random() > kwargs['epsilon']:
                # exploit
                # take argmax
                self.action_idx = retro_branching.argmax(self.preds)
                self.action = self.action_set[self.action_idx.item()]
            else:
                # explore 
                # uniform random action selection
                self.action, self.action_idx = self.agent.exploration_agent.action_select(action_set=self.action_set, obs=self.obs, model=kwargs['model'], done=kwargs['done'], sample_stochastically=self.sample_exploration_network_stochastically)

        # print('final | action set: {} action: {} | action idx: {}'.format(self.action_set, self.action, self.action_idx))

        return self.action, self.action_idx

        

    def _mask_nan_logits(self, logits, mask_val=-1e8): 
        if type(logits) == list:
            # Q-head DQN
            for head in range(len(logits)):
                logits[head][logits[head] != logits[head]] = mask_val
        else:
            logits[logits != logits] = mask_val
        return logits

    def parameters(self):
        return self.agent.parameters()

    def train(self):
        self.agent.train()

    def eval(self):
        self.agent.eval()









