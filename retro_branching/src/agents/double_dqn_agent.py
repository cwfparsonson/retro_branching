from retro_branching.agents import DQNAgent
from retro_branching.networks import BipartiteGCN, BipartiteGCNNoHeads

import torch
import torch.nn.functional as F

import itertools
import copy
import random
import ml_collections
import collections.abc
import numpy as np
import json




class DoubleDQNAgent:
    def __init__(self,
                 value_network_1=None,
                 value_network_2=None,
                 exploration_network=None,
                 sample_exploration_network_stochastically=False,
                 config=None,
                 device=None,
                 default_agent_idx=None,
                 head_aggregator='add',
                 default_epsilon=0,
                 default_muchausen_tau=0,
                 profile_time=False,
                 name='dqn_gnn',
                 **kwargs):
        '''
        Implementation of Clipped Double Q-learning in Fujimoto et al. 2018
        (see https://towardsdatascience.com/double-deep-q-networks-905dd8325412).

        Args:
            sample_exploration_network_stochastically (bool): If True, will sample
                exploration networks agent policy stochastically.
            head_aggregator ('add'): If using multiple heads, use this aggregator
                to reduce the multiple Q-heads to a single scalar value in order
                to evaluate actions for action selection. E.g. If head_aggregator='add',
                will sum output across heads and e.g. choose highest sum action (if exploiting).
            profile_time (bool): If True, will make calls to 
                torch.cuda.synchronize() to enable timing.
        '''
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if config is not None:
            self.init_from_config(config)
        else:
            if value_network_1 is None:
                raise Exception('Must provide value_network_1.')
            if value_network_2 is None:
                raise Exception('Must provide value_network_2.')
            self.value_network_1 = value_network_1.to(self.device)
            self.value_network_2 = value_network_2.to(self.device)
            self.exploration_network = exploration_network
            self.sample_exploration_network_stochastically = sample_exploration_network_stochastically

            self.agent_1 = DQNAgent(device=self.device, value_network=self.value_network_1, exploration_network=copy.deepcopy(self.exploration_network), sample_stochastically=self.sample_exploration_network_stochastically, profile_time=profile_time)
            self.agent_2 = DQNAgent(device=self.device, value_network=self.value_network_2, exploration_network=copy.deepcopy(self.exploration_network), sample_stochastically=self.sample_exploration_network_stochastically, profile_time=profile_time)
            
            self.default_agent_idx = default_agent_idx
            self.head_aggregator = head_aggregator
            self.default_epsilon = default_epsilon
            self.default_muchausen_tau = default_muchausen_tau

            self.name = name
            self.kwargs = kwargs
        self.profile_time = profile_time

    def init_from_config(self, config):
        if type(config) == str:
            # load from json
            with open(config, 'r') as f:
                json_config = json.load(f)
                config = ml_collections.ConfigDict(json.loads(json_config))

        # TEMPORARY: For where have different networks implementations
        if 'num_heads' in config.value_network_1.keys():
            NET = BipartiteGCN
        else:
            NET = BipartiteGCNNoHeads

        self.value_network_1 = NET(device=self.device,
                                   config=config.value_network_1)
        self.value_network_1.to(self.device)
        self.value_network_2 = NET(device=self.device,
                                   config=config.value_network_2)
        self.value_network_2.to(self.device)

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

        if 'profile_time' not in config:
            config.profile_time = False

        self.agent_1 = DQNAgent(device=self.device, value_network=self.value_network_1, exploration_network=copy.deepcopy(self.exploration_network), sample_stochastically=config.sample_exploration_network_stochastically)
        self.agent_1.eval()
        self.agent_2 = DQNAgent(device=self.device, value_network=self.value_network_2, exploration_network=copy.deepcopy(self.exploration_network), sample_stochastically=config.sample_exploration_network_stochastically)
        self.agent_2.eval()

        for key, val in config.agent.items():
            self.__dict__[key] = val

    def get_networks(self):
        return {'value_network_1': self.agent_1.value_network,
                'value_network_2': self.agent_2.value_network,
                'exploration_network': self.agent_1.exploration_network}

    def to(self, device):
        for net in self.get_networks().values():
            if net is not None:
                net.to(device)

    def create_config(self):
        '''Returns config dict so that can re-initialise easily.'''
        # create agent dict of self.<attribute> key-value pairs
        agent_dict = {}
        for key, val in self.__dict__.items():
            # remove NET() networks to avoid circular references and no need to save torch tensors
            if type(val) != torch.Tensor and key not in list(self.get_networks().keys())+['agent_1', 'agent_2']:
                agent_dict[key] = val

        # create config dict
        config = {'agent': ml_collections.ConfigDict(agent_dict),
                  'value_network_1': self.agent_1.value_network.create_config(),
                  'value_network_2': self.agent_2.value_network.create_config(),
                  'target_network_1': self.agent_1.target_network.create_config(),
                  'target_network_2': self.agent_2.target_network.create_config()}
        if self.agent_1.exploration_network is not None:
            if type(self.agent_1.exploration_network) != str:
                config['exploration_network'] = self.agent_1.exploration_network.create_config()
            else:
                config['exploration_network'] = self.agent_1.exploration_network
        else:
            config['exploration_network'] = None

        config = ml_collections.ConfigDict(config)

        return config

    def before_reset(self, model):
        self.agent_1.before_reset(model)
        self.agent_2.before_reset(model)

    def update_target_network(self, tau=None):
        self.agent_1.update_target_network(tau=tau)
        self.agent_2.update_target_network(tau=tau)

    def calc_Q_values(self, obs, use_target_network=False, agent_idx=-1, **kwargs):
        if agent_idx == -1:
            agent_idx = self.default_agent_idx

        agents = [self.agent_1, self.agent_2]
        if agent_idx is None:
            # print('\nget logits from both agents')
            # with torch.autograd.profiler.profile(use_cuda=True) as prof:
            logits = [agent.calc_Q_values(obs=obs, use_target_network=use_target_network) for agent in agents]
            # print(prof.table(sort_by='cuda_time_total'))

            # print('\nfind min logits')
            # torch.cuda.synchronize(device=self.device)
            # with torch.autograd.profiler.profile(use_cuda=True) as prof:
            if type(logits) == list:
                # Q-head DQN
                # import pdb; pdb.set_trace()
                min_logits = []
                for head in range(len(logits[0])):
                    head_logits = []
                    for a_idx in range(len(logits)):
                        head_logits.append(logits[a_idx][head])
                    min_logits.append(torch.stack(head_logits, dim=-1).min(dim=-1))
                logits = [_logits.values for _logits in min_logits]
            else:
                min_logits = torch.stack(logits, dim=-1).min(dim=-1)
                logits = min_logits.values
            # print(prof.table(sort_by='cuda_time_total'))
            # prof.export_chrome_trace('chrome_trace.json')

        else:
            logits = agents[agent_idx].calc_Q_values(obs=obs, use_target_network=use_target_network)

        return logits


    def action_select(self, **kwargs):
        '''
        state must be either action_set and obs, or a BipartiteNode object these
        attributes.

        Use agent_idx=0 to use specific agent. This should be done for e.g. generating
        q values and at test time. Use agent_idx=None for generating TD target 
        (will generate Q values with both value networks and then pick lowest 
        magnitude Q values as TD target to help with overestimation bias).

        kwargs:
            state
            action_set
            obs
            munchausen_tau
            epsilon
            model
            done
        '''
        # check kwargs
        if 'state' not in kwargs:
            if 'action_set' not in kwargs and 'obs' not in kwargs:
                raise Exception('Must provide either state or action_set and obs as kwargs.')
        if 'munchausen_tau' not in kwargs:
            kwargs['munchausen_tau'] = self.default_muchausen_tau
        if 'epsilon' not in kwargs:
            kwargs['epsilon'] = self.default_epsilon
        if 'agent_idx' not in kwargs:
            kwargs['agent_idx'] = 0
        if kwargs['agent_idx'] == -1:
            kwargs['agent_idx'] = self.default_agent_idx

        if 'state' in kwargs:
            # self.obs = (kwargs['state'].constraint_features, kwargs['state'].edge_index, kwargs['state'].edge_attr, kwargs['state'].variable_features)
            self.obs = (kwargs['state'].constraint_features, kwargs['state'].edge_index, kwargs['state'].variable_features)
            self.action_set = torch.as_tensor(kwargs['state'].candidates)
        else:
            # unpack
            self.action_set, self.obs = kwargs['action_set'], kwargs['obs']
            if isinstance(self.action_set, np.ndarray):
                self.action_set = torch.as_tensor(self.action_set)
            # self.action_set = copy.deepcopy(action_set)
            # if type(self.action_set) != list:
                # # self.action_set = torch.LongTensor(self.action_set.tolist())
                # self.action_set = torch.empty(self.action_set.tolist(), device=self.device)
            # else:
                # # self.action_set = torch.LongTensor(self.action_set)
                # self.action_set = torch.empty(self.action_set, device=self.device)
        # if self.profile_time:
            # torch.cuda.synchronize(self.device)

        self.logits = self.calc_Q_values(self.obs, use_target_network=False, agent_idx=kwargs['agent_idx'])
        # if self.profile_time:
            # torch.cuda.synchronize(self.device)
        # print(f'logits: {self.logits}')

        # print(f'action_set: {self.action_set}')
        if type(self.logits) == list:
            # Q-head DQN, need to aggregate to get values for each action
            self.preds = [self.logits[head][self.action_set] for head in range(len(self.logits))]
            # print(f'logits: {self.logits[0].shape} {type(self.logits[0])}')
            # print(f'action_set: {self.action_set.shape} {type(self.action_set)}')
            if self.head_aggregator == 'add':
                self.preds = torch.stack(self.preds, dim=0).sum(dim=0)
            else:
                raise Exception(f'Unrecognised head_aggregator {self.head_aggregator}')
        else:
            # no heads
            self.preds = self.logits[self.action_set]
        # print(f'preds: {self.preds}')

        # if self.profile_time:
            # torch.cuda.synchronize(self.device)

        if 'state' in kwargs:
            # batch of observations
            self.preds = self.preds.split_with_sizes(tuple(kwargs['state'].num_candidates))
            # self.action_set = kwargs['state'].candidates.split_with_sizes(tuple(kwargs['state'].num_candidates))
            self.action_set = kwargs['state'].raw_candidates.split_with_sizes(tuple(kwargs['state'].num_candidates))

            if kwargs['epsilon'] > 0:
                # explore
                raise Exception('Have not yet implemented exploration for batched scenarios.')
                if kwargs['munchausen_tau'] > 0:
                    # use softmax policy to select actions
                    self.preds = [F.softmax(preds / kwargs['munchausen_tau'], dim=0) for preds in self.preds]
                    dists = [torch.distributions.Categorical(preds) for preds in self.preds] # init discrete categorical distribution from softmax probs
                    self.action_idx = torch.stack([m.sample() for m in dists]) # sample action from categorical distribution
                    self.action = torch.stack([_action_set[idx.item()] for _action_set, idx in zip(self.action_set, self.action_idx)])
                else:
                    # act_randomly = (np.random.rand(len(self.action)) < kwargs['epsilon'])
                    # rand_action_idxs = [np.random.choice(np.random.randint(low=0, high=len(acts), size=(1,))[0]) for acts, rand in zip(self.action_set, act_randomly) if rand]
                    # rand_actions = [acts[idx] for acts, idx, rand in zip(self.action_set, rand_action_idxs, act_randomly) if rand]
                    # self.action_idx = torch.LongTensor(rand_action_idxs)
                    # self.action[act_randomly] = torch.LongTensor(rand_actions)
                    pass
            else:
                # exploit
                # use argmax
                self.action_idx = torch.stack([q.argmax() for q in self.preds])
                self.action = torch.stack([_action_set[idx.item()] for _action_set, idx in zip(self.action_set, self.action_idx)])

        else:
            # single observation
            if random.random() > kwargs['epsilon']:
                # exploit
                # take argmax
                self.action_idx = torch.argmax(self.preds)
                self.action = self.action_set[self.action_idx.item()]
            else:
                # explore 
                if kwargs['munchausen_tau'] > 0:
                    # use softmax policy to select action
                    self.preds = F.softmax(self.preds / kwargs['munchausen_tau'], dim=0)
                    m = torch.distributions.Categorical(self.preds) # init discrete categorical distribution from softmax probs
                    self.action_idx = m.sample() # sample action from categorical distribution
                    self.action = self.action_set[self.action_idx.item()]
                else:
                    # uniform random action selection
                    _, self.action_idx = self.agent_1.exploration_agent.action_select(action_set=self.action_set, obs=self.obs, model=kwargs['model'], done=kwargs['done'])
                    self.action = self.action_set[self.action_idx.item()]

        # print('action set: {} action: {} | action idx: {}'.format(self.action_set, self.action, self.action_idx))
        # if self.profile_time:
            # torch.cuda.synchronize(self.device)

        return self.action, self.action_idx

    # def _mask_nan_logits(self, logits, mask_val=-1e8):
        # if type(logits) == list:
            # # Q-head DQN
            # for head in range(len(logits)):
                # logits[head][logits[head] != logits[head]] = mask_val
        # else:
            # logits[logits != logits] = mask_val
        # return logits

    def parameters(self):
        return itertools.chain(self.agent_1.parameters(), self.agent_2.parameters())

    def train(self):
        self.agent_1.train()
        self.agent_2.train()

    def eval(self):
        self.agent_1.eval()
        self.agent_2.eval()
