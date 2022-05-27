from retro_branching.networks import BipartiteGCN, BipartiteGCNNoHeads
from retro_branching.agents import RandomAgent, REINFORCEAgent, StrongBranchingAgent, PseudocostBranchingAgent

import torch
import torch_geometric
import torch.nn.functional as F

import copy
import random
import ml_collections
import collections.abc
import numpy as np
import json






class DQNAgent:
    def __init__(self,
                 value_network=None,
                 exploration_network=None,
                 sample_stochastically=False,
                 config=None,
                 device=None,
                 head_aggregator='add',
                 default_muchausen_tau=0,
                 default_epsilon=0,
                 deterministic_mdqn=False,
                 profile_time=False,
                 name='dqn_gnn',
                 **kwargs):
        '''
        Args:
            exploration_agent (torch networks, str, None): Agent to use for exploration. If None,
                will explore with random action. For str, can be one of: 'strong_branching_agent',
                'pseudocost_branching_agent'.
            sample_stochastically (bool): If True, will sample
                agent policy stochastically.
            head_aggregator ('add'): If using multiple heads, use this aggregator
                to reduce the multiple Q-heads to a single scalar value in order
                to evaluate actions for action selection. E.g. If head_aggregator='add',
                will sum output across heads and e.g. choose highest sum action (if exploiting).
            deterministic_mdqn (bool): If False and if munchausen_tau != 0 passed to
                DQNAgent.action_select(), will stochastically sample from softmax policy
                to get greedy action.
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
            if value_network is None:
                raise Exception('Must provide value_network.')
            self.value_network = value_network.to(self.device)
            self.default_epsilon = default_epsilon
            self.default_muchausen_tau = default_muchausen_tau 
            self.deterministic_mdqn = deterministic_mdqn
            self.exploration_network = exploration_network
            self.init_exploration_agent()
            self.sample_stochastically = sample_stochastically 
            self.head_aggregator = head_aggregator
            self.name = name
            self.kwargs = kwargs

        self.target_network = copy.deepcopy(self.value_network)
        self.update_target_network()
        self.profile_time = profile_time

    def init_exploration_agent(self):
        if self.exploration_network is None:
            self.exploration_agent = RandomAgent()
        elif self.exploration_network == 'strong_branching_agent':
            self.exploration_agent = StrongBranchingAgent()
        elif self.exploration_network == 'pseudocost_branching_agent':
            self.exploration_agent = PseudocostBranchingAgent()
        else:
            self.exploration_agent = REINFORCEAgent(device=self.device, policy_network=self.exploration_network)
            self.exploration_agent.eval()

    def init_from_config(self, config):
        if type(config) == str:
            # load from json
            with open(config, 'r') as f:
                json_config = json.load(f)
                config = ml_collections.ConfigDict(json.loads(json_config))

        # TEMPORARY: For where have different networks implementations
        if 'num_heads' in config.value_network.keys():
            NET = BipartiteGCN
        else:
            NET = BipartiteGCNNoHeads

        self.value_network = NET(device=self.device,
                                          config=config.value_network)
        self.value_network.to(self.device)

        self.target_network = copy.deepcopy(self.value_network)
        self.update_target_network()

        if 'exploration_network' in config.keys():
            if config.exploration_network is not None and type(config.exploration_network) != str:
                self.exploration_network = NET(device=self.device,
                                                        config=config.exploration_network)
                self.exploration_network.to(self.device)
            else:
                self.exploration_network = None
        else:
            self.exploration_network = None
            config['sample_exploration_network_stochastically'] = False

        if 'profile_time' not in config:
            config.profile_time = False

        if 'deterministic_mdqn' not in config:
            config.deterministic_mdqn = False

        for key, val in config.agent.items():
            if key != 'device':
                self.__dict__[key] = val

    def get_networks(self):
        return {'value_network': self.value_network,
                'target_network': self.target_network,
                'exploration_network': self.exploration_network}

    def to(self, device):
        for net in self.get_networks().values():
            if net is not None:
                net.to(device)

    def before_reset(self, model):
        self.exploration_agent.before_reset(model)


    def create_config(self):
        '''Returns config dict so that can re-initialise easily.'''
        # create agent dict of self.<attribute> key-value pairs
        agent_dict = {}
        for key, val in self.__dict__.items():
            # remove NET() networks to avoid circular references and no need to save torch tensors
            if type(val) != torch.Tensor and key not in list(self.get_networks().keys()):
                agent_dict[key] = val

        # create config dict
        config = {'agent': ml_collections.ConfigDict(agent_dict),
                  'value_network': self.value_network.create_config(),
                  'target_network': self.target_network.create_config()}

        config = ml_collections.ConfigDict(config)

        return config

    def update_target_network(self, tau=None):
        '''
        If tau is None, performs hard updated. 

        If not None, performs soft update:

        θ_target = τ*θ_value + (1 - τ)*θ_target

        Where tau is the interpolation parameter from PPO paper.
        '''
        if tau is None:
            # hard update
            self.target_network.load_state_dict(self.value_network.state_dict())
        else:
            # soft update
            for target_param, value_param in zip(self.target_network.parameters(), self.value_network.parameters()):
                target_param.data.copy_(tau * value_param.data + (1.0 - tau) * target_param.data)

        # dont track target networks tradients
        for param in self.target_network.parameters():
            param.requires_grad = False
        self.target_network.to(self.device)


    def calc_Q_values(self, obs, use_target_network=False, **kwargs):
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        if not use_target_network:
            if type(obs) == tuple:
                logits = self.value_network(*obs)
            else:
                logits = self.value_network(obs)
        else:
            if type(obs) == tuple:
                logits = self.target_network(*obs)
            else:
                logits = self.target_network(obs)
        # print(prof.table(sort_by='cuda_time_total'))

        # if self.profile_time:
            # torch.cuda.synchronize(self.device)

        return logits

    def action_select(self, **kwargs):
        '''
        state must be either action_set and obs, or a BipartiteNode object with these
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
        if 'munchausen_tau' not in kwargs:
            kwargs['munchausen_tau'] = self.default_muchausen_tau
        if 'epsilon' not in kwargs:
            kwargs['epsilon'] = self.default_epsilon

        if 'state' in kwargs:
            self.obs = (kwargs['state'].constraint_features, kwargs['state'].edge_index, kwargs['state'].edge_attr, kwargs['state'].variable_features)
            self.action_set = torch.as_tensor(kwargs['state'].candidates)
        else:
            # unpack
            self.action_set, self.obs = kwargs['action_set'], kwargs['obs']
            if isinstance(self.action_set, np.ndarray):
                self.action_set = torch.as_tensor(self.action_set)

        self.logits = self.calc_Q_values(self.obs, use_target_network=False)

        if type(self.logits) == list:
            # Q-heads DQN, need to aggregate to get values for each action
            self.preds = [self.logits[head][self.action_set] for head in range(len(self.logits))]

            # get head aggregator
            if isinstance(self.head_aggregator, dict):
                if self.value_network.training:
                    head_aggregator = self.head_aggregator['train']
                else:
                    head_aggregator = self.head_aggregator['test']
            else:
                head_aggregator = self.head_aggregator

            if head_aggregator is None:
                self.preds = torch.stack(self.preds).squeeze(0)
            elif head_aggregator == 'add':
                self.preds = torch.stack(self.preds, dim=0).sum(dim=0)
            elif isinstance(head_aggregator, int):
                self.preds = self.preds[head_aggregator]
            else:
                raise Exception(f'Unrecognised head_aggregator {self.head_aggregator}')
        else:
            # no heads
            self.preds = self.logits[self.action_set]

        if 'state' in kwargs:
            # batch of observations
            self.preds = self.preds.split_with_sizes(tuple(kwargs['state'].num_candidates))
            self.action_set = kwargs['state'].raw_candidates.split_with_sizes(tuple(kwargs['state'].num_candidates))

            if kwargs['epsilon'] > 0:
                # explore
                raise Exception('Have not yet implemented exploration for batched scenarios.')
            else:
                # exploit
                if kwargs['munchausen_tau'] == 0 or self.deterministic_mdqn:
                    # deterministically select greedy action
                    self.action_idx = torch.stack([q.argmax() for q in self.preds])
                else:
                    # use softmax policy to stochastically select action
                    self.preds = [F.softmax(preds/kwargs['munchausen_tau'], dim=0) for preds in self.preds]
                    m = [torch.distributions.Categorical(preds) for preds in self.preds] # init discrete categorical distribution from softmax probs
                    self.action_idx = torch.stack([_m.sample() for _m in m]) # sample action from categorical distribution
                self.action = torch.stack([_action_set[idx] for _action_set, idx in zip(self.action_set, self.action_idx)])
        else:
            # single observation
            if random.random() > kwargs['epsilon']:
                # exploit
                if kwargs['munchausen_tau'] == 0 or self.deterministic_mdqn:
                    # deterministically select greedy action
                    self.action_idx = torch.argmax(self.preds)
                else:
                    # use softmax policy to stochastically select greedy action
                    self.preds = F.softmax(self.preds / kwargs['munchausen_tau'], dim=0)
                    m = torch.distributions.Categorical(self.preds) # init discrete categorical distribution from softmax probs
                    self.action_idx = m.sample() # sample action from categorical distribution
                self.action = self.action_set[self.action_idx.item()]
            else:
                # explore 
                # uniform random action selection
                self.action, self.action_idx = self.exploration_agent.action_select(action_set=self.action_set, obs=self.obs, model=kwargs['model'], done=kwargs['done'], sample_stochastically=self.sample_stochastically)

            # self.action = self.action_set[self.action_idx.item()]

        # print('final | action set: {} action: {} | action idx: {}'.format(self.action_set, self.action, self.action_idx))
        # if self.profile_time:
            # torch.cuda.synchronize(self.device)

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
        return self.value_network.parameters()

    def train(self):
        self.value_network.train()

    def eval(self):
        self.value_network.eval()
