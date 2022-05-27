from retro_branching.networks import BipartiteGCN, BipartiteGCNNoHeads

import torch
import torch_geometric
import torch.nn.functional as F

import copy
import random
import ml_collections
import collections.abc
import numpy as np
import json






class Agent:
    def __init__(self,
                 network=None,
                 config=None,
                 device=None,
                 head_aggregator='add',
                 network_name='networks',
                 print_forward_dim_warning=True,
                 name='agent'):
        '''
        Use this class for loading a pre-trained networks and doing test-time inference
        with it. Network can have been trained with any method.

        To select an action, passes observation to networks and selects action
        with highest logit output.
        '''
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.print_forward_dim_warning = print_forward_dim_warning

        if config is not None:
            self.init_from_config(config)
        else:
            if network is None:
                raise Exception('Must provide networks.')
            self.network = network.to(self.device)
            self.head_aggregator = head_aggregator
            self.network_name = network_name
        self.name = name


    def init_from_config(self, config):
        if type(config) == str:
            # load from json
            with open(config, 'r') as f:
                json_config = json.load(f)
                config = ml_collections.ConfigDict(json.loads(json_config))

        # find networks in config
        if 'networks' in config.keys():
            self.network_name, net_config = 'networks', config.network
        elif 'policy_network' in config.keys():
            self.network_name, net_config = 'policy', config.policy_network
        elif 'value_network' in config.keys():
            self.network_name, net_config = 'value_network', config.value_network
        elif 'value_network_1' in config.keys():
            self.network_name, net_config = 'value_network_1', config.value_network_1
        elif 'actor_network' in config.keys():
            self.network_name, net_config = 'actor_network', config.actor_network
        else:
            # config is networks config (is case for e.g. supervised learning, where didn't train with specific agent)
            self.network_name, net_config = 'networks', config

        # TEMPORARY: For where have different networks implementations
        if 'num_heads' in net_config.keys():
            NET = BipartiteGCN
        else:
            NET = BipartiteGCNNoHeads

        self.network = NET(device=self.device,
                           config=net_config)
        self.network.to(self.device)

        if 'head_aggregator' in config:
            self.head_aggregator = config.head_aggregator
        else:
            self.head_aggregator = None

    def get_networks(self):
        return {self.network_name: self.network}
    
    def forward(self, obs, **kwargs):
        '''Useful for compatability with some DQN custom test scripts.'''
        if type(obs) == tuple:
            return self.network(*obs, print_warning=self.print_forward_dim_warning)
        else:
            return self.network(obs, print_warning=self.print_forward_dim_warning)

    def before_reset(self, model):
        pass

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def _mask_nan_logits(self, logits, mask_val=-1e8): 
        if type(logits) == list:
            for head in range(len(logits)):
                logits[head][logits[head] != logits[head]] = mask_val
        else:
            logits[logits != logits] = mask_val
        return logits

    def parameters(self):
        return self.network.parameters()

    def action_select(self, **kwargs):
        # check args valid
        if 'state' not in kwargs:
            if 'action_set' not in kwargs and 'obs' not in kwargs:
                raise Exception('Must provide either state or action_set and obs as kwargs.')

        # process observation
        if 'state' in kwargs:
            self.obs = (kwargs['state'].constraint_features, kwargs['state'].edge_index, kwargs['state'].edge_attr, kwargs['state'].variable_features)
            self.action_set = torch.as_tensor(kwargs['state'].candidates)
        else:
            # unpack
            self.action_set, self.obs = kwargs['action_set'], kwargs['obs']
            if isinstance(self.action_set, np.ndarray):
                self.action_set = torch.as_tensor(self.action_set)

        # forward pass through NN
        self.logits = self.forward(self.obs)

        # filter invalid actions
        if type(self.logits) == list:
            # Q-heads DQN, need to aggregate to get values for each action
            self.preds = [self.logits[head][self.action_set] for head in range(len(self.logits))]

            # get head aggregator
            if isinstance(self.head_aggregator, dict):
                if self.network.training:
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

        # get agent action
        if 'state' in kwargs:
            # batch of observations
            self.preds = self.preds.split_with_sizes(tuple(kwargs['state'].num_candidates))
            self.action_set = kwargs['state'].raw_candidates.split_with_sizes(tuple(kwargs['state'].num_candidates))

            # exploit
            self.action_idx = torch.stack([q.argmax() for q in self.preds])
            self.action = torch.stack([_action_set[idx] for _action_set, idx in zip(self.action_set, self.action_idx)])
        else:
            # single observation, exploit
            self.action_idx = torch.argmax(self.preds)
            self.action = self.action_set[self.action_idx.item()]

        return self.action, self.action_idx


