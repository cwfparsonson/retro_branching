from retro_branching.networks import BipartiteGCN, BipartiteGCNNoHeads

import ecole
import torch
import torch.nn.functional as F

from collections import defaultdict
import abc
import math
import copy
import numpy as np
import ml_collections
import json






######################## ECOLE AGENTS ###############################
class REINFORCEAgent:
    def __init__(self, 
                 policy_network=None,
                 filter_network=None,
                 config=None,
                 device=None,
                 temperature=1, 
                 name='rl_gnn',
                 **kwargs):
        '''
        Args:
            config (str, ml_collections.ConfigDict()): If not None, will initialise 
                from config dict. Can be either string (path to config.json) or
                ml_collections.ConfigDict object.
            policy_network (obj)
            filter_network (obj): Before passing the observation to the policy_network,
                filter_network will filter out actions it considers less promising.
            temperature (float, int): 0 < temperature < inf
        
        Kwargs:
            filter_method ('method_1', 'method_2'):
        '''
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if config is not None:
            self.init_from_config(config)
        else:
            self.policy_network = policy_network.to(self.device)
            if filter_network is not None:
                filter_network.to(self.device)
                filter_network.eval() # turn on test mode
            self.filter_network = filter_network
            self.temperature = temperature
            self.name = name
            self.kwargs = kwargs

    def get_networks(self):
        return {'filter_network': self.filter_network,
                'policy_network': self.policy_network}

    def init_from_config(self, config):
        if type(config) == str:
            # load from json
            with open(config, 'r') as f:
                json_config = json.load(f)
                config = ml_collections.ConfigDict(json.loads(json_config))

        # TEMPORARY: For where have different networks implementations
        if 'num_heads' in config.policy_network.keys():
            NET = BipartiteGCN
        else:
            NET = BipartiteGCNNoHeads

        self.policy_network = NET(device=self.device,
                                        config=config.policy_network)
        self.policy_network.to(self.device)

        if config.filter_network is not None:
            self.filter_network = NET(device=self.device,
                                            config=config.filter_network)
            self.filter_network.to(self.device)
            self.filter_network.eval() # turn on test mode
        else:
            self.filter_network = None
        
        for key, val in config.agent.items():
            self.__dict__[key] = val


    def before_reset(self):
        pass

    def get_logits(self, obs, temperature=None):
        if temperature is None:
            temperature = self.temperature
        return self._mask_nan_logits(self.policy_network(obs)) / temperature

    def get_action_probs(self, candidates, logits):
        if type(candidates) != list:
            candidates = torch.LongTensor(candidates.tolist())
        else:
            candidates = torch.LongTensor(candidates)
        return F.softmax(logits[candidates], dim=0)

    def filter_actions(self, obs, action_set):
        if 'filter_method' in self.kwargs:
            filter_method = self.kwargs['filter_method']
        else:
            filter_method = 'method_1'

        # generate action mask filter
        self.filter_logits = self._mask_nan_logits(self.filter_network(obs))
        self.filter_output = F.sigmoid(self.filter_logits)
        threshold = min(torch.max(self.filter_output), 0.5)
        self.filter_mask = np.array([torch.where(self.filter_output >= threshold, 1, 0).detach().cpu().numpy()]).T # column vector

        # add filter mask predictions as feature to variable (column) features
        obs.column_features = np.hstack((obs.column_features, self.filter_mask))

        if filter_method == 'method_1':
            # agent can only choose actions permitted by filter_network -> update action_set
            _action_set = []
            _action_set_lookup = {action for action in action_set}
            for idx, filter in enumerate(self.filter_mask):
                if filter == 1 and idx in _action_set_lookup:
                    _action_set.append(idx)
            if len(_action_set) == 0:
                print('All valid actions were filtered! Occurs if filter_network has only let through invalid actions. Will ignore filter_network and use original action_set.')
                _action_set = copy.deepcopy(action_set)
        elif filter_method == 'method_2':
            # agent can choose any action even if not permitted by filter_network -> use original action_set
            _action_set = action_set
        else:
            raise Exception('Unrecognised filter method {}'.format(filter_method))

        return obs, _action_set

    def create_config(self):
        '''Returns config dict so that can re-initialise easily.'''
        # create agent dict of self.<attribute> key-value pairs
        agent_dict = copy.deepcopy(self.__dict__)

        # remove NET() networks to avoid circular references
        del agent_dict['policy_network']
        del agent_dict['filter_network']

        # create config dict
        config = {'agent': ml_collections.ConfigDict(agent_dict),
                  'policy_network': self.policy_network.create_config()}
        if self.filter_network is not None:
            config['filter_network'] = self.filter_network.create_config()
        else:
            config['filter_network'] = None
        config = ml_collections.ConfigDict(config)

        return config


    def action_select(self, action_set, obs, temperature=None, **kwargs):
        '''
        Args:
            temperature (float, int): 0 < temperature < inf

        Kwargs:
            sample_stochastically (bool): If True, regardless of whether in train
                or evaluation mode, will sample the agent's policy stochastically.
        '''
        if 'sample_stochastically' not in kwargs:
            kwargs['sample_stochastically'] = False

        self.action_set = copy.deepcopy(action_set)
        self.obs = obs
        if self.filter_network is not None:
            # use filter networks to filter actions and update action (variable) features in obs accordingly
            self.obs, self.action_set = self.filter_actions(self.obs, self.action_set)

        if temperature is None:
            temperature = self.temperature
        self.logits = self.get_logits(self.obs, temperature=temperature)
        self.probs = self.get_action_probs(self.action_set, self.logits)
        
        if self.policy_network.training or kwargs['sample_stochastically']:
            m = torch.distributions.Categorical(self.probs) # init discrete categorical distribution from softmax probs
            self.action_idx = m.sample() # sample action from categorical distribution
            action = self.action_set[self.action_idx.item()]
        else:
            self.action_idx = torch.argmax(self.probs)
            action = self.action_set[self.action_idx]

        return action, self.action_idx

    def _mask_nan_logits(self, logits, mask_val=-1e8):
        logits[logits != logits] = mask_val
        return logits

    def parameters(self):
        return self.policy_network.parameters()

    def train(self):
        self.policy_network.train()

    def eval(self):
        self.policy_network.eval()
