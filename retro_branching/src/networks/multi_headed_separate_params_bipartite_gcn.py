import retro_branching

import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np
import ml_collections
import copy
import json
import time

class MultiHeadedSeparateParamsBipartiteGCN(torch.nn.Module):
    def __init__(self,
                 device, 
                 config=None,
                 emb_size=64,
                 num_rounds=1,
                 aggregator='add',
                 activation=None,
                 cons_nfeats=5,
                 edge_nfeats=1,
                 var_nfeats=19,
                 num_heads=2,
                 linear_weight_init=None,
                 linear_bias_init=None,
                 layernorm_weight_init=None,
                 layernorm_bias_init=None,
                 head_aggregator=None,
                 profile_time=False,
                 name='gnn',
                 include_edge_features = False,
                 **kwargs):
        '''
        Args:
            aggregator: Which GNN aggregator to use after message passing.
            head_aggregator: How to aggregate output of heads.
                int: Will index head outputs with heads[int]
                'add': Sum heads to get output
                None: Will not aggregate heads
                dict: Specify different head aggregation for training and testing 
                    e.g. head_aggregator={'train': None, 'test': 0} to not aggregate
                    heads during training, but at test time only return output
                    of 0th index head.

        '''

        super().__init__()
        self.device = device

        if num_heads < 2:
            raise Exception('num_heads cannot be < 2 for multiple heads....')

        if config is not None:
            self.init_from_config(config)
        else:
            self.name = name
            self.init_nn_modules(emb_size=emb_size, 
                                 num_rounds=num_rounds, 
                                 cons_nfeats=cons_nfeats, 
                                 edge_nfeats=edge_nfeats, 
                                 var_nfeats=var_nfeats, 
                                 aggregator=aggregator, 
                                 activation=activation, 
                                 num_heads=num_heads,
                                 linear_weight_init=linear_weight_init,
                                 linear_bias_init=linear_bias_init,
                                 layernorm_weight_init=layernorm_weight_init,
                                 layernorm_bias_init=layernorm_bias_init,
                                 head_aggregator=head_aggregator,
                                 profile_time=profile_time,
                                 include_edge_features=include_edge_features)

        self.profile_time = profile_time

    def init_from_config(self, config):
        raise Exception('Not implemented')

    def get_networks(self):
        nets = {}
        for idx, net in enumerate(self.heads_module):
            nets[idx] = net.get_networks()


    def init_nn_modules(self, 
                        emb_size=64, 
                        num_rounds=1, 
                        cons_nfeats=5, 
                        edge_nfeats=1, 
                        var_nfeats=19, 
                        aggregator='add', 
                        activation=None,
                        num_heads=1,
                        linear_weight_init=None,
                        linear_bias_init=None,
                        layernorm_weight_init=None,
                        layernorm_bias_init=None,
                        head_aggregator='add',
                        include_edge_features=False,
                        profile_time=False):
        self.emb_size = emb_size
        self.num_rounds = num_rounds
        self.cons_nfeats = cons_nfeats
        self.edge_nfeats = edge_nfeats
        self.var_nfeats = var_nfeats
        self.aggregator = aggregator
        self.activation = activation
        self.num_heads = num_heads
        self.linear_weight_init = linear_weight_init
        self.linear_bias_init = linear_bias_init
        self.layernorm_weight_init = layernorm_weight_init
        self.layernorm_bias_init = layernorm_bias_init
        self.head_aggregator = head_aggregator
        self.include_edge_features = include_edge_features
        
        self.heads_module = torch.nn.ModuleList([
            retro_branching.src.networks.bipartite_gcn.BipartiteGCN(
                device=self.device,
                emb_size=self.emb_size,
                num_rounds=self.num_rounds,
                aggregator=self.aggregator,
                activation=None,
                cons_nfeats=self.cons_nfeats,
                edge_nfeats=self.edge_nfeats,
                var_nfeats=self.var_nfeats,
                num_heads=1,
                linear_weight_init=self.linear_weight_init,
                linear_bias_init=self.linear_bias_init,
                layernorm_weight_init=self.layernorm_weight_init,
                layernorm_bias_init=self.layernorm_bias_init,
                include_edge_features=self.include_edge_features,
                head_aggregator=None,
                profile_time=profile_time,
                )
            for _ in range(self.num_heads)
            ])

        if self.activation is None:
            self.activation_module = None
        elif self.activation == 'sigmoid':
            self.activation_module = torch.nn.Sigmoid()
        elif self.activation == 'relu':
            self.activation_module = torch.nn.ReLU()
        elif self.activation == 'leaky_relu' or self.activation == 'inverse_leaky_relu':
            self.activation_module = torch.nn.LeakyReLU()
        elif self.activation == 'elu':
            self.activation_module = torch.nn.ELU()
        elif self.activation == 'hard_swish':
            self.activation_module = torch.nn.Hardswish()
        elif self.activation == 'softplus':
            self.activation_module = torch.nn.Softplus()
        elif self.activation == 'mish':
            self.activation_module = torch.nn.Mish()
        elif self.activation == 'softsign':
            self.activation_module = torch.nn.Softsign()
        else:
            raise Exception(f'Unrecognised activation {self.activation}')

    def forward(self, *obs):
        head_output = [torch.stack(self.heads_module[head](*obs)).squeeze(0) for head in range(self.num_heads)]

        # get head aggregator
        if isinstance(self.head_aggregator, dict):
            if self.training:
                head_aggregator = self.head_aggregator['train']
            else:
                head_aggregator = self.head_aggregator['test']
        else:
            head_aggregator = self.head_aggregator

        # check if should aggregate head outputs
        if head_aggregator is None:
            # do not aggregate heads
            pass
        else:
            # aggregate head outputs
            if head_aggregator == 'add':
                head_output = [torch.stack(head_output, dim=0).sum(dim=0)]
            elif head_aggregator == 'mean':
                head_output = [torch.stack(head_output, dim=0).mean(dim=0)]
            elif isinstance(head_aggregator, int):
                head_output = [head_output[head_aggregator]]
            else:
                raise Exception(f'Unrecognised head_aggregator {head_aggregator}')

        # activation
        if self.activation_module is not None:
            head_output = [self.activation_module(head) for head in head_output]
            if self.activation == 'inverse_leaky_relu':
                # invert
                head_output = [-1 * head for head in head_output]

        return head_output

    def create_config(self):
        '''Returns config dict so that can re-initialise easily.'''
        # create networks dict of self.<attribute> key-value pairs
        network_dict = copy.deepcopy(self.__dict__)

        # remove module references to avoid circular references
        del network_dict['_modules']

        for h in range(self.num_heads):
            network_dict[f'head_{h}'] = self.heads_module[h].create_config()

        # create config dict
        config = ml_collections.ConfigDict(network_dict)

        return config