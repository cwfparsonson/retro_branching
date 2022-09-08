import retro_branching

import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np
import ml_collections
import copy
import json
import time

class BipartiteGCNNoHeads(torch.nn.Module):
    def __init__(self, 
                 device, 
                 config=None,
                 emb_size=64,
                 num_rounds=1,
                 aggregator='add',
                 activation=None,
                 mask_nan_logits=True,
                 cons_nfeats=5,
                 edge_nfeats=1,
                 var_nfeats=19,
                 name='gnn'):
        '''
        This is the old implementation of the GNN without any DQN heads. Keeping here
        so can still run old models without difficulty.

        Args:
            config (str, ml_collections.ConfigDict()): If not None, will initialise 
                from config dict. Can be either string (path to config.json) or
                ml_collections.ConfigDict object.
            activation (None, 'sigmoid', 'relu', 'leaky_relu', 'elu', 'hard_swish')
        '''
        super().__init__()
        self.device = device

        if config is not None:
            self.init_from_config(config)
        else:
            self.mask_nan_logits = mask_nan_logits
            self.name = name
            self.init_nn_modules(emb_size=emb_size, num_rounds=num_rounds, cons_nfeats=cons_nfeats, edge_nfeats=edge_nfeats, var_nfeats=var_nfeats, aggregator=aggregator, activation=activation)

        self.printed_warning = False
        self.to(self.device)

    def init_from_config(self, config):
        if type(config) == str:
            # load from json
            with open(config, 'r') as f:
                json_config = json.load(f)
                config = ml_collections.ConfigDict(json.loads(json_config))
        try:
            self.mask_nan_logits = config.mask_nan_logits
        except AttributeError:
            self.mask_nan_logits = False
        self.name = config.name
        if 'activation' in config.keys():
            pass
        else:
            config.activation = None
        self.init_nn_modules(emb_size=config.emb_size, num_rounds=config.num_rounds, cons_nfeats=config.cons_nfeats, edge_nfeats=config.edge_nfeats, var_nfeats=config.var_nfeats, aggregator=config.aggregator, activation=config.activation)

    def get_networks(self):
        # return {'networks': self}
        return {'network': self}

    def init_nn_modules(self, emb_size=64, num_rounds=1, cons_nfeats=5, edge_nfeats=1, var_nfeats=19, aggregator='add', activation=None):
        self.emb_size = emb_size
        self.num_rounds = num_rounds
        self.cons_nfeats = cons_nfeats
        self.edge_nfeats = edge_nfeats
        self.var_nfeats = var_nfeats
        self.aggregator = aggregator
        self.activation = activation

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = retro_branching.src.networks.bipartite_graph_convolution.BipartiteGraphConvolution(emb_size=emb_size, aggregator=aggregator)
        self.conv_c_to_v = retro_branching.src.networks.bipartite_graph_convolution.BipartiteGraphConvolution(emb_size=emb_size, aggregator=aggregator)

        output_layers = [
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
            ]
        if self.activation is None:
            pass
        elif self.activation == 'sigmoid':
            output_layers.append(torch.nn.Sigmoid())
        elif self.activation == 'relu':
            output_layers.append(torch.nn.ReLU())
        elif self.activation == 'leaky_relu':
            output_layers.append(torch.nn.LeakyReLU())
        elif self.activation == 'elu':
            output_layers.append(torch.nn.ELU())
        elif self.activation == 'hard_swish':
            output_layers.append(torch.nn.Hardswish())
        else:
            raise Exception(f'Unrecognised activation {self.activation}')
        self.output_module = torch.nn.Sequential(*output_layers)




    def _mask_nan_logits(self, logits, mask_val=-1e8):
        logits[logits != logits] = mask_val
        return logits

    def forward(self, *_obs, print_warning=True):

        if len(_obs) > 1:
            # no need to pre-process observation features
            constraint_features, edge_indices, edge_features, variable_features = _obs
            # constraint_features = constraint_features.to(self.device)
            # edge_indices = edge_indices.to(self.device)
            # edge_features = edge_features.to(self.device)
            # variable_features = variable_features.to(self.device)
        else:
            # need to pre-process observation features
            obs = _obs[0] # unpack
            constraint_features = torch.from_numpy(obs.row_features.astype(np.float32)).to(self.device)
            edge_indices = torch.from_numpy(obs.edge_features.indices.astype(np.int64)).to(self.device)
            edge_features = torch.from_numpy(obs.edge_features.values.astype(np.float32)).view(-1, 1).to(self.device)
            variable_features = torch.from_numpy(obs.variable_features.astype(np.float32)).to(self.device)

        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        
        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        if variable_features.shape[1] != self.var_nfeats:
            if print_warning:
                if not self.printed_warning:
                    print(f'WARNING: variable_features is shape {variable_features.shape} but var_nfeats is {self.var_nfeats}. Will index out extra features.')
                    self.printed_warning = True
            variable_features = variable_features[:, 0:self.var_nfeats]
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions (message passing round)
        for _ in range(self.num_rounds):
            constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
            variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)

        # A final MLP on the variable
        output = self.output_module(variable_features).clone().squeeze(-1) # must clone to avoid in place operation gradient error for some reason?

        return output

    def create_config(self):
        '''Returns config dict so that can re-initialise easily.'''
        # create networks dict of self.<attribute> key-value pairs
        network_dict = copy.deepcopy(self.__dict__)

        # remove module references to avoid circular references
        del network_dict['_modules']

        # create config dict
        config = ml_collections.ConfigDict(network_dict)

        return config
