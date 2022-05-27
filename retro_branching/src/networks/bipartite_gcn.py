import retro_branching

import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np
import ml_collections
import copy
import json
import time

class BipartiteGCN(torch.nn.Module):
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
                 num_heads=1,
                 head_depth=1,
                 linear_weight_init=None,
                 linear_bias_init=None,
                 layernorm_weight_init=None,
                 layernorm_bias_init=None,
                 head_aggregator=None,
                 include_edge_features=False,
                 use_old_heads_implementation=False,
                 profile_time=False,
                 print_warning=True,
                 name='gnn',
                 **kwargs):
        '''
        Args:
            config (str, ml_collections.ConfigDict()): If not None, will initialise 
                from config dict. Can be either string (path to config.json) or
                ml_collections.ConfigDict object.
            activation (None, 'sigmoid', 'relu', 'leaky_relu', 'inverse_leaky_relu', 'elu', 'hard_swish',
                'softplus', 'mish', 'softsign')
            num_heads (int): Number of heads (final layers) to use. Will use
                head_aggregator to reduce all heads.
            linear_weight_init (None, 'uniform', 'normal', 
                'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal')
            linear_bias_init (None, 'zeros', 'normal')
            layernorm_weight_init (None, 'normal')
            layernorm_bias_init (None, 'zeros', 'normal')
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
                                 head_depth=head_depth,
                                 linear_weight_init=linear_weight_init,
                                 linear_bias_init=linear_bias_init,
                                 layernorm_weight_init=layernorm_weight_init,
                                 layernorm_bias_init=layernorm_bias_init,
                                 head_aggregator=head_aggregator,
                                 include_edge_features=include_edge_features,
                                 use_old_heads_implementation=use_old_heads_implementation)

        self.profile_time = profile_time
        self.printed_warning = False
        self.to(self.device)

    def init_from_config(self, config):
        if type(config) == str:
            # load from json
            with open(config, 'r') as f:
                json_config = json.load(f)
                config = ml_collections.ConfigDict(json.loads(json_config))
        self.name = config.name
        if 'activation' not in config.keys():
            config.activation = None
        if 'num_heads' not in config.keys():
            config.num_heads = 1
        if 'linear_weight_init' not in config.keys():
            config.linear_weight_init = None
        if 'linear_bias_init' not in config.keys():
            config.linear_bias_init = None
        if 'layernorm_weight_init' not in config.keys():
            config.layernorm_weight_init = None
        if 'layernorm_bias_init' not in config.keys():
            config.layernorm_bias_init = None

        if 'head_aggregator' not in config:
            config.head_aggregator = None
        if 'head_depth' not in config:
            config.head_depth = 1

        if 'include_edge_features' not in config:
            config.include_edge_features = False
        if 'use_old_heads_implementation' not in config:
            config.use_old_heads_implementation = False

        self.init_nn_modules(emb_size=config.emb_size, 
                             num_rounds=config.num_rounds, 
                             cons_nfeats=config.cons_nfeats, 
                             edge_nfeats=config.edge_nfeats, 
                             var_nfeats=config.var_nfeats, 
                             aggregator=config.aggregator, 
                             activation=config.activation,
                             num_heads=config.num_heads,
                             head_depth=config.head_depth,
                             linear_weight_init=config.linear_weight_init,
                             linear_bias_init=config.linear_bias_init,
                             layernorm_weight_init=config.layernorm_weight_init,
                             layernorm_bias_init=config.layernorm_bias_init,
                             head_aggregator=config.head_aggregator,
                             include_edge_features=config.include_edge_features,
                             use_old_heads_implementation=config.use_old_heads_implementation)

        if isinstance(self.head_aggregator, ml_collections.config_dict.config_dict.ConfigDict):
            # convert to standard dictionary
            self.head_aggregator = self.head_aggregator.to_dict()

    def get_networks(self):
        return {'networks': self}

    def init_model_parameters(self, init_gnn_params=True, init_heads_params=True):

        def init_params(m):
            if isinstance(m, torch.nn.Linear):
                # weights
                if self.linear_weight_init is None:
                    pass
                elif self.linear_weight_init == 'uniform':
                    torch.nn.init.uniform_(m.weight, a=0.0, b=1.0)
                elif self.linear_weight_init == 'normal':
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                elif self.linear_weight_init == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain(self.activation))
                elif self.linear_weight_init == 'xavier_normal':
                    torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain(self.activation))
                elif self.linear_weight_init == 'kaiming_uniform':
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity=self.activation)
                elif self.linear_weight_init == 'kaiming_normal':
                    torch.nn.init.kaiming_normal_(m.weight, nonlinearity=self.activation)
                    # torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    raise Exception(f'Unrecognised linear_weight_init {self.linear_weight_init}')

                # biases
                if m.bias is not None:
                    if self.linear_bias_init is None:
                        pass
                    elif self.linear_bias_init == 'zeros':
                        torch.nn.init.zeros_(m.bias)
                    elif self.linear_bias_init == 'uniform':
                        torch.nn.init.uniform_(m.bias)
                    elif self.linear_bias_init == 'normal':
                        torch.nn.init.normal_(m.bias)
                    else:
                        raise Exception(f'Unrecognised bias initialisation {self.linear_bias_init}')

            elif isinstance(m, torch.nn.LayerNorm):
                # weights
                if self.layernorm_weight_init is None:
                    pass
                elif self.layernorm_weight_init == 'normal':
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    raise Exception(f'Unrecognised layernorm_weight_init {self.layernorm_weight_init}')

                # biases
                if self.layernorm_bias_init is None:
                    pass
                elif self.layernorm_bias_init == 'zeros':
                    torch.nn.init.zeros_(m.bias)
                elif self.layernorm_bias_init == 'normal':
                    torch.nn.init.normal_(m.bias)
                else:
                    raise Exception(f'Unrecognised layernorm_bias_init {self.layernorm_bias_init}')

        if init_gnn_params:
            # init base GNN params
            self.apply(init_params)

        if init_heads_params:
            # init head output params
            for h in self.heads_module:
                h.apply(init_params)

    def init_nn_modules(self, 
                        emb_size=64, 
                        num_rounds=1, 
                        cons_nfeats=5, 
                        edge_nfeats=1, 
                        var_nfeats=19, 
                        aggregator='add', 
                        activation=None,
                        num_heads=1,
                        head_depth=1,
                        linear_weight_init=None,
                        linear_bias_init=None,
                        layernorm_weight_init=None,
                        layernorm_bias_init=None,
                        head_aggregator='add',
                        include_edge_features=False,
                        use_old_heads_implementation=False):
        self.emb_size = emb_size
        self.num_rounds = num_rounds
        self.cons_nfeats = cons_nfeats
        self.edge_nfeats = edge_nfeats
        self.var_nfeats = var_nfeats
        self.aggregator = aggregator
        self.activation = activation
        self.num_heads = num_heads
        self.head_depth = head_depth
        self.linear_weight_init = linear_weight_init
        self.linear_bias_init = linear_bias_init
        self.layernorm_weight_init = layernorm_weight_init
        self.layernorm_bias_init = layernorm_bias_init
        self.head_aggregator = head_aggregator
        self.include_edge_features = include_edge_features
        self.use_old_heads_implementation = use_old_heads_implementation

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            # torch.nn.LayerNorm(emb_size, emb_size), # added
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            # torch.nn.LayerNorm(emb_size, emb_size), # added
            torch.nn.LeakyReLU(),
        )

        # EDGE EMBEDDING
        if self.include_edge_features:
            self.edge_embedding = torch.nn.Sequential(
                torch.nn.LayerNorm(edge_nfeats),
            )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            # torch.nn.LayerNorm(emb_size, emb_size), # added
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            # torch.nn.LayerNorm(emb_size, emb_size), # added
            torch.nn.LeakyReLU(),
        )

        self.conv_v_to_c = retro_branching.src.networks.bipartite_graph_convolution.BipartiteGraphConvolution(emb_size=emb_size, aggregator=aggregator, include_edge_features=self.include_edge_features)
        self.conv_c_to_v = retro_branching.src.networks.bipartite_graph_convolution.BipartiteGraphConvolution(emb_size=emb_size, aggregator=aggregator, include_edge_features=self.include_edge_features)


        # HEADS
        if self.use_old_heads_implementation:
            # OLD
            self.heads_module = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(emb_size, emb_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(emb_size, 1, bias=True)
                    )
                for _ in range(self.head_depth)
                for _ in range(self.num_heads)
                ])
        else:
            # NEW
            heads = []
            for _ in range(self.num_heads):
                head = []
                for _ in range(self.head_depth):
                    head.append(torch.nn.Linear(emb_size, emb_size))
                    head.append(torch.nn.LeakyReLU())
                head.append(torch.nn.Linear(emb_size, 1, bias=True))
                heads.append(torch.nn.Sequential(*head))
            self.heads_module = torch.nn.ModuleList(heads)


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
    
        self.init_model_parameters()





    def forward(self, *_obs, print_warning=True):
        '''Returns output of each head.'''
        forward_start = time.time()
        if len(_obs) > 1:
            # no need to pre-process observation features
            # if len(_obs) == 4:
                # # old obs where had pointless edge features
                # constraint_features, edge_indices, _, variable_features = _obs
            # else:
                # constraint_features, edge_indices, variable_features = _obs
            constraint_features, edge_indices, edge_features, variable_features = _obs

            # convert to tensors if needed
            if isinstance(constraint_features, np.ndarray):
                constraint_features = torch.from_numpy(constraint_features).to(self.device)
            if isinstance(edge_indices, np.ndarray):
                edge_indices = torch.LongTensor(edge_indices).to(self.device)
            if isinstance(edge_features, np.ndarray):
                edge_features = torch.from_numpy(edge_features).to(self.device).unsqueeze(1)
            if isinstance(variable_features, np.ndarray):
                variable_features = torch.from_numpy(variable_features).to(self.device)

        else:
            # need to pre-process observation features
            obs = _obs[0] # unpack
            start = time.time()
            constraint_features = torch.from_numpy(obs.row_features.astype(np.float32)).to(self.device)
            # edge_indices = torch.from_numpy(obs.edge_features.indices.astype(np.int16)).to(self.device)
            edge_indices = torch.LongTensor(obs.edge_features.indices.astype(np.int16)).to(self.device)
            edge_features = torch.from_numpy(obs.edge_features.values.astype(np.float32)).view(-1, 1).to(self.device)
            variable_features = torch.from_numpy(obs.column_features.astype(np.float32)).to(self.device)
            if self.profile_time:
                print(f'var feat: {variable_features[0][0]}')
                t = time.time() - start
                print(f'to_t: {t*1e3:.3f} ms')

        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        
        # First step: linear embedding layers to a common dimension (64)
        first_step_start = time.time()
        constraint_features = self.cons_embedding(constraint_features)
        if self.include_edge_features:
            edge_features = self.edge_embedding(edge_features)
        if variable_features.shape[1] != self.var_nfeats:
            if print_warning:
                if not self.printed_warning:
                    ans = None
                    while ans not in {'y', 'n'}:
                        ans = input(f'WARNING: variable_features is shape {variable_features.shape} but var_nfeats is {self.var_nfeats}. Will index out extra features. Continue? (y/n): ')
                    if ans == 'y':
                        pass
                    else:
                        raise Exception('User stopped programme.')
                self.printed_warning = True
            variable_features = variable_features[:, 0:self.var_nfeats]
        variable_features = self.var_embedding(variable_features)
        if self.profile_time:
            print(variable_features[0][0])
            first_step_t = time.time() - first_step_start
            print(f'first_step_t: {first_step_t*1e3:.3f} ms')

        # Two half convolutions (message passing round)
        conv_start = time.time()
        for _ in range(self.num_rounds):
            constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
            variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)
            # constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, constraint_features)
            # variable_features = self.conv_c_to_v(constraint_features, edge_indices, variable_features)
        if self.profile_time:
            print(f'{variable_features[0][0]}')
            conv_t = time.time() - conv_start
            print(f'conv_t: {conv_t*1e3:.3f} ms')

        # get output for each head
        head_output_start = time.time()
        head_output = [self.heads_module[head](variable_features).squeeze(-1) for head in range(self.num_heads)]
        if self.profile_time:
            print(f'{head_output[0][0]}')
            head_output_t = time.time() - head_output_start
            print(f'head_output_t: {head_output_t*1e3:.3f} ms')
        # print(f'head outputs: {head_output}')

        # get head aggregator
        head_output_agg_start = time.time()
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

        if self.profile_time:
            print(f'{head_output[0][0]}')
            head_output_agg_t = time.time() - head_output_agg_start
            print(f'head_output_agg_t: {head_output_agg_t*1e3:.3f} ms')

        # activation
        activation_start = time.time()
        if self.activation_module is not None:
            head_output = [self.activation_module(head) for head in head_output]
            if self.activation == 'inverse_leaky_relu':
                # invert
                head_output = [-1 * head for head in head_output]
        if self.profile_time:
            print(f'{head_output[0][0]}')
            activation_t = time.time() - activation_start
            print(f'activation_t: {activation_t*1e3:.3f}')
        # print(f'head outputs after activation: {head_output}')



        # # activation
        # if self.activation_module is not None:
            # head_output = self.activation_module(head_output)

        if self.profile_time:
            print(f'{head_output[0][0]}')
            forward_t = time.time() - forward_start
            print(f'>>> total forward time: {forward_t*1e3:.3f} ms <<<')

        return head_output

    def create_config(self):
        '''Returns config dict so that can re-initialise easily.'''
        # create networks dict of self.<attribute> key-value pairs
        network_dict = copy.deepcopy(self.__dict__)

        # remove module references to avoid circular references
        del network_dict['_modules']

        # create config dict
        config = ml_collections.ConfigDict(network_dict)

        return config