import ecole
import torch 

from retro_branching.networks import BipartiteGCN
from retro_branching.observations import NodeBipariteWith43VariableFeatures
from retro_branching.loss_functions import MeanSquaredError

import copy
import math
import random
import numpy as np
from collections import deque, defaultdict


class NovelD:
    def __init__(self, 
                 device,
                 prob_use_experience_for_predictor=1,
                 novelty_difference_scaling=0.1,
                 intrinsic_reward_scaling=1e3, # 0.5
                 clipping_threshold=0,
                 observation_function='default',
                 include_edge_features=True,
                 emb_size=64,
                 num_rounds=1,
                 aggregator='add',
                 activation='leaky_relu',
                 head_depth=1,
                 linear_weight_init='normal',
                 linear_bias_init='zeros',
                 layernorm_weight_init=None,
                 layernorm_bias_init=None,
                 head_aggregator=None,
                 learning_rate=1e-4,
                 batch_size=64,
                 debug_mode=False,
                 name='noveld'):
        '''
        Implementation of NovelD https://openreview.net/forum?id=CYUzpnOkFJp

        Returns alpha * intrinsic_reward term, where alpha is a scaling parameter.

        '''
        self.device = device
        self.name = name

        self.prob_use_experience_for_predictor = prob_use_experience_for_predictor
        self.novelty_difference_scaling = novelty_difference_scaling 
        self.intrinsic_reward_scaling = intrinsic_reward_scaling
        self.clipping_threshold = clipping_threshold 

        self.observation_function = observation_function
        self.include_edge_features = include_edge_features

        self.emb_size = emb_size
        self.num_rounds = num_rounds
        self.aggregator = aggregator
        self.activation = activation
        self.head_depth = head_depth
        self.linear_weight_init = linear_weight_init
        self.linear_bias_init = linear_bias_init
        self.layernorm_weight_init = layernorm_weight_init
        self.layernorm_bias_init = layernorm_bias_init
        self.head_aggregator = head_aggregator

        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.debug_mode = debug_mode

        self.init_networks()

    def init_networks(self):
        cons_nfeats = 5
        edge_nfeats = 1
        if self.observation_function == 'default':
            self.obs_generator = ecole.observation.NodeBipartite()
            var_nfeats = 19
        elif self.observation_function == '43_var_features':
            self.obs_generator = NodeBipariteWith43VariableFeatures()
            var_nfeats = 43
        else:
            raise Exception(f'Unrecognised observation_function {self.observation_function} in NovelD.')

        self.predictor_network = BipartiteGCN(device=self.device,
                                              emb_size=self.emb_size,
                                              num_rounds=self.num_rounds,
                                              cons_nfeats=cons_nfeats,
                                              edge_nfeats=edge_nfeats, 
                                              var_nfeats=var_nfeats,
                                              aggregator=self.aggregator,
                                              activation=self.activation,
                                              num_heads=1,
                                              head_depth=self.head_depth,
                                              linear_weight_init=self.linear_weight_init,
                                              linear_bias_init=self.linear_bias_init,
                                              layernorm_weight_init=self.layernorm_weight_init,
                                              layernorm_bias_init=self.layernorm_bias_init,
                                              head_aggregator=self.head_aggregator,
                                              include_edge_features=self.include_edge_features,
                                              profile_time=False)
        self.predictor_network.train()
        self.predictor_network.to(self.device)
        self.optimizer = torch.optim.Adam(self.predictor_network.parameters(), lr=self.learning_rate)

        self.target_network = BipartiteGCN(device=self.device,
                                           emb_size=self.emb_size,
                                           num_rounds=self.num_rounds,
                                           cons_nfeats=cons_nfeats,
                                           edge_nfeats=edge_nfeats, 
                                           var_nfeats=var_nfeats,
                                           aggregator=self.aggregator,
                                           activation=self.activation,
                                           num_heads=1,
                                           head_depth=self.head_depth,
                                           linear_weight_init=self.linear_weight_init,
                                           linear_bias_init=self.linear_bias_init,
                                           layernorm_weight_init=self.layernorm_weight_init,
                                           layernorm_bias_init=self.layernorm_bias_init,
                                           head_aggregator=self.head_aggregator,
                                           include_edge_features=self.include_edge_features,
                                           profile_time=False)
        self.target_network.eval()
        self.target_network.to(self.device)

        self.novelties = deque(maxlen=self.batch_size)
        self.steps_since_last_update = 0

    def before_reset(self, model=None):
        if model is not None:
            if self.observation_function == 'default':
                self.obs_generator = ecole.observation.NodeBipartite()
            elif self.observation_function == '43_var_features':
                self.obs_generator = NodeBipariteWith43VariableFeatures()
            else:
                raise Exception(f'Unrecognised observation_function {self.observation_function} in NovelD.')
            self.obs_generator.before_reset(model)
        else:
            # not using obs generator in intrinsic reward as will be supplied with obs when call extract
            pass

        self.prev_obs = None

    def get_latent_state(self, net, obs):
        if type(obs) == tuple:
            return net(*obs)[0]
        else:
            return net(obs)[0]

    def calc_novelty(self, obs):
        target = self.get_latent_state(self.target_network, obs)
        prediction = self.get_latent_state(self.predictor_network, obs)
        return torch.linalg.vector_norm(target - prediction, ord=2)

    def calc_intrinsic_reward(self, obs, prev_obs):
        prev_obs_novelty = self.calc_novelty(prev_obs)
        obs_novelty = self.calc_novelty(obs)

        novelty_difference = obs_novelty - (self.novelty_difference_scaling * prev_obs_novelty)

        if self.debug_mode:
            print(f's_t novelty: {prev_obs_novelty} | s_t+1 novelty: {obs_novelty} | Unclipped novelty difference: {novelty_difference}')

        return self.intrinsic_reward_scaling*(max(novelty_difference.detach().cpu().item(), self.clipping_threshold)), prev_obs_novelty

    def extract(self, model, done, obs=None, train_predictor=True):
        if done:
            return 0
        
        if model is None and obs is None:
            raise Exception('model and obs cannot both be None.')

        if self.prev_obs is None:
            # first step so do not have previous state with which to calc novelty difference
            if model is not None:
                self.prev_obs = self.obs_generator.extract(model, done)
            else:
                self.prev_obs = obs
            return 0

        if model is not None:
            obs = self.obs_generator.extract(model, done)
        else:
            pass

        if random.random() <= self.prob_use_experience_for_predictor and train_predictor:
            # track gradients and store for updating network
            with torch.enable_grad():
                intrinsic_reward, prev_obs_novelty = self.calc_intrinsic_reward(obs, self.prev_obs)
            self.novelties.append(prev_obs_novelty)
            self.steps_since_last_update += 1
        else:
            # do not track gradients
            with torch.no_grad():
                intrinsic_reward, _ = self.calc_intrinsic_reward(obs, self.prev_obs)
        if self.debug_mode:
            print(f'Intrinsic reward: {type(intrinsic_reward)} {intrinsic_reward} | Steps since last net update: {self.steps_since_last_update} | Batch size: {self.batch_size}')

        if self.batch_size - self.steps_since_last_update == 0:
            if self.debug_mode:
                print(f'{self.steps_since_last_update} steps since last update, updating predictor network...')

            # accumulate gradients
            self.optimizer.zero_grad()
            for idx in range(self.batch_size):
                with torch.enable_grad():
                    # loss = self.loss_function.extract(self.predictions[idx], self.targets[idx])
                    loss = self.novelties[idx]
                if self.debug_mode:
                    print(f'{idx} loss: {type(loss)} {loss}')
                loss.backward()

            # update network
            self.optimizer.step()
            if self.debug_mode:
                print(f'Updated predictor network with batch_size={self.batch_size}')
            self.steps_since_last_update = 0

        # update previous observation so can calc novelty difference at next step
        self.prev_obs = copy.deepcopy(obs)

        return intrinsic_reward