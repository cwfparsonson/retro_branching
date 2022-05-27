from retro_branching.learners import Learner
from retro_branching.environments import EcoleConfiguring, EcoleBranching
from retro_branching.agents import StrongBranchingAgent, PseudocostBranchingAgent, PseudocostBranchingAgent, REINFORCEAgent
from retro_branching.utils import turn_off_scip_heuristics, pad_tensor
from retro_branching.validators import ReinforcementLearningValidator
from retro_branching.networks import BipartiteGCN
from retro_branching.loss_functions import CrossEntropy

import torch
import torch.nn.functional as F
import ecole
import pyscipopt

from collections import defaultdict
import os
import numpy as np
import time
import pathlib
import gzip
import pickle
import json
import copy
import abc
import sigfig





class SupervisedLearner(Learner):
    def __init__(self,
                 agent,
                 train_loader,
                 valid_loader,
                 loss_function=CrossEntropy(),
                 imitation_target='expert_actions',
                 lr=3e-4,
                 bipartite_ranking_alpha=0.2,
                 epoch_log_frequency=1,
                 checkpoint_frequency=1,
                 save_logits_and_target=False,
                 path_to_save='.',
                 name='supervised_learner'):
        '''
        Args:
            imitation_target (str): Target for NN to imitate. Must be one of 
                'expert actions' (imitate action taken by expert), 
                'expert score' (predict score (e.g. reward) obtained by expert)
                'expert_scores' (imitate scores obtained by expert)'.
                'expert_bipartite_ranking' (imitate bipartite ranking)
            loss_function (object): Object with method Object.extract(logits, imitation_target) 
                which takes as arguments logits (dimensions batch_size x num_logits) and 
                imitation_target (dimension batch_size x target_dim) and returns the loss 
                which the NN must learn to minimise to imitate the target.
            save_logits_and_target (bool): If True, will save the logits and imitation
                target for the first batch of each epoch.
        '''
            
        super(SupervisedLearner, self).__init__(agent, path_to_save, name)

        self.agent = agent
        self.agent.train() # put in train mode
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.imitation_target = imitation_target
        self.loss_function = loss_function
        self.lr = lr
        self.bipartite_ranking_alpha = bipartite_ranking_alpha
        self.epoch_log_frequency = epoch_log_frequency
        self.checkpoint_frequency = checkpoint_frequency
        self.save_logits_and_target = save_logits_and_target
        self.path_to_save = path_to_save
        self.name = name

        self.optimizer = self.reset_optimizer(lr=self.lr)
        if self.path_to_save is not None:
            self.path_to_save = self.init_save_dir(path=self.path_to_save)
        
        self.epochs_log = self.init_epochs_log()

    def reset_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=lr)
        return self.optimizer

    def init_epochs_log(self):
        epochs_log = defaultdict(list)
        epochs_log['agent_name'] = self.agent.name
        epochs_log['agent_device'] = self.agent.device
        epochs_log['lr'] = self.lr
        epochs_log['learner_name'] = self.name

        return epochs_log

    def update_epoch_log(self, epoch_stats):
        for key, val in epoch_stats.items():
            self.epochs_log[key].append(val)
        self.epochs_log['epoch_counter'] = self.epoch_counter

    def get_epoch_log_str(self):
        log_str = 'Epoch: {}'.format(self.epoch_counter)
        log_str += ' | Run time: {} s'.format(sigfig.round(self.epochs_log['train_epoch_run_time'][-1]+self.epochs_log['valid_epoch_run_time'][-1], sigfigs=3))
        # log_str += ' | Train loss, acc: {}, {}'.format(round(self.epochs_log['mean_train_loss'][-1], 3), round(self.epochs_log['mean_train_acc'][-1], 3))
        # log_str += ' | Valid loss, acc: {}, {}'.format(round(self.epochs_log['mean_valid_loss'][-1], 3), round(self.epochs_log['mean_valid_acc'][-1], 3))
        log_str += ' | Train loss: {}'.format(sigfig.round(self.epochs_log['mean_train_loss'][-1], sigfigs=3))
        log_str += ' | Valid loss: {}'.format(sigfig.round(self.epochs_log['mean_valid_loss'][-1], sigfigs=3))

        return log_str

    def train(self, num_epochs):
        # save initial checkpoint of networks
        self.train_start = time.time()
        self.epoch_counter = 1
        for epoch in range(num_epochs):
            epoch_stats = defaultdict(lambda: 0)
            epoch_stats['train_logits'], epoch_stats['train_target'], epoch_stats['train_num_candidates'], = [], [], []
            epoch_stats['valid_logits'], epoch_stats['valid_target'], epoch_stats['valid_num_candidates'], = [], [], []

            epoch_stats = self.run_epoch(data_loader=self.train_loader, optimizer=self.optimizer, epoch_stats=epoch_stats)
            epoch_stats = self.run_epoch(data_loader=self.valid_loader, optimizer=None, epoch_stats=epoch_stats)

            self.update_epoch_log(epoch_stats)
            if self.epoch_counter % self.epoch_log_frequency == 0 and self.epoch_log_frequency != float('inf'):
                print(self.get_epoch_log_str())
            if self.path_to_save is not None:
                if self.epoch_counter % self.checkpoint_frequency == 0:
                    self.save_checkpoint({'epochs_log': self.epochs_log})
            self.epoch_counter += 1
        self.train_end = time.time()
        if self.path_to_save is not None:
            self.save_checkpoint({'epochs_log': self.epochs_log})
        print('Trained agent {} on {} epochs in {} s.'.format(self.agent.name, num_epochs, round(self.train_end-self.train_start, 3)))

    def conv_scores_to_bipartite_ranking(self, scores):
        '''Converts scores to bipartite ranking.

        For each element in scores, if element >= (1-alpha)*max_score, label
        as 1. Else, label as 0.
        '''
        threshold = (1 - self.bipartite_ranking_alpha) * torch.max(scores)
        return torch.where(scores >= threshold, 1, 0)

    def run_epoch(self, data_loader, optimizer, epoch_stats):
        '''If optimizer is not None, runs training epoch. If None, runs validation epoch.'''
        # DEBUG
        # torch.autograd.set_detect_anomaly(True)

        saved_logits_and_target = False # track if saved for first batch of epoch
        start = time.time()
        with torch.set_grad_enabled(optimizer is not None):
            for batch in data_loader:
                batch = batch.to(self.agent.device)
                # print(f'\nbatch: {batch}\n constraint_features: {batch.constraint_features.shape} {batch.constraint_features}\n variable_features: {batch.variable_features.shape} {batch.variable_features}')

                # Compute the logits (i.e. pre-softmax activations) according to the agent policy on the concatenated graphs
                logits = self.agent(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
                if type(logits) == list:
                    logits = torch.stack(logits).squeeze(0)
                # print(f'logits: {logits.shape} {logits}')

                # Index the results by the candidates, and split and pad them
                logits = pad_tensor(logits[batch.candidates], batch.num_candidates)
                # print(f'after index, split, and pad: logits: {logits.shape} {logits}')

                if self.imitation_target == 'expert_actions':
                    imitation_target = batch.candidate_choices
                elif self.imitation_target == 'expert_score':
                    imitation_target = batch.score
                    # get predicted score (reward) of best action
                    # print(f'\norig logits: {logits.shape}')
                    logits, _ = torch.max(logits, dim=1)
                    # print(f'\nnew logits: {logits.shape} {logits}')
                    # print(f'imitation target: {imitation_target.shape} {imitation_target}')
                elif self.imitation_target == 'expert_scores':
                    # organise candidate scores into padded tensor so can compare to softmax
                    imitation_target = pad_tensor(batch.candidate_scores, batch.num_candidates)
                elif self.imitation_target == 'expert_bipartite_ranking':
                    imitation_target = pad_tensor(batch.candidate_scores, batch.num_candidates)
                    for idx in range(len(imitation_target)):
                        imitation_target[idx] = self.conv_scores_to_bipartite_ranking(imitation_target[idx])
                else:
                    raise Exception('Unrecognised imitation_target {}.'.format(self.imitation_target))
                loss = self.loss_function.extract(logits, imitation_target)
                # print(f'loss: {loss}')

                if self.save_logits_and_target:
                    if not saved_logits_and_target:
                        # not yet saved logits and target for first batch of this epoch 
                        if optimizer is not None:
                            epoch_stats['train_logits'].append(logits)
                            epoch_stats['train_target'].append(imitation_target)
                            epoch_stats['train_num_candidates'].append(batch.num_candidates)
                        else:
                            epoch_stats['valid_logits'].append(logits)
                            epoch_stats['valid_target'].append(imitation_target)
                            epoch_stats['valid_num_candidates'].append(batch.num_candidates)
                        saved_logits_and_target = True

                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # true_scores = pad_tensor(batch.candidate_scores, batch.num_candidates)
                # true_bestscore = true_scores.max(dim=-1, keepdims=True).values
                
                # predicted_bestindex = logits.max(dim=-1, keepdims=True).indices
                # accuracy = (true_scores.gather(-1, predicted_bestindex) == true_bestscore).float().mean().item()

                if optimizer is not None:
                    # training
                    epoch_stats['mean_train_loss'] += loss.item() * batch.num_graphs
                    # epoch_stats['mean_train_acc'] += accuracy * batch.num_graphs
                    epoch_stats['n_train_samples_processed'] += batch.num_graphs
                else:
                    # validation
                    epoch_stats['mean_valid_loss'] += loss.item() * batch.num_graphs
                    # epoch_stats['mean_valid_acc'] += accuracy * batch.num_graphs
                    epoch_stats['n_valid_samples_processed'] += batch.num_graphs

        # finished epoch
        end = time.time()
        if optimizer is not None:
            # training
            epoch_stats['train_epoch_run_time'] = end - start
            epoch_stats['mean_train_loss'] /= epoch_stats['n_train_samples_processed']
            # epoch_stats['mean_train_acc'] /= epoch_stats['n_train_samples_processed']
        else:
            # validation
            epoch_stats['valid_epoch_run_time'] = end - start
            epoch_stats['mean_valid_loss'] /= epoch_stats['n_valid_samples_processed']
            # epoch_stats['mean_valid_acc'] /= epoch_stats['n_valid_samples_processed']

        return epoch_stats


