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
import glob
import gzip
import pickle
import json
import copy
import abc
from sqlitedict import SqliteDict


class Learner(abc.ABC):
    def __init__(self, agent, path_to_save='.', name='learner'):
        self.agent = agent
        self.agent.train() # turn on training mode
        self.path_to_save = path_to_save
        self.name = name

        self.checkpoint_counter = 1


    def save_checkpoint(self, logs, use_sqlite_database=False):
        '''
        Args:
            logs (dict): Dict of name (e.g. 'episode_log') to log (usually dict) pairs
                to be saved as a .pkl file.
        '''
        if self.path_to_save is None:
            raise Exception('Must provide path_to_save to save a checkpoint.')

        # make checkpoint dir
        path = self.path_to_save + 'checkpoint_{}/'.format(self.checkpoint_counter)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        # save checkpoint
        start = time.time()
        self.save(logs=logs, path=path, use_sqlite_database=use_sqlite_database)
        print(f'Saved checkpoint {self.checkpoint_counter} to {self.path_to_save} in {time.time()-start:.4f} s.')
        self.checkpoint_counter += 1

    def init_save_dir(self, path='.', agent_name=None, use_sqlite_database=False):
        # init agent name
        if agent_name is None:
            if self.agent.name is None:
                agent_name = 'agent'
            else:
                agent_name = self.agent.name
        
        # init folder to save data 
        _path = path + '/{}/{}/'.format(self.name, agent_name)
        pathlib.Path(_path).mkdir(parents=True, exist_ok=True)

        path_items = glob.glob(_path+'*')
        ids = sorted([int(el.split('_')[-1]) for el in path_items])
        if len(ids) > 0:
            _id = ids[-1] + 1
        else:
            _id = 0
        foldername = f'{agent_name}_{_id}/'

        os.mkdir(_path+foldername)

        if use_sqlite_database:
            os.mkdir(_path+foldername+'/database')

        return _path+foldername

    def save(self, logs, params_name='trained_params', path='.', use_sqlite_database=False):
        # save networks params
        for network_name, network in self.agent.get_networks().items():
            if network is not None and type(network) != str:
                filename = path+f'{network_name}_params.pkl'
                torch.save(network.state_dict(), filename)

        # save agent config
        config = self.agent.create_config().to_json_best_effort()
        with open(path+'config.json', 'w') as f:
            json.dump(config, f)

        # save logs
        for log_name, log in logs.items():
            if use_sqlite_database:
                # update database under database folder
                with SqliteDict(self.path_to_save+f'database/{log_name}.sqlite') as _log:
                    for key, val in log.items():
                        if key in _log and type(val) == list:
                            # extend vals list
                            _log[key] += val
                        else:
                            # create val
                            _log[key] = val
                    _log.commit()
                    _log.close()
            else:
                # save under checkpoint folder
                for log_name, log in logs.items():
                    filename = path+f'{log_name}.pkl'
                    with gzip.open(filename, 'wb') as f:
                        pickle.dump(log, f)

    @abc.abstractmethod
    def train(self, num_epochs):
        pass
