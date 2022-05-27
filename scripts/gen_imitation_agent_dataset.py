'''
Use an imitation agent to generate a data set to learn from for e.g. DQN.
Will save every prob_of_saving steps to ensure samples are drawn from a wide range
of different instances.
Will save the specified reward received at each step by the agent.
'''
from retro_branching.agents import REINFORCEAgent, StrongBranchingAgent
from retro_branching.environments import EcoleBranching
from retro_branching.networks import BipartiteGCN

import ecole
import torch

import gzip
import pickle
import numpy as np
from pathlib import Path
import time
import os
import glob
import random
import copy

import ray
import psutil
NUM_CPUS = psutil.cpu_count(logical=False)
try:
    ray.init(num_cpus=NUM_CPUS)
except RuntimeError:
    # already initialised ray in script calling dcn sim, no need to init again
    pass




@ray.remote
def run_sampler(_agent, nrows, ncols, agent_reward, observation_function, gamma, prob_of_saving):
    '''
    Args:
        branching (str): Branching scheme to use. Must be one of 'explore_then_strong_branch',
            'pure_strong_branch'
        max_steps (None, int): If not None, will terminate episode after max_steps.
    '''
    if _agent == 'strong_branching_agent':
        agent = StrongBranchingAgent() # initialise inside run_sampler() so dont have to pickle
    else:
        agent = copy.deepcopy(_agent)

    # N.B. Need to init instances and env here since ecole objects are not
    # serialisable and ray requires all args passed to it to be serialisable
    instances = ecole.instance.SetCoverGenerator(n_rows=nrows, n_cols=ncols, density=0.05)
    instance = next(instances)

    # init env
    env = EcoleBranching(observation_function=observation_function, # 'default' 'label_solution_values'
                         information_function='default',
                         reward_function='default',
                         scip_params='default') # 'default' 'ml4co_item_placement' 'ml4co_load_balancing' 'ml4co_anonymous'

    # generate sample(s)
    agent.before_reset(instance)
    obs, action_set, reward, done, info = env.reset(instance)
    prev_obs, prev_action_set = copy.deepcopy(obs), copy.deepcopy(action_set)
    rewards, data_to_save, saved_step_indices = [], [], []
    idx = 0
    while not done:
        if agent_reward == 'sb_scores':
            # extract strong branching agent scores -> set as target
            scores = agent.extract(env.model, done)
            target = scores
            action_idx = scores[action_set].argmax()
            action = action_set[action_idx]
        elif agent_reward == 'sb_score':
            # extract strong branching agent chosen action score -> set as target
            scores = agent.extract(env.model, done)
            target = scores[action_set].max()
            action_idx = scores[action_set].argmax()
            action = action_set[action_idx]
        else:
            # get agent action
            action, action_idx = agent.action_select(action_set=action_set, obs=obs, model=env.model, done=done) 

        # take step in env
        obs, action_set, reward, done, info = env.step(action)

        if agent_reward != 'sb_scores' and agent_reward != 'sb_score':
            # extract agent reward -> set as target
            target = reward[agent_reward]

        if random.random() < prob_of_saving:
            # save sample
            data_to_save.append([prev_obs, action, prev_action_set, target])
            saved_step_indices.append(idx)
        else:
            # do not save to ensure that save samples from wide range of different instances
            pass
        prev_obs, prev_action_set = copy.deepcopy(obs), copy.deepcopy(action_set)
        rewards.append(target)
        idx += 1

    if agent_reward != 'sb_scores':
        # discount rewards
        if len(data_to_save) > 0 and gamma > 0:
            # have saved some samples and need to discount rewards at each step in episode
            returns, R = [], 0
            for r in rewards[::-1]:
                R = r + (gamma * R)
                returns.insert(0, R)
            # update saved reward with discounted reward
            for idx, data in zip(saved_step_indices, data_to_save):
                data[-1] = returns[idx]

    return data_to_save


def init_save_dir(path, name):
    _path = path + name + '/'
    counter = 1
    foldername = '{}_{}/'
    while os.path.isdir(_path+foldername.format(name, counter)):
        counter += 1
    foldername = foldername.format(name, counter)
    Path(_path+foldername).mkdir(parents=True, exist_ok=True)
    return _path+foldername


if __name__ == '__main__':
    # init params
    nrows = 100 # 100 500
    ncols = 100 # 100 1000
    agent_reward = 'normalised_lp_gain' # 'num_nodes' 'dual_bound_frac' 'normalised_lp_gain' 'sb_scores' 'sb_score'
    gamma = 0.99 # 0.99 1.0
    imitation_name = 'strong_branching' # 'gnn_21_checkpoint_275' 'strong_branching'
    observation_function = '43_var_features' # 'default' '24_var_features' '43_var_features'
    prob_of_saving = 1 # 1 0.05
    min_samples = 255000 # 100000
    factor = 20
    path_to_save = f'/scratch/datasets/retro_branching/imitation_branching/{imitation_name}/nrows_{nrows}_ncols_{ncols}/{agent_reward}/gamma_{gamma}/obs_{observation_function}/'
    name = 'samples'

    # # IMITATE NEURAL NETWORK
    # path_to_params = '/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_21/checkpoint_275/trained_params.pkl'
    # # init policy
    # policy_network = BipartiteGCN(device='cpu',
                                   # emb_size=64,
                                   # num_rounds=1,
                                   # cons_nfeats=5,
                                   # edge_nfeats=1,
                                   # var_nfeats=19,
                                   # aggregator='add')
    # policy_network.load_state_dict(torch.load(path_to_params, map_location='cpu'))
    # # init agent
    # agent = REINFORCEAgent(policy_network=policy_network,
                           # filter_network=None,
                           # temperature=1.0,
                           # device='cpu')
    # agent.eval()


    # IMITATE STRONG BRANCHING
    agent = 'strong_branching_agent' # initialise inside run_sampler() so dont have to pickle




    # init save dir
    path = init_save_dir(path_to_save, name)
    print('Generating >={} samples in parallel on {} CPUs and saving to {}'.format(min_samples, NUM_CPUS, os.path.abspath(path)))

    # run episodes until gather enough samples
    episode_counter, sample_counter = 0, 0
    while sample_counter < min_samples:
        print('Starting {} parallel processes...'.format(NUM_CPUS*factor))

        # run parallel processes
        start = time.time()
        result_ids = []
        for _ in range(sample_counter, int(sample_counter+NUM_CPUS*factor)):
            result_ids.append(run_sampler.remote(_agent=agent, 
                                                 nrows=nrows, 
                                                 ncols=ncols, 
                                                 agent_reward=agent_reward, 
                                                 observation_function=observation_function,
                                                 gamma=gamma, 
                                                 prob_of_saving=prob_of_saving))
            episode_counter += 1
    
        # collect results
        runs_data_to_save = ray.get(result_ids)
        end = time.time()
        print('Completed {} parallel processes in {} s.'.format(NUM_CPUS*factor, round(end-start, 3)))

        # save collected samples
        for data_to_save in runs_data_to_save:
            for data in data_to_save:
                filename = f'{path}sample_{sample_counter}.pkl'
                with gzip.open(filename, 'wb') as f:
                    pickle.dump(data, f)
                sample_counter += 1

        print('Generated {} of {} samples after {} episodes.'.format(sample_counter, min_samples, episode_counter))





























