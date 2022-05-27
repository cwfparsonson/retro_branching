from retro_branching.utils import gen_co_name, ExploreThenStrongBranch, PureStrongBranch
from retro_branching.scip_params import gasse_2019_scip_params, default_scip_params

import ecole

import gzip
import pickle
import numpy as np
from pathlib import Path
import time
import os
import glob

import ray
import psutil
NUM_CPUS = psutil.cpu_count(logical=False)
try:
    ray.init(num_cpus=NUM_CPUS)
except RuntimeError:
    # already initialised ray in script calling dcn sim, no need to init again
    pass











@ray.remote
# def run_sampler(co_class, branching, nrows, ncols, max_steps=None, instance=None):
def run_sampler(co_class, co_class_kwargs, branching, max_steps=None, instance=None):
    '''
    Args:
        branching (str): Branching scheme to use. Must be one of 'explore_then_strong_branch',
            'pure_strong_branch'
        max_steps (None, int): If not None, will terminate episode after max_steps.
    '''
    if instance is None:
        # N.B. Need to init instances and env here since ecole objects are not
        # serialisable and ray requires all args passed to it to be serialisable
        if co_class == 'set_covering':
            instances = ecole.instance.SetCoverGenerator(**co_class_kwargs)
        elif co_class == 'combinatorial_auction':
            instances = ecole.instance.CombinatorialAuctionGenerator(**co_class_kwargs)
        elif co_class == 'capacitated_facility_location':
            instances = ecole.instance.CapacitatedFacilityLocationGenerator(**co_class_kwargs)
        elif co_class == 'maximum_independent_set':
            instances = ecole.instance.IndependentSetGenerator(**co_class_kwargs)
        else:
            raise Exception(f'Unrecognised co_class {co_class}')
        instance = next(instances)
    else:
        # already have an instance
        if type(instance) == str:
            # load instance from path
            instance = ecole.scip.Model.from_file(instance)

    # scip_params = default_scip_params
    scip_params = gasse_2019_scip_params

    if branching == 'explore_then_strong_branch':
        env = ecole.environment.Branching(observation_function=(ExploreThenStrongBranch(expert_probability=0.05), 
                                                                ecole.observation.NodeBipartite()), 
                                          scip_params=scip_params)
    elif branching == 'pure_strong_branch':
        env = ecole.environment.Branching(observation_function=(PureStrongBranch(), 
                                                                ecole.observation.NodeBipartite()), 
                                          scip_params=scip_params)
    else:
        raise Exception('Unrecognised branching {}'.format(branching))

    observation, action_set, _, done, _ = env.reset(instance)
    data_to_save = []
    t = 0
    while not done:
        if branching == 'explore_then_strong_branch':
            # only save samples if they are coming from the expert (strong branching)
            (scores, save_samples), node_observation = observation
        elif branching == 'pure_strong_branch':
            # always save samples since always using strong branching
            save_samples = True
            scores, node_observation = observation
        else:
            raise Exception('Unrecognised branching {}'.format(branching))

        action = action_set[scores[action_set].argmax()]

        if save_samples:
            data = [node_observation, action, action_set, scores]
            data_to_save.append(data)

        observation, action_set, _, done, _ = env.step(action)
        t += 1
        if max_steps is not None:
            if t >= max_steps:
                # stop episode
                break

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
    seed = 0 # new added seeding on 14/12/2021 (before did not use seeding

    # # SC 
    co_class = 'set_covering' # 'set_covering' 'combinatorial_auction' 'capacitated_facility_location' 'maximum_independent_set'
    co_class_kwargs = {'n_rows': 500, 'n_cols': 1000}

    # # CFL
    # co_class = 'capacitated_facility_location' # 'set_covering' 'combinatorial_auction' 'capacitated_facility_location' 'maximum_independent_set'
    # co_class_kwargs = {'n_customers': 5, 'n_facilities': 12}

    # # MIS 
    # co_class = 'maximum_independent_set' # 'set_covering' 'combinatorial_auction' 'capacitated_facility_location' 'maximum_independent_set'
    # co_class_kwargs = {'n_nodes': 58}

    branching = 'explore_then_strong_branch' # 'pure_strong_branch' 'explore_then_strong_branch'
    max_steps = None # None 3

    factor = 20 # 1 20
    path = f'/scratch/datasets/retro_branching/strong_branching/{branching}/{co_class}/max_steps_{max_steps}/{gen_co_name(co_class, co_class_kwargs)}/'
    min_samples = 126000 # 100000 1000

    # path = f'/scratch/datasets/retro_branching/strong_branching/{branching}/junk/'
    # factor = 1 # 1 20
    # NUM_CPUS = 1
    # min_samples = 1000 # 100000 1000

    name = 'samples'

    # (optional) load pre-generated instances (will automatically generate if set instances=None)
    instances = None
    # instances = iter(glob.glob(f'/scratch/datasets/retro_branching/gasse_2019/custom_data/data/instances/setcover/general_500r_1000c_0.05d/*.lp'))

    # init save dir
    path = init_save_dir(path, name)
    print('Generating >={} samples in parallel on {} CPUs and saving to {}'.format(min_samples, NUM_CPUS, os.path.abspath(path)))

    epoch_counter, sample_counter, loop_counter = 0, 0, 0
    ecole.seed(seed)
    # run epochs until gather enough samples
    orig_start = time.time()
    while sample_counter < min_samples:
        print('Starting {} parallel processes...'.format(NUM_CPUS*factor))

        # run parallel processes
        start = time.time()
        result_ids = []
        for _ in range(sample_counter, int(sample_counter+NUM_CPUS*factor)):
            if instances is not None:
                # load next pre-generated instance
                instance = next(instances)
            else:
                # will generate an instance
                instance = None
            result_ids.append(run_sampler.remote(co_class=co_class, 
                                                 co_class_kwargs=co_class_kwargs,
                                                 branching=branching, 
                                                 # nrows=nrows, 
                                                 # ncols=ncols, 
                                                 max_steps=max_steps,
                                                 instance=instance))
            epoch_counter += 1
    
        # collect results
        runs_data_to_save = ray.get(result_ids)
        end = time.time()
        print(f'Completed {NUM_CPUS*factor} parallel processes in {round(end-start, 3)} s.')

        # save collected samples
        for data_to_save in runs_data_to_save:
            for data in data_to_save:
                filename = f'{path}sample_{sample_counter}.pkl'
                with gzip.open(filename, 'wb') as f:
                    pickle.dump(data, f)
                sample_counter += 1

        loop_counter += 1

        run_time = round(time.time() - orig_start, 3)
        time_per_sample = round(run_time / sample_counter, 3)
        time_per_parallel_loop = round((run_time / loop_counter), 3)
        print(f'Generated {sample_counter} of {min_samples} samples after {epoch_counter} epochs / {run_time} s -> mean time per sample: {time_per_sample} s, mean time per parallel loop: {time_per_parallel_loop} s | Saved to {path}')
