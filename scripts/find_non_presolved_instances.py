'''
Use this script to generate instances, call env.reset(instance) and, if the 
instance is not pre-solved by env, save the instance. This saves having to
find non-pre-solved instances during ML training, which can be time-intensive.
'''

# disable annoying tensorboard numpy warning
import warnings
warnings.filterwarnings(action='ignore',
                        category=FutureWarning,
                        module='tensorboard')  # noqa

from retro_branching.environments import EcoleBranching

import ecole

import pathlib
import os
import time
import shutil

import ray
import psutil
NUM_CPUS = psutil.cpu_count(logical=False)
try:
    ray.init(num_cpus=NUM_CPUS)
except RuntimeError:
    # already initialised ray in script calling dcn sim, no need to init again
    pass









@ray.remote
def reset_env(instance,
              path_to_save,
               observation_function,
               information_function,
               reward_function,
               scip_params,
               ecole_seed,
               threshold_difficulty=None,
               threshold_agent=None,
               threshold_observation_function=None,
               threshold_information_function=None,
               threshold_reward_function=None,
               threshold_scip_params=None,
               max_attempts=10000):
    '''
    Takes instance, passes to env.reset(), and checks if the instance
    was pre-solved by env. If pre-solved, does not save instance and returns True.
    If not pre-solved, saves instance before reset and returns False.
    '''

    instance_name = instance.split('/')[-1]
    # load instance from path
    instance = ecole.scip.Model.from_file(instance)

    env = EcoleBranching(observation_function=observation_function,
                         information_function=information_function,
                         reward_function=reward_function,
                         scip_params=scip_params)
    env.seed(ecole_seed)
    counter = 1
    instance_before_reset = instance.copy_orig()
    obs, action_set, reward, done, info = env.reset(instance)

    if obs is not None:
        if threshold_difficulty is not None:
            # check difficulty using threshold agent
            threshold_agent.before_reset(instance_before_reset)
            threshold_env = EcoleBranching(observation_function=threshold_observation_function,
                                             information_function=threshold_information_function,
                                             reward_function=threshold_reward_function,
                                             scip_params=threshold_scip_params)
            threshold_env.seed(ecole_seed)
            _obs, _action_set, _reward, _done, _info = threshold_env.reset(instance_before_reset.copy_orig())
            while not _done:
                _action, _action_idx = threshold_agent.action_select(action_set=_action_set, obs=_obs, agent_idx=0)
                _obs, _action_set, _reward, _done, _info = threshold_env.step(_action)
                if _info['num_nodes'] > threshold_difficulty:
                    # already exceeded threshold difficulty
                    break
            if _info['num_nodes'] <= threshold_difficulty:
                # can give instance to agent to learn on
                instance_before_reset.write_problem(path_to_save+f'/{instance_name}')
                return False
        else:
            # can give instance to agent to learn on
            # return env, obs, action_set, reward, done, info, instance_before_reset
            instance_before_reset.write_problem(path_to_save+f'/{instance_name}')
            return False 

        counter += 1
        if counter > max_attempts:
            raise Exception('Unable to generate valid instance after {} attempts.'.format(max_attempts))
    else:
        # instance was pre-solved, cannot act in environments so dont save
        return True 


def init_save_dir(path, name):
    _path = path + name + '/'
    counter = 1
    foldername = '{}_{}/'
    while os.path.isdir(_path+foldername.format(name, counter)):
        counter += 1
    foldername = foldername.format(name, counter)
    pathlib.Path(_path+foldername).mkdir(parents=True, exist_ok=True)
    return _path+foldername


if __name__ == '__main__':
    # init params
    ecole_seed = 0
    nrows = 500 # 100 500
    ncols = 1000 # 100 1000
    scip_params = 'default'
    min_num_instances_to_find = int(500e3)
    paralellelisation_factor = 20
    threshold_difficulty = None
    threshold_agent = None
    
    path_to_save = f'/scratch/datasets/retro_branching/instances/training/nrows_{nrows}_ncols_{ncols}/scip_params_{scip_params}/threshold_difficulty_{threshold_difficulty}/threshold_agent_{threshold_agent}/seed_{ecole_seed}/'
    name = 'samples'

    path = init_save_dir(path_to_save, name)

    # init instances generator
    instances = ecole.instance.SetCoverGenerator(n_rows=nrows, n_cols=ncols, density=0.05)


    num_parallel_processes = min(min_num_instances_to_find, int(NUM_CPUS*paralellelisation_factor))
    print(f'Finding >= {min_num_instances_to_find} non-presolved instances in parallel with batches of {num_parallel_processes} parallel processes across {NUM_CPUS} CPUs...')
    print(f'Will save to {path}')
    env_counter, attempt_counter = 0, 0
    start = time.time()
    while env_counter < min_num_instances_to_find:
        result_ids = []
        for i in range(attempt_counter, attempt_counter+num_parallel_processes):
            # gen instance
            instance = next(instances)

            # temporarily write instance to a file so dont need to pickle
            tmp_instance_path = path + '/tmp_instances/'
            pathlib.Path(tmp_instance_path).mkdir(parents=True, exist_ok=True)
            instance_name = f'instance_{i}.mps'
            instance.write_problem(tmp_instance_path+instance_name)

            result_ids.append(reset_env.remote(instance=tmp_instance_path+instance_name,
                                                        path_to_save=path,
                                                      observation_function='default',
                                                      information_function='default',
                                                      reward_function='default',
                                                      scip_params=scip_params,
                                                       ecole_seed=ecole_seed,
                                                       threshold_difficulty=threshold_difficulty,
                                                       threshold_agent=threshold_agent,
                                                       threshold_observation_function='default',
                                                       threshold_information_function='default',
                                                       threshold_reward_function='default',
                                                       threshold_scip_params=scip_params))
            attempt_counter += 1

        # collect results
        results = ray.get(result_ids)
        for was_instance_presolved in results:
            if not was_instance_presolved:
                env_counter += 1

        # remove tmp instance folder
        shutil.rmtree(tmp_instance_path)

        print(f'Found {env_counter}/{min_num_instances_to_find} instances in {time.time()-start:.3f} s after {attempt_counter} env reset attempts.')

    end = time.time()
    print(f'Finished.\nFound {env_counter} non-presolved instances in {end-start:.3f} s after {attempt_counter} env reset attempts.')
    print(f'All {env_counter} instances saved to {path}')
















