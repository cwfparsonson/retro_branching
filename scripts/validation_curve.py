# disable annoying tensorboard numpy warning
import warnings
warnings.filterwarnings(action='ignore',
                        category=FutureWarning,
                        module='tensorboard')  # noqa

from retro_branching.utils import get_most_recent_checkpoint_foldername, gen_co_name
from retro_branching.networks import BipartiteGCN, BipartiteGCNNoHeads
from retro_branching.agents import Agent, REINFORCEAgent, DQNAgent, DoubleDQNAgent, AveragedDQNAgent
from retro_branching.environments import EcoleBranching
from retro_branching.validators import ReinforcementLearningValidator

import ecole
import torch
import numpy as np
import os
import shutil
import glob
import time

import psutil
import ray


@ray.remote
def run_rl_validator_in_training_dir(path, 
                                     agents,
                                     device,
                                     # nrows=100, 
                                     # ncols=100, 
                                     co_class,
                                     co_class_kwargs,
                                     threshold_difficulty=None, 
                                     max_steps=int(1e12), 
                                     max_steps_agent=None,
                                     observation_function='default',
                                     information_function='default',
                                     reward_function='default',
                                     scip_params='default',
                                     overwrite=False):

    start = time.time()
    agent = agents[path]

    # instances
#     files = glob.glob(f'/scratch/datasets/retro_branching/instances/set_cover_nrows_{nrows}_ncols_{ncols}_density_005_threshold_{threshold_difficulty}/*.mps')
    instances_path = f'/scratch/datasets/retro_branching/instances/{co_class}'
    for key, val in co_class_kwargs.items():
        instances_path += f'_{key}_{val}'
#     instances_path += f'_threshold_{threshold_difficulty}'
#     files = glob.glob(f'/scratch/datasets/retro_branching/instances/{co_class}_nrows_{nrows}_ncols_{ncols}_density_005_threshold_{threshold_difficulty}/*.mps')
    files = glob.glob(instances_path+f'/scip_{scip_params}/*.mps')
    instances = iter([ecole.scip.Model.from_file(f) for f in files])
    print('Initialised instances.')

    # env
    env = EcoleBranching(observation_function=observation_function,
                         information_function=information_function,
                         reward_function=reward_function,
                         scip_params=scip_params)
    env.seed(0)
    print('Initialised env.')

    # metrics
    # metrics = ['num_nodes', 'solving_time', 'lp_iterations', 'primal_dual_integral', 'primal_integral', 'dual_integral']
    metrics = ['num_nodes', 'solving_time', 'lp_iterations']
    print('Initialised metrics: {}'.format(metrics))

    # validator
    validator = ReinforcementLearningValidator(agents={agent.name: agent},
                                               envs={agent.name: env},
                                               instances=instances,
                                               # calibration_config_path='/home/zciccwf/phd_project/projects/retro_branching/scripts/',
                                               calibration_config_path=None,
                                               metrics=metrics,
                                               seed=0,
                                               max_steps=max_steps, # int(1e12), 10, 5, 3
                                               max_steps_agent=max_steps_agent,
                                               turn_off_heuristics=False,
                                               min_threshold_difficulty=None,
                                               max_threshold_difficulty=None, # None 250
                                               threshold_agent=None,
                                               threshold_env=None,
                                               episode_log_frequency=1,
                                               path_to_save=path,
                                               overwrite=overwrite,
                                               checkpoint_frequency=10)
    print('Initialised validator. Will save to {}'.format(validator.path_to_save))

    # run validation tests
    validator.test(len(files))
    end = time.time()
    print('Finished path {} validator in {} s'.format(path, round(end-start, 3)))






if __name__ == '__main__':
    '''
    Runs validator for different checkpoints of given agent to plot
    validation curve at each checkpoint for agent during training. Will
    save rl_validator data in rl_validator/ dir inside agent's training
    checkpoint dir. Will run these validation tests in parallel for
    each agent checkpoint.
    '''
    # init ray
    num_cpus = psutil.cpu_count(logical=False)
    ray.init(num_cpus=int(num_cpus))

    # init params and agents' training checkpoints to use for validation curve and other initialisation info

    learner = 'dqn_learner' # 'reinforce_learner' 'dqn_learner'
    base_name = 'dqn_gnn' # 'rl_gnn' 'dqn_gnn'
    
    # learner = 'ppo_learner' # 'reinforce_learner' 'dqn_learner'
    # base_name = 'ppo' # 'rl_gnn' 'dqn_gnn'

    AgentClass = Agent # REINFORCEAgent DQNAgent DoubleDQNAgent AveragedDQNAgent
    overwrite_rl_validator = False # True False

    # SC
    co_class = 'set_covering'
    # co_class_kwargs = {'n_rows': 100, 'n_cols': 100}
    # co_class_kwargs = {'n_rows': 165, 'n_cols': 230}
    # co_class_kwargs = {'n_rows': 250, 'n_cols': 500}
    co_class_kwargs = {'n_rows': 500, 'n_cols': 1000}
    # co_class_kwargs = {'n_rows': 1000, 'n_cols': 1000}

    # # CA
    # co_class = 'combinatorial_auction'
    # co_class_kwargs = {'n_items': 10, 'n_bids': 50}

    # # CFL
    # co_class = 'capacitated_facility_location' # 'set_covering' 'combinatorial_auction' 'capacitated_facility_location' 'maximum_independent_set'
    # co_class_kwargs = {'n_customers': 5, 'n_facilities': 5}

    # # MIS 
    # co_class = 'maximum_independent_set' # 'set_covering' 'combinatorial_auction' 'capacitated_facility_location' 'maximum_independent_set'
    # co_class_kwargs = {'n_nodes': 25}

    threshold_difficulty = None # None 100
    last_checkpoint_idx = None # None -1 -2
    max_steps = int(5000) # int(1e12) 10 5 3 2500 1250 5000
    observation_function = '43_var_features' # 'default' 'label_solution_values' '40_var_features' '45_var_features' '40_var_features' '43_var_features' 'custom_var_features'
    information_function = 'default'
    reward_function = 'default'
    # scip_params = 'default'
    scip_params = 'gasse_2019'

    # # init max_steps agent
    max_steps_agent = None
    # policy_network = BipartiteGCN(DEVICE,
                        # emb_size=64,
                        # num_rounds=1)
    # policy_network.load_state_dict(torch.load('/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_21/checkpoint_275/trained_params.pkl', map_location=DEVICE))
    # max_steps_agent = REINFORCEAgent(policy_network=policy_network, 
                                     # device=DEVICE, 
                                     # temperature=1.0,
                                     # name='max_steps_agent')
    # max_steps_agent.eval() # turn on evaluation mode

    #### NEW (using config.json file(s) to initialise) ####
    agent_info = {f'{base_name}_{i}': {'checkpoints': np.arange(5, 1000, 1)}
    # agent_info = {f'{base_name}_{i}': {'checkpoints': np.arange(1, 100, 1)}

            # for i in [1405]}
            # for i in [1479, 1481, 1484, 1485]}
            # for i in [1479, 1484]}
            # for i in [1481, 1484, 1485]}
            # for i in [1481, 1487]}

            # for i in [1481]}
            for i in [1481, 1488, 1489, 1490]}

            # for i in [1491]}








    ##### OLD #####

    # policy_networks = {f'{base_name}_{i}': {'filter_network': None, # None 'gnn_235
    #                                    'filter_method': 'method_2',
    #                                    'checkpoints': np.arange(1, 700, 50),
    #                                    'emb_size': 64,
    #                                    'num_rounds': 1,
    #                                    'cons_nfeats': 5,
    #                                    'edge_nfeats': 1,
    #                                    'var_nfeats': 19,
    #                                    'aggregator': 'add'} for i in [589]}

    # policy_networks = {f'{base_name}_{i}': {'filter_network': None, # None 'gnn_235
    #                                    'filter_method': 'method_2',
    #                                    'checkpoints': np.arange(1, 225, 25),
    #                                    'emb_size': 64,
    #                                    'num_rounds': 1,
    #                                    'cons_nfeats': 5,
    #                                    'edge_nfeats': 1,
    #                                    'var_nfeats': 19,
    #                                    'aggregator': 'add'} for i in [634]}
    # filter_networks = {f'gnn_{i}': {'checkpoint': cp,
    #                                 'emb_size': 128,
    #                                 'num_rounds': 2,
    #                                 'cons_nfeats': 5,
    #                                 'edge_nfeats': 1,
    #                                 'var_nfeats': 19,
    #                                 'aggregator': 'add'} for i, cp in zip([261], [58])}

    # load and initialise agents
    # DEVICE = 'cpu' # must be on CPU for ray to parallelise
    # agents, envs, agent_paths = {}, {}, []
    # for agent_name in policy_networks.keys():
    #     agent_path = '/scratch/datasets/retro_branching/{}/{}/{}/'.format(learner, base_name, agent_name)
    #     agent_paths.append(agent_path) # useful for overwriting later in script
    #     foldernames = [f'checkpoint_{cp}' for cp in policy_networks[agent_name]['checkpoints']]
    #     if last_checkpoint_idx is not None:
    #         foldernames.append(get_most_recent_checkpoint_foldername(agent_path, idx=last_checkpoint_idx))

    #     # collect agent NN training checkpoint parameters
    #     for foldername in foldernames:
    #         policy_network = BipartiteGCN(DEVICE,
    #                                     emb_size=policy_networks[agent_name]['emb_size'],
    #                                     num_rounds=policy_networks[agent_name]['num_rounds'],
    #                                     cons_nfeats=policy_networks[agent_name]['cons_nfeats'],
    #                                     edge_nfeats=policy_networks[agent_name]['edge_nfeats'],
    #                                     var_nfeats=policy_networks[agent_name]['var_nfeats'],
    #                                     aggregator=policy_networks[agent_name]['aggregator'])

    #         if policy_networks[agent_name]['filter_network'] is not None:
    #             filter_name = policy_networks[agent_name]['filter_network']
    #             filter_network = BipartiteGCN(DEVICE,
    #                                     emb_size=filter_networks[filter_name]['emb_size'],
    #                                     num_rounds=filter_networks[filter_name]['num_rounds'],
    #                                     cons_nfeats=filter_networks[filter_name]['cons_nfeats'],
    #                                     edge_nfeats=filter_networks[filter_name]['edge_nfeats'],
    #                                     var_nfeats=filter_networks[filter_name]['var_nfeats'],
    #                                     aggregator=filter_networks[filter_name]['aggregator'])
    #             filter_network.load_state_dict(torch.load('/scratch/datasets/retro_branching/supervised_learner/gnn/{}/checkpoint_{}/trained_params.pkl'.format(filter_name, filter_networks[filter_name]['checkpoint']), map_location=DEVICE))
    #         else:
    #             filter_network = None

    #         path = '{}{}/'.format(agent_path, foldername)
    #         policy_network.load_state_dict(torch.load(path+'trained_params.pkl', map_location=DEVICE))
    #         print('Loaded params from {}'.format(path))
    #         agent = REINFORCEAgent(policy_network=policy_network, filter_network=filter_network, device=DEVICE, name=agent_name, filter_method=policy_networks[agent_name]['filter_method'])
    #         agent.eval() # turn on evaluation mode
    #         agents[path] = agent




    # load and initialise agents
    DEVICE = 'cpu'
    agents, agent_paths = {}, []
    for agent_name in agent_info.keys():
        agent_path = f'/scratch/datasets/retro_branching/{learner}/{base_name}/{agent_name}/'
        agent_paths.append(agent_path) # useful for overwriting later in script
        foldernames = [f'checkpoint_{cp}' for cp in agent_info[agent_name]['checkpoints'] if os.path.exists(f'{agent_path}checkpoint_{cp}')]
        if last_checkpoint_idx is not None:
            foldernames.append(get_most_recent_checkpoint_foldername(agent_path, idx=last_checkpoint_idx))

        for foldername in foldernames:
            path = f'{agent_path}{foldername}'
            config = path + '/config.json'
            agent = AgentClass(device=DEVICE, config=config)
            agent.name = agent_name
            for network_name, network in agent.get_networks().items():
                # if networks is not None and type(networks) != str:
                    # agent.__dict__[network_name].load_state_dict(torch.load(path+f'/{network_name}_params.pkl', map_location=DEVICE))
                if network is not None:
                    try:
                        # see if networks saved under same var as 'network_name'
                        agent.__dict__[network_name].load_state_dict(
                            torch.load(path + f'/{network_name}_params.pkl', map_location=DEVICE))
                    except KeyError:
                        # networks saved under generic 'networks' var (as in Agent class)
                        if 'networks' in agent.__dict__.keys(): # TEMP handling typo
                            agent.__dict__['networks'].load_state_dict(
                                torch.load(path + f'/{network_name}_params.pkl', map_location=DEVICE))
                        else:
                            agent.__dict__['network'].load_state_dict(
                                    torch.load(path + f'/{network_name}_params.pkl', map_location=DEVICE))
                else:
                    print(f'{network_name} is None.')

            # # TEMPORARY (prev agents dont have this param)
            # agent.default_epsilon = 0

            agent.eval() # put in test mode
            agents[path] = agent


    if overwrite_rl_validator:
        # clear all old rl_validator/ folders even if would not be overwritten with current config to prevent testing inconsistencies
        for agent_path in agent_paths:
            paths = sorted(glob.glob(agent_path+'/checkpoint_*'))
            for path in paths:
                if os.path.isdir(path+'/rl_validator'):
                    print('Removing old {}'.format(path+'/rl_validator'))
                    shutil.rmtree(path+'/rl_validator')

    # run validators for agents' checkpoints in parallel
    result_ids = []
    for path in agents.keys():
        if os.path.isdir(path+'/rl_validator/') and not overwrite_rl_validator:
            print('Already have rl_validator/ in {} and overwrite_rl_validator set to False, skipping...'.format(path))
        else:
            print('Creating rl_validator/ in {}'.format(path))
            result_ids.append(run_rl_validator_in_training_dir.remote(path, 
                                                                      agents,
                                                                      DEVICE,
                                                                      # nrows, 
                                                                      # ncols, 
                                                                      co_class,
                                                                      co_class_kwargs,
                                                                      threshold_difficulty, 
                                                                      max_steps, 
                                                                      max_steps_agent,
                                                                      observation_function,
                                                                      information_function,
                                                                      reward_function,
                                                                      scip_params,
                                                                      overwrite_rl_validator))


    # collect results
    start = time.time()
    _ = ray.get(result_ids)
    end = time.time()
    print('\nFinished all parallel processes in {}.'.format(round(end-start, 3)))













