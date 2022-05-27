'''
N.B. For 100x100 and 500x1000 validation instances, so far have not used seeding,
so should re-do these at the end with seeding so is repeatable.
'''
from retro_branching.environments import EcoleBranching
from retro_branching.networks import BipartiteGCN
from retro_branching.agents import REINFORCEAgent

import ecole
import torch
from pathlib import Path




if __name__ == '__main__':
    # seed for reproducibility
    ecole.seed(1799)

    # init params

    # # SC 
    co_class = 'set_covering' # 'set_covering' 'combinatorial_auction' 'capacitated_facility_location' 'maximum_independent_set'
    kwargs = {'n_rows': 165, 'n_cols': 230}

    # # CA
    # co_class = 'combinatorial_auction' # 'set_covering' 'combinatorial_auction' 'capacitated_facility_location' 'maximum_independent_set'
    # kwargs = {'n_items': 37, 'n_bids': 83}

    # # CFL
    # co_class = 'capacitated_facility_location' # 'set_covering' 'combinatorial_auction' 'capacitated_facility_location' 'maximum_independent_set'
    # kwargs = {'n_customers': 5, 'n_facilities': 15}

    # MIS 
    # co_class = 'maximum_independent_set' # 'set_covering' 'combinatorial_auction' 'capacitated_facility_location' 'maximum_independent_set'
    # kwargs = {'n_nodes': 75}

    threshold_difficulty = None # None 100
    num_instances = 100
    path = '/scratch/datasets/retro_branching/instances/'

    foldername = f'{co_class}'
    # scip_params = 'default'
    scip_params = 'gasse_2019'
    for key, val in kwargs.items():
        foldername += f'_{key}_{val}'
    foldername += f'/scip_{scip_params}/'
    # foldername = f'{co_class}_nrows_{nrows}_ncols_{ncols}_density_005_threshold_{threshold_difficulty}/'

    # create instances
    if co_class == 'set_covering':
        instances = ecole.instance.SetCoverGenerator(**kwargs)
    elif co_class == 'combinatorial_auction':
        instances = ecole.instance.CombinatorialAuctionGenerator(**kwargs)
    elif co_class == 'capacitated_facility_location':
        instances = ecole.instance.CapacitatedFacilityLocationGenerator(**kwargs)
    elif co_class == 'maximum_independent_set':
        instances = ecole.instance.IndependentSetGenerator(**kwargs)
    else:
        raise Exception(f'Unrecognised co_class {co_class}')

    # init dir
    Path(path+foldername).mkdir(parents=True, exist_ok=True)

    # create instances
    env = EcoleBranching(observation_function='default',
                         information_function='default',
                         reward_function='default',
                         scip_params=scip_params)
    env.seed(0)


    if threshold_difficulty is not None:
        threshold_env = EcoleBranching(observation_function=env.str_observation_function,
                                        information_function=env.str_information_function,
                                        reward_function=env.str_reward_function,
                                        scip_params=env.str_scip_params)
        # set threshold difficulty params (optional)
        RLGNN_DEVICE = 'cuda:0'
        threshold_agent = BipartiteGCN(RLGNN_DEVICE)
        threshold_agent.load_state_dict(
            retro_branching.load('/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_1/checkpoint_1/trained_params.pkl'))
        # threshold_agent.load_state_dict(torch.load('/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_21/checkpoint_275/trained_params.pkl'))
        threshold_agent = REINFORCEAgent(policy_network=threshold_agent, device=RLGNN_DEVICE, name='threshold_agent')
        threshold_agent.eval() # turn on evaluation mode
        threshold_env = EcoleBranching(observation_function=env.str_observation_function,
                                       information_function=env.str_information_function,
                                       reward_function=env.str_reward_function,
                                       scip_params=env.str_scip_params)
        threshold_env.seed(0)
        print('Initialised threshold difficulty parameters.')
    else:
        threshold_agent = None
        threshold_env = None


    for i in range(num_instances):
        # find instance not pre-solved by ecole scip env
        counter = 0
        while True:
            counter += 1
            instance = next(instances)
            instance_before_reset = instance.copy_orig()
            env.seed(0)
            obs, _, _, _, _ = env.reset(instance)

            if obs is not None:
                if threshold_difficulty is not None:
                    # check difficulty using threshold agent
                    threshold_env.seed(0)
                    _obs, _action_set, _reward, _done, _info = threshold_env.reset(instance_before_reset.copy_orig())
                    while not _done:
                        _action, _action_idx = threshold_agent.action_select(_action_set, _obs)
                        # _action = _action_set[_action]
                        _obs, _action_set, _reward, _done, _info = threshold_env.step(_action)
                        if _info['num_nodes'] > threshold_difficulty:
                            # already exceeded threshold difficulty
                            break
                    if _info['num_nodes'] < threshold_difficulty:
                        break
                else:
                    break

            if counter > 10000:
                raise Exception('Unable to find instance that is not pre-solved.')

        # save instance
        name = f'instance_{i}.mps'
        instance_before_reset.write_problem(path+foldername+name)
        print(f'Saved instance {i+1} of {num_instances} to {path+foldername+name}')



