from _old.retro_branching import EcoleBranching
from _old.retro_branching import BipartiteGCN
from _old.retro_branching import REINFORCEAgent, StrongBranchingAgent, PseudocostBranchingAgent
from _old.retro_branching import ReinforcementLearningValidator
from _old.retro_branching import get_most_recent_checkpoint_foldername

import ecole
import glob






if __name__ == '__main__':
    DEVICE = 'cpu'
    agents = {}

    ################################ AGENTS ##################################
    # LOAD RL AGENTS
    # id_to_checkpoints = {261: [1, 100, 200, 'latest']}
    # id_to_checkpoints = {412: [1, 'latest'],
                         # 414: [1]}
    # id_to_checkpoints = {412: [1],
                         # 414: [1]}
    # id_to_checkpoints = {412: [1]}
    # id_to_checkpoints = {441: [1]}
    # id_to_checkpoints = {553: [1, 1000]}
    # id_to_checkpoints = {583: [1590],
                         # 584: [1620]}
    id_to_checkpoints = {586: [1, 'latest']}
    for i in id_to_checkpoints.keys():
        for checkpoint in id_to_checkpoints[i]:
            policy_network = BipartiteGCN(DEVICE,
                                emb_size=128,
                                num_rounds=2,
                                aggregator='add')
            agent = f'rl_gnn_{i}'
            agent_path = '/scratch/datasets/torch/reinforce_learner/rl_gnn/{}/'.format(agent)
            if checkpoint == 'latest':
                foldername = get_most_recent_checkpoint_foldername(agent_path)
            else:
                foldername = f'checkpoint_{checkpoint}'
            path = '{}{}/'.format(agent_path, foldername)
            policy_network.load_state_dict(retro_branching.load(path + 'trained_params.pkl', map_location=DEVICE))
            print('Loaded params from {}'.format(path))
            agent = REINFORCEAgent(policy_network=policy_network, 
                                   device=DEVICE, 
                                   name=agent+'_{}'.format(foldername))
            agent.eval() # turn on evaluation mode
            agents[agent.name] = agent

    # LOAD SUPERVISED AGENTS
    id_to_checkpoints = {1: [1],
                         21: [275]}
                         # 113: [68]}
    # id_to_checkpoints = {21: [275]}
    for i in id_to_checkpoints:
        for checkpoint in id_to_checkpoints[i]:
            policy_network = BipartiteGCN(DEVICE,
                                emb_size=64,
                                num_rounds=1)
            agent = f'gnn_{i}'
            agent_path = '/scratch/datasets/torch/supervised_learner/gnn/{}/'.format(agent)
            if checkpoint == 'latest':
                foldername = get_most_recent_checkpoint_foldername(agent_path)
            else:
                foldername = f'checkpoint_{checkpoint}'
            path = '{}{}/'.format(agent_path, foldername)
            policy_network.load_state_dict(retro_branching.load(path + 'trained_params.pkl', map_location=DEVICE))
            print('Loaded params from {}'.format(path))
            agent = REINFORCEAgent(policy_network=policy_network, device=DEVICE, name=agent+'_{}'.format(foldername))
            agent.eval() # turn on evaluation mode
            agents[agent.name] = agent


    # LOAD STRONG BRANCHING AGENT
    agent = StrongBranchingAgent(name='sb')
    agents['sb'] = agent

    # LOAD PSEUDOCOST AGENT
    agent = PseudocostBranchingAgent(name='pc')
    agents['pc'] = agent

    # # RANDOM AGENT
    # agent = RandomAgent(name='random')
    # agents['random'] = agent

    # PRINT INITIALISED AGENTS
    print('Initialised agents dict: {}'.format(agents))






    ############################## VALIDATOR ############################
    # init instances
    # instances = ecole.instance.SetCoverGenerator(n_rows=100, n_cols=100, density=0.05)
    # instances = ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05)
    # files = glob.glob('/scratch/datasets/torch/instances/set_cover_nrows_500_ncols_1000_density_005/*.mps')

    nrows = 100 
    ncols = 100 
    threshold_difficulty = None
    max_steps = 3 # int(1e12) 10 5 3
    files = glob.glob(f'/scratch/datasets/torch/instances/set_cover_nrows_{nrows}_ncols_{ncols}_density_005_threshold_{threshold_difficulty}/*.mps')

    instances = iter([ecole.scip.Model.from_file(f) for f in files])
    # instances = ecole.instance.SetCoverGenerator(n_rows=100, n_cols=100, density=0.05)
    print('Initialised instances.')

    # init envs
    envs = {}
    for agent_name in agents.keys():
        envs[agent_name] = EcoleBranching(observation_function='default',
                                          information_function='default',
                                          reward_function='default',
                                          scip_params='default')
        # envs[agent_name] = EcoleBranching()
    print('Initialised agent envs.')

    # init metrics
    # metrics = ['num_nodes', 'solving_time', 'lp_iterations', 'dual_integral', 'primal_integral', 'primal_dual_integral']
    metrics = ['num_nodes', 'solving_time', 'lp_iterations']
    print('Initialised metrics: {}'.format(metrics))

    # set threshold difficulty params (optional)
    threshold_agent = BipartiteGCN(DEVICE,
                                emb_size=64,
                                num_rounds=1)
    threshold_agent.load_state_dict(
        retro_branching.load('/scratch/datasets/torch/supervised_learner/gnn/gnn_21/checkpoint_275/trained_params.pkl', map_location=DEVICE))
    threshold_agent = REINFORCEAgent(policy_network=threshold_agent, device=DEVICE, name='threshold_agent')
    threshold_agent.eval() # turn on evaluation mode
    threshold_env = EcoleBranching(observation_function=list(envs.values())[0].str_observation_function,
                                   information_function=list(envs.values())[0].str_information_function,
                                   reward_function=list(envs.values())[0].str_reward_function,
                                   scip_params=list(envs.values())[0].str_scip_params)
    # threshold_env = EcoleBranching()
    print('Initialised threshold difficulty parameters.')

    # init max_steps agent
    policy_network = BipartiteGCN(DEVICE,
                        emb_size=64,
                        num_rounds=1)
    policy_network.load_state_dict(
        retro_branching.load('/scratch/datasets/torch/supervised_learner/gnn/gnn_21/checkpoint_275/trained_params.pkl', map_location=DEVICE))
    max_steps_agent = REINFORCEAgent(policy_network=policy_network, 
                                     device=DEVICE, 
                                     temperature=1.0,
                                     name='max_steps_agent')
    max_steps_agent.eval() # turn on evaluation mode
    print('Initialised max_steps agent.')

    # init validator
    validator = ReinforcementLearningValidator(agents=agents,
                                               envs=envs,
                                               instances=instances,
                                               metrics=metrics,
                                               seed=0,
                                               max_steps=max_steps, # int(1e12), 10, 5, 3
                                               max_steps_agent=max_steps_agent,
                                               turn_off_heuristics=False,
                                               min_threshold_difficulty=None,
                                               max_threshold_difficulty=None, # None 250
                                               threshold_agent=threshold_agent,
                                               threshold_env=threshold_env,
                                               episode_log_frequency=1,
                                               path_to_save='/scratch/datasets/torch',
                                               checkpoint_frequency=1)
    print('Initialised validator. Will save to {}'.format(validator.path_to_save))

    # run validator
    print('Validating agents...')
    validator.test(100)




















