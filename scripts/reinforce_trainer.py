from retro_branching.src.learners.reinforce_learner import REINFORCELearner
# from torch.ray_learners import REINFORCELearner
# from torch.multiprocessing_learners import REINFORCELearner
from retro_branching.agents import REINFORCEAgent
from retro_branching.networks import BipartiteGCN
from retro_branching.environments import EcoleBranching

import pyscipopt
import ecole

if __name__ == '__main__':
    # init agent
    RLGNN_DEVICE = 'cuda:4'

    # policy networks
    policy_network = BipartiteGCN(RLGNN_DEVICE,
                               emb_size=128,
                               num_rounds=2,
                               cons_nfeats=5,
                               edge_nfeats=1,
                               var_nfeats=19, # 19 20
                               aggregator='add')
    # policy_network.load_state_dict(torch.load('trained_params.pkl'))
    # policy_network.load_state_dict(torch.load('trained_params_model_disabler.pkl'))
    # policy_network.load_state_dict(torch.load('trained_params_dict_disabler.pkl'))
    # policy_network.load_state_dict(torch.load('/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_1/checkpoint_1/trained_params.pkl'))
    # policy_network.load_state_dict(torch.load('/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_21/checkpoint_275/trained_params.pkl'))
    # policy_network.load_state_dict(torch.load('/scratch/datasets/retro_branching/reinforce_learner/rl_gnn/rl_gnn_583/checkpoint_1500/trained_params.pkl'))
    # policy_network.load_state_dict(torch.load('/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_177/checkpoint_154/trained_params.pkl'))
    # policy_network.load_state_dict(torch.load('/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_258/checkpoint_148/trained_params.pkl'))
    # policy_network.load_state_dict(torch.load('/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_259/checkpoint_124/trained_params.pkl'))
    # policy_network.load_state_dict(torch.load('/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_265/checkpoint_305/trained_params.pkl'))
    policy_network.load_state_dict(
        retro_branching.load('/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_266/checkpoint_235/trained_params.pkl'))

    # filter networks
    filter_network = None
    # filter_network = BipartiteGCN(RLGNN_DEVICE,
    #                           emb_size=128,
    #                           num_rounds=2,
    #                           cons_nfeats=5,
    #                           edge_nfeats=1,
    #                           var_nfeats=19,
    #                           aggregator='add')
    # filter_network.load_state_dict(torch.load('/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_184/checkpoint_212/trained_params.pkl'))
    # filter_network.load_state_dict(torch.load('/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_235/checkpoint_91/trained_params.pkl'))
    # filter_network.load_state_dict(torch.load('/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_260/checkpoint_64/trained_params.pkl'))
    # filter_network.load_state_dict(torch.load('/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_261/checkpoint_58/trained_params.pkl'))

    rlgnn_agent = REINFORCEAgent(policy_network=policy_network, 
                                 filter_network=filter_network,
                                 device=RLGNN_DEVICE, 
                                 temperature=1.0,
                                 name='rl_gnn',
                                 filter_method='method_2') # 'method_1' 'method_2'
    rlgnn_agent.train() # turn on train mode
    print('Initialised agent.')

    # init env
    env = EcoleBranching(observation_function='default', # 'default' 'label_solution_values'
                         information_function='default',
                         reward_function='default',
                         scip_params='default') # 'default' 'ml4co_item_placement' 'ml4co_load_balancing' 'ml4co_anonymous'
    # env = EcoleBranching()
    print('Initialised environments.')

    # # init milp instances generator for generalisation
    # # instances = ecole.instance.SetCoverGenerator(n_rows=100, n_cols=100, density=0.05)
    # instances = ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05)
    # print('Initialised instances.')

    # init single instance for overfitting
    instances = pyscipopt.Model()
    # instances.readProblem('instance_nrows100_ncols100_density005.mps')
    instances.readProblem('instance_nrows500_ncols1000_density005.mps')
    instances = ecole.scip.Model.from_pyscipopt(instances)
    print('Initialised instance.')

    # set threshold difficulty params (optional)
    # threshold_network_path = '/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_1/checkpoint_1'
    threshold_network_path = '/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_21/checkpoint_275'
    threshold_network = BipartiteGCN(device=RLGNN_DEVICE, config=threshold_network_path+'/config.json')
    threshold_network.load_state_dict(retro_branching.load(threshold_network_path + '/trained_params.pkl'))
    threshold_agent = REINFORCEAgent(device=RLGNN_DEVICE, policy_network=threshold_network, name='threshold_agent')
    threshold_agent.eval() # turn on evaluation mode

    # threshold_agent = 'baseline_agent'

    threshold_env = EcoleBranching(observation_function='default',
                                   information_function='default',
                                   reward_function='default',
                                   scip_params='default')
    print('Initialised threshold difficulty parameters.')

    # init max_steps agent
    max_steps_network_path = '/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_21/checkpoint_275'
    max_steps_network = BipartiteGCN(device=RLGNN_DEVICE, config=max_steps_network_path+'/config.json')
    max_steps_network.load_state_dict(retro_branching.load(max_steps_network_path + '/trained_params.pkl'))
    max_steps_agent = REINFORCEAgent(device=RLGNN_DEVICE, policy_network=max_steps_network, name='max_steps_agent')
    max_steps_agent.eval() # turn on evaluation mode
    print('Initialised max_steps agent.')


    # init learner
    learner = REINFORCELearner(agent=rlgnn_agent,
                               env=env,
                               instances=instances,
                               seed=0,
                               max_steps=int(1e12), # 5000 10 5 3
                               max_steps_agent=max_steps_agent,
                               batch_size=512, # 512 32 16 8 1
                               baseline='mean', # None 'sb' 'mean' 'pc' 'gr' 'sr'
                               greedy_rollout_frequency=2049, # 2049
                               greedy_rollout_evaluations=100, # 100
                               sampled_rollout_beams=1,
                               sampled_rollout_frequency=1,
                               apply_max_steps_to_rollout=True,
                               agent_reward='num_nodes', # 'num_nodes' 'primal_dual_integral' 'dual_integral' 'dual_bound' 'primal_dual_gap' 'primal_dual_gap_frac' 'dual_bound_frac'
                               # scale_episode_rewards=False,
                               lr=1e-4,
                               gamma=0.99,
                               # max_log_probs=float('inf'), # 10 25 40 float('inf')
                               turn_off_heuristics=False,
                               threshold_difficulty=None, # None 250 100 50 75 30
                               threshold_agent=threshold_agent,
                               threshold_env=threshold_env,
                               action_filter_agent=None, # None StrongBranchingAgent()
                               action_filter_percentile=10, # 90
                               validation_frequency=None,
                               episode_log_frequency=1,
                               path_to_save='/scratch/datasets/retro_branching',
                               checkpoint_frequency=100,
                               )
    print('Initialised learner with params {}. Will save to {}'.format(learner.episodes_log, learner.path_to_save))

    # train agent
    print('Training agent...')
    learner.train(2e5)



    # # getting strong branching result for an overfit instance
    # env.seed(0)
    # agent = StrongBranchingAgent()
    # agent.before_reset(instances)
    # obs, action_set, reward, done, info = env.reset(instances)
    # t = 0
    # while not done:
        # action, action_idx = agent.action_select(action_set, env.model, done)
        # obs, action_set, reward, done, info = env.step(action)
        # print('Step {} | Num nodes: {}'.format(t, info['num_nodes']))
        # t += 1
    # print('Final num nodes: {}'.format(env.model.as_pyscipopt().getNNodes()))






    # # TIME PROFILING
    # # 1. Generate a file called time_profile.prof
    # # 2. Transfer to /home/cwfparsonson/Downloads
    # # 3. Run snakeviz /home/cwfparsonson/Downloads/time_profile.prof to visualise
    # profiler = cProfile.Profile()
    # profiler.enable()
    # learner.train(256)
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats()
    # stats.dump_stats('time_profile2.prof')
