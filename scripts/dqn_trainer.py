# disable annoying tensorboard numpy warning
import warnings
warnings.filterwarnings(action='ignore',
                        category=FutureWarning,
                        module='tensorboard')  # noqa
# import numpy as np
# np.testing.suppress_warnings()
# import tensorflow as tf
# tf.get_logger().setLevel('ERROR')

from retro_branching.agents import DQNAgent, DoubleDQNAgent, AveragedDQNAgent, StrongBranchingAgent, REINFORCEAgent, PseudocostBranchingAgent, Agent
# DEBUG from retro_branching.src.agents.dqn_agent_2 import DQNAgent
# from retro_branching.src.agents.double_dqn_agent_2 import DoubleDQNAgent

from retro_branching.environments import EcoleBranching

from retro_branching.learners import DQNLearner
# # DEBUG from retro_branching.src.learners.dqn_learner_2 import DQNLearner

from retro_branching.utils import check_if_network_params_equal, seed_stochastic_modules_globally
from retro_branching.networks import BipartiteGCN, BipartiteGCNNoHeads, MultiHeadedSeparateParamsBipartiteGCN
from retro_branching.loss_functions import MeanSquaredError, SmoothL1Loss, HuberLoss

import ecole
import pyscipopt

import torch

import glob
import cProfile
import pstats

import random

# # debug
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'






if __name__ == '__main__':
    import ecole
    import pyscipopt
    import torch

    # set seeds
    default_seed = 0
    numpy_seed = default_seed
    random_seed = default_seed
    torch_seed = default_seed
    ecole_seed = default_seed
    seed_stochastic_modules_globally(default_seed=default_seed,
                                     numpy_seed=numpy_seed,
                                     random_seed=random_seed,
                                     torch_seed=torch_seed,
                                     ecole_seed=ecole_seed)

    device = 'cuda:1'

    # time profiling
    profile_time = False # False

    # DQN
    init_value_network_path = None 
    # init_value_network_path = '/scratch/datasets/retro_branching/dqn_learner/dqn_gnn/dqn_gnn_1236/checkpoint_457/value_network_params.pkl'
    # init_value_network_path = '/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_343/checkpoint_233/network_params.pkl'
    # init_value_network_path = '/scratch/datasets/retro_branching/dqn_learner/dqn_gnn/dqn_gnn_1448/checkpoint_25/value_network_params.pkl'

    # if init_value_network_path is not None:
        # reinitialise_heads = True # True False
    # else:
        # reinitialise_heads = False
    reinitialise_heads = False


    NET = BipartiteGCN # BipartiteGCN MultiHeadedSeparateParamsBipartiteGCN
    value_network = NET(device=device,
                        emb_size=64,
                        num_rounds=1,
                        cons_nfeats=5,
                        edge_nfeats=1, 
                        var_nfeats=43, # 19 20 28 45 40 43 29 37
                        aggregator='add',
                        activation='inverse_leaky_relu',
                        num_heads=1, # 1 2
                        # num_heads=2, # 1 2
                        head_depth=2, # 1 2
                        # head_depth=2, # 1 2
                        linear_weight_init='normal',
                        linear_bias_init='zeros',
                        layernorm_weight_init=None,
                        layernorm_bias_init=None,
                        head_aggregator='add', # 'add'
                        # head_aggregator={'train': None, 'test': 0}, # 'add'
                        include_edge_features=True, # True False
                        use_old_heads_implementation=False, # True False
                        profile_time=profile_time) # None 'add'
    if init_value_network_path is not None:
        value_network.load_state_dict(torch.load(init_value_network_path, map_location=device))
    if reinitialise_heads:
        # re-initialise heads
        value_network.init_model_parameters(init_gnn_params=False, init_heads_params=True)
    agent = DQNAgent(device=device,
                     value_network=value_network,
                     exploration_network=None,
                     head_aggregator='add', # 'add'
                     # head_aggregator={'train': 'add', 'test': 0},
                     deterministic_mdqn=False,
                     profile_time=profile_time,
                     name='dqn_gnn')
    agent.train()








    
    # # DOUBLE DQN
    # init_value_network_path = None 
    # NET = BipartiteGCN # BipartiteGCN MultiHeadedSeparateParamsBipartiteGCN
    # # init_value_network_path = '/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_265/checkpoint_305/trained_params.pkl'
    # # init_value_network_path = '/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_266/checkpoint_235/trained_params.pkl'
    # # init_value_network_path = '/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_296/checkpoint_102/network_params.pkl'
    # value_network_1 = NET(device=device,
                          # emb_size=128,
                          # num_rounds=2,
                          # cons_nfeats=5,
                          # edge_nfeats=1,
                          # var_nfeats=43, # 19 20 28 45 40
                          # aggregator='add',
                          # activation='inverse_leaky_relu',
                          # num_heads=1,
                          # linear_weight_init='normal',
                          # linear_bias_init='zeros',
                          # layernorm_weight_init=None,
                          # layernorm_bias_init=None,
                          # head_aggregator=None,
                          # profile_time=profile_time) # None 'add'
    # value_network_2 = NET(device=device,
                          # emb_size=128,
                          # num_rounds=2,
                          # cons_nfeats=5,
                          # edge_nfeats=1,
                          # var_nfeats=43, # 19 20 28 45 40
                          # aggregator='add',
                          # activation='inverse_leaky_relu',
                          # num_heads=1,
                          # linear_weight_init='normal',
                          # linear_bias_init='zeros',
                          # layernorm_weight_init=None,
                          # layernorm_bias_init=None,
                          # head_aggregator=None,
                          # profile_time=False) # None 'add'


    # if init_value_network_path is not None:
        # value_network_1.load_state_dict(torch.load(init_value_network_path, map_location=device))
        # # value_network_2.load_state_dict(torch.load(init_value_network_path, map_location=device))

    # # # init exploration networks
    # exploration_network = None
    # # exploration_network = 'strong_branching_agent'
    # # exploration_network = 'pseudocost_branching_agent'
    # # exploration_network = BipartiteGCN(device=device,
                                # # emb_size=64,
                                # # num_rounds=1,
                                # # cons_nfeats=5,
                                # # edge_nfeats=1,
                                # # var_nfeats=19,
                                # # aggregator='add')
    # # exploration_network.load_state_dict(torch.load('/scratch/datasets/retro_branching/reinforce_learner/rl_gnn/rl_gnn_641/checkpoint_1997/policy_network_params.pkl', map_location=device))
    # # exploration_network.load_state_dict(torch.load('/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_265/checkpoint_305/trained_params.pkl', map_location=device))

    # # init agent
    # agent = DoubleDQNAgent(device=device,
                          # value_network_1=value_network_1,
                          # value_network_2=value_network_2,
                          # exploration_network=exploration_network,
                          # head_aggregator='add',
                          # # sample_exploration_network_stochastically=False, # False True
                          # name='dqn_gnn',
                          # profile_time=profile_time)
    # agent.train()










    # # AVERAGED DQN
    # NET = MultiHeadedSeparateParamsBipartiteGCN # BipartiteGCN MultiHeadedSeparateParamsBipartiteGCN
    # value_network = NET(device=device,
                        # emb_size=128,
                        # num_rounds=2,
                        # cons_nfeats=5,
                        # edge_nfeats=1,
                        # var_nfeats=19, # 19 20 28 45 40
                        # aggregator='add',
                        # activation='leaky_relu',
                        # num_heads=2,
                        # linear_weight_init='normal',
                        # linear_bias_init='zeros',
                        # layernorm_weight_init=None,
                        # layernorm_bias_init=None,
                        # head_aggregator=None) # None 'add'
    # exploration_network = None # None 'strong_branching_agent'
    # agent = AveragedDQNAgent(device=device,
                             # value_network=value_network,
                             # exploration_network=exploration_network,
                             # averaged_dqn_k=10,
                             # averaged_dqn_k_freq=1,
                             # head_aggregator='add',
                             # # sample_exploration_network_stochastically=False, # False True
                             # name='dqn_gnn')
    # agent.train()














    # # DEBUG
    # print(f'net params equal: {check_if_network_params_equal(agent.agent_1.value_network, agent.agent_2.value_network)}')
    # raise Exception()


    # # # init single instance for overfitting
    # instances = pyscipopt.Model()
    # # instances.readProblem('instance_nrows100_ncols100_density005.mps')
    # instances.readProblem('/home/zciccwf/phd_project/projects/retro_branching/notebooks/instance_2642.cip')
    # instances = ecole.scip.Model.from_pyscipopt(instances)

    # # init instance generator for generalising

    # SET COVER
    # instances = ecole.instance.SetCoverGenerator(n_rows=100, n_cols=100, density=0.05)
    # instances = iter(glob.glob(f'/scratch/datasets/retro_branching/instances/training/nrows_100_ncols_100/scip_params_default/threshold_difficulty_None/threshold_agent_None/seed_0/samples/samples_1/*.mps'))
    # instances = ecole.instance.FileGenerator(f'/scratch/datasets/retro_branching/instances/training/nrows_100_ncols_100/scip_params_default/threshold_difficulty_None/threshold_agent_None/seed_0/samples/samples_1/', sampling_mode='remove_and_repeat')

    instances = ecole.instance.SetCoverGenerator(n_rows=165, n_cols=230, density=0.05)

    # instances = ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05)
    # instances = ecole.instance.FileGenerator(f'/scratch/datasets/retro_branching/instances/training/nrows_500_ncols_1000/scip_params_default/threshold_difficulty_None/threshold_agent_None/seed_0/samples/samples_1/', sampling_mode='remove_and_repeat')

    # instances = ecole.instance.FileGenerator('/scratch/datasets/retro_branching/gasse_2019/custom_data/data/instances/setcover/general_500r_1000c_0.05d', sampling_mode='remove_and_repeat')
    # instances = iter(glob.glob('/scratch/datasets/retro_branching/gasse_2019/custom_data/data/instances/setcover/general_500r_1000c_0.05d/*.lp'))

    # instances = ecole.instance.SetCoverGenerator(n_rows=250, n_cols=500, density=0.05)

    # instances = ecole.instance.SetCoverGenerator(n_rows=1000, n_cols=1000, density=0.05)


    # # COMBINATORIAL AUCTION
    # instances = ecole.instance.CombinatorialAuctionGenerator(n_items=10, n_bids=50)

    # # CAPACITATED FACILITY LOCATION
    # instances = ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=5, n_facilities=5)

    # # MAXIMUM INDEPENDENT SET
    # instances = ecole.instance.IndependentSetGenerator(n_nodes=25)

    # # DEBUG
    # for _ in range(2640):
        # _ = next(instances)


    reward_function = 'retro_binary_fathomed'
    env = EcoleBranching(observation_function='43_var_features', # 'default' 'label_solution_values' '28_var_features', '45_var_features', '40_var_features' '43_var_features' '24_var_features' '29_var_features' '37_var_features'
                         information_function='default',
                         reward_function=reward_function,
                         scip_params='gasse_2019') # 'default' 'ml4co_item_placement' 'ml4co_load_balancing' 'ml4co_anonymous' 'gasse_2019' 'dfs'

    # set threshold difficulty params (optional)
    threshold_agent, threshold_env = None, None
    # threshold_network_path = '/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_21/checkpoint_275'
    # threshold_network = BipartiteGCNNoHeads(device=device, config=threshold_network_path+'/config.json')
    # threshold_network.load_state_dict(torch.load(threshold_network_path+'/trained_params.pkl'))
    # threshold_agent = REINFORCEAgent(device=device, policy_network=threshold_network, name='threshold_agent')
    # threshold_agent.eval() # turn on evaluation mode
    # threshold_agent = PseudocostBranchingAgent()
    # threshold_env = EcoleBranching(observation_function='default',
                                   # information_function='default',
                                   # reward_function='default',
                                   # scip_params='default')

    # # init max_steps agent
    max_steps_agent = None
    # max_steps_network_path = '/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_21/checkpoint_275'
    # max_steps_network = BipartiteGCNNoHeads(device=device, config=max_steps_network_path+'/config.json')
    # max_steps_network.load_state_dict(torch.load(max_steps_network_path+'/trained_params.pkl', map_location=device))
    # max_steps_agent = REINFORCEAgent(device=device, policy_network=max_steps_network, name='max_steps_agent')
    # max_steps_agent.eval() # turn on evaluation mode


    # # demonstrator agent
    demonstrator_agent = None
    # demonstrator_agent = StrongBranchingAgent()

    # backtrack rollout expert
    backtrack_rollout_expert = None
    # expert_path = '/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_341/checkpoint_120' # 100x100
    # expert_path = '/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_343/checkpoint_233' # 500x1000 
    # expert_network = BipartiteGCN(device=device, config=expert_path+'/config.json')
    # expert_network.load_state_dict(torch.load(expert_path+'/network_params.pkl', map_location='cpu'))
    # expert_network = expert_network.to(device)
    # backtrack_rollout_expert = Agent(device=device,
                                     # network=expert_network,
                                     # print_forward_dim_warning=False)
    # backtrack_rollout_expert.eval()



    learner = DQNLearner(agent=agent,
                        env=env,
                        instances=instances,
                        reset_envs_batch=1,
                        max_steps=int(1e12), # int(1e12) 3 dont infinite loop while training
                        max_steps_agent=max_steps_agent,
                        buffer_min_length=20000, # 200 20000
                        buffer_capacity=100000, # 20000
                        use_per=True,
                        use_cer=False,
                        initial_per_beta=0.4,
                        final_per_beta=1.0,
                        final_per_beta_epoch=5000,
                        per_alpha=0.6,
                        min_agent_per_priority=0.001,
                        hard_update_target_frequency=10000, # 50 10000
                        soft_update_target_tau=1e-4, # None 1e-3 1e-4 1e-2
                        gradient_clipping_max_norm=None, # 1e-3 None
                        gradient_clipping_clip_value=10, # None
                        steps_per_update=10, # 25
                        prob_add_to_buffer=1, # 1.0 0.1
                        ecole_seed=ecole_seed,
                        reproducible_episodes=True,
                        batch_size=128, # 10 128
                        accumulate_gradient_factor=1,
                        save_gradients=True, # True False
                        # agent_reward='num_nodes',
                        # agent_reward=['primal_bound_frac', 'dual_bound_frac'], # 'num_nodes' 'dual_bound_frac' 'primal_dual_bound_frac_sum' ['primal_bound_frac', 'dual_bound_frac'] 'normalised_lp_gain'
                        # agent_reward=['primal_bound_gap_frac', 'dual_bound_gap_frac'], # 'num_nodes' 'dual_bound_frac' 'primal_dual_bound_frac_sum' ['primal_bound_frac', 'dual_bound_frac']
                        # agent_reward='primal_dual_bound_frac_sum',
                        # agent_reward='normalised_lp_gain',
                        # agent_reward='optimal_retro_trajectory_normalised_lp_gain',
                        # agent_reward='retro_branching',
                        agent_reward=reward_function, # step-retro
                        # agent_reward='retro_branching_mcts_backprop',
                        # intrinsic_reward='noveld', # None 'noveld'
                        # agent_reward='binary_solved', # step-orig baseline
                        # agent_reward='binary_fathomed',
                        intrinsic_reward=None, # None 'noveld'
                        intrinsic_extrinsic_combiner='list',
                        n_step_return=3, # 3 5
                        use_n_step_dqn=False,
                        lr=5e-5, # 5e-5 1e-4
                        gamma=0.99, # 0.99 0.9
                        # loss_function=SmoothL1Loss(beta=1, reduction='mean'), # MeanSquaredError(), SmoothL1Loss
                        loss_function=MeanSquaredError(), # MeanSquaredError(), SmoothL1Loss
                        # loss_function=HuberLoss(delta=1, reduction='mean'),
                        optimizer_name='adam', # 'adam', 'sgd'
                        munchausen_tau=0, # 0 0.03
                        munchausen_lo=-1,
                        munchausen_alpha=0.9,
                        initial_epsilon=0.025,
                        final_epsilon=0.025,
                        final_epsilon_epoch=5000, # 5000 10000
                        double_dqn_clipping=True, # True False
                        threshold_difficulty=None, # None 100
                        threshold_agent=threshold_agent,
                        threshold_env=threshold_env,

                        demonstrator_agent=demonstrator_agent, # set to None if dont wasn DQfD
                        save_demonstrator_buffer=False,
                        num_pretraining_epochs=10000, # 10000 
                        demonstrator_buffer_capacity=10000, # 5000
                        min_demonstrator_per_priority=1e-1,
                        demonstrator_n_step_return_loss_weight=1,
                        demonstrator_margin_loss_weight=1e-5,
                        demonstrator_margin=0.8,
                        # demonstrator_l2_regularization_weight=1e-6,
                        weight_decay=0, # 0 1e-6

                        backtrack_rollout_expert=backtrack_rollout_expert,
                        max_attempts_to_reach_expert=1,

                        episode_log_frequency=1,
                        checkpoint_frequency=2500,
                        # path_to_save=None, # '/scratch/datasets/retro_branching' None
                        path_to_save='/scratch/datasets/retro_branching', # '/scratch/datasets/retro_branching' None
                        use_sqlite_database=True,
                        name='dqn_learner',
                        debug_mode=False,
                        profile_time=profile_time)

    print('Initialised learner with params {}. Will save to {}'.format(learner.episodes_log, learner.path_to_save))

    learner.train(int(5e6)) # 5e5 100 500
    # learner.train(int(10)) # 5e5 100 500


    

    # # getting strong branching result for an overfit instance
    # env.seed(0)
    # agent = StrongBranchingAgent()
    # agent.before_reset(instances)
    # obs, action_set, reward, done, info = env.reset(instances)
    # while not done:
        # action, action_idx = agent.action_select(action_set, env.model, done)
        # obs, action_set, reward, done, info = env.step(action)
    # print(env.model.as_pyscipopt().getNNodes())





    # # TIME PROFILING
    # # 1. Generate a file called <name>.prof
    # # 2. Transfer to /home/cwfparsonson/Downloads
    # # 3. Run snakeviz /home/cwfparsonson/Downloads/<name>.prof to visualise
    # profiler = cProfile.Profile()
    # profiler.enable()
    # learner.train(100)
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # # stats.print_stats()
    # stats.dump_stats('large_profiling_training_speed.prof')







