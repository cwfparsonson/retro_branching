# disable annoying warnings
import warnings
warnings.filterwarnings(action='ignore',
                        category=FutureWarning,
                        module='tensorboard')  # noqa
# warnings.filterwarnings(action='ignore',
                        # category=FutureWarning,
                        # module='ray')  # noqa

from retro_branching.agents import PPOAgent
from retro_branching.learners import PPOLearner
from retro_branching.networks import BipartiteGCN
from retro_branching.environments import EcoleBranching
from retro_branching.utils import seed_stochastic_modules_globally
from retro_branching.loss_functions import MeanSquaredError

import ecole

import glob



if __name__ == '__main__':
    seed = 0
    seed_stochastic_modules_globally(seed)

    device = 'cuda:3'

    actor_network = BipartiteGCN(device=device,
                            emb_size=64,
                            num_rounds=1,
                            cons_nfeats=5,
                            edge_nfeats=1, 
                            var_nfeats=43, # 19 20 28 45 40 43 29 37 23
                            aggregator='add',
                            activation='inverse_leaky_relu',
                            num_heads=1, # 1 2
                            # num_heads=2, # 1 2
                            # head_depth=1, # 1 2
                            head_depth=2, # 1 2
                            linear_weight_init=None,
                            linear_bias_init=None,
                            layernorm_weight_init=None,
                            layernorm_bias_init=None,
                            head_aggregator='add', # 'add'
                            # head_aggregator={'train': None, 'test': 0}, # 'add'
                            include_edge_features=True, 
                            profile_time=False) # None 'add'

    critic_network = BipartiteGCN(device=device,
                            emb_size=64,
                            num_rounds=1,
                            cons_nfeats=5,
                            edge_nfeats=1, 
                            var_nfeats=43, # 19 20 28 45 40 43 29 37 23
                            aggregator='add',
                            activation='inverse_leaky_relu',
                            num_heads=1, # 1 2
                            # num_heads=2, # 1 2
                            # head_depth=1, # 1 2
                            head_depth=2, # 1 2
                            linear_weight_init=None,
                            linear_bias_init=None,
                            layernorm_weight_init=None,
                            layernorm_bias_init=None,
                            head_aggregator='add', # 'add'
                            # head_aggregator={'train': None, 'test': 0}, # 'add'
                            include_edge_features=True, 
                            profile_time=False) # None 'add'

    agent = PPOAgent(device=device,
                     actor_network=actor_network,
                     critic_network=critic_network)
    agent.train()

    env = EcoleBranching(observation_function='43_var_features') # custom_var_features 43_var_features


    # SC

    # 100x100
    # instances = ecole.instance.SetCoverGenerator(n_rows=100, n_cols=100)
    # instances = iter(glob.glob(f'/scratch/datasets/retro_branching/instances/training/nrows_100_ncols_100/scip_params_default/threshold_difficulty_None/threshold_agent_None/seed_0/samples/samples_1/*.mps'))

    # 500x1000
    # instances = ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000)
    instances = iter(glob.glob(f'/scratch/datasets/retro_branching/instances/training/nrows_500_ncols_1000/scip_params_default/threshold_difficulty_None/threshold_agent_None/seed_0/samples/samples_1/*.mps'))




    learner = PPOLearner(agent=agent,
                         env=env,
                         instances=instances,
                         ecole_seed=seed,
                         reproducible_episodes=True,

                         # extrinsic_reward='num_nodes',
                         # extrinsic_reward='dual_bound_gap_frac',
                         # extrinsic_reward='retro_binary_fathomed',
                         # extrinsic_reward='retro_branching_mcts_backprop',
                         extrinsic_reward='binary_fathomed',
                         intrinsic_reward=None,
                         intrinsic_extrinsic_combiner='list',

                         value_function_coeff=0.5,
                         entropy_coeff=0.01,
                         eps_clip=0.2,
                         whiten_rewards=True,

                         ppo_update_freq=10, # 500 10
                         ppo_epochs_per_update=3,
                         batch_size=128,
                         gradient_accumulation_factor=1,

                         actor_gradient_clipping_clip_value=1.0,
                         critic_gradient_clipping_clip_value=5.0e-12,
                         # actor_gradient_clipping_clip_value=None,
                         # critic_gradient_clipping_clip_value=None,

                         num_workers=None,
                         # num_workers=8,
                         
                         gamma=0.9,
                         lr_actor=1e-4,
                         lr_critic=1e-4,
                         critic_loss_function=MeanSquaredError(),

                         episode_log_freq=1,
                         epoch_log_freq=1,
                         checkpoint_freq=2,
                         # path_to_save=None,
                         path_to_save='/scratch/datasets/retro_branching',
                         use_sqlite_database=True,

                         name='ppo_learner',
                         debug_mode=False)

    learner.train(int(5e6))
    # learner.train(10)
























