from retro_branching.learners import SupervisedLearner
from retro_branching.networks import BipartiteGCN, BipartiteGCNNoHeads
from retro_branching.utils import GraphDataset, seed_stochastic_modules_globally, gen_co_name
from retro_branching.loss_functions import CrossEntropy, JensenShannonDistance, KullbackLeiblerDivergence, BinaryCrossEntropyWithLogits, BinaryCrossEntropy, MeanSquaredError

import torch_geometric 
import pathlib
import glob
import numpy as np
import os







if __name__ == '__main__':
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

    # init agent
    device = 'cuda:1'
    # agent = BipartiteGCN(device,
                        # emb_size=128,
                        # num_rounds=2,
                        # aggregator='add',
                        # name='gnn')
    # agent = BipartiteGCNNoHeads(device,
                        # emb_size=64,
                        # num_rounds=1,
                        # aggregator='add',
                        # name='gnn')
    agent = BipartiteGCN(device=device,
                          emb_size=64,
                          num_rounds=1,
                          cons_nfeats=5,
                          edge_nfeats=1,
                          var_nfeats=19, # 19 20 28 45 40
                          aggregator='add',
                          activation=None,
                          num_heads=1,
                          head_depth=2,
                          linear_weight_init='normal',
                          linear_bias_init='zeros',
                          layernorm_weight_init=None,
                          layernorm_bias_init=None,
                          head_aggregator=None,
                          include_edge_features=True,
                          name='gnn') # None 'add'
    agent.to(device)
    agent.train() # turn on train mode
    print('Initialised agent.')

    # SC
    co_class = 'set_covering' # 'set_covering' 'combinatorial_auction' 'capacitated_facility_location' 'maximum_independent_set'
    co_class_kwargs = {'n_rows': 500, 'n_cols': 1000}

    # get paths to labelled training and validation data
    num_samples = 120000 # 200000 100000 1000 100 120000
    # path = '/scratch/datasets/retro_branching/strong_branching/samples/aggregated_samples/'

    # # CFL
    # co_class = 'capacitated_facility_location' # 'set_covering' 'combinatorial_auction' 'capacitated_facility_location' 'maximum_independent_set'
    # co_class_kwargs = {'n_customers': 5, 'n_facilities': 12}

    # # MIS 
    # co_class = 'maximum_independent_set' # 'set_covering' 'combinatorial_auction' 'capacitated_facility_location' 'maximum_independent_set'
    # co_class_kwargs = {'n_nodes': 58}

    branching = 'explore_then_strong_branch' # 'pure_strong_branch' 'explore_then_strong_branch'
    max_steps = None # None 3
    folder_name = 'samples_1' # 'aggregated_samples' 'samples_1'
    # path = f'/scratch/datasets/retro_branching/strong_branching/{branching}/{co_class}/max_steps_{max_steps}/nrows_{nrows}_ncols_{ncols}/samples/{folder_name}/'
    # path = f'/scratch/datasets/retro_branching/strong_branching/{branching}/{co_class}/max_steps_{max_steps}/nrows_{nrows}_ncols_{ncols}/samples/{folder_name}/'

    path = f'/scratch/datasets/retro_branching/strong_branching/{branching}/{co_class}/max_steps_{max_steps}/{gen_co_name(co_class, co_class_kwargs)}/samples/{folder_name}/'
    # path = f'/scratch/datasets/retro_branching/gasse_2019/custom_data/data/strong_branching/{branching}/{co_class}/general_{nrows}r_{ncols}c_0.05d/samples/{folder_name}/'

    # path = f'/scratch/datasets/retro_branching/imitation_branching/gnn_21_checkpoint_275/nrows_{nrows}_ncols_{ncols}/num_nodes/gamma_0/samples/aggregated_samples/'
    # path = f'/scratch/datasets/retro_branching/imitation_branching/gnn_21_checkpoint_275/nrows_{nrows}_ncols_{ncols}/dual_bound_frac/gamma_0/samples/aggregated_samples/'
    # path = f'/scratch/datasets/retro_branching/imitation_branching/gnn_21_checkpoint_275/nrows_{nrows}_ncols_{ncols}/num_nodes/gamma_0.99/samples/aggregated_samples/'
    # path = f'/scratch/datasets/retro_branching/imitation_branching/gnn_21_checkpoint_275/nrows_{nrows}_ncols_{ncols}/dual_bound_frac/gamma_0.99/samples/aggregated_samples/'
    # path = f'/scratch/datasets/retro_branching/imitation_branching/strong_branching/nrows_{nrows}_ncols_{ncols}/normalised_lp_gain/gamma_0/obs_43_var_features/samples/aggregated_samples/'

    if not os.path.isdir(path):
        raise Exception(f'Path {path} does not exist')
    files = np.array(glob.glob(path+'*.pkl'))
    # np.random.shuffle(files)
    # sample_files = [str(path) for path in pathlib.Path('dict_disabler_samples/').glob('sample_*.pkl')]
    sample_files = files[:num_samples]
    files = [] # clear memory
    train_files = sample_files[:int(0.83*len(sample_files))]
    valid_files = sample_files[int(0.83*len(sample_files)):]

    # init training and validaton data loaders
    train_data = GraphDataset(train_files)
    print(train_data)
    train_loader = torch_geometric.data.DataLoader(train_data, batch_size=32, shuffle=True)
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.data.DataLoader(valid_data, batch_size=512, shuffle=False)
    print('Initialised training and validation data loaders.')

    # init learner
    learner = SupervisedLearner(agent=agent,
                                train_loader=train_loader,
                                valid_loader=valid_loader,
                                imitation_target='expert_actions', # 'expert_scores' 'expert_score' 'expert_actions' 'expert_bipartite_ranking'
                                loss_function=CrossEntropy(), # MeanSquaredError() CrossEntropy() JensenShannonDistance() KullbackLeiblerDivergence()
                                lr=1e-4,
                                bipartite_ranking_alpha=0.5,
                                epoch_log_frequency=1,
                                checkpoint_frequency=1,
                                save_logits_and_target=True,
                                path_to_save='/scratch/datasets/retro_branching',
                                # path_to_save=None,
                                name='supervised_learner')
    print('Initialised learner with params {}. Will save to {}'.format(learner.epochs_log, learner.path_to_save))

    # train agent
    print('Training agent...')
    learner.train(1000)
