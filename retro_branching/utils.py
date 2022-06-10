import pyscipopt
import ecole
import torch
import torch.nn.functional as F
import torch_geometric

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import gzip
import pickle
import os
from ordered_set import OrderedSet
from collections import defaultdict

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout



def gen_co_name(co_class, co_class_kwargs):
    _str = f'{co_class}'
    for key, val in co_class_kwargs.items():
        _str += f'_{key}_{val}'
    return _str



############################# ECOLE #################################
class SearchTree:
    '''
    Tracks SCIP search tree. Call SearchTree.update_tree(ecole.Model) each
    time the ecole environment (and therefore the ecole.Model) is updated.

    N.B. SCIP does not store nodes which were pruned, infeasible, outside
    the search tree's optimality bounds, or which node was optimal, therefore these nodes will not be
    stored in the SearchTree. This is why m.getNTotalNodes() (the total number
    of nodes processed by SCIP) will likely be more than the number of nodes in
    the search tree when an instance is solved.
    '''
    def __init__(self, model):       
        self.tree = nx.DiGraph()
        
        self.tree.graph['root_node'] = None
        self.tree.graph['visited_nodes'] = []
        self.tree.graph['visited_node_ids'] = OrderedSet()
        
        m = model.as_pyscipopt()
        if m.getCurrentNode() is not None:

            self.tree.graph['optimum_nodes'] = [m.getCurrentNode()]
            self.tree.graph['optimum_node_ids'] = OrderedSet([m.getCurrentNode().getNumber()])
            self.init_primal_bound = m.getPrimalbound()
            self.tree.graph['incumbent_primal_bound'] = self.init_primal_bound

            self.tree.graph['fathomed_nodes'] = []
            self.tree.graph['fathomed_node_ids'] = OrderedSet()

            self.prev_primal_bound = None
            self.prev_node_id = None

            self.step_idx = 0

            self.update_tree(model)
            
        else:
            # instance was pre-solved
            pass
            
    def update_tree(self, model):
        '''
        Call this method after each update to the ecole environment. Pass
        the updated ecole.Model, and the B&B tree tracker will be updated accordingly.
        '''
        m = model.as_pyscipopt()
                
        # get current node (i.e. next node to be branched at)
        _curr_node = m.getCurrentNode()
        if _curr_node is not None:
            self.curr_node_id = _curr_node.getNumber()
        else:
            # branching finished, no curr node
            self.curr_node_id = None
        
        if len(self.tree.graph['visited_node_ids']) >= 1:
            self.prev_node_id, self.prev_node = self.tree.graph['visited_node_ids'][-1], self.tree.graph['visited_nodes'][-1]
            
            # check if previous branching at previous node changed global primal bound. If so, set previous node as optimum
            if m.getPrimalbound() < self.tree.graph['incumbent_primal_bound']:
                # branching at previous node led to finding new incumbent solution
                self.tree.graph['optimum_nodes'].append(self.prev_node)
                self.tree.graph['optimum_node_ids'].add(self.prev_node_id)
                self.tree.graph['incumbent_primal_bound'] = m.getPrimalbound()
            
        self.curr_node = {self.curr_node_id: _curr_node}
        if self.curr_node_id is not None:
            if self.curr_node_id not in self.tree.graph['visited_node_ids']:
                self._add_nodes(self.curr_node)
                self.tree.graph['visited_nodes'].append(self.curr_node)
                self.tree.graph['visited_node_ids'].add(self.curr_node_id)
                self.tree.nodes[self.curr_node_id]['step_visited'] = self.step_idx
        
        if self.curr_node_id is not None:
            _parent_node = list(self.curr_node.values())[0].getParent()
            if _parent_node is not None:
                parent_node_id = _parent_node.getNumber()
            else:
                # curr node is root node
                parent_node_id = None
            self.parent_node = {parent_node_id: _parent_node}
        else:
            self.parent_node = {None: None} 
            
        # add open nodes to tree
        open_leaves, open_children, open_siblings = m.getOpenNodes()
        self.open_leaves = {node.getNumber(): node  for node in open_leaves}
        self.open_children = {node.getNumber(): node for node in open_children}
        self.open_siblings = {node.getNumber(): node for node in open_siblings}
        
        self._add_nodes(self.open_leaves)
        self._add_nodes(self.open_children)
        self._add_nodes(self.open_siblings)
        
        # check if previous branching at previous node led to fathoming
        if len(self.tree.graph['visited_node_ids']) > 2 or self.curr_node_id is None:
            if self.curr_node_id is not None:
                # in above code, have added current node to visited node ids, therefore prev node is at idx=-2
                self.prev_node_id, self.prev_node = self.tree.graph['visited_node_ids'][-2], self.tree.graph['visited_nodes'][-2]
            else:
                # branching finished, previous node was fathomed
                self.prev_node_id, self.prev_node = self.tree.graph['visited_node_ids'][-1], self.tree.graph['visited_nodes'][-1]
            if len(list(self.tree.successors(self.prev_node_id))) == 0 and self.prev_node_id != self.curr_node_id:
                # branching at previous node led to fathoming
                self.tree.graph['fathomed_nodes'].append(self.prev_node)
                self.tree.graph['fathomed_node_ids'].add(self.prev_node_id)

        self.step_idx += 1

    def _add_nodes(self, nodes, parent_node_id=None):
        '''Adds nodes if not already in tree.'''
        for node_id, node in nodes.items():
            if node_id not in self.tree:
                # add node
                self.tree.add_node(node_id,
                                   _id=node_id,
                                   lower_bound=node.getLowerbound())

                # add edge
                _parent_node = node.getParent()
                if _parent_node is not None:
                    if parent_node_id is None:
                        parent_node_id = _parent_node.getNumber()
                    else:
                        # parent node id already given
                        pass
                    self.tree.add_edge(parent_node_id,
                                       node_id)
                else:
                    # is root node, has no parent
                    self.tree.graph['root_node'] = {node_id: node}
                    
    def _get_node_groups(self):
        node_groups = defaultdict(lambda: [])
        for node in self.tree.nodes:
            if node not in self.tree.graph['visited_node_ids'] or self.curr_node_id == node:
                node_groups['Unvisited'].append(node)
            else:
                node_groups['Visited'].append(node)
            if node in self.tree.graph['fathomed_node_ids']:
                node_groups['Fathomed'].append(node)
            if node == self.tree.graph['optimum_node_ids'][-1]:
                node_groups['Incumbent'].append(node)
        return node_groups
                                    
    def render(self,
               unvisited_node_colour='#FFFFFF',
               visited_node_colour='#A7C7E7',
               fathomed_node_colour='#FF6961',
               incumbent_node_colour='#C1E1C1',
               next_node_colour='#FFD700',
               node_edge_colour='#000000',
               use_latex_font=True,
               font_scale=0.75,
               context='paper',
               style='ticks'
              ):
        '''Renders B&B search tree.'''
        if use_latex_font:
            sns.set(rc={'text.usetex': True},
                    font='times')
        sns.set_theme(font_scale=font_scale, context=context, style=style)
        
        group_to_colour = {'Unvisited': unvisited_node_colour,
                           'Visited': visited_node_colour,
                           'Fathomed': fathomed_node_colour,
                           'Incumbent': incumbent_node_colour}
        
        f, ax = plt.subplots()
        
        pos = graphviz_layout(self.tree, prog='dot')

        node_groups = self._get_node_groups()
        for group_label, nodes in node_groups.items():
            nx.draw_networkx_nodes(self.tree,
                                   pos,
                                   nodelist=nodes,
                                   node_color=group_to_colour[group_label],
                                   edgecolors=node_edge_colour,
                                   label=group_label)
            
        if self.curr_node_id is not None:
            nx.draw_networkx_nodes(self.tree,
                                   pos,
                                   nodelist=[self.curr_node_id],
                                   node_color=unvisited_node_colour,
                                   edgecolors=next_node_colour,
                                   linewidths=3,
                                   label='Next')
            num_groups = len(list(node_groups.keys())) + 1
        else:
            num_groups = len(list(node_groups.keys()))
    
        nx.draw_networkx_edges(self.tree,
                               pos)
        
        nx.draw_networkx_labels(self.tree, pos, labels={node: node for node in self.tree.nodes})
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=num_groups)
        
        plt.show()

############################### HELPER FUNCTIONS #############################
def seed_stochastic_modules_globally(default_seed=0, 
                                     numpy_seed=None, 
                                     random_seed=None, 
                                     torch_seed=None, 
                                     ecole_seed=None):
    '''Seeds any stochastic modules so get reproducible results.'''
    if numpy_seed is None:
        numpy_seed = default_seed
    if random_seed is None:
        random_seed = default_seed
    if torch_seed is None:
        torch_seed = default_seed
    if ecole_seed is None:
        ecole_seed = default_seed

    np.random.seed(numpy_seed)

    random.seed(random_seed)

    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ecole.seed(ecole_seed)


def turn_off_scip_heuristics(ecole_instance):
    # write ecole instance to mps
    ecole_instance.write_problem('tmp_instance.mps')
    
    # read mps into pyscip model
    pyscipopt_instance = pyscipopt.Model()
    pyscipopt_instance.readProblem('tmp_instance.mps')
    
    # turn off heuristics
    pyscipopt_instance.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
    pyscipopt_instance.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
    pyscipopt_instance.disablePropagation()
    pyscipopt_instance.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF)
    
    return ecole.scip.Model.from_pyscipopt(pyscipopt_instance)


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    """
    This utility function splits a tensor and pads each split to make them all the same size, then stacks them.
    """
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack([F.pad(slice_, (0, max_pad_size - slice_.size(0)), 'constant', pad_value)
                          for slice_ in output], dim=0)
    return output

def get_most_recent_checkpoint_foldername(path, idx=-1):
    '''
    Given a path to a folders named <name>_<number>, will sort checkpoints (i.e. most recently saved checkpoint
    last) and return name of idx provided (default idx=-1 to return 
    most recent checkpoint folder).
    '''
    foldernames = [name.split('_') for name in os.listdir(path)]
    idx_to_num = {idx: int(num) for idx, num in zip(range(len(foldernames)), [name[-1] for name in foldernames if name[0] == 'checkpoint'])}
    sorted_indices = np.argsort(list(idx_to_num.values()))
    _idx = sorted_indices[idx]
    foldername = [name for name in os.listdir(path) if name.split('_')[0] == 'checkpoint'][_idx]
    return foldername

def create_milp_curriculum(num_levels, init_matrix_size, final_matrix_size, density=0.05):
    '''
    Args:
        num_levels (int): Number of difficulty levels in curriculum.
        init_matrix (list): Initial [nrows, ncols] MILP problem size at first level.
        final_matrix (list): Final [nrows, ncols] MILP problem size at final level.
        
    Return:
        (dict): Maps level to problem size.
    '''
    init_nrows, init_ncols = init_matrix_size[0], init_matrix_size[1]
    final_nrows, final_ncols = final_matrix_size[0], final_matrix_size[1]

    nrows_distance, ncols_distance = final_nrows-init_nrows, final_ncols-init_ncols
    nrows_delta, ncols_delta = int(nrows_distance/num_levels), int(ncols_distance/num_levels)

    curriculum = {0: [init_nrows, init_ncols]}
    for level in range(1, num_levels-1):
        curriculum[level] = [curriculum[level-1][0]+nrows_delta, curriculum[level-1][1]+ncols_delta]
    curriculum[num_levels-1] = [final_nrows, final_ncols]
    
    return curriculum
    

def check_if_network_params_equal(net_1, net_2):
    all_equal = True
    for (name_1, tensor_1), (name_2, tensor_2) in zip(net_1.named_parameters(), net_2.named_parameters()):
        print(f'\n{name_1}:\n{tensor_1}:')
        print(f'{name_2}:\n{tensor_2}:')
        if not torch.equal(tensor_1, tensor_2):
            # not equal
            all_equal = False 
    return all_equal








########################### PYTORCH DATA LOADERS #############################
class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite` 
    observation function in a format understood by the pytorch geometric data handlers.
    """
    def __init__(self, 
                 constraint_features=None, 
                 edge_indices=None, 
                 edge_features=None, 
                 variable_features=None,
                 candidates=None, 
                 candidate_choice=None, 
                 candidate_scores=None, 
                 score=None):
        super().__init__()
        if constraint_features is not None:
            self.constraint_features = torch.FloatTensor(constraint_features)
        if edge_indices is not None:
            self.edge_index = torch.LongTensor(edge_indices.astype(np.int64))
        if edge_features is not None:
            self.edge_attr = torch.FloatTensor(edge_features).unsqueeze(1)
        if variable_features is not None:
            self.variable_features = torch.FloatTensor(variable_features)
        if candidates is not None:
            self.candidates = torch.LongTensor(candidates)
            self.num_candidates = len(candidates)
        if candidate_choice is not None:
            self.candidate_choices = torch.LongTensor(candidate_choice)
        if candidate_scores is not None:
            self.candidate_scores = torch.FloatTensor(candidate_scores)
        if score is not None:
            self.score = torch.FloatTensor(score)

    def __inc__(self, key, value, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs 
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'candidates':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value)

class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        sample_observation, sample_action, sample_action_set, sample_scores = sample
        
        # We note on which variables we were allowed to branch, the scores as well as the choice 
        # taken by strong branching (relative to the candidates)
        candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
        try:
            candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])
            score = []
        except (TypeError, IndexError):
            # only given one score and not in a list so not iterable
            score = torch.FloatTensor([sample_scores])
            candidate_scores = []
        candidate_choice = torch.where(candidates == sample_action)[0][0]

        graph = BipartiteNodeData(sample_observation.row_features, sample_observation.edge_features.indices, 
                                  sample_observation.edge_features.values, sample_observation.column_features,
                                  candidates, candidate_choice, candidate_scores, score)
        
        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = sample_observation.row_features.shape[0]+sample_observation.column_features.shape[0]
        
        return graph












################################### PLOTTING #################################

class PlotAesthetics:
    def __init__(self):
        pass

    def set_icml_paper_plot_aesthetics(self,
                                      context='paper',
                                      style='ticks',
                                      linewidth=0.75,
                                      font_scale=1,
                                      palette='colorblind',
                                      desat=1,
                                      dpi=300):
        
        # record params
        self.context = context
        self.linewidth = linewidth
        self.font_scale = font_scale
        self.palette = palette
        self.desat = desat
        self.dpi = dpi

        # apply plot config
        sns.set(rc={'text.usetex': True,
                    'figure.dpi': dpi,
                    'savefig.dpi': dpi})
        sns.set(font="times")
        sns.set_theme(font_scale=font_scale,
                      context=context,
                      style=style,
                      palette=palette)

        
    def get_standard_fig_size(self,
                              col_width=3.25, 
                              col_spacing=0.25, 
                              n_cols=1,
                              scaling_factor=1,
                              width_scaling_factor=1,
                              height_scaling_factor=1):
        
        # save params
        self.col_width = col_width
        self.col_spacing = col_spacing
        self.n_cols = n_cols
        self.scaling_factor=scaling_factor
        self.width_scaling_factor = width_scaling_factor
        self.height_scaling_factor = height_scaling_factor
    
        # calc fig size
        self.fig_width = ((col_width * n_cols) + ((n_cols - 1) * col_spacing))
        golden_mean = (np.sqrt(5) - 1.0) / 2.0    # Aesthetic ratio
        self.fig_height = self.fig_width * golden_mean
        return (scaling_factor * width_scaling_factor * self.fig_width, scaling_factor * height_scaling_factor * self.fig_height)

    def get_winner_bar_fig_size(self,
                                col_width=3.25, 
                                col_spacing=0.25, 
                                n_cols=1):
        # save params
        self.col_width = col_width
        self.col_spacing = col_spacing
        self.n_cols = n_cols

        # calc fig size
        self.fig_width = ((col_width * n_cols) + ((n_cols - 1) * col_spacing))
        self.fig_height = self.fig_width * 1.25

        return (self.fig_width, self.fig_height)





def get_plot_params_config(font_size):
    params = {'legend.fontsize': font_size*0.75,
              'axes.labelsize': font_size,
              'axes.titlesize': font_size,
              'xtick.labelsize': font_size*0.75,
              'ytick.labelsize': font_size*0.75}
    return params

def smooth_line(data, axis, res=0.01):
    '''
    res=0.1 will sample mean every 10% of data, res=0.01 will sample mean every 1% of data,
    etc. res=1 will take mean of whole data set -> get straight line, res=0 will
    return original data.
    '''
    if res < 0 or res > 1:
        raise Exception('0 <= res <= 1 but is {}'.format(res))
    smoothed_data = []
    if res != 1:
        num_pts = max(1, int(res*len(data)))
        for idx in range(0, len(data), num_pts):
            if axis == 'y':
                smoothed_data.append(np.mean(data[idx:idx+num_pts]))
            elif axis == 'x':
                smoothed_data.append(np.max(data[idx:idx+num_pts]))
            else:
                raise Exception('axis must be x or y, not {}'.format(axis))
    else:
        if axis == 'y':
            smoothed_data.append(np.mean(data))
        elif axis == 'x':
            smoothed_data.append(np.max(data))
        else:
            raise Exception('axis must be x or y, not {}'.format(axis))
    return smoothed_data

def plot_val_line(plot_dict={},
                 xlabel='Random Variable',
                 ylabel='Random Variable Value',
                 ylim=None,
                 linewidth=1,
                 alpha=1,
                 ylogscale=False,
                 title=None,
                 vertical_lines=[],
                 gridlines=True,
                 aspect='auto',
                 plot_style='default',
                 font_size=10,
                 figsize=(6.4, 4.8),
                 plot_legend=True,
                 legend_ncol=1,
                 smooth_data_res=None,
                 show_fig=False):
    '''Plots line plot.
    plot_dict= {'class_1': {'x_values': [0.1, 0.2, 0.3], 'y_values': [20, 40, 80]},
                'class_2': {'x_values': [0.1, 0.2, 0.3], 'y_values': [80, 60, 20]}}
    '''

    keys = list(plot_dict.keys())

    fig = plt.figure(figsize=figsize)
    plt.style.use(plot_style)
    plt.rcParams.update(get_plot_params_config(font_size=font_size))

    class_colours = iter(sns.color_palette(palette='hls', n_colors=len(keys), desat=None))
    for _class in sorted(plot_dict.keys()):
        if smooth_data_res is not None:
            ydata = smooth_line(plot_dict[_class]['y_values'], axis='y', res=smooth_data_res)
            xdata = smooth_line(plot_dict[_class]['x_values'], axis='x', res=smooth_data_res)
        else:
            ydata = plot_dict[_class]['y_values']
            xdata = plot_dict[_class]['x_values']
        if len(ydata) == 1:
            ydata.insert(0, ydata[0])
            xdata.insert(0, 0)
        plt.plot(xdata, ydata, color=next(class_colours), linewidth=linewidth, alpha=alpha, label=str(_class))
        for vline in vertical_lines:
            plt.axvline(x=vline, color='r', linestyle='--')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax = plt.gca()
    if ylogscale:
        ax.set_yscale('log')
    if ylim is not None:
        plt.ylim(bottom=ylim[0], top=ylim[1])
    if plot_legend:
        plt.legend(ncol=legend_ncol)

    if gridlines:
        plt.grid(which='both', axis='both', color='gray', linestyle='dashed', alpha=0.3)
    if aspect != 'auto':
        plt.gca().set_aspect(aspect=_get_matplotlib_aspect_ratio(fig, aspect_ratio=aspect))
    if title is not None:
        plt.title(title)

    if show_fig:
        plt.show()
    
    return fig




def sns_plot_val_line(plot_dict,
                      xlabel='Random Variable',
                      ylabel='Random Variable Value',
                      horizontal_lines={},
                      title=None,
                      moving_average_window=None,
                      xlim=None,
                      ylim=None,
                      ylogscale=False,
                      context='paper',
                      font_scale=1,
                      style='whitegrid',
                      palette='hls',
                      linewidth=1,
                      alpha=1,
                      ci=95,
                      rc=None,
                      show_fig=True,
                      **kwargs):
    '''Plots line plot.
    plot_dict= {'class_1': {'x_values': [0.1, 0.2, 0.3], 'y_values': [20, 40, 80]},
                'class_2': {'x_values': [0.1, 0.2, 0.3], 'y_values': [80, 60, 20]}}
    horizontal_lines (dict): Map label to horizontal line value to plot.

    **kwargs:
        plot_unfiltered_data (bool): If moving_average_window is not None, whether
            or not to plot the original unaveraged data.
        unfiltered_data_alpha (float): Alpha value of original unfiltered data.
    '''
    if 'plot_unfiltered_data' not in kwargs:
        kwargs['plot_unfiltered_data'] = True
    if 'unfiltered_data_alpha' not in kwargs:
        kwargs['unfiltered_data_alpha'] = 0.3

    sns.set_context(context=context, rc=rc) # paper notebook talk poster
    sns.set_style(style=style, rc=rc)

    fig = plt.figure()

    class_colours = iter(sns.color_palette(palette=palette, n_colors=len(list(plot_dict.keys()))+len(list(horizontal_lines.keys())), desat=None))
    for _class in sorted(plot_dict.keys()):
        color = next(class_colours)

        data = pd.DataFrame(plot_dict[_class])
        if moving_average_window is not None:
            if kwargs['plot_unfiltered_data']:
                # plot unfiltered data
                sns.lineplot(x='x_values', 
                             y='y_values', 
                             data=data,
                             color=color,
                             # style=True,
                             # dashes=[(2,2)],
                             linestyle='--',
                             linewidth=linewidth*0.5,
                             ci=ci,
                             alpha=kwargs['unfiltered_data_alpha'],)
            # apply moving average
            data['y_values'] = data.y_values.rolling(moving_average_window).mean()

        sns.lineplot(x='x_values', 
                     y='y_values', 
                     data=data,
                     color=color,
                     linewidth=linewidth,
                     err_style='band', 
                     alpha=alpha,
                     ci=ci,
                     label=str(_class))

    for label in sorted(horizontal_lines.keys()):
        color = next(class_colours)
        plt.axhline(y=horizontal_lines[label], color=color, linestyle='--', label=label)
    plt.legend()

    ax = plt.gca()
    if ylogscale:
        ax.set_yscale('log')

    if xlim is not None:
        plt.xlim(left=xlim[0], right=xlim[1])
    if ylim is not None:
        plt.ylim(bottom=ylim[0], top=ylim[1])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)

    if show_fig:
        plt.show()


    return fig











class ExploreThenStrongBranch:
    """
    This custom observation function class will randomly return either strong branching scores (expensive expert) 
    or pseudocost scores (weak expert for exploration) when called at every node.
    """
    def __init__(self, expert_probability):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.strong_branching_function = ecole.observation.StrongBranchingScores()
    
    def before_reset(self, model):
        """
        This function will be called at initialization of the environments (before dynamics are reset).
        """
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)
    
    def extract(self, model, done):
        """
        Should we return strong branching or pseudocost scores at time node?
        """
        probabilities = [1-self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            return (self.strong_branching_function.extract(model, done), True)
        else:
            return (self.pseudocosts_function.extract(model, done), False)


class PureStrongBranch:
    def __init__(self):
        self.strong_branching_function = ecole.observation.StrongBranchingScores()

    def before_reset(self, model):
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        return (self.strong_branching_function.extract(model, done))















