import ecole
import numpy as np
from collections import defaultdict

class NodeBipariteWith28VariableFeatures(ecole.observation.NodeBipartite):
    '''
    Adds (mostly global) features to variable node features.

    Adds 9 extra variable features to each variable on top of standard ecole
    NodeBipartite obs variable features (19), so each variable will have
    28 features in total.

    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def before_reset(self, model):
        super().before_reset(model)
        
        self.init_dual_bound = None
        self.init_primal_bound = None
        
        
    def extract(self, model, done):
        # get the NodeBipartite obs
        obs = super().extract(model, done)
        
        m = model.as_pyscipopt()
        
        if self.init_dual_bound is None:
            self.init_dual_bound = m.getDualbound()
            self.init_primal_bound = m.getPrimalbound()
            
        # dual/primal bound features
        dual_bound_frac_change = self.init_dual_bound / m.getDualbound()
        primal_bound_frac_change = self.init_primal_bound / m.getPrimalbound()
        curr_primal_dual_bound_gap_frac = m.getGap()
        
        # global tree features
        num_leaves_frac = m.getNLeaves() / m.getNNodes()
        num_feasible_leaves_frac = m.getNFeasibleLeaves() / m.getNNodes()
        num_infeasible_leaves_frac = m.getNInfeasibleLeaves() / m.getNNodes()
        # getNSolsFound() raises attribute error for some reason. Not supported by Ecole?
#         num_feasible_sols_found_frac = m.getNSolsFound() / m.getNNodes() # gives idea for how hard problem is, since harder problems may have more sparse feasible solutions?
#         num_feasible_best_sols_found_frac = m.getNBestSolsFound() / m.getNSolsFound()
        num_lp_iterations_frac = m.getNNodes() / m.getNLPIterations()
        
        # focus node features
        num_children_frac = m.getNChildren() / m.getNNodes()
        num_siblings_frac = m.getNSiblings() / m.getNNodes()
        
        # add feats to each variable
        feats_to_add = np.array([[dual_bound_frac_change,
                                 primal_bound_frac_change,
                                 curr_primal_dual_bound_gap_frac,
                                 num_leaves_frac,
                                 num_feasible_leaves_frac,
                                 num_infeasible_leaves_frac,
                                 num_lp_iterations_frac,
                                 num_children_frac,
                                 num_siblings_frac] for _ in range(obs.variable_features.shape[0])])
        
        obs.variable_features = np.column_stack((obs.variable_features, feats_to_add))
                
        return obs