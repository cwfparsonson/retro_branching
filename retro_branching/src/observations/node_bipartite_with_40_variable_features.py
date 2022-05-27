import ecole
import numpy as np
from collections import defaultdict

class NodeBipariteWith40VariableFeatures(ecole.observation.NodeBipartite):
    '''
    Adds (mostly global) features to variable node features.

    Adds 21 extra variable features to each variable on top of standard ecole
    NodeBipartite obs variable features (19), so each variable will have
    40 features in total.

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
        dual_bound_frac_change = min(self.init_dual_bound, m.getDualbound()) / max(self.init_dual_bound, m.getDualbound())
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
        num_siblings_frac = m.getNSiblings() / m.getNNodes()
        curr_node = m.getCurrentNode()
        best_node = m.getBestNode()
        if best_node is not None:
            if curr_node.getNumber() == best_node.getNumber():
                is_curr_node_best = 1
            else:
                is_curr_node_best = 0
        else:
            # no best node found yet
            is_curr_node_best = 0
        parent_node = curr_node.getParent()
        if parent_node is not None and best_node is not None:
            if parent_node.getNumber() == best_node.getNumber():
                is_curr_node_parent_best = 1
            else:
                is_curr_node_parent_best = 0
        else:
            # node has no parent node or no best node found yet
            is_curr_node_parent_best = 0
        curr_node_depth = m.getDepth() / m.getNNodes()
        curr_node_lower_bound_relative_to_init_dual_bound = self.init_dual_bound / curr_node.getLowerbound()
        curr_node_lower_bound_relative_to_curr_dual_bound =  m.getDualbound() / curr_node.getLowerbound()
        num_branching_changes, num_constraint_prop_changes, num_prop_changes = curr_node.getNDomchg()
        total_num_changes = num_branching_changes + num_constraint_prop_changes + num_prop_changes
        try:
            branching_changes_frac = num_branching_changes / total_num_changes
        except ZeroDivisionError:
            branching_changes_frac = 0
        try:
            constraint_prop_changes_frac = num_constraint_prop_changes / total_num_changes
        except ZeroDivisionError:
            constraint_prop_changes_frac = 0
        try:
            prop_changes_frac = num_prop_changes / total_num_changes
        except ZeroDivisionError:
            prop_changes_frac = 0
        parent_branching_changes_frac = curr_node.getNParentBranchings() / m.getNNodes()
        best_sibling = m.getBestSibling()
        if best_sibling is None:
            is_best_sibling_none = 1
            is_best_sibling_best_node = 0
        else:
            is_best_sibling_none = 0
            if best_node is not None:
                if best_sibling.getNumber() == best_node.getNumber():
                    is_best_sibling_best_node = 1
                else:
                    is_best_sibling_best_node = 0
            else:
                is_best_sibling_best_node = 0
        if best_sibling is not None:
            best_sibling_lower_bound_relative_to_init_dual_bound = self.init_dual_bound / best_sibling.getLowerbound()
            best_sibling_lower_bound_relative_to_curr_dual_bound = m.getDualbound() / best_sibling.getLowerbound()
            best_sibling_lower_bound_relative_to_curr_node_lower_bound = best_sibling.getLowerbound() / curr_node.getLowerbound()
        else:
            best_sibling_lower_bound_relative_to_init_dual_bound = 0
            best_sibling_lower_bound_relative_to_curr_dual_bound = 0
            best_sibling_lower_bound_relative_to_curr_node_lower_bound = 0
        
        # add feats to each variable
        feats_to_add = np.array([[dual_bound_frac_change,
                                 curr_primal_dual_bound_gap_frac,
                                 num_leaves_frac,
                                 num_feasible_leaves_frac,
                                 num_infeasible_leaves_frac,
                                 num_lp_iterations_frac,
                                 num_siblings_frac,
                                 is_curr_node_best,
                                 is_curr_node_parent_best,
                                 curr_node_depth,
                                 curr_node_lower_bound_relative_to_init_dual_bound,
                                 curr_node_lower_bound_relative_to_curr_dual_bound,
                                 branching_changes_frac,
                                 constraint_prop_changes_frac,
                                 prop_changes_frac,
                                 parent_branching_changes_frac,
                                 is_best_sibling_none,
                                 is_best_sibling_best_node,
                                 best_sibling_lower_bound_relative_to_init_dual_bound,
                                 best_sibling_lower_bound_relative_to_curr_dual_bound,
                                 best_sibling_lower_bound_relative_to_curr_node_lower_bound] for _ in range(obs.column_features.shape[0])])
        
        obs.column_features = np.column_stack((obs.column_features, feats_to_add))
                
        return obs