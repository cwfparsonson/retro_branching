import ecole
import numpy as np
from collections import defaultdict

class NodeBipariteWith24VariableFeatures(ecole.observation.NodeBipartite):
    '''
    Adds (mostly global) features to variable node features.

    Adds 5 extra variable features to each variable on top of standard ecole
    NodeBipartite obs variable features (19), so each variable will have
    24 features in total.

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
        # dual_bound_frac_change = abs(1-(min(self.init_dual_bound, m.getDualbound()) / max(self.init_dual_bound, m.getDualbound())))
        # primal_bound_frac_change = abs(1-(min(self.init_primal_bound, m.getPrimalbound()) / max(self.init_primal_bound, m.getPrimalbound())))
        dual_bound_frac_change = abs(self.init_dual_bound - m.getDualbound()) / self.init_dual_bound
        primal_bound_frac_change = abs(self.init_primal_bound - m.getPrimalbound()) / self.init_primal_bound

        primal_dual_gap = abs(m.getPrimalbound() - m.getDualbound())
        max_dual_bound_frac_change = primal_dual_gap / self.init_dual_bound
        max_primal_bound_frac_change = primal_dual_gap / self.init_primal_bound

        curr_primal_dual_bound_gap_frac = m.getGap()
        
        # add feats to each variable
        feats_to_add = np.array([[dual_bound_frac_change,
                                 primal_bound_frac_change,
                                 max_primal_bound_frac_change,
                                 max_dual_bound_frac_change,
                                 curr_primal_dual_bound_gap_frac,
                                 ] for _ in range(obs.variable_features.shape[0])])
        
        obs.variable_features = np.column_stack((obs.variable_features, feats_to_add))
                
        return obs