import ecole
import numpy as np
from collections import defaultdict

class NodeBipariteWithCustomFeatures(ecole.observation.NodeBipartite):
    '''
    Adds (mostly global) features to variable node features.

    Adds 24 extra variable features to each variable on top of standard ecole
    NodeBipartite obs variable features (19), so each variable will have
    43 features in total.

    '''

    variable_obs_dim = 23
    constraint_obs_dim = 5

    def before_reset(self, model):
        super().before_reset(model)

        self.init_dual_bound = None
        self.init_primal_bound = None

    def extract(self, model, done):
        # get the NodeBipartite obs, model, and current node.
        obs = super().extract(model, done)
        m = model.as_pyscipopt()
        curr_node = m.getCurrentNode()

        if self.init_dual_bound is None:
            # Set init stats if first observation.
            self.init_dual_bound = m.getDualbound()
            self.init_primal_bound = m.getPrimalbound()
            self.init_primal_dual_gap = self.init_primal_bound - self.init_dual_bound

            obj = m.getObjective()
            var_coeffs = np.array([obj[var] for var in m.getVars()])
            # normaliser of V:coef
            self.obj_coeffs_norm = np.sum(var_coeffs**2)**0.5
            self.obj_coeffs_renorm = self.obj_coeffs_norm / np.sum(var_coeffs)

        # print("row_features", obs.row_features.shape) # [500,5]
        # print("col_features", obs.variable_features.shape) # [1000,19]
        # print("edge_features", obs.edge_features.shape) # [500,1000]

        # remove scaled age
        # obs.variable_features = np.delete(obs.variable_features, 12, -1)

        # Global primal/dual.
        glob_primal_bound, glob_dual_bound = m.getPrimalbound(), m.getDualbound()
        # Focus node dual bound.
        dual_bound = curr_node.getLowerbound()

        # Current primal-dual gap as fraction of initial gap.
        primal_dual_gap = abs( (glob_primal_bound - dual_bound) / self.init_primal_dual_gap )
        dual_change_gap = abs( (self.init_primal_bound - dual_bound) / self.init_primal_dual_gap)
        primal_change_gap = abs( (glob_primal_bound - self.init_dual_bound) / self.init_primal_dual_gap)

        # Depth of current node.
        if m.getDepth() != 0:
            curr_node_depth = 1 / m.getDepth()
        else:
            curr_node_depth = 1

        # # Statistics over tree nodes.
        # num_leaves_frac = m.getNLeaves() / m.getNNodes()
        # num_feasible_leaves_frac = m.getNFeasibleLeaves() / m.getNNodes()
        # num_infeasible_leaves_frac = m.getNInfeasibleLeaves() / m.getNNodes()

        features_to_add = np.array([
            primal_dual_gap,
            dual_change_gap,
            primal_change_gap,
            curr_node_depth,
        ])
        features_to_add = features_to_add[None].repeat(obs.variable_features.shape[0], 0)

        # obs.variable_features[..., 0] = obs.variable_features[..., 0] * self.obj_coeffs_renorm
        # obs.variable_features[..., 7] = obs.variable_features[..., 7] * self.obj_coeffs_renorm

        obs.variable_features = np.column_stack([
            obs.variable_features,
            features_to_add,
        ])

        return obs