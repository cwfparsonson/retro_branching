import ecole
import numpy as np
from collections import defaultdict

class NodeBipariteWithSolutionLabels(ecole.observation.NodeBipartite):
    '''
    Before resetting, pre-solves the instance and labels the variables in obs
    with their final solution value thereafter.
    '''
    def __init__(self, presolve_env, seed=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.presolve_env = presolve_env
        self.seed = seed

    def before_reset(self, model):
        super().before_reset(model)

        # pre-solve with presolve env
        self.presolve_env.seed(self.seed)
        _ = self.presolve_env.reset(model.copy_orig())
        _ = self.presolve_env.step({})

        # get solution
        self.presolve_m = self.presolve_env.model.as_pyscipopt()
        self.presolve_solution = self.presolve_m.getBestSol()
        self.presolve_sol_vals = np.array([self.presolve_solution[var] for var in self.presolve_m.getVars()]).T

    def extract(self, model, done):
        # get the NodeBipartite obs
        obs = super().extract(model, done)

        # label vars in obs with final solution values
        obs.variable_features = np.column_stack((obs.variable_features, self.presolve_sol_vals))

        return obs