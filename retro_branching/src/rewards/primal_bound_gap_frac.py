import copy

class PrimalBoundGapFrac:
    '''
    Evaluates change in primal bound normalised w.r.t. initial primal-dual gap.
    '''
    def __init__(self, sense=-1):
        '''sense=-1 if want to minimisation objective function, 1 if maximise.'''
        self.sense = sense
    
    def before_reset(self, model):
        self.init_gap = None

    def extract(self, model, done):
        m = model.as_pyscipopt()
        if self.init_gap is None:
            init_dual_bound = m.getDualbound()
            init_primal_bound = m.getPrimalbound()
            self.init_gap = abs(init_dual_bound - init_primal_bound)
            self.primal_bound = m.getPrimalbound()
            return 0
        if self.init_gap == 0:
            # was pre-solved
            return 0
        else:
            self.prev_primal_bound = copy.deepcopy(self.primal_bound)
            self.primal_bound = m.getPrimalbound()
            reward = (self.prev_primal_bound - self.primal_bound) / self.init_gap
            if self.sense == 1:
                reward *= -1
            return reward