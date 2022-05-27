import copy

class PrimalBoundFrac:
    '''
    Evaluates change in primal bound normalised w.r.t. initial primal bound.
    '''
    def __init__(self, sense=-1):
        '''sense=-1 if want to minimisation objective function, 1 if maximise.'''
        self.sense = sense
    
    def before_reset(self, model):
        self.init_primal_bound = None

    def extract(self, model, done):
        m = model.as_pyscipopt()
        if self.init_primal_bound is None:
            self.init_primal_bound = m.getPrimalbound()
            self.primal_bound = m.getPrimalbound()
            return 0
        self.prev_primal_bound = copy.deepcopy(self.primal_bound)
        self.primal_bound = m.getPrimalbound()
        reward = (self.prev_primal_bound - self.primal_bound) / self.init_primal_bound
        if self.sense == 1:
            reward *= -1
        return reward