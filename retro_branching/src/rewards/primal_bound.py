import copy

class PrimalBound:
    def __init__(self, sense=-1):
        '''
        Args:
            sense (-1, 1): -1 to minimise, +1 to maximise.
        '''
        self.sense = sense

    def before_reset(self, model):
        # m = model.as_pyscipopt()
        # self.internal_primal_bound = m.getPrimalbound()
        self.internal_primal_bound = None

    def extract(self, model, done):
        '''Updates the internal primal bound and returns the difference multiplied by the sense.'''
        m = model.as_pyscipopt()
        if self.internal_primal_bound is None:
            self.internal_primal_bound = m.getPrimalbound()
        self.prev_internal_primal_bound = copy.deepcopy(self.internal_primal_bound)
        self.internal_primal_bound = m.getPrimalbound()
        return self.sense * (self.prev_internal_primal_bound - self.internal_primal_bound)