import copy

class DualBoundFrac:
    '''
    Evaluates change in dual bound normalised w.r.t. initial dual bound.
    '''
    def __init__(self, sense=-1):
        '''If minimising, must set sense=-1 to incentivise largest delta in dual bound.'''
        self.sense = sense

    def before_reset(self, model):
        self.init_dual_bound = None

    def extract(self, model, done):
        '''Updates the internal dual bound and returns the fractional difference.'''
        m = model.as_pyscipopt()
        if self.init_dual_bound is None:
            self.init_dual_bound = m.getDualbound()
            self.dual_bound = m.getDualbound()
            return 0
        self.prev_dual_bound = copy.deepcopy(self.dual_bound)
        self.dual_bound = m.getDualbound()
        reward = self.sense*(self.prev_dual_bound-self.dual_bound)/self.init_dual_bound
        return reward 