import copy

class DualBoundGapFrac:
    '''
    Evaluates change in dual bound normalised w.r.t. initial primal-dual gap.
    '''
    def __init__(self, sense=-1):
        '''If minimising, must set sense=-1 to incentivise largest delta in dual bound.'''
        self.sense = sense

    def before_reset(self, model):
        self.init_gap = None

    def extract(self, model, done):
        '''Updates the internal dual bound and returns the fractional difference.'''
        m = model.as_pyscipopt()
        if self.init_gap is None:
            init_dual_bound = m.getDualbound()
            init_primal_bound = m.getPrimalbound()
            self.init_gap = abs(init_dual_bound - init_primal_bound)
            self.dual_bound = m.getDualbound()
            return 0
        if self.init_gap == 0:
            # was pre-solved
            return 0
        else:
            self.prev_dual_bound = copy.deepcopy(self.dual_bound)
            self.dual_bound = m.getDualbound()
            reward = self.sense*(self.prev_dual_bound-self.dual_bound)/self.init_gap
            return reward 