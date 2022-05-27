import copy

class PrimalDualGapFrac:
    '''
    Evaluates change in primal-dual gap normalised w.r.t. initial primal-dual gap.
    '''
    def __init__(self):
        pass

    def before_reset(self, model):
        self.init_gap = None

    def extract(self, model, done):
        '''Updates the internal primal-dual gap and returns the the fractional difference.'''
        m = model.as_pyscipopt()
        if self.init_gap is None:
            self.init_gap = abs(m.getDualbound() - m.getPrimalbound())
            self.gap = copy.deepcopy(self.init_gap)
            return 0
        if self.init_gap == 0:
            # was pre-solved
            return 0
        else:
            self.prev_gap = copy.deepcopy(self.gap)
            self.gap = abs(m.getDualbound() - m.getPrimalbound())
            reward = (self.prev_gap - self.gap) / self.init_gap
            return reward