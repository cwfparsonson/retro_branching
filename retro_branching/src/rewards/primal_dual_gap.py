class PrimalDualGap:
    def __init__(self):
        pass

    def before_reset(self, model):
        # self.internal_gap = None
        pass

    def extract(self, model, done):
        '''Returns the negative of the current primal-dual gap.'''
        m = model.as_pyscipopt()
        return -m.getGap()