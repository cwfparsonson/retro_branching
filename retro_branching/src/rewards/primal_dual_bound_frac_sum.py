import retro_branching

class PrimalDualBoundFracSum:
    def __init__(self):
        self.primal_reward = retro_branching.src.rewards.primal_bound_frac.PrimalBoundFrac()
        self.dual_reward = retro_branching.src.rewards.dual_bound_frac.DualBoundFrac()

    def before_reset(self, model):
        self.primal_reward.before_reset(model)
        self.dual_reward.before_reset(model)

    def extract(self, model, done):
        primal = self.primal_reward.extract(model, done)
        dual = self.dual_reward.extract(model, done)
        return primal + dual