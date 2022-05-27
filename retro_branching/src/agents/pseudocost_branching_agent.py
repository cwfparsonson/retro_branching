import ecole


class PseudocostBranchingAgent:
    def __init__(self, name='pc'):
        self.name = name
        self.pc_branching_function = ecole.observation.Pseudocosts()

    def before_reset(self, model):
        self.pc_branching_function.before_reset(model)

    def extract(self, model, done):
        return self.pc_branching_function.extract(model, done)

    def action_select(self, action_set, model, done, **kwargs):
        scores = self.extract(model, done)
        action_idx = scores[action_set].argmax()
        return action_set[action_idx], action_idx
