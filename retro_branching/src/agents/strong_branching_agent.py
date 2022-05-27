import ecole

class StrongBranchingAgent:
    def __init__(self, pseudo_candidates=False, name='sb'):
        self.name = name
        self.pseudo_candidates = pseudo_candidates
        self.strong_branching_function = ecole.observation.StrongBranchingScores(pseudo_candidates=pseudo_candidates)

    def before_reset(self, model):
        """
        This function will be called at initialization of the environments (before dynamics are reset).
        """
        self.strong_branching_function.before_reset(model)
    
    def extract(self, model, done):
        return self.strong_branching_function.extract(model, done)

    def action_select(self, action_set, model, done, **kwargs):
        scores = self.extract(model, done)
        action_idx = scores[action_set].argmax()
        return action_set[action_idx], action_idx

