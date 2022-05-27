import numpy as np

class RandomAgent:
    def __init__(self, name='random'):
        self.name = name

    def before_reset(self, model):
        pass

    def action_select(self, action_set, **kwargs):
        action_idx = np.random.choice([i for i in range(len(action_set))])
        return action_set[action_idx], action_idx
