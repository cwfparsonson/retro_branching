import torch.nn.functional as F

class CrossEntropy:
    def __init__(self):
        pass

    def extract(self, _input, target):
        return F.cross_entropy(_input, target)