import torch
import torch.nn.functional as F

class BinaryCrossEntropy:
    def __init__(self, weight=None, reduction='mean'):
        self.loss_function = torch.nn.BCELoss(weight=weight, reduction=reduction)
        
    def extract(self, _input, target, reduction='mean'):
        return self.loss_function(F.sigmoid(_input), target)