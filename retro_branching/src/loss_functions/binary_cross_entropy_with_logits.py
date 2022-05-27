import torch

class BinaryCrossEntropyWithLogits:
    def __init__(self, weight=None, reduction='mean', pos_weight=None):
        self.loss_function = torch.nn.BCEWithLogitsLoss(weight=weight, reduction=reduction, pos_weight=pos_weight)
        
    def extract(self, _input, target, reduction='mean'):
        return self.loss_function(_input, target)