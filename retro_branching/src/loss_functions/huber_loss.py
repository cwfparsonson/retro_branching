import torch

class HuberLoss:
    def __init__(self,
                 delta=1,
                 reduction='mean'):
        self.delta = 1
        self.reduction = reduction

    def extract(self, _input, target, reduction='default'):
        if reduction == 'default':
            reduction = self.reduction
        self.loss_function = torch.nn.HuberLoss(delta=self.delta, reduction=reduction)
        return self.loss_function(_input, target)