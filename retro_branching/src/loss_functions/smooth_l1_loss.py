import torch

class SmoothL1Loss:
    def __init__(self,
                 beta=1,
                 reduction='mean'):
        self.beta = 1
        self.reduction = reduction
        self.loss_function = torch.nn.SmoothL1Loss(beta=beta, reduction=reduction)

    def extract(self, _input, target, reduction='default'):
        if reduction == 'default':
            reduction = self.reduction
        return self.loss_function(_input, target, reduction=reduction)