import torch.nn.functional as F

class MeanSquaredError:
    def __init__(self,
                 reduction='mean'):
        self.reduction = reduction

    def extract(self, _input, target, reduction='default'):
        if reduction == 'default':
            reduction = self.reduction
        return F.mse_loss(_input, target, reduction=reduction)