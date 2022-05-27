import torch.nn.functional as F

class KullbackLeiblerDivergence:
    def __init__(self, reduction='mean', eps=1e-4):
        self.reduction = reduction
        self.eps = eps

    def compute_kullback_leibler_divergence(self, p, q):
        return (p * ((p+self.eps) / (q+self.eps)).log()).sum()

    def extract(self, logits, imitation_target):
        '''Returns the KL divergence between (softmax'd) logits and imitation_target.

        Args:
            reduction (str): Reduction operation with which to reduce KL divergence
                across multiple batches to a single number. Must be one
                of 'mean', 'sum'.
        '''
        self.logits, self.imitation_target = logits, imitation_target

        loss = 0.0
        for self.idx in range(self.logits.shape[0]):
            p, q = F.softmax(self.logits[self.idx]), F.softmax(self.imitation_target[self.idx])
            loss += (self.compute_kullback_leibler_divergence(p, q) + self.eps)

        if self.reduction == 'mean':
            loss /= logits.shape[0]
        elif self.reduction == 'sum':
            pass
        else:
            raise Exception('Unrecognised JSD batch reduction {}'.format(self.reduction))

        return loss