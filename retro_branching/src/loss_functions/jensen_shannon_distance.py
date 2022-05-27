import torch
import torch.nn.functional as F

import numpy as np
import math
import scipy


class JensenShannonDistance:
    def __init__(self,
                 reduction='mean',
                 eps=1e-4):
        self.reduction = reduction
        self.eps = eps

    def compute_kullback_leibler_divergence(self, p, q):
        return (p * ((p+self.eps) / (q+self.eps)).log()).sum()

    def _compute_numpy_jsd(self, p, q, eps=1e-7):
        '''For debugging.'''
        p, q = p.detach().cpu().numpy(), q.detach().cpu().numpy()
        m = (p + q) / 2
        _p, _q = [], []
        for idx in range(len(p)):
            _p.append(np.sum(np.where(p[idx] != 0, p[idx] * np.log((p[idx]+eps) / (m[idx]+eps)), 0)))
            _q.append(np.sum(np.where(q[idx] != 0, q[idx] * np.log((q[idx]+eps) / (m[idx]+eps)), 0)))
        div = (np.array(_p) + np.array(_q)) / 2
        jsd = np.mean(np.sqrt(div))
        print('numpy jsd: {}'.format(jsd))
        return jsd


    def _compute_scipy_jsd(self, p, q):
        '''For debugging.'''
        p, q = p.detach().cpu().numpy(), q.detach().cpu().numpy()
        jsd = np.mean(scipy.spatial.distance.jensenshannon(p, q, axis=1))
        print('scipy jsd: {}'.format(jsd))
        return jsd

    def compute_jensen_shannon_distance(self, p, q):
        # calc m
        m = (p + q) / 2

        # compute jensen-shannon divergence
        _p = self.compute_kullback_leibler_divergence(p, m)
        _q = self.compute_kullback_leibler_divergence(q, m)
        divergence = (_p + _q) / 2

        # compute jensen-shannon distance
        distance = torch.sqrt(divergence + self.eps)

        if math.isnan(distance) or not torch.isfinite(_p) or not torch.isfinite(_q) or not torch.isfinite(distance):
            # print('Invalid value found, assuming is because distributions are the same and so should return JSD=0. Check below output to check this is the case. If no, there is a bug somewhere!')
            print('Invalid value found.')
            print('idx: {}'.format(self.idx))
            print('logits: shape {} {}'.format(self.logits[self.idx].shape, self.logits[self.idx]))
            print('imitation_target: shape {} {}'.format(self.imitation_target[self.idx].shape, self.imitation_target[self.idx]))
            print('p: shape {} sum {} {}'.format(p.shape, torch.sum(p), p))
            print('q: shape {} sum {} {}'.format(q.shape, torch.sum(q), q))
            print('m: shape {} {}'.format(m.shape, m))
            print('_p: {}'.format(_p))
            print('_q: {}'.format(_q))
            print('divergence: {}'.format(divergence))
            print('distance: {}'.format(distance))
            # return 0.0
            raise Exception()

        return distance

    def extract(self, logits, imitation_target):
        '''Returns the Jensen-Shannon distance between (softmax'd) logits and imitation_target.'''
        self.logits, self.imitation_target = logits, imitation_target
        loss = 0.0
        for self.idx in range(self.logits.shape[0]):
            p, q = F.softmax(self.logits[self.idx]), F.softmax(self.imitation_target[self.idx])
            distance = self.compute_jensen_shannon_distance(p, q)
            loss += distance

        if self.reduction == 'mean':
            loss /= logits.shape[0]
        elif self.reduction == 'sum':
            pass
        else:
            raise Exception('Unrecognised JSD batch reduction {}'.format(self.reduction))

        return loss