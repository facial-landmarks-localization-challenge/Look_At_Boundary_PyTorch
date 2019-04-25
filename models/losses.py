import torch
import torch.nn as nn
import numpy as np


class HeatmapLoss(nn.Module):
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        assert pred.size() == gt.size()
        loss = ((pred - gt)**2)
        loss = loss.sum(dim=3).sum(dim=2).sum(dim=1).mean() / 2.
        return loss


class WingLoss(nn.Module):

    def __init__(self, w=10, epsilon=2, weight=None):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = self.w - self.w * np.log(1 + self.w / self.epsilon)
        self.weight = weight

    def forward(self, predictions, targets):
        x = predictions - targets
        if self.weight is not None:
            x = x * self.weight
        t = torch.abs(x)

        return torch.mean(torch.where(t < self.w, self.w * torch.log(1 + t / self.epsilon), t - self.C))
