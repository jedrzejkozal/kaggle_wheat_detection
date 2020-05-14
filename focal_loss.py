import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        self.gamma = gamma

    def forward(self, input, target):
        log_pt = F.cross_entropy(input, target, reduction='none')
        p_t = torch.exp(-log_pt)
        loss = (1 - p_t)**self.gamma * log_pt
        return loss.mean()
