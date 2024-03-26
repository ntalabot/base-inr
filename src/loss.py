"""
Module used for defining losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_recon(loss='l1', reduction='mean'):
    """Return the reconstruction loss to apply on the generator's output."""
    if loss.lower() in ['l1', 'mae']:
        return nn.L1Loss(reduction=reduction)
    elif loss.lower() in ['l2', 'mse']:
        return nn.MSELoss(reduction=reduction)
    elif loss.lower() == 'l1-hard':
        return LossHard(loss='l1', reduction=reduction)
    elif loss.lower() == 'l1-hard-linear':
        return LossHard(loss='l1', reduction=reduction, linear_weight=True)
    elif loss.lower() == 'l2-hard':
        return LossHard(loss='l2', reduction=reduction)
    elif loss.lower() == 'l2-hard-linear':
        return LossHard(loss='l2', reduction=reduction, linear_weight=True)
    elif loss.lower() == 'curriculum':
        return CurriculumLoss(reduction=reduction)
    else:
        raise NotImplementedError(f"Unkown reconstruction loss \"{loss}\".")


class LossHard(nn.Module):
    """L1 loss with hard samples re-weighting, from Duan et al., ECCV 2020."""

    def __init__(self, loss='l1', lambda_=0.5, reduction='mean', linear_weight=False) -> None:
        super().__init__()
        self.reduction = reduction
        self.lambda_ = lambda_
        self.linear_weight = linear_weight
        self.loss_fn = F.l1_loss if loss.lower() in ['l1', 'mae'] else F.mse_loss

    def forward(self, input, target):
        loss = self.loss_fn(input, target, reduction="none")
        if self.linear_weight:
            weights = 1 + self.lambda_ * F.relu(torch.sign(target) * (target - input)).detach() * 100
        else:
            weights = 1 + self.lambda_ * torch.sign(target) * torch.sign(target - input)
        loss = loss * weights

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss


class CurriculumLoss(nn.Module):
    """Full curriculum loss, from Duan et al., ECCV 2020."""

    def __init__(self, epsilon=[0.025, 0.01, 0.0025, 0.], lambda_=[0., 0.1, 0.2, 0.5], 
                 checkpoints=[0.1, 0.3, 0.5], reduction='mean') -> None:
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.checkpoints = checkpoints

    def forward(self, input, target, epoch_frac=1.):
        loss = F.l1_loss(input, target, reduction="none")

        epsilon = None
        for i in range(len(self.checkpoints)):
            if epoch_frac < self.checkpoints[i]:
                epsilon = self.epsilon[i]
                lambda_ = self.lambda_[i]
                break
        if epsilon is None:
            epsilon = self.epsilon[-1]
            lambda_ = self.lambda_[-1]
        
        weights = 1 + lambda_ * torch.sign(target) * torch.sign(target - input)
        loss = weights * (loss - epsilon).clamp(min=0)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
