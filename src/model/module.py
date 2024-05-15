"""
Implementation of useful modules to use in models.
"""

from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F


class ModulatedLinear(nn.Module):
    """
    Modulated linear module from Dupont et al., ICML 2022.
    
    Effectively similar to concatenating x and modulation before a single linear layer.
    The interpretation of shift modulations appears when using Sine activations.
    """

    def __init__(self, in_dim, out_dim, modulation_dim, init_fn=None, weight_norm=False):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        if init_fn is not None:
            self.linear.apply(init_fn)
        if weight_norm:
            self.linear = nn.utils.weight_norm(self.linear)

        if modulation_dim > 0:
            self.modulation = nn.Linear(modulation_dim, out_dim, bias=False)
            if weight_norm:
                self.modulation = nn.utils.weight_norm(self.modulation)
        else:
            self.modulation = None

    def forward(self, modulations, x):
        shifts = self.modulation(modulations) if self.modulation is not None else 0.
        return self.linear(x) + shifts


class DemodulatedLinear(nn.Module):
    """
    Demodulated linear module from StyleGAN2, Karras et al., 2020.

    Modulate scaling (or std) by scaling the weights and also renormalize them.
    """

    def __init__(self, in_dim, out_dim, modulation_dim, bias=True, weight_norm=False):
        super().__init__()

        # For linear
        self.weight = nn.Parameter(torch.rand(out_dim, in_dim))
        self._init_param(self.weight, in_dim)
        if bias:
            self.bias = nn.Parameter(torch.rand(out_dim))
            self._init_param(self.bias, in_dim)
        else:
            self.bias = 0.

        # For modulations
        self.modulation = nn.Linear(modulation_dim, in_dim, bias=bias)
        if weight_norm:
            self.modulation = nn.utils.weight_norm(self.modulation)
    
    @torch.no_grad()
    def _init_param(self, param, in_features):
        """Initialize the parameter, assuming it was sampled in [0,1)."""
        k = sqrt(1. / in_features)
        param *= 2 * k
        param -= k
    
    def forward(self, modulations, x):
        # Modulate weights
        scales = self.modulation(modulations).unsqueeze(-2)  # [B]x1xI
        weight_1 = self.weight * scales  # [B]xOxI

        # Demodulate/normalize weights (rsqrt() := 1/sqrt())
        weight_2 = weight_1 * torch.rsqrt(weight_1.square().sum(-2, keepdims=True) + 1e-8)  # [B]xOxI

        # Linear layer
        return (x.unsqueeze(-2) @ weight_2.transpose(-2, -1) + self.bias).squeeze(-2)  # [B]xO
