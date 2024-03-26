"""
Module implementing activation functions for 
PyTorch models.
"""

import torch
from torch import nn
import torch.nn.functional as F


def get_activation(activation):
    """Return the Module corresponding to the activation."""
    if isinstance(activation, nn.Module):
        return activation
    
    # Separate activation from its arguments (if any)
    activation = activation.split('-')
    activation, args = activation[0], activation[1:]
    args = [float(arg) for arg in args]  # convert arguments to floats

    if activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() in ["leaky", "lrelu", "leakyrelu"]:
        if len(args) == 0:
            args.append(0.1)
        return nn.LeakyReLU(*args)
    elif activation.lower() == "celu":
        return nn.CELU(*args)
    elif activation.lower() == "gelu":
        return nn.GELU(approximate='tanh', *args)
    elif activation.lower() == "softplus":
        if len(args) == 0:
            args.append(100.)
        return nn.Softplus(*args)
    elif activation.lower() == "gaussian":
        return Gaussian(*args)
    elif activation.lower() in ["sin", "sine", "sinus", "siren"]:
        return Sine(*args)
    elif activation.lower() == "sawtooth":
        return Sawtooth(*args)
    elif activation.lower() == "geglu":
        return GEGLU(*args)
    else:
        raise NotImplementedError(f"Unknown activation \"{activation}\".")


class Gaussian(nn.Module):
    """Gaussian activation function, as proposed in Ramasinghe and Lucey, arXiv 2022."""

    def __init__(self, a=1.) -> None:
        super().__init__()
        self.a = a
        self.half_inv_a2 = 0.5 / a**2
    

    def forward(self, x):
        return torch.exp(- x.square() * self.half_inv_a2)
    
    def __repr__(self):
        return f"Gaussian(a={self.a})"


class Sine(nn.Module):
    """Sinus activation for Siren networks, Sitzmann et al., NeurIPS 2020."""

    def __init__(self, factor=1.):
        super().__init__()
        self.factor = factor
    
    def forward(self, x):
        return torch.sin(self.factor * x)
    
    def __repr__(self):
        return f"Sine(factor={self.factor})"


class Sawtooth(nn.Module):
    """Sawtooth (:=piece-wise linear) activation."""

    def __init__(self, factor=1.):
        super().__init__()
        self.factor = factor
    
    def forward(self, x):
        x = self.factor * x
        offset = torch.floor((x + 1) / 2).detach()
        sign = 1. - 2. * offset.remainder(2)
        return sign * (x - 2. * offset)
    
    def __repr__(self):
        return f"Sawtooth(factor={self.factor})"


class GEGLU(nn.Module):
    """
    GeGLU activation function.

    Taken from 3DShape2VecSet, Zhang et al., SIGGRAPH23.
    https://github.com/1zb/3DShape2VecSet/blob/master/models_ae.py
    """

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)