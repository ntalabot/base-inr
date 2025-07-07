"""
Implementation of SIREN (Sitzmann et al., NeurIPS 2020)
and similar models.
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import Sine
from .features import get_input_features
from .module import ModulatedLinear


def sine_init(m, factor=30.):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-sqrt(6 / num_input) / factor, sqrt(6 / num_input) / factor)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class Siren(nn.Module):
    """SIREN from Sitzmann et al., NeurIPS 2020."""
    
    def __init__(self, latent_dim=256, hidden_dim=512, n_layers=8, in_insert=[],
                 factor=30., dropout=0., weight_norm=True, last_tanh=False, 
                 layer_norm=False, features=None,
                 input_dim=3, output_dim=1, output_scale=None,
                 **kwargs):
        """
        Args:
            latent_dim (int): dimension of the latent code.
            hidden_dim (int): dimension of the hidden layers.
            n_layers (int): number of layers.
            in_insert (list of int): list of layers where to reinsert the input.
            factor (float): factor for the initialization of the weights.
            dropout (float): dropout rate.
            weight_norm (bool): whether to use weight normalization.
            last_tanh (bool): whether to apply a tanh to the output.
            layer_norm (bool): whether to use layer normalization.
            features (str): type of input features to use.
            input_dim (int): dimension of the input features.
            output_dim (int): dimension of the output.
            output_scale (float): scale to apply to the output.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.in_insert = in_insert
        self.register_buffer("output_scale", torch.tensor(output_scale) if output_scale is not None else None)

        if features is None or features == "none":
            self.features = None
            feats_dim = input_dim
        else:
            self.features = get_input_features(features)
            feats_dim = self.features.outdim

        self.fcs = nn.ModuleList()
        for n in range(n_layers):
            layer = []

            # Fully-connected
            in_d = out_d = hidden_dim
            if n == 0:
                in_d = self.latent_dim + feats_dim
            if n == n_layers - 1:
                out_d = output_dim
            elif n + 1 in self.in_insert:
                out_d = hidden_dim - (self.latent_dim + feats_dim)
            
            fc = nn.Linear(in_d, out_d)
            if n == 0:
                fc.apply(first_layer_sine_init)
            elif n < n_layers - 1:
                fc.apply(lambda m: sine_init(m, factor))
            if weight_norm:
                fc = nn.utils.weight_norm(fc)
            layer.append(fc)
            
            # Normalization
            if not weight_norm and layer_norm and n < n_layers - 1:
                layer.append(nn.LayerNorm(out_d))

            # Activation
            if n < n_layers - 1:
                layer.append(Sine(factor))
            elif last_tanh:
                layer.append(nn.Tanh())
            
            # Dropout
            if dropout > 0. and n < n_layers - 1:
                layer.append(nn.Dropout(dropout))

            # Combine them
            self.fcs.append(nn.Sequential(*layer))


    def forward(self, lat, xyz):
        """
        Args:
            lat (torch.Tensor): latent code. Should have singleton dimensions
                where the latents need to be repeated. E.g., [B, 1, 256] if
                xyz.shape = [B, N, 3] with B:=batch and N:=points per shape.
            xyz (torch.Tensor): input positions.
        """
        # Compute input features, then concatenate them with the latent code
        feats = self.features(xyz) if self.features is not None else xyz
        out = torch.cat([lat.expand(feats.shape[:-1] + (-1,)), feats], dim=-1)
        
        for i, fc in enumerate(self.fcs):
            out = fc(out)

            if (i + 1) in self.in_insert:
                out = torch.cat([out, feats], dim=-1)
        
        if self.output_scale is not None:
            out = out * self.output_scale
        return out


class LatentModulatedSiren(nn.Module):
    """Latent Modulated SIREN from Dupont et al., ICML 2022."""
    
    def __init__(self, latent_dim=256, hidden_dim=512, n_layers=8, factor=30.,
                 dropout=0., weight_norm=True, last_tanh=False, features=None,
                 input_dim=3, output_dim=1, output_scale=None,
                 **kwargs):
        """
        Args:
            latent_dim (int): dimension of the latent code.
            hidden_dim (int): dimension of the hidden layers.
            n_layers (int): number of layers.
            factor (float): factor for the initialization of the weights.
            dropout (float): dropout rate.
            weight_norm (bool): whether to use weight normalization.
            last_tanh (bool): whether to apply a tanh to the output.
            features (str): type of input features to use.
            input_dim (int): dimension of the input features.
            output_dim (int): dimension of the output.
            output_scale (float): scale to apply to the output.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.last_tanh = last_tanh
        self.register_buffer("output_scale", torch.tensor(output_scale) if output_scale is not None else None)

        if features is None or features == "none":
            self.features = None
            feats_dim = input_dim
        else:
            self.features = get_input_features(features)
            feats_dim = self.features.outdim

        self.layers = nn.ModuleList()
        self.layers.append(
            ModulatedLinear(feats_dim, hidden_dim, self.latent_dim, first_layer_sine_init,
                            weight_norm=weight_norm)
        )
        for _ in range(1, n_layers-1):
            self.layers.append(
                ModulatedLinear(hidden_dim, hidden_dim, self.latent_dim,
                                lambda m: sine_init(m, factor), weight_norm=weight_norm)
            )
        self.layers.append(
            ModulatedLinear(hidden_dim, output_dim, self.latent_dim, None,
                            weight_norm=weight_norm)
        )
        self.sine = Sine(factor)
        

    def forward(self, lat, xyz):
        """
        Args:
            lat (torch.Tensor): latent code. Should have singleton dimensions
                where the latents need to be repeated. E.g., [B, 1, 256] if
                xyz.shape = [B, N, 3] with B:=batch and N:=points per shape.
            xyz (torch.Tensor): input positions.
        """
        # Compute input features
        out = self.features(xyz) if self.features is not None else xyz

        for layer in self.layers[:-1]:
            out = self.sine(layer(lat, out))
            if self.dropout > 0. and self.training:
                out = F.dropout(out, self.dropout)
        
        out = self.layers[-1](lat, out)
        if self.last_tanh:
            out = F.tanh(out)
        if self.output_scale is not None:
            out = out * self.output_scale
        return out