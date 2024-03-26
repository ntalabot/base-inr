"""
Implementation of DeepSDF (Park et al., CVPR 2019)
and similar models.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import get_activation
from .features import get_input_features
from .module import ModulatedLinear, DemodulatedLinear


class DeepSDF(nn.Module):
    """DeepSDF network from Park et al., CVPR 2019."""
    
    def __init__(self, latent_dim=256, hidden_dim=512, n_layers=8, in_insert=[4],
                 dropout=0.2, weight_norm=True, last_tanh=False,
                 layer_norm=False, activation="relu", features=None,
                 input_dim=3, output_dim=1, output_scale=None,
                 **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_insert = in_insert
        self.output_scale = self.register_buffer("output_scale", 
                                                 torch.tensor(output_scale) if output_scale is not None else None)

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
                in_d = latent_dim + feats_dim
            if n == n_layers - 1:
                out_d = output_dim
            elif n + 1 in self.in_insert:
                out_d = hidden_dim - (latent_dim + feats_dim)
            
            fc = nn.Linear(in_d, out_d)
            if weight_norm:
                fc = nn.utils.weight_norm(fc)
            layer.append(fc)
            
            # Normalization
            if not weight_norm and layer_norm and n < n_layers - 1:
                layer.append(nn.LayerNorm(out_d))

            # Activation
            if n < n_layers - 1:
                layer.append(get_activation(activation))
            elif last_tanh:
                layer.append(nn.Tanh())
            
            # Dropout
            if dropout > 0. and n < n_layers - 1:
                layer.append(nn.Dropout(dropout))

            # Combine them
            self.fcs.append(nn.Sequential(*layer))

    def forward(self, x):
        # Separate latent from positions
        lat = x[..., :self.latent_dim]
        xyz = x[..., self.latent_dim:]

        feats = self.features(xyz) if self.features is not None else xyz
        x = torch.cat([lat, feats], dim=-1)

        out = x
        for i, fc in enumerate(self.fcs):
            out = fc(out)

            if (i + 1) in self.in_insert:
                out = torch.cat([out, x], dim=-1)
        
        if self.output_scale is not None:
            out = out * self.output_scale
        return out


class LatentModulatedDeepSDF(nn.Module):
    """Latent Modulated DeepSDF."""
    
    def __init__(self, latent_dim=256, hidden_dim=512, n_layers=8,
                 dropout=0., weight_norm=True, last_tanh=False,
                 activation="relu", features=None,
                 input_dim=3, output_dim=1, output_scale=None,
                 **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.last_tanh = last_tanh
        self.output_scale = self.register_buffer("output_scale", 
                                                 torch.tensor(output_scale) if output_scale is not None else None)

        if features is None or features == "none":
            self.features = None
            feats_dim = input_dim
        else:
            self.features = get_input_features(features)
            feats_dim = self.features.outdim

        self.layers = nn.ModuleList()
        self.layers.append(
            ModulatedLinear(feats_dim, hidden_dim, self.latent_dim, None,
                            weight_norm=weight_norm)
        )
        for _ in range(1, n_layers-1):
            self.layers.append(
                ModulatedLinear(hidden_dim, hidden_dim, self.latent_dim, None,
                                weight_norm=weight_norm)
            )
        self.layers.append(
            ModulatedLinear(hidden_dim, output_dim, self.latent_dim, None,
                            weight_norm=weight_norm)
        )
        self.activ = get_activation(activation)
            

    def forward(self, x):
        # Separate latent from positions
        lat = x[..., :self.latent_dim]
        xyz = x[..., self.latent_dim:]

        if self.features is not None:
            out = self.features(xyz)
        else:
            out = xyz

        for layer in self.layers[:-1]:
            out = self.activ(layer(out, lat))
            if self.dropout > 0. and self.training:
                out = F.dropout(out, self.dropout)
        
        out = self.layers[-1](out, lat)
        if self.last_tanh:
            out = F.tanh(out)
        if self.output_scale is not None:
            out = out * self.output_scale
        return out


class InputModulatedDeepSDF(nn.Module):
    """Input Modulated DeepSDF. Basically concat the input at each layer."""
    
    def __init__(self, latent_dim=256, hidden_dim=512, n_layers=8,
                 dropout=0., weight_norm=True, last_tanh=False,
                 activation="relu", features=None,
                 input_dim=3, output_dim=1, output_scale=None,
                 **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.last_tanh = last_tanh
        self.output_scale = self.register_buffer("output_scale", 
                                                 torch.tensor(output_scale) if output_scale is not None else None)

        if features is None or features == "none":
            self.features = None
            feats_dim = input_dim
        else:
            self.features = get_input_features(features)
            feats_dim = self.features.outdim

        input_dim = self.latent_dim + feats_dim
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Linear(input_dim, hidden_dim)
        )
        if weight_norm:
            self.layers[0] = nn.utils.weight_norm(self.layers[0])
        for _ in range(1, n_layers-1):
            self.layers.append(
                ModulatedLinear(hidden_dim, hidden_dim, input_dim, None,
                                weight_norm=weight_norm)
            )
        self.layers.append(
            ModulatedLinear(hidden_dim, output_dim, input_dim, None,
                            weight_norm=weight_norm)
        )
        self.activ = get_activation(activation)
            

    def forward(self, x):
        # Separate latent from positions
        lat = x[..., :self.latent_dim]
        xyz = x[..., self.latent_dim:]

        if self.features is not None:
            out = self.features(xyz)
        else:
            out = xyz

        inp = torch.cat([lat, out], dim=-1)
        out = self.activ(self.layers[0](inp))
        for layer in self.layers[1:-1]:
            out = self.activ(layer(out, inp))
            if self.dropout > 0. and self.training:
                out = F.dropout(out, self.dropout)
        
        out = self.layers[-1](out, inp)
        if self.last_tanh:
            out = F.tanh(out)
        if self.output_scale is not None:
            out = out * self.output_scale
        return out


class DemodulatedDeepSDF(nn.Module):
    """DeepSDF with demodulated-linear layers."""
    
    def __init__(self, latent_dim=256, hidden_dim=512, n_layers=8,
                 dropout=0., weight_norm=True, last_tanh=False,
                 activation="relu", features=None,
                 input_dim=3, output_dim=1, output_scale=None,
                 **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.last_tanh = last_tanh
        self.output_scale = self.register_buffer("output_scale", 
                                                 torch.tensor(output_scale) if output_scale is not None else None)

        if features is None or features == "none":
            self.features = None
            feats_dim = input_dim
        else:
            self.features = get_input_features(features)
            feats_dim = self.features.outdim

        self.layers = nn.ModuleList()
        self.layers.append(
            DemodulatedLinear(feats_dim, hidden_dim, self.latent_dim,
                              weight_norm=weight_norm)
        )
        for _ in range(1, n_layers-1):
            self.layers.append(
                DemodulatedLinear(hidden_dim, hidden_dim, self.latent_dim,
                                  weight_norm=weight_norm)
            )
        self.layers.append(
            DemodulatedLinear(hidden_dim, output_dim, self.latent_dim,
                              weight_norm=weight_norm)
        )
        self.activ = get_activation(activation)
            

    def forward(self, x):
        # Separate latent from positions
        lat = x[..., :self.latent_dim]
        xyz = x[..., self.latent_dim:]

        if self.features is not None:
            out = self.features(xyz)
        else:
            out = xyz

        for layer in self.layers[:-1]:
            out = self.activ(layer(out, lat))
            if self.dropout > 0. and self.training:
                out = F.dropout(out, self.dropout)
        
        out = self.layers[-1](out, lat)
        if self.last_tanh:
            out = F.tanh(out)
        if self.output_scale is not None:
            out = out * self.output_scale
        return out