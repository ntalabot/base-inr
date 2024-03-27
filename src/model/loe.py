"""
Implementation of Levels-of-Experts (Hao et al., NeurIPS 2022)
and similar models.

Code not yet available, but a snippet was provided on OpenReview:
see `positional_dependent_linear_1d()` for the 1D case.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import get_activation
from .features import get_input_features


# Following was the provided snippet from the authors on OpenReview.
def positional_dependent_linear_1d(weight, bias, in_feats, in_coords, alpha, beta):
    r"""Linear layer with position-dependent weight.
    Assuming the input coordinate is 1D.
    Args:
        weight (N * Cout * Cin tensor): Tile of N weight matrices
        bias (Cout tensor): Bias vector
        in_feats (B * Cin tensor): Batched input features
        in_coords (B * 1 tensor): Batched input coordinates
        alpha (scalar): Scale of input coordinates
        beta (scalar): Translation of input coordinates
    Returns:
        out_feats (B * Cout tensor): Batched output features
    """
    B = in_feats.size(0) # Batch size
    N = weight.size(0) # Tile size
    Cout = weight.size(1) # Out channel count
    Cin = weight.size(2) # In channel count
    # In the actual implementation, the following lines are fused into a CUDA kernel.
    tile_id = torch.floor(alpha * in_coords + beta).long() % N
    out_feats = torch.empty([B, Cout])
    for t in range(N):
        mask = tile_id == t
        sel_in_feats = torch.masked_select(in_feats, mask).reshape(-1, Cin)
        sel_weight = weight[t]
        sel_out_feats = sel_in_feats @ sel_weight.T
        out_feats.masked_scatter_(mask, sel_out_feats)
        
    return out_feats + bias


class LevelsOfExperts(nn.Module):
    """
    Levels-of-Experts network from Hao et al., NeurIPS 2022.
    
    Network structure is taken from DeepSDF, Park et al. (CVPR 2019), with some
    simplification (no dropout, normalization).
    Note: unlike the paper, the bias is not shared.
    """
    
    def __init__(self, latent_dim=256, hidden_dim=512, n_layers=8, in_insert=[4],
                 n_experts_per_dim=2,  # number of experts per layer, per dimension (so is taken to the power of 3)
                 tiling_type="coarsetofine", interp_type="nearest",
                 activation="relu", features=None,
                 input_dim=3, output_dim=1, output_scale=None,
                 **kwargs):
        """
        Args:
            latent_dim (int): dimension of the latent code.
            hidden_dim (int): dimension of the hidden layers.
            n_layers (int): number of layers.
            in_insert (list of int): list of layers where to reinsert the input.
            n_experts_per_dim (int): number of experts per layer, per dimension.
            tiling_type (str): type of tiling, either "coarsetofine" or "finetocoarse".
            interp_type (str): type of interpolation, currently only "nearest" is supported.
            activation (str): type of activation function.
            features (str): type of input features.
            input_dim (int): dimension of the input coordinates.
            output_dim (int): dimension of the output.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.in_insert = in_insert
        self.n_layers = n_layers
        self.N_per_d = n_experts_per_dim
        self.N = self.N_per_d ** input_dim
        self.interp_type = interp_type
        self.output_scale = self.register_buffer("output_scale", 
                                                 torch.tensor(output_scale) if output_scale is not None else None)

        if features is None or features == "none":
            self.features = None
            feats_dim = input_dim
        else:
            self.features = get_input_features(features)
            feats_dim = self.features.outdim

        # Activation
        self.activ = get_activation(activation)

        self.linears = nn.ModuleList()
        self.biases = nn.ParameterList()
        self.alphas, self.betas = [], []
        for n in range(n_layers):
            # Fully-connected
            in_d = out_d = hidden_dim
            if n == 0:
                in_d = latent_dim + feats_dim
            if n == n_layers - 1:
                out_d = output_dim
            elif n + 1 in self.in_insert:
                out_d = hidden_dim - (latent_dim + feats_dim)
            
            self.linears.append(
                nn.ModuleList([nn.Linear(in_d, out_d, bias=False) for _ in range(self.N)])
            )
            self.biases.append(nn.parameter.Parameter(torch.zeros(1, out_d)))
        
            # Tiling parameters
            if tiling_type == "coarsetofine":
                self.alphas.append(2. ** (n+1))
                self.betas.append(0.)
            elif tiling_type == "finetocoarse":
                self.alphas.append(2. ** (n_layers - (n+1)))
                self.betas.append(0.)
            else:
                raise NotImplementedError(f"Unknown tiling type \"{tiling_type}\".")

        if self.interp_type not in ["nearest"]:
            raise NotImplementedError(f"Unknown interpolation type \"{self.interp_type}\".")


    def forward(self, lat, xyz):
        """
        Args:
            lat (torch.Tensor): latent code. Should have singleton dimensions
                where the latents need to be repeated. E.g., [B, 1, 256] if
                xyz.shape = [B, N, 3] with B:=batch and N:=points per shape.
            xyz (torch.Tensor): input positions.
        """
        batch_shape = x.shape[:-1]
        # Compute input features, then concatenate them with the latent code
        feats = self.features(xyz) if self.features is not None else xyz
        x = torch.cat([lat.expand(batch_shape + (-1,)), feats], dim=-1)

        # Flatten the batch dimensions
        xyz = xyz.flatten(0, -2)
        x = x.flatten(0, -2)
        inp = x
        for i in range(self.n_layers):
            D_out = self.linears[i][0].out_features

            # Find the tile indices of the samples based on their positions
            with torch.no_grad():
                alpha, beta = self.alphas[i], self.betas[i]
                if self.interp_type == "nearest":
                    tile_id = torch.floor(alpha * xyz + beta).long() % self.N_per_d
                    tile_id = tile_id[...,0] + self.N_per_d * tile_id[...,1] + (self.N_per_d**2) * tile_id[...,2]
            
            # Loop over the tiles, applying the correct layer
            # (this should be fused into a single CUDA kernel in their implementation)
            out = torch.empty((inp.shape[0], D_out), device=inp.device)
            for n in range(self.N):
                mask = tile_id == n
                out[mask] = self.linears[i][n](inp[mask])
            out = out + self.biases[i]

            if i < self.n_layers - 1:
                inp = self.activ(out)
                if (i + 1) in self.in_insert:
                    inp = torch.cat([inp, x], dim=-1)

        if self.output_scale is not None:
            out = out * self.output_scale
        return out.view(batch_shape + (-1,))