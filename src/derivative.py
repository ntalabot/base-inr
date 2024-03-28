"""
Module for computing derivatives of SDFs.

NOTE: it is assumed that the model outputs a single scalar value for each input point!
"""

import igl
import torch
from torch import autograd

from utils import get_device


_device = get_device()


# Autograd
##########
# Use the exact gradients and Hessians computed by autograd.

def get_gradient(model, latents, xyz, create_graph=False, retain_graph=None):
    """Return the gradient w.r.t. xyz."""
    grads, = autograd.grad(model(latents, xyz).sum(), xyz, create_graph=create_graph, retain_graph=retain_graph)
    return grads


def get_hessian(model, latents, xyz, create_graph=False):
    """Return the Hessian w.r.t. xyz."""
    # Here, need to flatten the points and repeat the latents
    xyz_flat = xyz.view(-1, 3)
    if latents is not None and latents.nelement() > 0:
        latents = latents.view(-1, latents.shape[-1])
    hessian = torch.autograd.functional.hessian(
        lambda xyz: model(latents, xyz).sum(), xyz_flat, 
        vectorize=True, create_graph=create_graph
    ).sum(-2)
    return hessian.view(xyz.shape + (3,))


def get_laplacian(model, latents, xyz, create_graph=False):
    """Return the Laplacian w.r.t. xyz."""
    hessian = get_hessian(model, latents, xyz, create_graph=create_graph)
    # Laplacian is the trace of the Hessian
    return torch.einsum('...ii->...', hessian)


# Finite-Difference
###################
# Use finite-differences to compute the gradients and Hessians.
# Based on central differences: https://en.wikipedia.org/wiki/Finite_difference

# Point offsets for the FD computations.
_offset_fd = torch.tensor([[  # for first derivative (1x3x2x3)
    [[-1, 0, 0], [1, 0, 0]],
    [[0, -1, 0], [0, 1, 0]],
    [[0, 0, -1], [0, 0, 1]],
]]).float().to(_device)
_offset_2nd_fd = torch.tensor([[  # for second derivative (1x19x3)
    [0, 0, 0],
    [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1],
    [-1, -1, 0], [1, 1, 0], [-1, 0, -1], [1, 0, 1], [0, -1, -1], [0, 1, 1],
    [-1, 1, 0], [1, -1, 0], [-1, 0, 1], [1, 0, -1], [0, 1, -1], [0, -1, 1],
]]).float().to(_device)


def _build_hessian_fd(sdf):
    """
    Build the Hessian matrix from the SDF values around the points.

    It is based on the order of offsets from `_offset_2nd_fd`.
    """
    # sdf is [B]x19
    hessian = torch.zeros(sdf.shape[:-1] + (3,3), device=sdf.device)
    hessian[..., 0, 0] = sdf[..., 2] - 2*sdf[..., 0] + sdf[..., 1]
    hessian[..., 1, 1] = sdf[..., 4] - 2*sdf[..., 0] + sdf[..., 3]
    hessian[..., 2, 2] = sdf[..., 6] - 2*sdf[..., 0] + sdf[..., 5]
    hessian[..., 0, 1] = hessian[..., 1, 0] = 0.25*(sdf[...,  8] - sdf[..., 14] - sdf[..., 13] + sdf[...,  7])
    hessian[..., 0, 2] = hessian[..., 2, 0] = 0.25*(sdf[..., 10] - sdf[..., 16] - sdf[..., 15] + sdf[...,  9])
    hessian[..., 1, 2] = hessian[..., 2, 1] = 0.25*(sdf[..., 12] - sdf[..., 18] - sdf[..., 17] + sdf[..., 11])
    return hessian.view(sdf.shape[:-1] + (3,3))


## For an SDF model
def get_gradient_fd(model, latents, xyz, h):
    """Return the pseudo-gradient w.r.t. xyz using finite-differences."""
    xyz_fd = xyz.unsqueeze(-2).unsqueeze(-2) + _offset_fd.to(xyz.device) * h
    if latents is not None and latents.nelement() > 0:
        latents = latents.unsqueeze(-2).unsqueeze(-2)
    sdf = model(latents, xyz_fd).squeeze(-1)

    grads = (sdf[..., 1] - sdf[..., 0]) / (2. * h)
    return grads.view(xyz.shape)


def get_hessian_fd(model, latents, xyz, h):
    """Return the pseudo-Hessian w.r.t. xyz using finite-differences."""
    # Here, need to flatten the points and repeat the latents
    xyz_flat = (xyz.view(-1, 1, 3) + _offset_2nd_fd.to(xyz.device) * h).view(-1, 3)
    if latents is not None and latents.nelement() > 0:
        latents = latents.expand(*xyz.shape[:-1], -1)
        latents = latents.view(-1, 1, latents.shape[-1]).expand(-1, 19, -1).view(-1, latents.shape[-1])
    sdf = model(latents, xyz_flat).view(-1, 19)
    
    hessian = _build_hessian_fd(sdf) / (h * h)
    return hessian.view(xyz.shape + (3,))


def get_laplacian_fd(model, latents, xyz, h):
    """Return the pseudo-Laplacian w.r.t. xyz using finite-differences."""
    # Append the offset points to the original ones
    xyz_fd = xyz.unsqueeze(-2) + _offset_fd.to(xyz.device).view(1, 3*2, 3) * h
    xyz_fd = torch.cat([xyz.unsqueeze(-2), xyz_fd], dim=-2)
    if latents is not None and latents.nelement() > 0:
        latents = latents.unsqueeze(-2)
    sdf = model(latents, xyz_fd).squeeze(-1)

    laplacian = (sdf[..., 1:7].sum(-1) - 6 * sdf[..., 0]) / (h * h)
    return laplacian.view(xyz.shape[:-1])


## For a mesh (done on cpu for memory issues; non-differentiable)
@torch.no_grad()
def get_gradient_fd_mesh(mesh, xyz, h):
    """Return the pseudo-gradient of the mesh's SDF w.r.t. xyz using finite-differences."""
    xyz_flat = (xyz.view(-1, 1, 1, 3).detach().cpu() + _offset_fd.cpu() * h).view(-1, 3).numpy()
    sdf = igl.signed_distance(xyz_flat, mesh.vertices, mesh.faces)[0]
    sdf = torch.from_numpy(sdf).float().view(-1, 3, 2)
    grads = (sdf[:, :, 1] - sdf[:, :, 0]) / (2 * h)
    return grads.view(xyz.shape).detach().cpu()

@torch.no_grad()
def get_hessian_fd_mesh(mesh, xyz, h):
    """Return the pseudo-Hessian of the mesh's SDF w.r.t. xyz using finite-differences."""
    xyz_flat = (xyz.view(-1, 1, 3).detach().cpu() + _offset_2nd_fd.cpu() * h).view(-1, 3).numpy()
    sdf = igl.signed_distance(xyz_flat, mesh.vertices, mesh.faces)[0]
    sdf = torch.from_numpy(sdf).float().view(-1, 19)
    
    hessian = _build_hessian_fd(sdf) / (h * h)
    return hessian.view(xyz.shape + (3,)).detach().cpu()

@torch.no_grad()
def get_laplacian_fd_mesh(mesh, xyz, h):
    """Return the pseudo-Laplacian of the mesh's SDF w.r.t. xyz using finite-differences."""
    xyz_flat = xyz.view(-1, 3).detach().cpu()
    len_xyz = len(xyz_flat)
    xyz_offset = (xyz.view(-1, 1, 1, 3) + _offset_fd.cpu() * h).view(-1, 3)
    xyz_flat = torch.cat([xyz_flat, xyz_offset], dim=0).numpy()
    sdf = igl.signed_distance(xyz_flat, mesh.vertices, mesh.faces)[0]
    sdf = torch.from_numpy(sdf).float().view(-1)
    sdf_0 = sdf[:len_xyz]
    sdf = sdf[len_xyz:].view(-1, 3*2)
    
    laplacian = (sdf.sum(1) - 6 * sdf_0) / (h * h)
    return laplacian.view(xyz.shape[:-1]).detach().cpu()