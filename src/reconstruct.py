"""
Module for test and inference.
"""

import time

import torch
from torch import optim

from .data import remove_nans, samples_from_tensor
from .loss import get_loss_recon
from .utils import clamp_sdf, get_device


def reconstruct(model, sdf_data, n_iters, n_samples, lr, loss_fn_recon="l1", 
                latent_reg=None, clampD=None, latent_init=None, latent_size=256,
                max_norm=None, verbose=False, device=get_device()):
    """Reconstruct the shape by optimizing the latent wrt to SDF data."""
    sdf_data = [sdf_data]
    return reconstruct_batch(model, sdf_data, n_iters, n_samples, lr, loss_fn_recon=loss_fn_recon, 
                             latent_reg=latent_reg, clampD=clampD, latent_init=latent_init, latent_size=latent_size,
                             max_norm=max_norm, verbose=verbose, device=device)


def reconstruct_batch(model, sdf_data, n_iters, n_samples, lr, loss_fn_recon="l1", 
                      latent_reg=None, clampD=None, latent_init=None, latent_size=256,
                      max_norm=None, verbose=False, device=get_device()):
    """Reconstruct the batch of shapes by optimizing their latents wrt to SDF data."""
    if verbose:
        start_time = time.time()
    # Data
    n_shapes = len(sdf_data)
    sdf_pos = [remove_nans(sdf['pos']) for sdf in sdf_data]
    sdf_neg = [remove_nans(sdf['neg']) for sdf in sdf_data]
    sdf_pos = [torch.from_numpy(sdf).float().to(device) for sdf in sdf_pos]
    sdf_neg = [torch.from_numpy(sdf).float().to(device) for sdf in sdf_neg]

    # Initialize the latent
    if latent_init is None:
        latent = torch.ones(n_shapes, latent_size).normal_(0, 0.01).to(device)
    elif isinstance(latent_init, float):
        latent = torch.ones(n_shapes, latent_size).normal_(0, latent_init).to(device)
    elif isinstance(latent_init, torch.Tensor):
        latent = latent_init.clone().detach()
    latent = latent.view(n_shapes, -1)
    latent.requires_grad_(True)

    # Optimizer and scheduler
    if isinstance(loss_fn_recon, str):
        loss_fn_recon = get_loss_recon(loss_fn_recon, 'none').to(device)
    optimizer = optim.Adam([latent], lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, n_iters//2, 0.1)

    # Model in evaluation mode and frozen
    model.eval()
    p_state = []
    for p in model.parameters():
        p_state.append(p.requires_grad)
        p.requires_grad = False
    
    for _ in range(n_iters):
        # Sample SDF
        xyz, sdf_gt = [], []
        for pos, neg in zip (sdf_pos, sdf_neg):
            _xyz, _sdf_gt = samples_from_tensor(pos, neg, n_samples)
            xyz.append(_xyz)
            sdf_gt.append(_sdf_gt)
        xyz, sdf_gt = torch.stack(xyz), torch.stack(sdf_gt)

        # Forward pass
        preds = model(latent.unsqueeze(1), xyz)[..., 0:1]  # SDF output
        if clampD is not None and clampD > 0:
            preds = clamp_sdf(preds, clampD, ref=sdf_gt)
            sdf_gt = clamp_sdf(sdf_gt, clampD)

        loss = loss_fn_recon(preds, sdf_gt).mean()
        if latent_reg is not None:
            loss = loss + latent_reg * latent.square().sum()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Latent max norm
        if max_norm is not None:
            with torch.no_grad():
                latent_norm = latent.norm(dim=-1, keepdim=True)
                latent *= latent_norm.clamp(max=max_norm) / latent_norm

    # Restore model's parameter state
    for p, state in zip(model.parameters(), p_state):
        p.requires_grad = state
    
    if verbose:
        print(f"reconstruction took {time.time() - start_time:.3f}s.") 
    return loss.detach().cpu().numpy(), latent.detach().clone()