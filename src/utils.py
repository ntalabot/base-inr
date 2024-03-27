"""
Utilities module.
"""

import sys
import random
import logging
import io
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from PIL import Image
from sklearn.decomposition import PCA
import igl
import torch


###########
# General #
###########

def set_seed(seed):
    """Set the manual seeds for python, numpy, and pytorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_logging(args=None, logfile=None):
    logger = logging.getLogger()
    if args is None:
        logger.setLevel(logging.INFO)
    elif "debug" in args and args.debug:
        logger.setLevel(logging.DEBUG)
    elif "verbose" in args and not args.verbose:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)

    # Edit display of level names
    for level, levelname in zip([logging.CRITICAL, logging.ERROR, logging.WARNING, logging.DEBUG],
                                ['CRITICAL', 'ERROR', 'WARNING', 'DEBUG']):
        logging.addLevelName(level, levelname + ': ')
    logging.addLevelName(logging.INFO, '')
    formatter = logging.Formatter("%(levelname)s%(message)s")

    # Log in stream
    logger_handler = logging.StreamHandler()
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    # Log in file
    if logfile is not None:
        file_logger_handler = logging.FileHandler(logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)
    
    # Automatically log exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = handle_exception

    # Deal with verbose loggers
    plt.set_loglevel(level='info')
    pil_logger = logging.getLogger('PIL')  
    pil_logger.setLevel(logging.INFO)


#######
# SDF #
#######

def compute_sdf(model, latent, xyz, max_batch=32**3, verbose=False, device="cuda:0"):
    """
    Compute the SDF values for a single shape at the given positions.

    Args:
    -----
    model: nn.Module
        The model to evaluate.
    latent: torch.Tensor, shape (1, latent_size)
        The latent vector of the shape.
    xyz: torch.Tensor, shape ([N], 3)
        The positions to evaluate the SDF at.
    max_batch: int (default=32**3)
        The maximum number of points to evaluate at once.
    verbose: bool (default=False)
        If True, print the time taken for the computation.
    device: str (default="cuda:0")
        The device to use to store the final SDF values.
    
    Returns:
    --------
    sdf: torch.Tensor, shape ([N], 1)
        The SDF values at the given positions.
    """
    if verbose:
        start_time = time.time()
    model.eval()

    # Prepare data
    xyz_all = xyz.view(-1, 3)
    n_points = len(xyz_all)
    sdf = torch.zeros(n_points, device=device)

    # Predict SDF on a subset of points at a time
    latent = latent.view(1, -1)
    for i in range(0, n_points, max_batch):
        xyz_subset = xyz_all[i : i + max_batch].to(device)
        sdf[i : i + max_batch] = model(latent, xyz_subset)[:, 0].detach().to(device)

    if verbose:
        print(f"sdf-prediction took {time.time() - start_time:.3f}s.")    
    return sdf.view(xyz.shape[:-1] + (1,))


def clamp_sdf(sdf, clampD, ref=None):
    """
    Clamp the SDF, optionally based on a reference.
    If ref is given, then the clamping is dependent on it:
    e.g., sdf is clamped by +clampD only where ref >= +clampD.
    """
    if ref is None:
        return torch.clamp(sdf, -clampD, clampD)
    sdf = torch.where(ref > clampD, torch.clamp(sdf, max=clampD), sdf)
    sdf = torch.where(ref < -clampD, torch.clamp(sdf, min=-clampD), sdf)
    return sdf


def make_grid(bbox, N, device=None):
    """Create points on a regular grid."""
    if isinstance(N, int):
        N = [N] * len(bbox[0])
    coords = []
    for i in range(len(bbox[0])):
        coords.append(torch.linspace(bbox[0][i], bbox[1][i], N[i], device=device))
    coords = torch.meshgrid(*coords, indexing='ij')
    coords = torch.stack(coords, dim=-1)
    return coords


def make_grid2d(bbox, N, axis=0, value=0., device=None):
    """
    Create points on a regular grid of a 2D slice of a 3D world.
    
    axis: int (default=0)
        The axis index of the normal to the plane, e.g. 0 for x.
    value: float (default=0.)
        The second is the "fill" value along that axis, e.g. 0. for the x_i=0. plane.
    """
    assert axis in [0, 1, 2]

    if isinstance(N, int):
        N = [N] * 2
    x1 = torch.linspace(bbox[0][0], bbox[1][0], N[0], device=device)
    x2 = torch.linspace(bbox[0][1], bbox[1][1], N[1], device=device)
    grid = torch.meshgrid(x1, x2, indexing='ij')
    plane = torch.zeros_like(grid[0]) + value
    if axis == 0:
        xyz = torch.stack((plane, grid[0], grid[1]), dim=-1)
    elif axis == 1:
        xyz = torch.stack((grid[0], plane, grid[1]), dim=-1)
    elif axis == 2:
        xyz = torch.stack((grid[0], grid[1], plane), dim=-1)
    return xyz


def make_grid_image(bbox, N_max, axis=1, value=0., device=None):
    """
    Create points on a regular grid of a 2D slice, with number of pixels adapting from bbox.
    
    axis: int (default=0)
        The axis index of the normal to the plane, e.g. 0 for x.
    value: float (default=0.)
        The second is the "fill" value along that axis, e.g. 0. for the x_i=0. plane.
    """
    assert axis in [0, 1, 2]

    # Get resolution per axis
    if not isinstance(bbox, np.ndarray):
        bbox = np.array(bbox)
    extent = np.abs(bbox[1] - bbox[0])
    ratio = extent.min() / extent.max()
    N_min = np.round(N_max * ratio).astype(int)
    N = [N_max, N_min] if extent[0] > extent[1] else [N_min, N_max]
    
    x1 = torch.linspace(bbox[0][0], bbox[1][0], N[0], device=device)
    x2 = torch.linspace(bbox[0][1], bbox[1][1], N[1], device=device)
    grid = torch.meshgrid(x1, x2, indexing='ij')
    plane = torch.zeros_like(grid[0]) + value
    if axis == 0:
        xyz = torch.stack((plane, grid[0], grid[1]), dim=-1)
    elif axis == 1:
        xyz = torch.stack((grid[0], plane, grid[1]), dim=-1)
    elif axis == 2:
        xyz = torch.stack((grid[0], grid[1], plane), dim=-1)
    return xyz


def get_sdf_mesh(mesh, xyz):
    """Return the SDF of the mesh at the queried positions, using IGL."""
    sdf = igl.signed_distance(xyz.reshape(-1, 3), mesh.vertices, mesh.faces)[0]
    return sdf.reshape(xyz.shape[:-1])

def get_winding_number_mesh(mesh, xyz, method='fast'):
    """Return the winding number of the mesh at the queried positions, using IGL."""
    if method == 'fast':
        wn = igl.fast_winding_number_for_meshes(mesh.vertices, mesh.faces, xyz.reshape(-1, 3).astype(mesh.vertices.dtype))
    else:
        wn = igl.winding_number(mesh.vertices, mesh.faces, xyz.reshape(-1, 3).astype(mesh.vertices.dtype))
    return wn.reshape(xyz.shape[:-1])


##########
# Meshes #
##########

def get_color_mesh(mesh, vertex_values, vmin=None, vmax=None, cmap=colormaps['viridis'], symmetric=False):
    """Return a new mesh with the vertex colors set according to the values."""
    vmin = vertex_values.min() if vmin is None else vmin
    vmax = vertex_values.max() if vmax is None else vmax
    if symmetric:
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
    mesh = mesh.copy()
    vertex_values = (vertex_values - vmin) / (vmax - vmin)
    mesh.visual.vertex_colors = cmap(vertex_values)
    return mesh


##########
# Latent #
##########

def sample_latents(latents, n_samples=1, expvar=None, pca=None):
    """
    Sample latents from the PCA of training latents.

    If expvar is given, then the number of components is chosen such 
    that their explained variance is at least expvar (between 0 and 1).
    """
    if pca is None:
        pca = PCA(whiten=True).fit(latents.weight.detach().cpu().numpy())
    samples = np.random.randn(n_samples, min(latents.embedding_dim, latents.num_embeddings))
    if expvar is not None:
        n_comp = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= expvar)
        samples[:, n_comp:] *= 0.
    samples = pca.inverse_transform(samples)
    return torch.from_numpy(samples).float().to(latents.weight.device)


###########
# PyTorch #
###########

def get_gradient(inputs, outputs, retain_graph=True):
    grads = torch.autograd.grad(
        outputs.sum(), inputs,
        create_graph=True,
        retain_graph=retain_graph,
        only_inputs=True
    )
    if isinstance(inputs, torch.Tensor):
        grads = grads[0]
    return grads


#########
# Misc. #
#########

def fig2img(fig):
    """Convert a figure to an image."""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img