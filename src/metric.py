"""
Evaluation metrics for reconstructed shapes.
"""

import numpy as np
from scipy.spatial import cKDTree

from .utils import make_grid, get_winding_number_mesh


def chamfer_distance(pc1, pc2, square_dist=True, return_idx=False,
                     val1=None, val2=None, val_fn=np.abs):
    """
    Compute the symmetric L2-Chamfer Distance between the two point clouds.

    Can optionally gives other values to be compared on the points.
    
    Args:
    -----
    pc1, pc2: (N, 3) arrays
        The point clouds to compare.
    square_dist: bool (default=True)
        If True, compute the squared distance.
    return_idx: bool (default=False)
        If True, also return the indices of the closest points.
    val1, val2: (N, D) arrays (optional)
        If given, compare these values on the closest points.
    val_fn: callable (default=np.abs)
        The function to compare the values. Should take the difference as input.
    
    Returns:
    --------
    chamfer: float
        The symmetric L2-Chamfer distance.
    idx1, idx2: (N,) arrays (optional)
        The indices of the closest points.
    val_diff: float (optional)
        The average difference between the values on the closest points.
    """
    tree1 = cKDTree(pc1)
    dist1, idx1 = tree1.query(pc2)
    if square_dist:
        dist1 = np.square(dist1)
    chamfer2to1 = np.mean(dist1)

    tree2 = cKDTree(pc2)
    dist2, idx2 = tree2.query(pc1)
    if square_dist:
        dist2 = np.square(dist2)
    chamfer1to2 = np.mean(dist2)

    results = (chamfer2to1 + chamfer1to2,)
    if return_idx:
        results += (idx1, idx2)

    # Compare values on the points
    if val1 is not None and val2 is not None:
        val2to1 = np.mean(val_fn(val1[idx1] - val2))
        val1to2 = np.mean(val_fn(val2[idx2] - val1))
        results = results + (val2to1 + val1to2,)

    return results[0] if len(results) == 1 else results


def mesh_iou(mesh1, mesh2, N=256, bbox=None):
    """
    Compute volumetric IoU between the meshes using occupancy samples.

    Args:
    -----
    mesh1, mesh2: trimesh.base.Trimesh
        The meshes to compare.
    N: int or tuple of int (default=256)
        The resolution of the grid to compute occupancy on.
    bbox: (2, 3) array (optional)
        The bounding box to use for the grid. If None, take the bbox of both meshes.
    
    Returns:
    --------
    iou: float
        The volumetric IoU.
    """
    if bbox is None:  # take a bbox including both meshes
        bbox = np.stack([np.minimum(mesh1.bounds[0], mesh2.bounds[0]), 
                         np.maximum(mesh1.bounds[1], mesh2.bounds[1])])
    
    # Compute occupancy on a regular grid
    xyz = make_grid(bbox, N).numpy()
    wn1, wn2 = get_winding_number_mesh(mesh1, xyz), get_winding_number_mesh(mesh2, xyz)
    occ1, occ2 = (wn1 >= 0.5), (wn2 >= 0.5)
    
    intersection = np.sum(occ1 & occ2)
    union = np.sum(occ1 | occ2)
    return intersection / union