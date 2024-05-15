"""
Visualization utility module, based on Pytorch3D.

Most of the values here assumes the shape is in the [-1,1]^3 cube,
aligned with the x-axis, and up is the y-axis.
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import trimesh
import torch
import torch.nn.functional as F
try:
    # Data structures and functions for rendering
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        look_at_view_transform,
        FoVPerspectiveCameras, 
        PointLights, DirectionalLights, AmbientLights,
        RasterizationSettings, 
        MeshRenderer, 
        MeshRasterizer,  
        SoftPhongShader, HardFlatShader, HardPhongShader,
        TexturesVertex
    )
    renderer_available = True
except ImportError:
    warnings.warn("PyTorch3D not found, rendering functions will not work.")
    renderer_available = False

from .mesh import create_mesh
from .utils import make_grid2d, compute_sdf, get_device


#############
# Rendering #
#############

def convert_meshes_pytorch3d(meshes, use_texture=False, device=get_device()):
    """Convert a list of (non-empty) meshes from Trimesh to PyTorch3D."""
    assert renderer_available, "PyTorch3D not installed, cannot render."

    verts = [torch.tensor(mesh.vertices).float().to(device) for mesh in meshes]
    faces = [torch.tensor(mesh.faces).float().to(device) for mesh in meshes]

    if use_texture:
        verts_rgb = [torch.tensor(mesh.visual.vertex_colors[:,:3]).float().to(device) / 255.
                     for mesh in meshes]
    else:  # initialize each vertex to be white in color
        verts_rgb = [torch.ones_like(v) for v in verts]  # (B, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb)

    # Load mesh in pytorch3d
    return Meshes(verts, faces, textures=textures)


def get_renderer(size=512, ambient_light=False, eye=((1.2, 0.6, 1.8),), at=((0., 0., 0.),), 
                 up=((0., 1., 0.),), light_loc=((4., 2., 2.),), device=get_device()):
    """
    Return a default PyTorch3D renderer.

    Args:
    -----
    size: int or tuple
        Size of the rendered image.
    ambient_light: bool
        If True, use an ambient light instead of a point light.
    eye, at, up: tuples of tuples of 3 floats
        Cameras parameters for the views transforms.
    light_loc: tuple of tuples 3 floats
        Locations of the point lights.
    
    Returns:
    --------
    renderer: MeshRenderer
        The default renderer.
    """
    assert renderer_available, "PyTorch3D not installed, cannot render."

    # Initialize a camera
    R, T = look_at_view_transform(eye=eye, at=at, up=up) 
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Define the settings for rasterization and shading
    raster_settings = RasterizationSettings(
        image_size=size, 
        blur_radius=0.0, 
        faces_per_pixel=8,
        cull_backfaces=False
    )

    # Place a point light
    if ambient_light:
        lights = AmbientLights(device=device, ambient_color=((1., 1., 1.),))
    else:
        lights = PointLights(device=device, location=light_loc)

    shader=SoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=shader
    )

    return renderer


@torch.no_grad()
def render_meshes(meshes, size=512, use_texture=False, ambient_light=False,
                  eye=((1.2, 0.6, 1.8),), at=((0., 0., 0.),), up=((0., 1., 0.),), aa_factor=1,
                  device=get_device(), **kwargs_renderer):
    """
    Render all meshes based on the default renderer.

    Args:
    -----
    meshes: list of Trimesh or PyTorch3D Meshes
        List of meshes to render.
    size: int or tuple
        Size of the rendered image.
    use_texture: bool
        If True, use the texture of the meshes, else they will be white.
    ambient_light: bool
        If True, use an ambient light instead of a point light.
    eye, at, up: tuples of tuples of 3 floats
        Cameras parameters for the views transforms.
    aa_factor: int
        Anti-aliasing factor: rendering at size * aa_factor and then average pooling.
    device: str or Device
        Device to use for the rendering.
    **kwargs_renderer: dict
        Additional parameters for the renderer.
    
    Returns:
    --------
    images: np.array, shape (N, size[0], size[1], 3)
        The rendered images.
    """
    assert renderer_available, "PyTorch3D not installed, cannot render."
    
    # Separate empty meshes (renderer does work with them)
    n_mesh = len(meshes)
    if isinstance(meshes[0], trimesh.Trimesh):
        idx = [i for i in range(n_mesh) if not meshes[i].is_empty]
        meshes = [mesh for mesh in meshes if not mesh.is_empty]
        if len(meshes) > 0:
            meshes = convert_meshes_pytorch3d(meshes, use_texture=use_texture, device=device)
    else:  # assumes PyTorch3D Meshes
        idx = [i for i in range(n_mesh) if not meshes[i].isempty()]
        meshes = [mesh for mesh in meshes if not mesh.isempty()]

    if isinstance(size, int):
        size = (size, size)
    images = torch.ones((n_mesh, size[0], size[1], 4))
    size = (size[0] * aa_factor, size[1] * aa_factor)
    if len(meshes) > 0:
        renderer = get_renderer(size=size, ambient_light=ambient_light, eye=eye, at=at, up=up, 
                                device=device, **kwargs_renderer)

        with warnings.catch_warnings():
            # Ignore "R not valid rotation matrix" warning
            warnings.simplefilter('ignore')
            renders = renderer(meshes).cpu()
            if aa_factor > 1:
                # Average pool for anti-aliasing
                renders = renders.permute(0, 3, 1, 2)  # NHWC -> NCHW
                renders = F.avg_pool2d(renders, kernel_size=aa_factor, stride=aa_factor)
                renders = renders.permute(0, 2, 3, 1)  # NCHW -> NHWC
        # Regroup renders with empty ones
        images[idx] = renders

    # Post-processing
    images = images[..., :3].clamp(0., 1.)
    return images.numpy()

def render_mesh(mesh, *args, **kwargs):
    """
    Render a single mesh based on the default renderer.

    Args:
    -----
    mesh: Trimesh or PyTorch3D Mesh
        Mesh to render.
    size: int or tuple
        Size of the rendered image.
    use_texture: bool
        If True, use the texture of the mesh, else it will be white.
    ambient_light: bool
        If True, use an ambient light instead of a point light.
    eye, at, up: tuples of tuples of 3 floats
        Cameras parameters for the views transforms.
    aa_factor: int
        Anti-aliasing factor: rendering at size * aa_factor and then average pooling.
    device: str or Device
        Device to use for the rendering.
    **kwargs_renderer: dict
        Additional parameters for the renderer.
    
    Returns:
    --------
    image: np.array, shape (size[0], size[1], 3)
        The rendered image.
    """
    return render_meshes([mesh], *args, **kwargs)[0]


def render_latents(model, latents, size=512, aa_factor=1, **kwargs):
    """Reconstruct and render the meshes from the latent vectors."""
    meshes = []
    for latent in latents:
        meshes.append(create_mesh(model, latent, **kwargs))
    return render_meshes(meshes, size=size, aa_factor=aa_factor)


def image_grid(images, rows=2):
    """Concatenate the same-sized images into a grid."""
    if len(images) == 1:
        return images[0]
    
    # Number of images per row:
    N = max(1, len(images) // rows)

    # Add empty images if needed
    to_add = len(images) % N
    if to_add > 0:
        images = np.concatenate([images, (np.zeros_like(images[0]),) * to_add], axis=0)

    return np.concatenate([
        np.concatenate(images[N * i: N * (i+1)], axis=1)
        for i in range(rows)
    ], axis=0)


#######
# SDF #
#######

def plot_sdf_slices(model, latent, clampD=None, cmap='bwr', contour=False, device=get_device()):
    """Return the figure ploting 2D slices of the SDF."""
    if isinstance(cmap, str):
        cmap = colormaps[cmap]
    model.eval()
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.5))
    for i, (ax, ax_name) in enumerate(zip(axs.flatten(), ['x', 'y', 'z'])):
        xyz = make_grid2d([[-1, -1], [1, 1]], 512, i, 0.)
        with torch.no_grad():
            sdf = compute_sdf(model, latent, xyz.to(device)).squeeze().detach().cpu().T
        vmax = sdf.abs().max()
        if clampD is not None and clampD > 0.:
            vmax = min(vmax, clampD)
        ax.set_title(f"SDF at {ax_name}=0.")
        if contour:
            im = ax.contourf(sdf, levels=20, cmap=cmap, vmin=-vmax, vmax=vmax, extent=[-1,1,-1,1])
        else:
            im = ax.imshow(sdf.flip(0), cmap=cmap, vmin=-vmax, vmax=vmax, extent=[-1,1,-1,1])
        plt.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_render(meshes, use_texture=None, titles=None, max_cols=3, **kwargs):
    """
    Plot renders of the meshes, optionally using the mesh's texture.

    Args:
    -----
    meshes: list
        List of meshes to render.
    use_texture: list of bool (optional)
        If given, should be a list so that use_texture[i] indicates if
        the render of meshes[i] should use its texture.
    titles: list of str (optional)
        List of title for the plots.
    max_cols: int
        Maximum number of columns in the plot.
    **kwargs: dict
        Additional arguments for render_meshes.
    """
    N = len(meshes)
    if use_texture is None:
        use_texture = [False] * N
    images = [render_mesh(mesh, use_texture=texture, **kwargs) if mesh is not None else None
              for (mesh, texture) in zip(meshes, use_texture)]

    n_cols = min(N, max_cols)
    n_rows = int(np.ceil(N / max_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    for ax in axs.flat:  # only enable axis of rendered shapes
        ax.axis('off')
    for i in range(N):
        if images[i] is None:
            continue
        axs.flat[i].axis('on')
        axs.flat[i].imshow(images[i])
        if titles is not None:
            axs.flat[i].set_title(titles[i])
    return fig