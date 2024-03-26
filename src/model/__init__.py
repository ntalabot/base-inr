from math import sqrt

from torch import nn

from .deepsdf import (
    DeepSDF, LatentModulatedDeepSDF, InputModulatedDeepSDF, 
    DemodulatedDeepSDF,
)
from .siren import Siren, LatentModulatedSiren
from .loe import LevelsOfExperts


def get_model(network, **kwargs):
    if network.lower() == "deepsdf":
        return DeepSDF(**kwargs)
    elif network.lower() == "latentmodulateddeepsdf":
        return LatentModulatedDeepSDF(**kwargs)
    elif network.lower() == "inputmodulateddeepsdf":
        return InputModulatedDeepSDF(**kwargs)
    elif network.lower() == "demodulateddeepsdf":
        return DemodulatedDeepSDF(**kwargs)
    elif network.lower() == "siren":
        return Siren(**kwargs)
    elif network.lower() == "latentmodulatedsiren":
        return LatentModulatedSiren(**kwargs)
    elif network.lower() in ["levelsofexperts", "loe"]:
        return LevelsOfExperts(**kwargs)
    else:
        raise NotImplementedError(f"Unkown model \"{network}\"")


def get_latents(n_shapes, dim, max_norm=None, std=None):
    """Create and initialize latent vectors as embeddings."""
    latents = nn.Embedding(n_shapes, dim, max_norm=max_norm).cuda()
    if std is None:
        std = 1. / sqrt(dim) if dim > 0 else 1.
    nn.init.normal_(latents.weight.data, 0., std)
    return latents