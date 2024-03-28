"""
Main reconstruction script.

Reconstruct a set of shapes with a deep implicit model.
"""

import os, os.path
import sys
import argparse
import logging
import json
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import workspace as ws
from src.loss import get_loss_recon
from src.mesh import create_mesh
from src.model import get_model
from src.reconstruct import reconstruct
from src.utils import configure_logging, set_seed, get_device


def parser(argv=None):
    parser = argparse.ArgumentParser(description="Reconstruct shapes with a deep implicit model.")

    parser.add_argument("experiment", help="path to the experiment directory")

    parser.add_argument('--debug', action='store_true', help="increase verbosity to print debugging messages")
    parser.add_argument('-i', '--iters', type=int, default=800, help="number of iterations for latent optimization")
    parser.add_argument('--load-epoch', default='latest', help="epoch to load, default to latest available")
    parser.add_argument('--lr', type=float, default=5e-3, help="learning rate for latent optimization")
    parser.add_argument('--max-norm', action='store_true', help="use the max norm from specs to bound the reconstructed latent (otherwise is unbounded)")
    parser.add_argument('-n', '--n-samples', type=int, default=8000, help="number of sdf samples used per iteration")
    parser.add_argument('--overwrite', action='store_true', help="overwrite shapes that are already reconstructed")
    parser.add_argument('-q', '--quiet', dest="verbose", action='store_false',  help="disable verbosity and run in quiet mode")
    parser.add_argument('-r', '--resolution', type=int, default=256, help="resolution for the reconstruction with marching cubes")
    parser.add_argument('--seed', type=int, default=0, help="initial seed for the RNGs (default=0)")
    parser.add_argument('-s', '--split', help="split to reconstruct, default to \"TestSplit\" in specs file")
    parser.add_argument('-t', '--test', action='store_true', help="reconstruct the test set, otherwise reconstruct the validation set (--split override this)")

    args = parser.parse_args(argv)

    return args


def main(args=None):
    # Initialization
    if args is None:
        args = parser()
    set_seed(args.seed)
    device = get_device()
    start_time = time.time()

    expdir = args.experiment
    specs = ws.load_specs(expdir)
    device = specs.get("Device", get_device())
    configure_logging(args, os.path.join(ws.get_log_dir(expdir), "reconlog.txt"))

    logging.info(f"Command:  python {' '.join(sys.argv)}")
    logging.info(f"Reconstructing shapes in {expdir}. (on {device})")
    logging.info(f"arguments = {args}")
    
    # Data
    if args.split is None:
        args.split = specs["TestSplit"] if args.test else specs["ValidSplit"]
    with open(args.split) as f:
        instances = json.load(f)
    n_shapes = len(instances)
    n_samples = args.n_samples
    datasource = os.path.join(specs["DataSource"], specs["SamplesDir"])
    samplefile = specs["SamplesFile"]

    logging.info(f"{n_shapes} shapes in {args.split} to reconstruct.")

    # Model
    latent_dim = specs["LatentDim"]
    model = get_model(
        specs.get("Network", "DeepSDF"),
        **specs.get("NetworkSpecs", {}),
        latent_dim=latent_dim
    ).to(device)
    # Evaluation mode with frozen model
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    logging.info(f"Model has {sum([x.nelement() for x in model.parameters()]):,} parameters.")

    # Loss
    loss_recon = get_loss_recon(specs.get("ReconLoss", "L1-Hard"), reduction='none').to(device)
    latent_reg = specs["LatentRegLambda"]

    # Resume from checkpoint
    if args.load_epoch == 'latest':
        args.load_epoch = ws.load_history(expdir)['epoch']
    ws.load_model(expdir, model, args.load_epoch)

    logging.info(f"Loaded checkpoint from epoch={args.load_epoch}.")

    # Parameters and directories
    clampD = specs["ClampingDistance"]
    max_norm = specs["LatentBound"] if args.max_norm else None
    latent_subdir = ws.get_recon_latent_subdir(expdir, args.load_epoch)
    mesh_subdir = ws.get_recon_mesh_subdir(expdir, args.load_epoch)
    os.makedirs(latent_subdir, exist_ok=True)
    os.makedirs(mesh_subdir, exist_ok=True)

    # Reconstruction
    for i, instance in enumerate(instances):
        logging.info(f"Shape {i+1}/{n_shapes} ({instance})")
        if not args.overwrite and \
           os.path.isfile(os.path.join(latent_subdir, instance + ".pth")) and \
           os.path.isfile(os.path.join(mesh_subdir, instance + ".obj")):
            logging.info(f"already existing, skipping...")
            continue
        
        # Load sdf data
        filename = os.path.join(datasource, instance, samplefile)
        npz = np.load(filename)

        # Optimize latent
        err, latent = reconstruct(model, npz, args.iters, n_samples, args.lr, loss_recon, 
                                  latent_reg, clampD, latent_init=None, latent_size=specs["LatentDim"],
                                  max_norm=max_norm, verbose=args.verbose, device=device)
        logging.info(f"Final error={err:.6f}")

        # Reconstruct the mesh
        mesh = create_mesh(model, latent, N=args.resolution, max_batch=32**3, verbose=args.verbose, device=device)

        # Save results
        torch.save(latent, os.path.join(latent_subdir, instance + ".pth"))
        mesh.export(os.path.join(mesh_subdir, instance + ".obj"))
    
    torch.cuda.empty_cache()  # release unused GPU memory

    duration = time.time() - start_time
    duration_msg = "{:.0f}h {:02.0f}min {:02.0f}s".format(duration // 3600, (duration // 60) % 60, duration % 60)
    logging.info(f"End of reconstruction after {duration_msg}.")
    logging.info(f"Results saved in {expdir}.")


if __name__ == "__main__":
    args = parser()
    main(args)