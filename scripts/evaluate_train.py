"""
Secondary evaluation script.

Evaluate the training set of shapes reconstructed with a deep implicit model.
"""

import os, os.path
import sys
import argparse
import logging
import json
import time

import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import workspace as ws
from src.mesh import create_mesh
from src.metric import chamfer_distance
from src.model import get_model, get_latents
from src.utils import configure_logging, set_seed


def parser(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate training shapes reconstructed with a deep implicit model.")

    parser.add_argument("experiment", help="path to the experiment directory")

    parser.add_argument('--debug', action='store_true', help="increase verbosity to print debugging messages")
    parser.add_argument('--load-epoch', default='latest', help="epoch to load, default to latest available")
    parser.add_argument('-q', '--quiet', dest="verbose", action='store_false',  help="disable verbosity and run in quiet mode")
    parser.add_argument('-r', '--resolution', type=int, default=256, help="resolution for the reconstruction with marching cubes")
    parser.add_argument('--seed', type=int, default=0, help="initial seed for the RNGs (default=0)")
    parser.add_argument('--skip', action='store_true', help="skip meshes that are already evaluated")

    # Chamfer-Distance
    parser.add_argument('--chamfer-samples', type=int, default=30000, help="number of surface samples for chamfer distance (default=30000)")
    parser.add_argument('--no-square-dist', action='store_false', dest="square_dist", help="do not use square distances for the chamfer-distance")

    args = parser.parse_args(argv)

    return args


def main(args=None):
    # Initialization
    if args is None:
        args = parser()
    set_seed(args.seed)
    start_time = time.time()

    expdir = args.experiment
    specs = ws.load_specs(expdir)
    configure_logging(args, os.path.join(ws.get_log_dir(expdir), "evaltrainlog.txt"))

    logging.info(f"Command:  python {' '.join(sys.argv)}")
    logging.info(f"Evaluating training shapes in {expdir}.")
    logging.info(f"arguments = {args}")
    
    # Data
    split = specs["TrainSplit"]
    with open(split) as f:
        instances = json.load(f)
    n_shapes = len(instances)
    datasource = os.path.join(specs["DataSource"], specs["SamplesDir"])

    logging.info(f"{n_shapes} shapes in {split} to evaluate.")

    # Model and latent vectors
    latent_dim = specs["LatentDim"]
    model = get_model(
        specs.get("Network", "DeepSDF"),
        **specs.get("NetworkSpecs", {}),
        latent_dim=latent_dim
    ).cuda()
    latents = get_latents(n_shapes, latent_dim, specs.get("LatentBound", None))

    # Resume from checkpoint
    if args.load_epoch == 'latest':
        args.load_epoch = ws.load_history(expdir)['epoch']
    
    try:
        ws.load_model(expdir, model, args.load_epoch)
        ws.load_latents(expdir, latents, args.load_epoch)
    except FileNotFoundError as err:
        checkpoint = ws.load_checkpoint(expdir)
        model.load_state_dict(checkpoint['model_state_dict'])
        latents.load_state_dict(checkpoint['latents_state_dict'])
        print(f"File not found: {err.filename}.\nLoading checkpoint instead (epoch={checkpoint['epoch']}).")
        del checkpoint

    # Parameters and directories
    eval_dir = ws.get_eval_dir(expdir, args.load_epoch)
    os.makedirs(eval_dir, exist_ok=True)

    logging.info(f"Evaluating checkpoint from epoch={args.load_epoch}.")

    ## Evaluation metrics
    results = {}
    filenames = {}
    metrics = [
        "chamfer"  # Chamfer-Distance
    ]
    for metric in metrics:
        results[metric] = {}
        filenames[metric] = os.path.join(eval_dir, metric+"_train.json")
        if os.path.isfile(filenames[metric]):
            with open(filenames[metric]) as f:
                results[metric].update(json.load(f))

    # Evaluation
    for i, instance in enumerate(instances):
        logging.info(f"Shape {i+1}/{n_shapes} ({instance})")

        # Chamger-Distance
        if args.skip and instance in results["chamfer"]:
            logging.info(f"chamfer = {results["chamfer"][instance]} (existing)")
        else:
            # Load GT surface samples
            gt_samples = np.load(os.path.join(datasource, instance, "surface.npz"))['all']
            gt_samples = np.random.permutation(gt_samples)[:args.chamfer_samples, :3]

            # Reconstruct latent
            recon_mesh = create_mesh(model, latents(torch.tensor([i]).cuda()), N=args.resolution)
            recon_samples = recon_mesh.sample(args.chamfer_samples)

            # Chamfer-Distance
            chamfer_val = chamfer_distance(gt_samples, recon_samples, square_dist=args.square_dist)
            results["chamfer"][instance] = float(chamfer_val)
            logging.info(f"chamfer = {chamfer_val}")
    
    # Save results
    for metric in metrics:
        with open(filenames[metric], "w") as f:
            json.dump(results[metric], f, indent=2)
    
    # Average and median metrics
    for metric in metrics:
        all_values = list(results[metric].values())
        all_values = [v for v in all_values if not np.isnan(v)]
        logging.info(f"Average {metric} = {np.mean(all_values)}  ({len(all_values)}/{len(instances)} shapes)")
        logging.info(f"Median  {metric} = {np.median(all_values)}  ({len(all_values)}/{len(instances)} shapes)")

    duration = time.time() - start_time
    duration_msg = "{:.0f}h {:02.0f}min {:02.0f}s".format(duration // 3600, (duration // 60) % 60, duration % 60)
    logging.info(f"End of evaluation after {duration_msg}.")
    logging.info(f"Results saved in {expdir}.")


if __name__ == "__main__":
    args = parser()
    main(args)