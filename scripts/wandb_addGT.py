"""
Make a pseudo experiment to add the ground-truth to Weights & Biases.
"""

import os, os.path
import sys
import argparse
import logging
import json

import trimesh
import wandb


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import visualization as viz
from src import workspace as ws
from src.utils import set_seed


def parser(argv=None):
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="Add GT meshes to Weight & Biases.")

    parser.add_argument("experiment", help="path to the experiment directory. If existing, will try to resume it (see --no-resume)")

    parser.add_argument('--debug', action='store_true', help="increase verbosity to print debugging messages")
    parser.add_argument('-q', '--quiet', dest="verbose", action='store_false',  help="disable verbosity and run in quiet mode")
    parser.add_argument('--seed', type=int, default=0, help="initial seed for the RNGs (default=0)")

    args = parser.parse_args(argv)

    return args


def main(args=None):
    # Initialization
    if args is None:
        args = parser()
    set_seed(args.seed)

    expdir = args.experiment
    specs = ws.load_specs(expdir)
    os.makedirs(os.path.join(expdir, ws.LOG_DIR), exist_ok=True)
    os.makedirs(os.path.join(expdir, ws.RENDER_SUBDIR), exist_ok=True)
    os.makedirs(os.path.join(expdir, "meshes"), exist_ok=True)
    if args.debug:
        os.environ['WANDB_MODE'] = 'offline'  # don't sync W&B with cloud
    wandb_project = specs.get("WandbProject", "inr")
    wandb.init(project=wandb_project, name=expdir, config=specs,
               dir=ws.get_log_dir(expdir), resume=True, 
               mode=("disabled" if args.debug else None))

    logging.info(f"Command:  python {' '.join(sys.argv)}")
    logging.info(f"Running experiment in {expdir}.")
    logging.info(f"arguments = {args}")

    for name, split in zip(["training", "valid", "test"],
                           [specs["TrainSplit"], specs["ValidSplit"], specs["TestSplit"]]):
        if split is None:
            continue
        # Load the data
        with open(split) as f:
            instances = json.load(f)
        # First 8 shapes
        instances = instances[:8]
        meshes = []
        for instance in instances:
            filename = os.path.join(specs["DataSource"], "meshes", instance+".obj")
            meshes.append(trimesh.load(filename))
        # Render and save
        renders = viz.render_meshes(meshes, size=224, aa_factor=2)
        ws.save_renders(expdir, renders, name)
        wandb.log({f"render/{name}": [wandb.Image(render) for render in renders]}, step=1)
        # wandb.log({f"point_cloud/{name}": [wandb.Object3D(mesh.sample(10000) if not mesh.is_empty else np.empty((0,3))) 
        #                                    for mesh in meshes]}, step=1)
        mesh_subdir = os.path.join(expdir, "meshes")
        for i, instance in enumerate(instances):
            meshes[i].export(os.path.join(mesh_subdir, instance + ".obj"))


if __name__ == "__main__":
    args = parser()
    main(args)