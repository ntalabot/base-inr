"""
Overfit training script.

Train a deep implicit shape representation model on a single shape.

The main modification is pre-loading all samples and sampling among them 
for each batch. This is done to avoid creating a new dataloader for each iteration.
"""

import os, os.path
import sys
import argparse
import logging
import time
import json

import numpy as np
import torch
import wandb


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import workspace as ws
from src.data import samples_from_tensor
from src.loss import get_loss_recon
from src.mesh import create_mesh
from src.model import get_model, get_latents
from src.optimizer import get_optimizer, get_scheduler
from src.utils import configure_logging, set_seed, clamp_sdf, get_gradient
from src import visualization as viz


def parser(argv=None):
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="Train a deep implicit representation network.")

    parser.add_argument("experiment", help="path to the experiment directory. If existing, will try to resume it (see --no-resume)")

    parser.add_argument('--debug', action='store_true', help="increase verbosity to print debugging messages")
    parser.add_argument('--idx', default=None, type=int, help="index of the shape to overfit in the training split")
    parser.add_argument('--load-epoch', type=int, default=None, help="specific epoch to resume from (will throw an error if not possible)")
    parser.add_argument('--no-resume', action='store_true', help="do not resume the experiment if existing and start form epoch 0")
    parser.add_argument('-q', '--quiet', dest="verbose", action='store_false',  help="disable verbosity and run in quiet mode")
    parser.add_argument('--reset', action='store_true', help="force a reset of the experimental directory")
    parser.add_argument('--seed', type=int, default=0, help="initial seed for the RNGs (default=0)")

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
    if args.reset:
        ws.reset_experiment_dir(expdir)
    else:
        ws.build_experiment_dir(expdir)
    configure_logging(args, os.path.join(ws.get_log_dir(expdir), "trainlog.txt"))
    if args.debug:
        os.environ['WANDB_MODE'] = 'offline'  # don't sync W&B with cloud
    wandb_project = specs.get("WandbProject", "inr")
    wandb.init(project=wandb_project, name=expdir, config=specs,
               dir=ws.get_log_dir(expdir), resume=True, 
               mode=("disabled" if args.debug else None))

    logging.info(f"Command:  python {' '.join(sys.argv)}")
    logging.info(f"Running experiment in {expdir}.")
    logging.info(f"arguments = {args}")
    
    # Data
    if args.idx is None:
        args.idx = specs.get("ShapeIndex", 0)
    n_samples = specs["SamplesPerScene"]
    with open(specs["TrainSplit"]) as f:
        instance = json.load(f)[args.idx]
    all_samples = np.load(os.path.join(specs["DataSource"], specs["SamplesDir"], instance, specs["SamplesFile"]))
    all_samples = {k: torch.from_numpy(all_samples[k]).float().cuda() for k in ['pos', 'neg']}

    logging.info(f"Overfitting shape {args.idx}: {instance}.")

    # Model and latent vectors (get a single latent vector)
    latent_dim = 0
    model = get_model(
        specs.get("Network", "DeepSDF"),
        **specs.get("NetworkSpecs", {}),
        latent_dim=latent_dim
    ).cuda()
    
    latent = torch.zeros(1, latent_dim).cuda()

    # If using pre-trained network and latents, load them (note: will get overwritten by existing checkpoints!)
    model_pretrain = specs.get("NetworkPretrained", None)
    if model_pretrain is not None:
        model.load_state_dict(torch.load(model_pretrain))

    logging.info(f"Model has {sum([x.nelement() for x in model.parameters()]):,} parameters." + \
                 (" (pretrained)" if model_pretrain is not None else ""))

    # Loss and optimizer
    loss_recon = get_loss_recon(specs.get("ReconLoss", "L1-Hard"), reduction='none')
    latent_reg = specs["LatentRegularizationLambda"]
    eikonal_lambda = specs.get("EikonalLossLambda", None)
    
    optimizer = get_optimizer(model, type=specs["Optimizer"].pop("Type"),
                              lrs=specs["Optimizer"].pop("LearningRates"),
                              kwargs=specs["Optimizer"])
    scheduler = get_scheduler(optimizer, **specs["LearningRateSchedule"])

    # Resume from checkpoint
    history = {'epoch': 0}
    if not args.no_resume:
        if args.load_epoch is None and os.path.isfile(ws.get_checkpoint_filename(expdir)):
            checkpoint = ws.load_checkpoint(expdir)
            ws.use_checkpoint(checkpoint, model, None, optimizer, scheduler)
            history = ws.load_history(expdir, checkpoint['epoch'])
            del checkpoint

        elif args.load_epoch is not None and os.path.isfile(os.path.join(expdir, ws.HISTORY_FILE)):
            ws.load_experiment(expdir, args.load_epoch, model, None, optimizer, scheduler)
            history = ws.load_history(expdir, args.load_epoch)

        if history['epoch'] > 0:
            logging.info(f"Loaded checkpoint from epoch={history['epoch']}.")
    
    # Prepare checkpointing and logging
    log_frequency = specs.get("LogFrequency", 10)
    snapshot_epochs = set(range(
        specs["SnapshotFrequency"],
        specs["NumEpochs"] + 1,
        specs["SnapshotFrequency"]
    ))
    for cp in specs["AdditionalSnapshots"]:
        snapshot_epochs.add(cp)
    render_frequency = specs.get("RenderFrequency", None)

    # Training parameters
    n_epochs = specs['NumEpochs']
    clampD = specs["ClampingDistance"]

    # Training
    loss_names = ['loss', 'loss_reg']
    if eikonal_lambda is not None and eikonal_lambda > 0.:
        loss_names += ['loss_eik']
    for key in loss_names + ['lr', 'lr_lat', 'lat_norm']:
        if key not in history:
            history[key] = []
    for epoch in range(history['epoch']+1, n_epochs+1):
        time_epoch = time.time()
        running_losses = {name: 0. for name in loss_names}
        model.train()
        optimizer.zero_grad()

        # Sample a batch of points
        xyz, sdf_gt = samples_from_tensor(all_samples['pos'], all_samples['neg'], n_samples)

        xyz = xyz.unsqueeze(0).requires_grad_(eikonal_lambda is not None and eikonal_lambda > 0.)  # 1xNx3
        sdf_gt = sdf_gt.unsqueeze(0)  # 1xNx1
        batch_latents = latent.view(1, 1, -1).expand(1, xyz.shape[1], 1)  # 1xNxL

        inputs = torch.cat([batch_latents, xyz], dim=-1)  # 1xNx(L+3)
        sdf_pred = model(inputs)
        sdf_pred_noclamp = sdf_pred
        if clampD is not None and clampD > 0.:
            sdf_pred = clamp_sdf(sdf_pred, clampD, ref=sdf_gt)
            sdf_gt = clamp_sdf(sdf_gt, clampD)
        
        loss = loss_recon(sdf_pred, sdf_gt).mean()
        running_losses['loss'] += loss.item()
        # Eikonal loss
        if eikonal_lambda is not None and eikonal_lambda > 0.:
            grads = get_gradient(xyz, sdf_pred_noclamp)
            loss_eikonal = (grads.norm(dim=-1) - 1.).square().mean()
            loss = loss + eikonal_lambda * loss_eikonal
            running_losses['loss_eik'] += loss_eikonal.item()
        # Latent regularization
        if latent_reg is not None and latent_reg > 0.:
            loss_reg = min(1, epoch / 100) * batch_latents[:,0,:].square().sum()
            loss = loss + latent_reg * loss_reg
            running_losses['loss_reg'] += loss_reg.item()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        history['epoch'] += 1
        for name in loss_names:
            history[name].append(running_losses[name])
        history["lr"].append(optimizer.state_dict()["param_groups"][0]["lr"])
        history["lr_lat"].append(optimizer.state_dict()["param_groups"][1]["lr"])

        # Apply lr-schedule
        if scheduler is not None:
            scheduler.step()
        
        # WandB
        for name in loss_names:
            wandb.log({f"loss/{name}": history[name][-1]}, step=epoch)
        for lr in ['lr', 'lr_lat']:
            wandb.log({f"learning_rate/{lr}": history[lr][-1]}, step=epoch)
        
        # Renders, snapshot, log and checkpoint
        if render_frequency is not None and epoch % render_frequency == 0:
            meshes = [create_mesh(model, latent)]
            renders = viz.render_meshes(meshes, size=224, aa_factor=2)
            ws.save_renders(expdir, renders, epoch)
            wandb.log({"render/training": [wandb.Image(render) for render in renders]}, step=epoch)
            # wandb.log({"point_cloud/training": [wandb.Object3D(mesh.sample(10000) if not mesh.is_empty else np.empty((0,3))) 
            #                                     for mesh in meshes]}, step=epoch)
        if epoch in snapshot_epochs:
            ws.save_experiment(expdir, epoch, model, None, optimizer, scheduler)
        if epoch % log_frequency == 0:
            ws.save_history(expdir, history)
            checkpoint = ws.build_checkpoint(epoch, model, None, optimizer, scheduler)
            ws.save_checkpoint(expdir, checkpoint)
            del checkpoint
        
        msg = f"Epoch {epoch}/{n_epochs}:"
        for name in loss_names:
            msg += f"{name}={history[name][-1]:.6f} - "
        msg = msg[:-3] + f" ({time.time() - time_epoch:.0f}s/epoch)"
        logging.info(msg)

    # End of training
    last_epoch = history['epoch']
    checkpoint = ws.build_checkpoint(last_epoch, model, None, optimizer, scheduler)
    ws.save_checkpoint(expdir, checkpoint)
    ws.save_history(expdir, history)
    ws.save_experiment(expdir, last_epoch, model, None, optimizer, scheduler)
    torch.cuda.empty_cache()  # release unused GPU memory

    duration = time.time() - start_time
    duration_msg = "{:.0f}h {:02.0f}min {:02.0f}s".format(duration // 3600, (duration // 60) % 60, duration % 60)
    logging.info(f"End of training after {duration_msg}.")
    logging.info(f"Results saved in {expdir}.")


if __name__ == "__main__":
    args = parser()
    main(args)