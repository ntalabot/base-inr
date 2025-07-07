# Implicit Neural Representations
Template for Implicit Neural Representations (INR) projects, mostly with the Signed Distance Function (SDF) in mind.


## File structure
This repository is organized as follow:

    base-inr/
    ├── experiments/        <- Experimental config and results (untracked by default)
    │   └── template/
    │       └── specs.json  <- Template experimental config
    ├── notebooks/          <- Jupyter notebooks for testing code and visualizing results
    ├── scripts/            <- Python scripts
    └── src/                <- Source code


See below for the data files structure.


## Installation
This repo was tested with **Python 3.10** and **PyTorch 2.0.0+cu118**. If you wish to use different versions, you may need to adapt the following commands.

Install the requirements with
```bash
pip install --upgrade pip
pip install numpy==1.24.2 matplotlib scipy imageio pillow scikit-image trimesh libigl jupyterlab
```
Then, install PyTorch and PyTorch3d (which is used for renderings):
```bash
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118
pip install --no-index pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt200/download.html
```


## How to use
**Note:** the scripts can be launched from the main directory with:
```bash
python3 scripts/script.py [--option [VALUE]]
```
Please, refer to the script itself or use the `--help` options for details regarding its options.


### Data
The expected dataset structure is as follow:

    dataset/
    ├── meshes/        <- meshes in .obj format (normalized between [-1,1]^3)
    ├── samples/       <- 3D and SDF samples
    └── splits/        <- Split files (each is a list of instance names)
  

### Training
To train a model, first create an experiment directory, *e.g.*, under `experiments`, then copy there the template specifications `experiments/template/specs.json`, and adapt it as needed (such as data paths). Then, launch the training with:
```bash
python3 scripts/train.py <experiments/expdir>
```
The models, latents, and poses will be saved in `<experiments/expdir>`.


### Reconstruction and evaluation
After training, reconstruct the test shapes with: (omit `--test` to reconstruct the validation set)
```bash
python3 scripts/reconstruct.py <experiments/expdir> --test
```
The reconstructions (meshes, parts, latents, and poses) will be saved in `<experiments/expdir>/reconstruction/<epoch>/`.

Once that is done, you can evaluate them by running:
```bash
python3 scripts/evaluate.py <experiments/expdir> --test
```
The metric values will be saved per-shape under `<experiments/expdir>/evaluation/<epoch>/`.

---
### TODO
* [X] Update models to take (latent, xyz) without repeating the latents
* [X] Look into IoU metric and Winding-Number (`libigl`)
* [ ] ~~Speed up reconstruction function~~
* [X] Add `device` instead of defaulting to CUDA
* [X] Change all `register_buffer` to not have None
* [ ] ~~Make clamping part of SDF loss (make sure when it applies)?~~
* [ ] Restructure training (and inference/eval?) as `Trainer` classes
* [X] Docs: Expected data structure
* [ ] Make more general to arbitrary input dimension (e.g., to work in 3D and 2D)
* [ ] Rework `deepsdf` network classes into a single main one?
* [ ] Rework latents so that the training embeddings are part of the model?