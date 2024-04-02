# Implicit Neural Representations
Template for Implicit Neural Representations (INR) projects, mostly with the Signed Distance Function (SDF) in mind.

## TODO
* [X] Update models to take (latent, xyz) without repeating the latents
* [X] Look into IoU metric and Winding-Number (`libigl`)
* [ ] ~~Speed up reconstruction function~~
* [X] Add `device` instead of defaulting to CUDA
* [ ] Restructure training (and inference/eval?) as `Trainer` classes
* [ ] Docs
  * [ ] Expected data structure
* [ ] Make more general to arbitrary input dimension (e.g., to work in 3D and 2D)
* [ ] Rework `deepsdf` network classes into a single main one?


## File structure
    base-inr/
    ├── experiments/        <- Experimental config and results (untracked by default)
    │   └── template/
    │       └── specs.json  <- Template experimental config
    ├── notebooks/          <- Jupyter notebooks for testing code and visualizing results
    ├── scripts/            <- Python scripts
    └── src/                <- Source code


## How to use
The scripts can be launched from the main directory with:
```bash
python scripts/script.py [--option [VALUE]]
```
