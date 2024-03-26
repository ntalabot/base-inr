# Implicit Neural Representations
Template for Implicit Neural Representations (INR) projects, mostly with the Signed Distance Function (SDF) in mind.

## TODO
* [ ] Update models to take (latent, xyz) without repeating the latents
* [ ] Look into IoU metric and Winding-Number (`libigl`)
* [ ] Add `device` instead of defaulting to CUDA
* [ ] Speed up reconstruction function
* [ ] Restructure training (and inference/eval?) as `Trainer` class
* [ ] Docs
  * [ ] Expected data structure
* [ ] Make more generalize for arbitrary input dimension (e.g., to work in 3D and 2D)


## File structure
    base-inr/
    ├── experiments/        <- Experimental config and results (untracked by default)
    │   └── template/
    │       └── specs.json  <- Template experimental config
    ├── notebooks/          <- Jupyter notebooks for testing code and visualizing results
    ├── scripts/            <- Python scripts
    └── src/                <- Source code


## How to use
The scripts can be launched with:
```bash
python scripts/script.py [--option [VALUE]]
```