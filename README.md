# Solving Differential Equation with (Science) Constrained Learning

All experiments presented in the paper can be reproduced by running the scripts available in the `scripts` folder. These scripts are organized as follows:

- **Unsupervised (PINN):** This category is further divided into three sub-folders:
  - `specific_bvp`: Scripts corresponding to the **Solving a specific BVP** section of the paper.
  - `parametric_solution`: Scripts corresponding to the **Solving Parametric Families of BVPs** section. 
  - `invariance`: Scripts corresponding to the **Leveraging invariance when solving BVPs** section.

- **Supervised (FNO):** This folder contains a subfolder `supervised_sol_bvp` with scripts for the **Supervised Solutions of BVPs** section of the paper.

Each folder is further organized by the specific PDEs considered, such as Burger's equation and the Navier-Stokes equation.


## Software Dependencies

1. **Install PyTorch:** 
    Install PyTorch according to your OS, system etc. (tested with PyTorch version 2.5.0 and python 3.11.11), e.g., `pip3 install torch torchvision torchaudio`
2. **Install other dependencies:**
   Install the required Python packages: `pip install -r requirements.txt`

A python3 environment can be created prior to this, e.g. `conda create -n scl python=3.11.11; conda activate scl`

## Citation

If you find this repository useful for your research, please consider citing our paper:

```bibtex
@inproceedings{
moro2025solving,
title={Solving Differential Equations with Constrained Learning},
author={Viggo Moro and Luiz F. O. Chamon},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=5KqveQdXiZ}
}
