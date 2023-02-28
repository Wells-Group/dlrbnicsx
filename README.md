# DLRBniCSx

PyTorch-RBniCSx-FEniCSx based open source library for deep learning based reduced order modelling.

## Dependencies

- DOLFINx (v0.6.0)
- [RBniCSx](https://github.com/RBniCS/RBniCSx)
  ```
  pip install git+https://github.com/RBniCS/RBniCSx.git
  ```
- ufl4rom
  ```
  pip install git+https://github.com/RBniCS/ufl4rom.git
  ```
- [MDFEniCSx](https://github.com/niravshah241/MDFEniCSx)
- [DLRBniCSx](https://github.com/Wells-Group/dlrbnicsx)
  ```
  pip install git+https://github.com/Wells-Group/dlrbnicsx.git
  ```
- [pytorch](https://github.com/pytorch/pytorch)
  ```
  pip install torch
  ```

### Other

- plotly
- Matplolib

## Overview

The finite element calculations are performed using dolfinx. We use RBniCSx for Proper Orthogonal Decomposition (POD) and construction of reduced basis dataset. Once the dataset has been constructed, typical workflow in DLRBniCSx is as follow: 

- Create training dataset and validation dataset using ```CustomDataset```
- Use datasets ```DataLoader``` as train_loader and valid_loader for easy access to samples
- Initialise neural network model using ```HiddenLayersNet```
- Use train_loader for training of the neural network using ```train_nn``` function and valid_loader for validation of the neural network using ```validate_nn``` function
- Perform error analysis using ```error_analysis``` function
- Compute reduced basis solution at a given online parameter using ```online_nn``` function

## Disclaimer

In downloading this SOFTWARE you are deemed to have read and agreed to
the following terms: This SOFT- WARE has been designed with an exclusive
focus on civil applications. It is not to be used for any illegal,
deceptive, misleading or unethical purpose or in any military
applications. This includes ANY APPLICATION WHERE THE USE OF THE
SOFTWARE MAY RESULT IN DEATH, PERSONAL INJURY OR SEVERE PHYSICAL OR
ENVIRONMENTAL DAMAGE. Any redistribution of the software must retain
this disclaimer. BY INSTALLING, COPYING, OR OTHERWISE USING THE
SOFTWARE, YOU AGREE TO THE TERMS ABOVE. IF YOU DO NOT AGREE TO THESE
TERMS, DO NOT INSTALL OR USE THE SOFTWARE.
