## POD-ANN for the Poisson equation with geometric parametrization ##

### 1. Problem statement

We consider **Geometrically parametrized Poisson equation**:

$$ - \nabla \cdot \left( \nabla (u(\mu))\right) = s \ \text{in} \ \Omega \ ,$$

$$u_D = u \ \text{on} \ \partial \Omega \ .$$

The source term $s$ is adjusted to reproduce the **actual solution** $u$:

$$u(\mu) = 1 + x^2 + 2 y^2 \ , \ x,y \in \Omega \ .$$

The reference domain $\hat{\Omega}$ considered in this problem is shown in the figure below. First a square with vertices (0, 0) -- (0, 10) -- (10, 10) -- (10, 0) is constructed. Next, bottom right quarter of the circle with center at (0, 10) and radius of 5 is cut from the square.

* **Reference domain**:

![alt text](https://github.com/Wells-Group/dlrbnicsx/blob/main/demo/poisson/mesh_data/domain.png)

* **Mesh and boundary markers**: 1: Bottom boundary, 2: Right boundary, 3: Top boundary, 4: Curved boundary, 5: Left boundary

![alt text](https://github.com/Wells-Group/dlrbnicsx/blob/main/demo/poisson/mesh_data/boundaries.png)

The domain $\Omega$ is parametrized by 2 geometric parameters $\mu = \lbrace \mu_0, \mu_1 \rbrace$. These parameters are used to deform the domain. The domain boundaries are deformed as:

$\text{On } \partial \hat{\Omega}: (\mu_0 x, \mu_1 y)$.

This boundary deformation is then propagated througth the entire domain using Harmonic mesh deformation.

### 2. Implementation

* **Mesh deformation**

The harmonic mesh deformation is performed using [MDFEniCSx](https://github.com/niravshah241/mdfenicsx).

* **Finite element**

Finite element implementation is performed using [FEniCSx](https://fenicsproject.org/). It demonstrates the computation of single finite element solution at given parameter set $\mu$.

* **POD-ANN implementation**


We use [RBniCSx](https://github.com/RBniCS/RBniCSx) for Proper Orthogonal Decomposition (POD) and [DLRBniCSx](https://github.com/niravshah241/dlrbnicsx) for Artificial Neural Network (ANN). MDFEniCSx is used to perform ```HarmonicMeshMotion```.

   - Parametric formulation ```ProblemOnDeformedDomain```

First, the ```ProblemOnDeformedDomain``` class, with ```solve``` method, is created to compute the finite element solution at a given parameter $\mu$. This class resembles to dolfinx implementation (dolfinx_linear_poisson.py).

   - Reduced problem ```PODANNReducedProblem```

The ```PODANNReducedProblem``` class, with ```project_snapshot``` method, is used to project the computed finite element solution at parameter $\mu$, on the reduced basis space. ```PODANNReducedProblem``` class also contains attributes ```input_scaling_range```, ```output_scaling_range```, ```input_range``` and ```output_range``` which are used for teh scaling of ANN input-output. ```PODANNReducedProblem``` also contains attributes ```optimizer```, ```learning_rate```, ```loss_fn``` and ```regularisation``` which are used during training and validation of ANN.

After initialisation of the classes, POD is performed on the snapshot matrix. The snapshot matrix is constructed by computing finite element solution at each of the parameters given by ```generate_training_set```. Next, input training and validation dataset is created by sampling input parameters from the parameter set using ```generate_ann_input_set```. The output training and validation dataset is created using ```generate_ann_output_set``` which primarily uses the ```solve``` method from ```ProblemOnDeformedDomain``` class and ```project_snapshot``` method from ```PODANNReducedProblem``` class.

```CustomDataset``` scales the computed data. The ```DataLoader``` wraps an iterable around ```CustomDataset```. During training, the data is loaded in batches and shuffled to improve learning of the ANN. During validation, the data is not divided into batches and is not shuffled, as no backpropagation is performed during validation. ANN model is constructed using ```HiddenLayersNet```. The training and validation of neural network is performed using methods ```train_nn``` and ```validate_nn``` respectively, for maximum number of epochs ```max_epochs``` or until the early stopping criteria is invoked. Finally, error analysis is performed using ```error_analysis```. ```error_analysis``` uses ```reconstruct_solution``` and ```norm_error``` methods from ```PODANNReducedProblem```. ```error_analysis``` also uses ```solve``` method from ```ProblemOnDeformedDomain```. ```online_nn``` is used to perform feed-forward of the ANN at parameter ```online_mu``` during the **online phase**.
