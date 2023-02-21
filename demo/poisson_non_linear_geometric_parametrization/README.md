## POD-ANN for the Nonlinear Poisson equation with geometric parametrization ##

### 1. Problem statement

We consider **Geometrically parametrized Nonlinear Poisson equation**:

$$ - \nabla \cdot \left( exp(u (\mu))  \nabla (u(\mu))\right) = s \ \text{in} \ \Omega \ ,$$

$$u_D = u \ \text{on} \ \partial \Omega \ .$$

The source term $s$ is adjusted to reproduce the **actual solution** $u$:

$$u(\mu) = y sin(x \pi) cos(y \pi) \ , \ x,y \in \Omega \ .$$

The reference domain considered in this problem is the unit square with boundaries as shown in the figure.

* **Reference domain**: Unit square with vertices (0,0) -- (0,1) -- (1,1) -- (1,0)
![alt text](https://github.com/Wells-Group/dlrbnicsx/blob/main/demo/poisson_non_linear_geometric_parametrization/mesh_data/domain.png)

* **Mesh and boundary markers**: 1: Bottom boundary, 2: Right boundary, 3: Top boundary, 4: Left boundary
![alt text](https://github.com/Wells-Group/dlrbnicsx/blob/main/demo/poisson_non_linear_geometric_parametrization/mesh_data/mesh_boundaries.png)

The domain $\Omega$ is parametrized by 3 geometric parameters $\mu = \lbrace \mu_0, \mu_1, \mu_2 \rbrace$. These parameters are used to deform the domain. The parameters $\mu_0$ and $\mu_1$ are used to deform the bottom and top boundaries. Specifically, the deformations applied are:
$$\text{On } \Gamma_1:  \left(0, \mu_0 sin(x \pi) \right)$$
$$\text{On } \Gamma_3:  \left(0, -\mu_1 sin(x \pi) \right)$$

The parameter $\mu_2$ is used to stretch the domain along x direction. Specifically, the deformation applied is:
$$\text{On } \Omega: \left( (\mu_2 - 1)x, 0 \right)$$

### 2. Implementation

* **Mesh deformation**

For the mesh deformation, first, the boundary deformation are applied on the boundaries $\Gamma_1$ and $\Gamma_3$. The harmonic extension is used to compute the shape parametrization corressponding to the deformation on the boundaries $\Gamma_1$ and $\Gamma_3$. The domain is then stretched along x-direction by factor $\mu_2$. The computed shape parametrization is then applied to the stretched domain. The mesh deformation is performed using [MDFEniCSx](https://github.com/niravshah241/mdfenicsx).

* **Finite element**

Finite element implementation is performed using dolfinx (dolfinx_nonlinear_poisson.py). It demonstrates the computation of single finite element solution at given parameter $\mu$.

* **POD-ANN implementation**

We use rbnicsx for Proper Orthogonal Decomposition (POD) and dlrbnicsx for Artificial Neural Network (ANN). First, ```CustomMeshDeformation``` class is created based on MDFEniCSx. Next, the ```ProblemOnDeformedDomain``` class, with ```solve``` method, is created to compute the finite element solution at given parameter $\mu$. The ```PODANNReducedProblem``` class, with ```project_snapshot``` method, is used to project the computed finite element solution at parameter $\mu$, on the reduced basis space. ```PODANNReducedProblem``` class also contains attributes ```input_scaling_range```, ```output_scaling_range```, ```input_range``` and ```output_range``` which are used for scaling of the training set. ```PODANNReducedProblem``` also contains attributes ```optimizer```, ```learning_rate```, ```loss_fn``` and ```regularisation``` which are used during training and validation of ANN.

After initialisation of the classes, POD is performed on the snapshot matrix. The snapshot matrix is constructed by computing finite element solution at each of the parameters given by ```generate_training_set```. Next, input training and validation dataset is created by sampling input parameters from the parameter set using ```generate_ann_input_set```. The output training and validation dataset is created using ```generate_ann_output_set``` which primarily uses the ```solve``` method from ```ProblemOnDeformedDomain``` class and ```project_snapshot``` method from ```PODANNReducedProblem``` class.

```CustomDataset``` scales the computed data. The ```DataLoader``` wraps an iterable around ```CustomDataset```. The data is stored in the directory ```ann_data```. During training, the data is loaded in batches and shuffled to improve learning of the ANN. During validation, the data is not loaded in batches and is not shuffled, as no backpropagation is performed during validation. ANN model is constructed using ```HiddenLayersNet```. The training and validation of neural network is performed using methods ```train_nn``` and ```validate_nn``` respectively, for maximum number of epochs ```max_epochs``` or until the early stopping criteria is invoked. Finally, error analysis is performed using ```error_analysis```. ```error_analysis``` uses ```reconstruct_solution``` and ```norm_error``` methods from ```PODANNReducedProblem```. ```error_analysis``` also uses ```solve``` method from ```ProblemOnDeformedDomain```. ```online_nn``` is used to perform feed-forward of the ANN during **online phase**.
