## POD-ANN for the Stokes equation with geometric parametrization ##

### 1. Problem statement

We consider **Geometrically parametrized Stokes equation** as applied to flow over airfoil. We consider in our analysis naca0012 airfoil.

$$ \nu \nabla^2 u + \nabla p_{\rho} = f \ \text{in} \ \Omega(\mu) \ ,$$

$$ \nabla \cdot u = 0 \ \text{in} \ \Omega(\mu) \ .$$

$$ u_D = u_{i} = (1., \ 0.) \ \text{on} \ \Gamma_2 \ \text{Inlet boundary} \ .$$

$$ u_D = u_{i} = (1., \ 0.) \ \text{on} \ \Gamma_1 \cup \Gamma_4 \ \text{Free velocity boundary} \ .$$

$$ p = 0 \ \text{on} \ \Gamma_3 \ \text{Free boundary at atmospheric pressure} \ .$$

$$ u_D = u_{n} = (0, 0) \ \text{on} \ \Gamma_5 \cup \Gamma_6 \ \text{No slip boundary} \ .$$

**NOTE**: 

1. We have replaced negative of pressure in the equation for symmetric systems of equations. Also, we have not considered density in the pressure term.

2. We consider no gravity force in our analysis. 

3. The pressure gradient is scaled with density.

First a **reference domain** $\hat{\Omega}$, whose configuration is known entirely, is selected. The **parametric domain** $\Omega$ is obtained by specifying the boundary location the reference domain $\hat{\Omega}$ at given parameter $\mu$.

$$\hat{\Omega} \times \mu \to \Omega \ .$$

The reference domain $\hat{\Omega}$ considered in this problem is shown in the figure below. The mesh is constructed on the reference domain $\hat{\Omega}$. At a given parameter $\mu$, the mesh is deformed to obtain mesh of the parametric domain $\Omega$.

* **Reference domain**:

![alt text](https://github.com/Wells-Group/dlrbnicsx/blob/main/demo/stokes_flow_dlrbnicsx/mesh_data/domain.png)

* **Mesh and boundary markers**: 

1 (\Gamma_1): Bottom boundary
2 (\Gamma_2): Left boundary
3 (\Gamma_3): Right boundary
4 (\Gamma_4): Top boundary
5 (\Gamma_5): Airfoil upper boundary
6 (\Gamma_6): Airfoil lower boundary

![alt text](https://github.com/Wells-Group/dlrbnicsx/blob/main/demo/stokes_flow_dlrbnicsx/mesh_data/boundaries.png)

The domain $\Omega$ is parametrized by 2 geometric parameters $\mu = \lbrace \mu_0, \mu_1 \rbrace$. These parameters are used to deform the domain. We apply 

$$\text{On } \Gamma_5 \cup \Gamma_6: (x, y) \to (\mu_0 x, \mu_1 y) \ .$$

This boundary deformation is then propagated inside the entire domain using Harmonic mesh deformation.

### 2. Implementation

* **Mesh deformation**

The harmonic mesh deformation is performed using [MDFEniCSx](https://github.com/niravshah241/mdfenicsx).

* **Finite element**

Finite element implementation is performed using [FEniCSx](https://fenicsproject.org/) (dolfinx_stokes_flow.py). It demonstrates the computation of single finite element solution at given parameter set $\mu$.

* **POD-ANN implementation**


We use [RBniCSx](https://github.com/RBniCS/RBniCSx) for Proper Orthogonal Decomposition (POD) and [DLRBniCSx](https://github.com/niravshah241/dlrbnicsx) for Artificial Neural Network (ANN). MDFEniCSx is used to perform ```HarmonicMeshMotion```.

   - Parametric formulation ```ProblemOnDeformedDomain```

First, the ```ProblemOnDeformedDomain``` class, with ```solve``` method, is created to compute the finite element solution at a given parameter $\mu$. This class resembles to dolfinx implementation (dlrbnicsx_stokes_flow.py).

   - Reduced problem ```PODANNReducedProblem```

The ```PODANNReducedProblem``` class, with ```project_snapshot``` method, is used to project the computed finite element solution at parameter $\mu$, on the reduced basis space. ```PODANNReducedProblem``` class also contains attributes ```input_scaling_range```, ```output_scaling_range```, ```input_range``` and ```output_range``` which are used for teh scaling of ANN input-output. ```PODANNReducedProblem``` also contains attributes ```optimizer```, ```learning_rate```, ```loss_fn``` and ```regularisation``` which are used during training and validation of ANN.

After initialisation of the classes, POD is performed on the separate snapshot matrices of pressure and velocity. The snapshot matrix is constructed by computing finite element solutions (velocity and pressure) at each of the parameters given by ```generate_training_set```. Next, input training and validation dataset is created by sampling input parameters from the parameter set using ```generate_ann_input_set```. The output training and validation dataset is created using ```generate_ann_output_set``` which primarily uses the ```solve``` method from ```ProblemOnDeformedDomain``` class and ```project_snapshot``` method from ```PODANNReducedProblem``` class.

```CustomDataset``` scales the computed data. The ```DataLoader``` wraps an iterable around ```CustomDataset```.

**NOTE**: 
It should be noted that additional keyword arguments are provided for ```CustomDataset``` such as ```input_scaling_range```, ```output_scaling_range```, ```input_range``` and ```output_range```. This allows user to use different scaling range and different input-output data range for pressure and velocity.

During training, the data is loaded in batches and shuffled to improve learning of the ANN. During validation, the data is not divided into batches and is not shuffled, as no backpropagation is performed during validation. ANN model is constructed using ```HiddenLayersNet```. The training and validation of neural network is performed using methods ```train_nn``` and ```validate_nn``` respectively, for maximum number of epochs ```max_epochs``` or until the early stopping criteria is invoked. Finally, error analysis is performed using ```error_analysis```. ```error_analysis``` uses ```reconstruct_solution``` and ```norm_error``` methods from ```PODANNReducedProblem```. ```error_analysis``` also uses ```solve``` method from ```ProblemOnDeformedDomain```. ```online_nn``` is used to perform feed-forward of the ANN at parameter ```online_mu``` during the **online phase**.
