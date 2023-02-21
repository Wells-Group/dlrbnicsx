# Nonlinear Poisson equation with geometric parametrization

We consider **Geometrically parametrized Nonlinear Poisson equation**:

$$ - \nabla \left( exp(u (\mu))  \nabla (u(\mu))\right) = s \ \text{in} \ \Omega \ ,$$
$$ u(\mu) = y sin(x \pi) cos(y \pi) \ \text{on} \ \partial \Omega \ .$$

The source term $s$ is adjusted to reproduce the **actual solution**:

$$u(\mu) = y sin(x \pi) cos(y \pi) \ , \ x,y \in \Omega \ .$$

The reference domain considered in this problem is the unit square with boundaries as shown in the figure.

* **Reference domain**: Unit square with vertices (0,0) -- (0,1) -- (1,1) -- (1,0)
![alt text](https://github.com/Wells-Group/dlrbnicsx/blob/main/demo/poisson_non_linear_geometric_parametrization/mesh_data/domain.png)

* **Mesh and boundary markers**: 1: Bottom boundary, 2: Right boundary, 3: Top boundary, 4: Left boundary
![alt text](https://github.com/Wells-Group/dlrbnicsx/blob/main/demo/poisson_non_linear_geometric_parametrization/mesh_data/mesh_boundaries.png)

The domain $\Omega$ is parametrized by 3 geometric parameters $\mu = \lbrace \mu_0, \mu_1, \mu_2 \rbrace$. These parameters are used to deform the domain. The parameters $\mu_0$ and $\mu_1$ are used to deform the bottom and top boundaries. Specifically, the deformations applied are:
$$\text{On } \Gamma_1:  \left(0, \mu_0 sin(x \pi) \right)$$
$$\text{On } \Gamma_3:  \left(0, -\mu_1 sin(x \pi) \right)$$

The parameter $\mu_2$ is used to stretch the domain along x directions. Specifically, the deformation applied is:
$$\text{On } \Omega: \left( (\mu_2 - 1)x, 0 \right)$$
