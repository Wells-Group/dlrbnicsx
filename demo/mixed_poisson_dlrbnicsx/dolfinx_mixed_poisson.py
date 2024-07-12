from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

from basix.ufl import element, mixed_element

import basix
import ufl
import dolfinx
from dolfinx.fem.petsc import LinearProblem

# Import mesh in dolfinx
# Boundary markers: x=1 is 22, x=0 is 30, y=1 is 26, y=0 is 18, z=1 is 31, z=0 is 1
gdim = 3
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
mesh, subdomains, boundaries = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/3d_mesh.msh", mesh_comm,
                                    gmsh_model_rank, gdim=gdim)

mu = np.array([-2., 0.5, 0.5, 0.5, 3.])

k = 1
Q_el = basix.ufl.element("BDMCF", mesh.basix_cell(), k)
P_el = basix.ufl.element("DG", mesh.basix_cell(), k - 1)
V_el = basix.ufl.mixed_element([Q_el, P_el])
V = dolfinx.fem.FunctionSpace(mesh, V_el)

(sigma, u) = ufl.TrialFunctions(V)
(tau, v) = ufl.TestFunctions(V)

x = ufl.SpatialCoordinate(mesh)
f = 10. * ufl.exp(-mu[0] * ((x[0] - mu[1]) * (x[0] - mu[1]) + (x[1] - mu[2]) * (x[1] - mu[2]) + (x[2] - mu[3]) * (x[2] - mu[3])))

ds = ufl.Measure("ds", domain=mesh, subdomain_data=boundaries)
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)

a = ufl.inner(sigma, tau) * dx + ufl.inner(u, ufl.div(tau)) * dx + ufl.inner(ufl.div(sigma), v) * dx
L = -ufl.inner(f, v) * dx

# Get subspace of V
V0 = V.sub(0)
Q, _ = V0.collapse()

dofs_x0 = dolfinx.fem.locate_dofs_topological((V0, Q), gdim-1, boundaries.find(30))

def f1(x):
    values = np.zeros((3, x.shape[1]))
    values[0, :] = np.sin(mu[4] * x[0])
    return values


f_h1 = dolfinx.fem.Function(Q)
f_h1.interpolate(f1)
bc_x0 = dolfinx.fem.dirichletbc(f_h1, dofs_x0, V0)

dofs_y0 = dolfinx.fem.locate_dofs_topological((V0, Q), gdim-1, boundaries.find(18))

def f2(x):
    values = np.zeros((3, x.shape[1]))
    values[1, :] = np.sin(mu[4] * x[1])
    return values


f_h2 = dolfinx.fem.Function(Q)
f_h2.interpolate(f2)
bc_y0 = dolfinx.fem.dirichletbc(f_h2, dofs_y0, V0)

dofs_z0 = dolfinx.fem.locate_dofs_topological((V0, Q), gdim-1, boundaries.find(1))

def f3(x):
    values = np.zeros((3, x.shape[1]))
    values[2, :] = np.sin(mu[4] * x[2])
    return values


f_h3 = dolfinx.fem.Function(Q)
f_h3.interpolate(f3)
bc_z0 = dolfinx.fem.dirichletbc(f_h3, dofs_z0, V0)

# NOTE
bcs = [bc_x0, bc_y0, bc_z0]

# TODO solver and preconditioner
problem = LinearProblem(
    a,
    L,
    bcs=bcs,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
)
try:
    w_h = problem.solve()
except PETSc.Error as e:  # type: ignore
    if e.ierr == 92:
        print("The required PETSc solver/preconditioner is not available. Exiting.")
        print(e)
        exit(0)
    else:
        raise e

sigma_h, u_h = w_h.split()
sigma_h = sigma_h.collapse()
u_h = u_h.collapse()

with dolfinx.io.XDMFFile(mesh.comm, "out_mixed_poisson/sigma.xdmf", "w") as sol_file:
    sol_file.write_mesh(mesh)
    sol_file.write_function(sigma_h)

with dolfinx.io.XDMFFile(mesh.comm, "out_mixed_poisson/u.xdmf", "w") as sol_file:
    sol_file.write_mesh(mesh)
    sol_file.write_function(u_h)

print(sigma_h.x.array, np.linalg.norm(sigma_h.x.array))
print(u_h.x.array, np.linalg.norm(u_h.x.array))
