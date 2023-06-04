import ufl
import numpy as np
import math

from mpi4py import MPI

import dolfinx
from petsc4py import PETSc
from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion

class CustomMeshDeformation(HarmonicMeshMotion):
    def __init__(self, mesh, boundaries, bc_marerks_list, bc_function_list, mu, reset_reference = True, is_deformation = False):
        super().__init__(mesh, boundaries, bc_marerks_list, bc_function_list, reset_reference, is_deformation)
        self.mu = mu

    def __enter__(self):
        gdim = self._mesh.geometry.dim
        mu = self.mu

        # Compute shape parametrization such that 
        #   mu_1 defines the length of the horizontal edge,
        #   mu_2 the slanting (possibly vertical) edges, and
        #   mu_3 the angle between the oblique sides and the positive x-semiaxis
        self.shape_parametrization = self.solve()
        if self._is_deformation:
            self._mesh.geometry.x[:, :gdim] += self.shape_parametrization.x.array.reshape(self._reference_coordinates.shape[0], gdim)
        else:
            self._mesh.geometry.x[:, :gdim] = self.shape_parametrization.x.array.reshape(self._reference_coordinates.shape[0], gdim)
        
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh_data/changed_mesh.xdmf", "w") as mesh_file_xdmf:
            mesh_file_xdmf.write_mesh(self._mesh)
        
        return self
        
# Read unit square mesh with Triangular elements
mesh_comm = MPI.COMM_WORLD
gdim = 2
gmsh_model_rank = 0
mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh", mesh_comm, gmsh_model_rank, gdim = gdim)

# Mesh deformation parameters
mu = np.array([1.0, 2/np.sqrt(3), math.pi/3])
# mu = np.array([1.0, 1.0, math.pi/2])

# Boundary conditions for custom mesh deformation (not for problem)
def u_bc_bottom(x): return (mu[0] * x[0], x[1])
def u_bc_right(x): return (mu[0]* x[0] + np.cos(mu[2]) * mu[1] * x[1], np.sin(mu[2]) * mu[1] * x[1])
def u_bc_left(x): return (np.cos(mu[2]) * mu[1] * x[1] , np.sin(mu[2]) * mu[1] * x[1])
def u_bc_top(x): return (mu[0]* x[0] + np.cos(mu[2]) * mu[1], np.sin(mu[2]) * mu[1] +  0.0 * x[1])


# Boundary markers
boundary_markers = [1, 2, 3, 4]
bcs=list()

x_elem = ufl.VectorElement("CG", mesh.ufl_cell(), 2) # Ansatz space for velocity
q_elem = ufl.FiniteElement("CG", mesh.ufl_cell(), 1) # Ansatz space for pressure

v_elem = ufl.MixedElement([x_elem, q_elem])
W = dolfinx.fem.FunctionSpace(mesh, v_elem)

V0, _ = W.sub(0).collapse()

vq = ufl.TestFunction(W)
(v, q) = ufl.split(vq)

up = dolfinx.fem.Function(W)
(u, p) =ufl.split(up)

Re = 400
u_x = 1.0

nu = max(mu[0], mu[1]) / Re

def lid_velocity_expression(x):
    return np.stack((u_x * np.ones(x.shape[1]), np.zeros(x.shape[1])))

def noslip_velocity_expression(x):
    return np.zeros(x.shape)

passing_velocity = dolfinx.fem.Function(V0)
noslip = dolfinx.fem.Function(V0)
passing_velocity.interpolate(lid_velocity_expression)

dofs_bottom = dolfinx.fem.locate_dofs_topological((W.sub(0),V0), gdim -1, facet_tags.find(1))
dofs_right = dolfinx.fem.locate_dofs_topological((W.sub(0),V0), gdim -1, facet_tags.find(2))
dofs_left = dolfinx.fem.locate_dofs_topological((W.sub(0),V0), gdim -1, facet_tags.find(4))
dofs_top = dolfinx.fem.locate_dofs_topological((W.sub(0),V0), gdim -1, facet_tags.find(3))

bcs.append(dolfinx.fem.dirichletbc(passing_velocity, dofs_top, W.sub(0)))
bcs.append(dolfinx.fem.dirichletbc(noslip, dofs_bottom,W.sub(0)))
bcs.append(dolfinx.fem.dirichletbc(noslip, dofs_left,W.sub(0)))
bcs.append(dolfinx.fem.dirichletbc(noslip, dofs_right,W.sub(0)))

F = ufl.inner(ufl.grad(u)* u,v) * ufl.dx\
    + nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx\
    - ufl.div(v) * p * ufl.dx\
    - q * ufl.div(u) * ufl.dx

problem = dolfinx.fem.petsc.NonlinearProblem(F, up, bcs=bcs)

with CustomMeshDeformation(mesh, facet_tags, boundary_markers, [u_bc_bottom, u_bc_right, u_bc_top, u_bc_left], mu, reset_reference=True, is_deformation=False) as mesh_class:
    solver = dolfinx.nls.petsc.NewtonSolver(mesh.comm, problem)
    solver.max_it = 100
    solver.rtol = 1e-6

    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

    n, converged = solver.solve(up)
    (v_1, p_1) = up.split()
    v_1.name = "Velocity"
    p_1.name = "Pressure"

    assert(converged)

    with dolfinx.io.XDMFFile(mesh.comm, "results/lid_driven_cavity.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(v_1, 0.0)
        xdmf.write_function(p_1, 0.0)