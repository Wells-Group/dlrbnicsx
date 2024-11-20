import ufl
import numpy as np
import math
import itertools

from mpi4py import MPI

import dolfinx
from petsc4py import PETSc
from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion
        
# Read unit square mesh with Triangular elements
mesh_comm = MPI.COMM_WORLD
gdim = 2
gmsh_model_rank = 0
mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh", mesh_comm, gmsh_model_rank, gdim = gdim)

# Mesh deformation parameters
mu = np.array([1.0, 2/np.sqrt(3), math.pi/3])

def generate_training_set(samples=[4, 4, 4]):
    # Todo: was sind das für Parameter?
    training_set_0 = np.linspace(1.0, 2.0, samples[0])
    training_set_1 = np.linspace(1.0, 2.0, samples[1])
    training_set_2 = np.linspace(np.pi/6, 5*np.pi/6, samples[2])
    training_set = np.array(list(itertools.product(training_set_0,
                                                   training_set_1,
                                                   training_set_2)))
    return training_set

parameters = generate_training_set()
mu = [1.0,1.0,np.pi/6]
#mu = np.round(mu, 8)


"""
Comment one of the following two options to choose between reset_reference=True and reset_reference=False
"""
# Boundary conditions for custom mesh deformation (not for problem) with reset_reference=True and is_deformation=False
def u_bc_bottom(x): return (mu[0] * x[0] , 0.0*x[1])
def u_bc_right(x): return (mu[0] + np.cos(mu[2]) * mu[1] * x[1] , np.sin(mu[2]) * mu[1] * x[1])
def u_bc_top(x): return (mu[0]* x[0] + np.cos(mu[2]) * mu[1], np.sin(mu[2]) * mu[1] +  0.0 * x[1])
def u_bc_left(x): return (np.cos(mu[2]) * mu[1] * x[1] , np.sin(mu[2]) * mu[1] * x[1])

# Boundary conditions for custom mesh deformation (not for problem) with reset_reference=False and is_deformation=False
def u_bc_bottom(x): return (mu[0] * x[0] / np.max(x[0]) , 0.0*x[1])
def u_bc_right(x): return (mu[0] + np.cos(mu[2]) * mu[1] * x[1]/ np.max(x[1]) , np.sin(mu[2]) * mu[1] * x[1] / np.max(x[1]) )
def u_bc_top(x): return (mu[0]* x[0] / np.max(x[0]) + np.cos(mu[2]) * mu[1] , np.sin(mu[2]) * mu[1] +  0.0 * x[1])
def u_bc_left(x): return (np.cos(mu[2]) * mu[1] * x[1] / np.max(x[1]) , np.sin(mu[2]) * mu[1] * x[1] / np.max(x[1]))

# Boundary markers
boundary_markers = [1, 2, 3, 4]
bcs=list()

x_elem = ufl.VectorElement("CG", mesh.ufl_cell(), 2) # Ansatz space for velocity
q_elem = ufl.FiniteElement("CG", mesh.ufl_cell(), 1) # Ansatz space for pressure

v_elem = ufl.MixedElement([x_elem, q_elem])
W = dolfinx.fem.FunctionSpace(mesh, v_elem)

V, _ = W.sub(0).collapse()

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

def reference_point(x):
    bools = np.isclose(x, np.asarray([[0.0]*x.shape[1]]*x.shape[0]))
    return np.logical_and(bools[0], bools[1])

passing_velocity = dolfinx.fem.Function(V)
noslip = dolfinx.fem.Function(V)
passing_velocity.interpolate(lid_velocity_expression)
reference = dolfinx.fem.Function(W.sub(1).collapse()[0])
reference.interpolate(lambda x: 0.0*x[0])

dofs_bottom = dolfinx.fem.locate_dofs_topological((W.sub(0),V), gdim -1, facet_tags.find(1))
dofs_right = dolfinx.fem.locate_dofs_topological((W.sub(0),V), gdim -1, facet_tags.find(2))
dofs_left = dolfinx.fem.locate_dofs_topological((W.sub(0),V), gdim -1, facet_tags.find(4))
dofs_top = dolfinx.fem.locate_dofs_topological((W.sub(0),V), gdim -1, facet_tags.find(3))
dofs_reference = dolfinx.fem.locate_dofs_geometrical((W.sub(1), W.sub(1).collapse()[0]), reference_point)

bcs.append(dolfinx.fem.dirichletbc(passing_velocity, dofs_top, W.sub(0)))
bcs.append(dolfinx.fem.dirichletbc(noslip, dofs_bottom,W.sub(0)))
bcs.append(dolfinx.fem.dirichletbc(noslip, dofs_left,W.sub(0)))
bcs.append(dolfinx.fem.dirichletbc(noslip, dofs_right,W.sub(0)))
bcs.append(dolfinx.fem.dirichletbc(reference, dofs_reference, W.sub(1)))

F = ufl.inner(ufl.grad(u)* u,v) * ufl.dx\
    + nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx\
    - ufl.div(v) * p * ufl.dx\
    - q * ufl.div(u) * ufl.dx

problem = dolfinx.fem.petsc.NonlinearProblem(F, up, bcs=bcs)

with HarmonicMeshMotion(mesh, facet_tags, boundary_markers, [u_bc_bottom, u_bc_right, u_bc_top, u_bc_left], reset_reference=False, is_deformation=False) as mesh_class:
    solver = dolfinx.nls.petsc.NewtonSolver(mesh.comm, problem)
    solver.max_it = 100
    solver.rtol = 1e-6

    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    print("Solving problem with parameters: ", mu)
    n, converged = solver.solve(up)
    (v_1, p_1) = up.split()
    v_1.name = "Velocity"
    p_1.name = "Pressure"

    assert(converged)

    with dolfinx.io.XDMFFile(mesh.comm, "results/lid_driven_cavity.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(v_1, 0.0)
        xdmf.write_function(p_1, 0.0)

mu = [2.0, 2.0 , 5*np.pi/6]
up.vector[:] = np.zeros(np.shape(up.vector[:]))


with HarmonicMeshMotion(mesh, facet_tags, boundary_markers, [u_bc_bottom, u_bc_right, u_bc_top, u_bc_left], reset_reference=False, is_deformation=False) as mesh_class:
    solver = dolfinx.nls.petsc.NewtonSolver(mesh.comm, problem)
    solver.max_it = 100
    solver.rtol = 1e-6

    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    print("Solving problem with parameters: ", mu)
    n, converged = solver.solve(up)
    (v_1, p_1) = up.split()
    v_1.name = "Velocity"
    p_1.name = "Pressure"

    assert(converged)

    with dolfinx.io.XDMFFile(mesh.comm, "results/lid_driven_cavity_2.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(v_1, 0.0)
        xdmf.write_function(p_1, 0.0)