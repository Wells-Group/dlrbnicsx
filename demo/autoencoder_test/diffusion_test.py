import ufl
import dolfinx
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

# Create mesh
gdim = 2
mesh_comm = MPI.COMM_WORLD
gmsh_model_rank = 0
mesh, subdomains, boundaries = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh", mesh_comm,
                                    gmsh_model_rank, gdim=gdim)

VD = dolfinx.fem.FunctionSpace(mesh, ("CG", 2))
uD, vD = ufl.TestFunction(VD), ufl.TrialFunction(VD)
u_prev = dolfinx.fem.Function(VD)
u = dolfinx.fem.Function(VD)
u_init = dolfinx.fem.Function(VD)
x = ufl.SpatialCoordinate(mesh)
t = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.))
dt = 0.2
t_start, t_final = 0, 1
f = dolfinx.fem.Function(VD)
f.interpolate(lambda x: t * np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]))

a = ufl.inner(uD, vD) * ufl.dx + dt * ufl.inner(ufl.grad(uD), ufl.grad(vD)) * ufl.dx
l = ufl.inner(u_init, vD) * ufl.dx - ufl.inner(dt * f, vD) * ufl.dx
a_form = dolfinx.fem.form(a)
l_form = dolfinx.fem.form(l)

dofs_1 = dolfinx.fem.locate_dofs_topological(VD, gdim-1, boundaries.find(1))
dofs_2 = dolfinx.fem.locate_dofs_topological(VD, gdim-1, boundaries.find(2))
dofs_3 = dolfinx.fem.locate_dofs_topological(VD, gdim-1, boundaries.find(3))
dofs_4 = dolfinx.fem.locate_dofs_topological(VD, gdim-1, boundaries.find(4))

u_dirichlet = dolfinx.fem.Function(VD)

bc_1 = dolfinx.fem.dirichletbc(u_dirichlet, dofs_1)
bc_2 = dolfinx.fem.dirichletbc(u_dirichlet, dofs_2)
bc_3 = dolfinx.fem.dirichletbc(u_dirichlet, dofs_3)
bc_4 = dolfinx.fem.dirichletbc(u_dirichlet, dofs_4)
bcs = [bc_1, bc_2, bc_3, bc_4]

A = assemble_matrix(a_form, bcs=[bcs])
A.assemble()
b = create_vector(l_form)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
