from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

import ufl
import dolfinx

from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion

# Read mesh
mesh_comm = MPI.COMM_WORLD
gdim = 2
gmsh_model_rank = 0
mesh, subdomains, boundaries = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh", mesh_comm,
                                    gmsh_model_rank, gdim=gdim)


V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
Q = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))
w = dolfinx.fem.Function(Q)
omega_1_cells = subdomains.find(1)
w.x.array[omega_1_cells] = np.full_like(omega_1_cells, 1., dtype=PETSc.ScalarType)

vol = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(w, w) * ufl.dx))
print(vol - 5.9501)
