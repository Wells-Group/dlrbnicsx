import dolfinx
import ufl

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

# Read mesh
gdim = 2
comm = MPI.COMM_WORLD
gmsh_model_rank = 0
mesh, subdomains, boundaries = dolfinx.io.gmshio.read_from_msh(
    "mesh_data/mesh.msh", comm, gmsh_model_rank, gdim=gdim)

V_element = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
Q_element = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)

V = dolfinx.fem.FunctionSpace(mesh, V_element)
Q = dolfinx.fem.FunctionSpace(mesh, Q_element)

(v, q) = (ufl.TestFunction(V), ufl.TestFunction(Q))
(du, dp) = (ufl.TrialFunction(V), ufl.TrialFunction(Q))

(u, p) = (dolfinx.fem.Function(V), dolfinx.fem.Function(Q))

residual = [(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
            - ufl.inner(p, ufl.div(v)) * ufl.dx), ufl.inner(ufl.div(u), q) * ufl.dx]
residual_cpp = dolfinx.fem.form(residual)

jacobian = [[ufl.derivative(residual[0], u, du), ufl.derivative(residual[0], p, dp)],
            [ufl.derivative(residual[1], u, du), ufl.derivative(residual[1], p, dp)]]
jacobian_cpp = dolfinx.fem.form(jacobian)

u_top = dolfinx.fem.Function(V)
u_noSlip = dolfinx.fem.Function(V)

def top(x):
    return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))


def no_slip(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))


u_top.interpolate(top)
u_noSlip.interpolate(no_slip)

dofs_1 = dolfinx.fem.locate_dofs_topological(V, gdim-1, boundaries.find(1))
dofs_2 = dolfinx.fem.locate_dofs_topological(V, gdim-1, boundaries.find(2))
dofs_3 = dolfinx.fem.locate_dofs_topological(V, gdim-1, boundaries.find(3))
dofs_4 = dolfinx.fem.locate_dofs_topological(V, gdim-1, boundaries.find(4))

bc_left = dolfinx.fem.dirichletbc(u_noSlip, dofs_1)
bc_top = dolfinx.fem.dirichletbc(u_top, dofs_2)
bc_right = dolfinx.fem.dirichletbc(u_noSlip, dofs_3)
bc_bottom = dolfinx.fem.dirichletbc(u_noSlip, dofs_4)

bcs = [bc_left, bc_top, bc_right, bc_bottom]

def assemble_residual(snes, x, residual_vec):
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                  mode=PETSc.ScatterMode.FORWARD)
    with residual_vec.localForm() as residual_vec_local:
            residual_vec_local.set(0.0)
    dolfinx.fem.petsc.assemble_vector_block(residual_vec, residual_cpp,
                                            jacobian_cpp, bcs, x0=x,
                                            scale=-1.0)


def assemble_jacobian(snes, x, jacobian_mat, preconditioner_mat):
    jacobian_mat.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix_block(jacobian_mat, jacobian_cpp,
                                            bcs, diagonal=1.0)
    jacobian_mat.assemble()


snes = PETSc.SNES().create(mesh.comm)
snes.setTolerances(max_it=200)
snes.getKSP().setType("preonly")
snes.getKSP().getPC().setType("lu")
snes.getKSP().getPC().setFactorSolverType("mumps")
residual_vec = dolfinx.fem.petsc.create_vector_block(residual_cpp)
snes.setFunction(assemble_residual, residual_vec)
jacobian_mat = dolfinx.fem.petsc.create_matrix_block(jacobian_cpp)
snes.setJacobian(assemble_jacobian, J=jacobian_mat, P=None)
snes.setMonitor(lambda _, it, residual: print(it, residual))
solution = dolfinx.fem.petsc.create_vector_block(residual_cpp)

snes.solve(None, solution)
solution.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                     mode=PETSc.ScatterMode.FORWARD)
residual_vec.destroy()
jacobian_mat.destroy()
solution.destroy()
snes.destroy()

print(solution.getArray())
