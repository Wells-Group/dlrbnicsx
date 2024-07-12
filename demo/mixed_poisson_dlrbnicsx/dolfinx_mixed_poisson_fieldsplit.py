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
Q, VQ_map = V0.collapse()
V1 = V.sub(1)
W, VW_map = V1.collapse()

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
a_cpp = dolfinx.fem.form(a)
l_cpp = dolfinx.fem.form(L)
A = dolfinx.fem.petsc.assemble_matrix(a_cpp, bcs=bcs)
A.assemble()
L = dolfinx.fem.petsc.assemble_vector(l_cpp)
dolfinx.fem.petsc.apply_lifting(L, [a_cpp], [bcs])
L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.petsc.set_bc(L, bcs)

# Solver setup
ksp = PETSc.KSP()
ksp.create(mesh.comm)
ksp.setOperators(A)
ksp.setType("fgmres")
pc = ksp.getPC()
pc.setType("fieldsplit")
# NOTE see https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.PC.CompositeType.html
pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
# CAUTION it is "assumed" that 1 means full see https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.PC.FieldSplitSchurFactType.html #petsc4py.PETSc.PC.FieldSplitSchurFactType
pc.setFieldSplitSchurFactType(1)
# CAUTION it is "assumed" that 3 means selfp see https://web.cels.anl.gov/projects/petsc/vault/petsc-3.20/docs/petsc4py/reference/petsc4py.PETSc.PC.FieldSplitSchurPreType.html
pc.setFieldSplitSchurPreType(3)
# ksp.getPC().setFactorSolverType("mumps")

# NOTE Since setFieldSplitIS for ISq is called zero-th and for ISw is called first --> subksps[0] corressponds to ISq and subksps[1] corressponds to ISw
ISq = PETSc.IS().createGeneral(VQ_map, mesh.comm)
ISw = PETSc.IS().createGeneral(VW_map, mesh.comm)
pc.setFieldSplitIS(("sigma",ISq))
pc.setFieldSplitIS(("u",ISw))
pc.setUp()

subksps = pc.getFieldSplitSubKSP()
print(subksps)
subksps[0].setType("cg")
subksps[0].getPC().setType("ilu")
subksps[0].rtol = 1.e-12
subksps[1].setType("cg")
subksps[1].getPC().setType("none")
subksps[1].rtol = 1.e-12
ksp.rtol = 1.e-8 # NOTE or ksp.setTolerances(1e-8) # rtol is first argument of setTolerances

# ksp.setConvergenceHistory()
ksp.setFromOptions()
w_h = dolfinx.fem.Function(V)
ksp.solve(L, w_h.vector)
print(f"Number of iterations: {ksp.getIterationNumber()}")
print(f"Convergence reason: {ksp.getConvergedReason()}")
# print(f"Convergence history: {ksp.getConvergenceHistory()}")
ksp.destroy()
w_h.x.scatter_forward()
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

# NOTE references
# https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.PC.FieldSplitSchurPreType.html#petsc4py.PETSc.PC.FieldSplitSchurPreType.SELFP
# https://fenicsproject.org/qa/5287/using-the-petsc-pcfieldsplit-in-fenics/
# https://fenicsproject.discourse.group/t/robustness-issue-two-the-same-runs-behave-differently/14347/3
# https://petsc.org/main/manualpages/PC/PCFieldSplitSetIS/
# https://gitlab.com/rafinex-external-rifle/fenicsx-pctools
