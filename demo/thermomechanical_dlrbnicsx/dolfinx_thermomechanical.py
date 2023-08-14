from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import sympy

import matplotlib.pyplot as plt
import time

import ufl
import dolfinx

from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion
# TODO Gometric parametrization

# Read mesh
mesh_comm = MPI.COMM_WORLD
gdim = 2
gmsh_model_rank = 0

mesh, subdomains, boundaries = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh", mesh_comm,
                                     gmsh_model_rank, gdim=gdim)

VT = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
uT, vT = ufl.TrialFunction(VT), ufl.TestFunction(VT)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=boundaries)

ds_sf = ds(11) + ds(20) + ds(21) + ds(22) + ds(23)
ds_bottom = ds(1) + ds(31)
ds_out = ds(30)
ds_sym = ds(5) + ds(9) + ds(12)
ds_top = ds(18) + ds(19) + ds(27) + ds(28) + ds(29)

x = ufl.SpatialCoordinate(mesh)

T_f = 1773.
T_out = 300.
T_bottom = 300.
h_cf = 2000.
h_cout = 200.
h_cbottom = 2000.
q_source = dolfinx.fem.Function(VT)
q_source.x.array[:] = 0.
q_top = dolfinx.fem.Function(VT)
q_top.x.array[:] = 0.

# TODO DG space for different material k, alpha, E, nu

QT = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))#(mesh, ("DG", 0))
thermal_conductivity_func = dolfinx.fem.Function(QT)
thermal_conductivity_func_diff = dolfinx.fem.Function(QT)
temperature_field = dolfinx.fem.Function(VT, name="Temperature (K)")
temperature_field.x.array[:] = 300.
solution_field = dolfinx.fem.Function(VT)
max_iteration = 10
rtol = 1.e-5
atol = 1.e-6

JaT = ufl.inner(thermal_conductivity_func * ufl.grad(uT), ufl.grad(vT)) * x[0] * ufl.dx + \
    ufl.inner(uT * thermal_conductivity_func_diff * ufl.grad(temperature_field), ufl.grad(vT)) * x[0] * ufl.dx
cT = ufl.inner(h_cf * uT, vT) * x[0] * ds_sf + \
    ufl.inner(h_cout * uT, vT) * x[0] * ds_out + \
    ufl.inner(h_cbottom * uT, vT) * x[0] * ds_bottom
JlT = ufl.inner(temperature_field * thermal_conductivity_func_diff * ufl.grad(temperature_field), ufl.grad(vT)) * x[0] * ufl.dx
lT = ufl.inner(q_source, vT) * x[0] * ufl.dx + h_cf * vT * T_f * x[0] * ds_sf + \
    h_cout * vT * T_out * x[0] * ds_out + h_cbottom * vT * T_bottom * x[0] * ds_bottom - ufl.inner(q_top, vT) * x[0] * ds_top
aT_cpp = dolfinx.fem.form(JaT + cT)
lT_cpp = dolfinx.fem.form(JlT + lT)
bcsT = []

for iteration in range(max_iteration):

    print(f"Iteration {iteration + 1}/{max_iteration}")

    thermal_conductivity_func.x.array[:] = np.sin(temperature_field.x.array[:]/1773.) # TODO actual sympy interpolation
    thermal_conductivity_func_diff.x.array[:] = np.cos(temperature_field.x.array[:]/1773.)/1773. # TODO actual sympy interpolation

    residual = abs(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(thermal_conductivity_func * ufl.grad(temperature_field),
                                                                                                  ufl.grad(vT)) * x[0] * ufl.dx +
    h_cf * ufl.inner(temperature_field - T_f, vT) * x[0] * ds_sf + h_cbottom * ufl.inner(temperature_field - T_bottom, vT) * x[0] * ds_bottom +
    h_cout * ufl.inner(temperature_field - T_out, vT) * x[0] * ds_out)), op=MPI.SUM))

    if iteration == 0:
        initial_residual = residual

    print(f"Residual: {residual/initial_residual}")

    if residual/initial_residual < rtol:
        print(f"Residual tolerance {rtol} reached in iterations {iteration}")
        break

    # Bilinear side assembly
    A = dolfinx.fem.petsc.assemble_matrix(aT_cpp, bcs=bcsT)
    A.assemble()

    # Linear side assembly
    L = dolfinx.fem.petsc.assemble_vector(lT_cpp)
    dolfinx.fem.petsc.apply_lifting(L, [aT_cpp], [bcsT])
    L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(L, bcsT)

    # Solver setup
    ksp = PETSc.KSP()
    ksp.create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setFromOptions()
    ksp.solve(L, solution_field.vector)
    solution_field.x.scatter_forward()
    print(solution_field.x.array)

    update_abs = \
        (np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(solution_field - temperature_field,
                                                                                   solution_field - temperature_field) * x[0] * ufl.dx)), op=MPI.SUM)))/\
        (np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(temperature_field, temperature_field) * x[0] * ufl.dx)), op=MPI.SUM)))

    print(f"Absolute update: {update_abs}")

    temperature_field.x.array[:] = solution_field.x.array

    if update_abs < atol:
        print(f"Solver tolerance {atol} reached in iterations {iteration + 1}")
        break

computed_file = "solution_nonlinear_thermomechanical_thermal/solution_computed.xdmf"
with dolfinx.io.XDMFFile(mesh.comm, computed_file, "w") as solution_file:
    solution_file.write_mesh(mesh)
    solution_file.write_function(temperature_field)

VM = dolfinx.fem.VectorFunctionSpace(mesh, ("CG", 1))
uM = ufl.TrialFunction(VM)
vM = ufl.TestFunction(VM)
displacement_field = dolfinx.fem.Function(VM, name="Displacement magnitude (m)")
rho = 77106.
g = 9.8
alpha = 1.e-6
ymax = mesh_comm.allreduce(np.max(mesh.geometry.x), op=MPI.MAX)
T0 = 300.
n_vec = ufl.FacetNormal(mesh)

def epsilon(u):
    return ufl.sym(ufl.grad(u))
def sigma(u, E = 1.e6, nu=0.2, epsilon=epsilon):
    lambda_ = E * nu / ((1 - 2 * nu) * (1 + nu))
    mu = E / (2 * (1 + nu))
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2*mu*epsilon(u)

aM = ufl.inner(sigma(uM), epsilon(vM)) * x[0] * ufl.dx
lM = (temperature_field - T0) * alpha * ufl.div(vM) * x[0] * ufl.dx - rho * g * (ymax-x[1]) * ufl.dot(vM, n_vec) * x[0] * ds_sf

aM_cpp = dolfinx.fem.form(aM)
lM_cpp = dolfinx.fem.form(lM)

dofs_bottom_1 = dolfinx.fem.locate_dofs_topological(VM.sub(1), gdim-1, boundaries.find(1))
dofs_bottom_31 = dolfinx.fem.locate_dofs_topological(VM.sub(1), gdim-1, boundaries.find(31))
dofs_sym_5 = dolfinx.fem.locate_dofs_topological(VM.sub(0), gdim-1, boundaries.find(5))
dofs_sym_9 = dolfinx.fem.locate_dofs_topological(VM.sub(0), gdim-1, boundaries.find(9))
dofs_sym_12 = dolfinx.fem.locate_dofs_topological(VM.sub(0), gdim-1, boundaries.find(12))

bc_bottom_1 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_bottom_1, VM.sub(1))
bc_bottom_31 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_bottom_31, VM.sub(1))
bc_sym_5 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_sym_5, VM.sub(0))
bc_sym_9 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_sym_9, VM.sub(0))
bc_sym_12 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_sym_12, VM.sub(0))

bcsM = [bc_bottom_1, bc_bottom_31, bc_sym_5, bc_sym_9, bc_sym_12]

# Bilinear side assembly
A = dolfinx.fem.petsc.assemble_matrix(aM_cpp, bcs=bcsM)
A.assemble()

# Linear side assembly
L = dolfinx.fem.petsc.assemble_vector(lM_cpp)
dolfinx.fem.petsc.apply_lifting(L, [aM_cpp], [bcsM])
L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.petsc.set_bc(L, bcsM)

# Solver setup
ksp = PETSc.KSP()
ksp.create(mesh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
ksp.setFromOptions()
ksp.solve(L, displacement_field.vector)
displacement_field.x.scatter_forward()

computed_file = "solution_nonlinear_thermomechanical_mechanical/solution_computed.xdmf"
with dolfinx.io.XDMFFile(mesh.comm, computed_file, "w") as solution_file:
    solution_file.write_mesh(mesh)
    solution_file.write_function(displacement_field)
