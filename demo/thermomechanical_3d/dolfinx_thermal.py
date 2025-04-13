from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import sympy

import matplotlib.pyplot as plt
import time

import ufl
import dolfinx
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion

# Read mesh
mesh_comm = MPI.COMM_WORLD
gdim = 3
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
mesh, cell_tags, facet_tags = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh", mesh_comm,
                                    gmsh_model_rank, gdim=gdim)


VT = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
VT_plot = dolfinx.fem.FunctionSpace(mesh, ("CG", 2))
uT, vT = ufl.TrialFunction(VT), ufl.TestFunction(VT)
uT_func = dolfinx.fem.Function(VT)
uT_func_plot = dolfinx.fem.Function(VT_plot)
dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

ds_bottom = ds(4) + ds(10)
ds_outer_bottom = ds(5) + ds(11)
ds_outer_top = ds(15) + ds(19)
ds_1top = ds(8) + ds(14)
ds_2top = ds(16) + ds(20)
ds_3top = ds(23) + ds(26)
ds_inner = ds(22) + ds(25)

dx_sub_1 = dx(1) + dx(2)
dx_sub_2 = dx(3) + dx(4)
dx_sub_3 = dx(5) + dx(6)

T_f = 1773.
T_out = 300.
T_bottom = 300.
h_cf = 2000.
h_cout = 200.
h_cbottom = 200.
q_source = dolfinx.fem.Function(VT)
q_source.x.array[:] = 0.
q_top = dolfinx.fem.Function(VT)
q_top.x.array[:] = 0.

x = ufl.SpatialCoordinate(mesh)
n_vec = ufl.FacetNormal(mesh)
sym_T = sympy.Symbol("sym_T")  #  Sympy symbol for spline interpolation

# Geometric deformation boundary condition w.r.t. reference domain
# i.e. set reset_reference=True and is_deformation=True
# Parameter tuple (D_0, D_1, t_0, t_1)
# mu_ref = [0.6438, 0.4313, 1., 0.5]  # reference geometry
mu = [-0.27, 0.55, 0.8, 0.4] # Parametric geometry

def thermal_diffusivity_1(sym_T):
    conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 673.)), ufl.And(ufl.ge(sym_T, 673.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
    interps = [8.17484662576687e-6*sym_T**2 - 0.00926193251533741*sym_T + 18.0819438190184, 8.17484662576687e-6*sym_T**2 - 0.00926193251533741*sym_T + 18.0819438190184, 1.76073619631904e-6*sym_T**2 - 0.000628539877300632*sym_T + 15.176807196319, 1.76073619631904e-6*sym_T**2 - 0.000628539877300632*sym_T + 15.176807196319]
    assert len(conditions) == len(interps)
    d_func = ufl.conditional(conditions[0], interps[0], interps[0])
    for i in range(1, len(conditions)):
        d_func = ufl.conditional(conditions[i], interps[i], d_func)
    return d_func

def thermal_diffusivity_2(sym_T):
    conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 693.)), ufl.And(ufl.ge(sym_T, 673.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
    interps = [0.000299054192229039*sym_T**2 - 0.36574217791411*sym_T + 130.838954780164, 0.000299054192229039*sym_T**2 - 0.36574217791411*sym_T + 130.838954780164, -1.10434560327197e-5*sym_T**2 + 0.0516492566462166*sym_T - 9.61326294938646, -1.10434560327197e-5*sym_T**2 + 0.0516492566462166*sym_T - 9.61326294938646]
    assert len(conditions) == len(interps)
    d_func = ufl.conditional(conditions[0], interps[0], interps[0])
    for i in range(1, len(conditions)):
        d_func = ufl.conditional(conditions[i], interps[i], d_func)
    return d_func

def thermal_diffusivity_3(sym_T):
    conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 693.)), ufl.And(ufl.ge(sym_T, 673.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
    interps = [8.17484662576687e-6*sym_T**2 - 0.00926193251533741*sym_T + 18.0819438190184, 8.17484662576687e-6*sym_T**2 - 0.00926193251533741*sym_T + 18.0819438190184, 1.76073619631904e-6*sym_T**2 - 0.000628539877300632*sym_T + 15.176807196319, 1.76073619631904e-6*sym_T**2 - 0.000628539877300632*sym_T + 15.176807196319]
    assert len(conditions) == len(interps)
    d_func = ufl.conditional(conditions[0], interps[0], interps[0])
    for i in range(1, len(conditions)):
        d_func = ufl.conditional(conditions[i], interps[i], d_func)
    return d_func

a_T = ufl.inner(thermal_diffusivity_1(uT_func) * ufl.grad(uT_func), ufl.grad(vT)) * dx_sub_1 + \
      ufl.inner(thermal_diffusivity_2(uT_func) * ufl.grad(uT_func), ufl.grad(vT)) * dx_sub_2 + \
      ufl.inner(thermal_diffusivity_3(uT_func) * ufl.grad(uT_func), ufl.grad(vT)) * dx_sub_3 + \
      ufl.inner(h_cf * uT_func, vT) * (ds_inner + ds_1top) + \
      ufl.inner(h_cout * uT_func, vT) * (ds_outer_bottom + ds_outer_top) + \
      ufl.inner(h_cbottom * uT_func, vT) * ds_bottom

l_T = ufl.inner(q_source, vT) * (dx_sub_1 + dx_sub_2 + dx_sub_3) + \
      h_cf * vT * T_f * (ds_inner + ds_1top) + \
      h_cout * vT * T_out * (ds_outer_bottom + ds_outer_top) + \
      h_cbottom * vT * T_bottom * ds_bottom - \
      ufl.inner(q_top, vT) * (ds_2top + ds_3top)

uT_func.x.array[:] = 350.
uT_func.x.scatter_forward()
problem = NonlinearProblem(a_T - l_T, uT_func, bcs=[])

def bc_internal(x):
    return (mu[0] * np.sin(x[1] * 2. * np.pi), 0. * x[1], 0. * x[2])

def bc_external(x):
    return (0. * x[0], 0. * x[1], 0. * x[2])

# Mesh deformation (Harmonic mesh motion)
with HarmonicMeshMotion(mesh, facet_tags, [4, 10, 5, 11, 15, 19,
                                           8, 14, 16, 20, 23, 26,
                                           22, 25, 17, 21, 6, 12,
                                           7, 13],
                        [bc_external, bc_external, bc_external,
                         bc_external, bc_external, bc_external,
                         bc_external, bc_external, bc_external,
                         bc_external, bc_external, bc_external,
                         bc_external, bc_external, bc_internal,
                         bc_internal, bc_external, bc_external,
                         bc_external, bc_external],
                        reset_reference=True, is_deformation=True):
    solver = NewtonSolver(mesh.comm, problem)
    solver.convergence_criterion = "incremental"

    solver.rtol = 1e-10
    solver.report = True
    ksp = solver.krylov_solver
    ksp.setFromOptions()
    # dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

    thermal_start_time = time.process_time()
    n, converged = solver.solve(uT_func)
    thermal_end_time = time.process_time()
    # print(f"Computed solution array: {uT_func.x.array}")
    assert (converged)
    print(f"Number of iterations: {n:d}")

    uT_func_plot.interpolate(uT_func)
    uT_func_plot.x.scatter_forward()
    print(f"Temperature field norm: {mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(uT_func, uT_func) * ufl.dx + ufl.inner(ufl.grad(uT_func), ufl.grad(uT_func)) * ufl.dx)))}")
    computed_file = "solution_nonlinear_thermomechanical_thermal/solution_computed.xdmf"
    with dolfinx.io.XDMFFile(mesh.comm, computed_file, "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(uT_func_plot)

# Mechanical problem

VM = dolfinx.fem.VectorFunctionSpace(mesh, ("CG", 1))
VM_plot = dolfinx.fem.VectorFunctionSpace(mesh, ("CG", 2))
uM = ufl.TrialFunction(VM)
vM = ufl.TestFunction(VM)
uM_func = dolfinx.fem.Function(VM, name="Displacement")
uM_func_plot = dolfinx.fem.Function(VM_plot)
rho = 77106.
g = 9.8
T0 = 300.

def young_modulus_1(sym_T):
    conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 823.)), ufl.And(ufl.ge(sym_T, 823.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
    interps = [1698.65453106434*sym_T**2 - 2185320.53818741*sym_T + 10994471124.8516, 1698.65453106434*sym_T**2 - 2185320.53818741*sym_T + 10994471124.8516, -1586.66402849173*sym_T**2 + 3222313.81084183*sym_T + 8769229590.22603, -1586.66402849173*sym_T**2 + 3222313.81084183*sym_T + 8769229590.22603]
    assert len(conditions) == len(interps)
    d_func = ufl.conditional(conditions[0], interps[0], interps[0])
    for i in range(1, len(conditions)):
        d_func = ufl.conditional(conditions[i], interps[i], d_func)
    return d_func

def young_modulus_2(sym_T):
    conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 823.)), ufl.And(ufl.ge(sym_T, 823.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
    interps = [-547.091412742593*sym_T**2 - 2026218.83656489*sym_T + 16040649369.8061, -547.091412742593*sym_T**2 - 2026218.83656489*sym_T + 16040649369.8061, 8466.75900277084*sym_T**2 - 16863016.6205001*sym_T + 22145991657.8954, 8466.75900277084*sym_T**2 - 16863016.6205001*sym_T + 22145991657.8954]
    assert len(conditions) == len(interps)
    d_func = ufl.conditional(conditions[0], interps[0], interps[0])
    for i in range(1, len(conditions)):
        d_func = ufl.conditional(conditions[i], interps[i], d_func)
    return d_func

def young_modulus_3(sym_T):
    conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 823.)), ufl.And(ufl.ge(sym_T, 823.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
    interps = [-105360.110803325*sym_T**2 + 123741855.955679*sym_T + 30988696357.3406, -105360.110803325*sym_T**2 + 123741855.955679*sym_T + 30988696357.3406, 61686.9806094212*sym_T**2 - 151217656.509701*sym_T + 144134535736.845, 61686.9806094212*sym_T**2 - 151217656.509701*sym_T + 144134535736.845]
    assert len(conditions) == len(interps)
    d_func = ufl.conditional(conditions[0], interps[0], interps[0])
    for i in range(1, len(conditions)):
        d_func = ufl.conditional(conditions[i], interps[i], d_func)
    return d_func

poisson_ratio_1 = 0.3
poisson_ratio_2 = 0.2
poisson_ratio_3 = 0.1

thermal_expansion_coefficient_1 = 2.3e-6
thermal_expansion_coefficient_2 = 4.6e-6
thermal_expansion_coefficient_3 = 4.7e-6

ymax = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.))

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma_mech(u, young_modulus, poisson_ratio, uT_func):
    lambda_ = young_modulus(uT_func) * poisson_ratio / ((1. - 2. * poisson_ratio) * (1. + poisson_ratio))
    mu = young_modulus(uT_func) / (2. * (1. + poisson_ratio))
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

def sigma_thermal(young_modulus, poisson_ratio, thermal_coeff, T0, uT_func):
    lambda_ = young_modulus(uT_func) * poisson_ratio / ((1. - 2. * poisson_ratio) * (1. + poisson_ratio))
    mu = young_modulus(uT_func) / (2. * (1. + poisson_ratio))
    stress_thermal = (2 * mu + 3 * lambda_) * thermal_coeff * (uT_func - T0)
    return stress_thermal


aM = ufl.inner(sigma_mech(uM, young_modulus_1, poisson_ratio_1, uT_func), epsilon(vM)) * dx_sub_1 + \
     ufl.inner(sigma_mech(uM, young_modulus_2, poisson_ratio_2, uT_func), epsilon(vM)) * dx_sub_2 + \
     ufl.inner(sigma_mech(uM, young_modulus_3, poisson_ratio_3, uT_func), epsilon(vM)) * dx_sub_3

lM = \
      ufl.inner(sigma_thermal(young_modulus_1, poisson_ratio_1, thermal_expansion_coefficient_1, T0, uT_func), ufl.div(vM)) * dx_sub_1 + \
      ufl.inner(sigma_thermal(young_modulus_2, poisson_ratio_2, thermal_expansion_coefficient_2, T0, uT_func), ufl.div(vM)) * dx_sub_2 + \
      ufl.inner(sigma_thermal(young_modulus_3, poisson_ratio_3, thermal_expansion_coefficient_3, T0, uT_func), ufl.div(vM)) * dx_sub_3 - \
      rho * g * (ymax - x[1]) * ufl.dot(vM, n_vec) * (ds_inner + ds_1top)

aM_cpp = dolfinx.fem.form(aM)
lM_cpp = dolfinx.fem.form(lM)

dofs_bottom_x_4 = dolfinx.fem.locate_dofs_topological(VM.sub(0), gdim-1, facet_tags.find(4))
dofs_bottom_y_4 = dolfinx.fem.locate_dofs_topological(VM.sub(1), gdim-1, facet_tags.find(4))
dofs_bottom_z_4 = dolfinx.fem.locate_dofs_topological(VM.sub(2), gdim-1, facet_tags.find(4))
dofs_bottom_x_10 = dolfinx.fem.locate_dofs_topological(VM.sub(0), gdim-1, facet_tags.find(10))
dofs_bottom_y_10 = dolfinx.fem.locate_dofs_topological(VM.sub(1), gdim-1, facet_tags.find(10))
dofs_bottom_z_10 = dolfinx.fem.locate_dofs_topological(VM.sub(2), gdim-1, facet_tags.find(10))

bc_bottom_x_4 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_bottom_x_4, VM.sub(0))
bc_bottom_y_4 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_bottom_y_4, VM.sub(1))
bc_bottom_z_4 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_bottom_z_4, VM.sub(2))
bc_bottom_x_10 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_bottom_x_10, VM.sub(0))
bc_bottom_y_10 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_bottom_y_10, VM.sub(1))
bc_bottom_z_10 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_bottom_z_10, VM.sub(2))

dofs_top_y_16 = dolfinx.fem.locate_dofs_topological(VM.sub(1), gdim-1, facet_tags.find(16))
dofs_top_y_20 = dolfinx.fem.locate_dofs_topological(VM.sub(1), gdim-1, facet_tags.find(20))
dofs_top_y_23 = dolfinx.fem.locate_dofs_topological(VM.sub(1), gdim-1, facet_tags.find(23))
dofs_top_y_26 = dolfinx.fem.locate_dofs_topological(VM.sub(1), gdim-1, facet_tags.find(26))

bc_top_y_16 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_top_y_16, VM.sub(1))
bc_top_y_20 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_top_y_20, VM.sub(1))
bc_top_y_23 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_top_y_23, VM.sub(1))
bc_top_y_26 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_top_y_26, VM.sub(1))

bcsM = [bc_bottom_x_4, bc_bottom_y_4, bc_bottom_z_4,
        bc_bottom_x_10, bc_bottom_y_10, bc_bottom_z_10,
        bc_top_y_16, bc_top_y_20, bc_top_y_23, bc_top_y_26]

# Mesh deformation (Harmonic mesh motion)
with HarmonicMeshMotion(mesh, facet_tags, [4, 10, 5, 11, 15, 19,
                                           8, 14, 16, 20, 23, 26,
                                           22, 25, 17, 21, 6, 12,
                                           7, 13],
                        [bc_external, bc_external, bc_external,
                         bc_external, bc_external, bc_external,
                         bc_external, bc_external, bc_external,
                         bc_external, bc_external, bc_external,
                         bc_external, bc_external, bc_internal,
                         bc_internal, bc_external, bc_external,
                         bc_external, bc_external],
                        reset_reference=True, is_deformation=True):

    ymax.value = mesh_comm.allreduce(np.max(mesh.geometry.x[:, 1]), op=MPI.MAX)
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
    ksp.setType("cg")#("preonly")
    ksp.getPC().setType("gamg")#("lu")
    # ksp.getPC().setFactorSolverType("mumps")
    ksp.setFromOptions()
    mechanical_start_time = time.process_time()
    ksp.solve(L, uM_func.vector)
    mechanical_end_time = time.process_time()
    print(ksp.its)
    uM_func.x.scatter_forward()
    # print(displacement_field.x.array)
    print(f"Displacement field norm: {mesh_comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(uM_func, uM_func) * ufl.dx + ufl.inner(epsilon(uM_func), epsilon(uM_func)) * ufl.dx)))}")

    uM_func_plot.interpolate(uM_func)
    uM_func_plot.x.scatter_forward()
    computed_file = "solution_nonlinear_thermomechanical_mechanical/solution_computed.xdmf"
    with dolfinx.io.XDMFFile(mesh.comm, computed_file, "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(uM_func_plot)

print(f"Thermal solve time: {thermal_end_time - thermal_start_time}")
print(f"Mechanical solve time: {mechanical_end_time - mechanical_start_time}")

'''
# TODO s (Thermal problem)
1. Solver
2. Parametrised form
3. See and implement linearised equation (Jacobian-Residual form) in a_T and l_T, if needed
4. Hypre use

# TODO s (Mechanical problem)
1. Solver
2. Parametrised form
3. Check formulation
4. cg and amg set near nullspace from dolfinx demo: https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_elasticity.html
'''
