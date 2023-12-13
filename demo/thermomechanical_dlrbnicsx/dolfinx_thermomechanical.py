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
gdim = 2
gmsh_model_rank = 0

mesh, subdomains, boundaries = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh", mesh_comm,
                                     gmsh_model_rank, gdim=gdim)

VT = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
uT, vT = ufl.TrialFunction(VT), ufl.TestFunction(VT)
uT_func = dolfinx.fem.Function(VT)
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=boundaries)

ds_sf = ds(11) + ds(20) + ds(21) + ds(22) + ds(23)
ds_bottom = ds(1) + ds(31)
ds_out = ds(30)
ds_sym = ds(5) + ds(9) + ds(12)
ds_top = ds(18) + ds(19) + ds(27) + ds(28) + ds(29)

x = ufl.SpatialCoordinate(mesh)
n_vec = ufl.FacetNormal(mesh)

sym_T = sympy.Symbol("sym_T") #  Sympy symbol for spline interpolation

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

# Geometric deformation boundary condition w.r.t. reference domain
# i.e. set reset_reference=True and is_deformation=True
# Parameter tuple (D_0, D_1, t_0, t_1)
mu_ref = [0.6438, 0.4313, 1., 0.5]  # reference geometry
mu = mu_ref # [0.45, 0.56, 0.9, 0.7] # [0.8, 0.55, 0.8, 0.4]  # Parametric geometry

def thermal_diffusivity_1(sym_T):
    conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 693.)), ufl.And(ufl.ge(sym_T, 673.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
    interps = [8.17484662576687e-6*sym_T**2 - 0.00926193251533741*sym_T + 18.0819438190184, 8.17484662576687e-6*sym_T**2 - 0.00926193251533741*sym_T + 18.0819438190184, 1.76073619631904e-6*sym_T**2 - 0.000628539877300632*sym_T + 15.176807196319, 1.76073619631904e-6*sym_T**2 - 0.000628539877300632*sym_T + 15.176807196319]
    assert len(conditions) == len(interps)
    d_func = ufl.conditional(conditions[0], interps[0], interps[0])
    for i in range(1, len(conditions)):
        d_func = ufl.conditional(conditions[i], interps[i], d_func)
    return d_func

def thermal_diffusivity_2(sym_T):
    conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 693.)), ufl.And(ufl.ge(sym_T, 673.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
    interps = [8.17484662576687e-6*sym_T**2 - 0.00926193251533741*sym_T + 18.0819438190184, 8.17484662576687e-6*sym_T**2 - 0.00926193251533741*sym_T + 18.0819438190184, 1.76073619631904e-6*sym_T**2 - 0.000628539877300632*sym_T + 15.176807196319, 1.76073619631904e-6*sym_T**2 - 0.000628539877300632*sym_T + 15.176807196319]
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

def thermal_diffusivity_4(sym_T):
    conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 693.)), ufl.And(ufl.ge(sym_T, 673.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
    interps = [8.17484662576687e-6*sym_T**2 - 0.00926193251533741*sym_T + 18.0819438190184, 8.17484662576687e-6*sym_T**2 - 0.00926193251533741*sym_T + 18.0819438190184, 1.76073619631904e-6*sym_T**2 - 0.000628539877300632*sym_T + 15.176807196319, 1.76073619631904e-6*sym_T**2 - 0.000628539877300632*sym_T + 15.176807196319]
    assert len(conditions) == len(interps)
    d_func = ufl.conditional(conditions[0], interps[0], interps[0])
    for i in range(1, len(conditions)):
        d_func = ufl.conditional(conditions[i], interps[i], d_func)
    return d_func

def thermal_diffusivity_5(sym_T):
    conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 693.)), ufl.And(ufl.ge(sym_T, 673.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
    interps = [8.17484662576687e-6*sym_T**2 - 0.00926193251533741*sym_T + 18.0819438190184, 8.17484662576687e-6*sym_T**2 - 0.00926193251533741*sym_T + 18.0819438190184, 1.76073619631904e-6*sym_T**2 - 0.000628539877300632*sym_T + 15.176807196319, 1.76073619631904e-6*sym_T**2 - 0.000628539877300632*sym_T + 15.176807196319]
    assert len(conditions) == len(interps)
    d_func = ufl.conditional(conditions[0], interps[0], interps[0])
    for i in range(1, len(conditions)):
        d_func = ufl.conditional(conditions[i], interps[i], d_func)
    return d_func

def thermal_diffusivity_6(sym_T):
    conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 693.)), ufl.And(ufl.ge(sym_T, 673.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
    interps = [8.17484662576687e-6*sym_T**2 - 0.00926193251533741*sym_T + 18.0819438190184, 8.17484662576687e-6*sym_T**2 - 0.00926193251533741*sym_T + 18.0819438190184, 1.76073619631904e-6*sym_T**2 - 0.000628539877300632*sym_T + 15.176807196319, 1.76073619631904e-6*sym_T**2 - 0.000628539877300632*sym_T + 15.176807196319]
    assert len(conditions) == len(interps)
    d_func = ufl.conditional(conditions[0], interps[0], interps[0])
    for i in range(1, len(conditions)):
        d_func = ufl.conditional(conditions[i], interps[i], d_func)
    return d_func

def thermal_diffusivity_7(sym_T):
    conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 693.)), ufl.And(ufl.ge(sym_T, 673.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
    interps = [8.17484662576687e-6*sym_T**2 - 0.00926193251533741*sym_T + 18.0819438190184, 8.17484662576687e-6*sym_T**2 - 0.00926193251533741*sym_T + 18.0819438190184, 1.76073619631904e-6*sym_T**2 - 0.000628539877300632*sym_T + 15.176807196319, 1.76073619631904e-6*sym_T**2 - 0.000628539877300632*sym_T + 15.176807196319]
    assert len(conditions) == len(interps)
    d_func = ufl.conditional(conditions[0], interps[0], interps[0])
    for i in range(1, len(conditions)):
        d_func = ufl.conditional(conditions[i], interps[i], d_func)
    return d_func

a_T = \
    ufl.inner(thermal_diffusivity_1(uT_func) * ufl.grad(uT_func), ufl.grad(vT))  * x[0] * dx(1) + \
    ufl.inner(thermal_diffusivity_2(uT_func) * ufl.grad(uT_func), ufl.grad(vT))  * x[0] * dx(2) + \
    ufl.inner(thermal_diffusivity_3(uT_func) * ufl.grad(uT_func), ufl.grad(vT))  * x[0] * dx(3) + \
    ufl.inner(thermal_diffusivity_4(uT_func) * ufl.grad(uT_func), ufl.grad(vT))  * x[0] * dx(4) + \
    ufl.inner(thermal_diffusivity_5(uT_func) * ufl.grad(uT_func), ufl.grad(vT))  * x[0] * dx(5) + \
    ufl.inner(thermal_diffusivity_6(uT_func) * ufl.grad(uT_func), ufl.grad(vT))  * x[0] * dx(6) + \
    ufl.inner(thermal_diffusivity_7(uT_func) * ufl.grad(uT_func), ufl.grad(vT))  * x[0] * dx(7) + \
    ufl.inner(h_cf * uT_func, vT) * x[0] * ds_sf + \
    ufl.inner(h_cout * uT_func, vT) * x[0] * ds_out + \
    ufl.inner(h_cbottom * uT_func, vT) * x[0] * ds_bottom

l_T = \
    ufl.inner(q_source, vT) * x[0] * dx + h_cf * vT * T_f * x[0] * ds_sf + h_cout * vT * T_out * x[0] * ds_out + \
    h_cbottom * vT * T_bottom * x[0] * ds_bottom - ufl.inner(q_top, vT) * x[0] * ds_top

problem = NonlinearProblem(a_T - l_T, uT_func, bcs=[])
solver = NewtonSolver(mesh.comm, problem)
solver.convergence_criterion = "incremental"

solver.rtol = 1e-10
solver.report = True
ksp = solver.krylov_solver
ksp.setFromOptions()
dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

n, converged = solver.solve(uT_func)
# print(f"Computed solution array: {uT_func.x.array}")
assert (converged)
print(f"Number of interations: {n:d}")

computed_file = "solution_nonlinear_thermomechanical_thermal/solution_computed.xdmf"
with dolfinx.io.XDMFFile(mesh.comm, computed_file, "w") as solution_file:
    solution_file.write_mesh(mesh)
    solution_file.write_function(uT_func)

'''
# TODO
1. interps of diffusivity 2 to diffusvity 7
2. geometric parametrization
3. dlrbnicx multiprocess format (serial, cpu parallel, gpu parallel)
4. time measurements
5. paper update
'''
