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
VT_plot = dolfinx.fem.FunctionSpace(mesh, ("CG", 2))
uT, vT = ufl.TrialFunction(VT), ufl.TestFunction(VT)
uT_func = dolfinx.fem.Function(VT)
uT_func_plot = dolfinx.fem.Function(VT_plot)
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
mu = [0.8, 0.55, 0.8, 0.4] # [0.45, 0.56, 0.9, 0.7] # Parametric geometry

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
    interps = [3.08018064076346e-5*sym_T**2 - 0.0376497392638036*sym_T + 31.7270693260054, 3.08018064076346e-5*sym_T**2 - 0.0376497392638036*sym_T + 31.7270693260054, -2.79311520109062e-6*sym_T**2 + 0.00756902522154049*sym_T + 16.5109550766871, -2.79311520109062e-6*sym_T**2 + 0.00756902522154049*sym_T + 16.5109550766871]
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
    interps = [0.000299054192229039*sym_T**2 - 0.36574217791411*sym_T + 130.838954780164, 0.000299054192229039*sym_T**2 - 0.36574217791411*sym_T + 130.838954780164, -1.10434560327197e-5*sym_T**2 + 0.0516492566462166*sym_T - 9.61326294938646, -1.10434560327197e-5*sym_T**2 + 0.0516492566462166*sym_T - 9.61326294938646]
    assert len(conditions) == len(interps)
    d_func = ufl.conditional(conditions[0], interps[0], interps[0])
    for i in range(1, len(conditions)):
        d_func = ufl.conditional(conditions[i], interps[i], d_func)
    return d_func

a_T = \
    ufl.inner(thermal_diffusivity_1(uT_func) * ufl.grad(uT_func), ufl.grad(vT))  * x[0] * dx(1) + \
    ufl.inner(thermal_diffusivity_2(uT_func) * ufl.grad(uT_func), ufl.grad(vT))  * x[0] * dx(2) + \
    ufl.inner(5.3 * ufl.grad(uT_func), ufl.grad(vT))  * x[0] * dx(3) + \
    ufl.inner(4.75 * ufl.grad(uT_func), ufl.grad(vT))  * x[0] * dx(4) + \
    ufl.inner(thermal_diffusivity_5(uT_func) * ufl.grad(uT_func), ufl.grad(vT))  * x[0] * dx(5) + \
    ufl.inner(45.6 * ufl.grad(uT_func), ufl.grad(vT))  * x[0] * dx(6) + \
    ufl.inner(thermal_diffusivity_7(uT_func) * ufl.grad(uT_func), ufl.grad(vT))  * x[0] * dx(7) + \
    ufl.inner(h_cf * uT_func, vT) * x[0] * ds_sf + \
    ufl.inner(h_cout * uT_func, vT) * x[0] * ds_out + \
    ufl.inner(h_cbottom * uT_func, vT) * x[0] * ds_bottom

l_T = \
    ufl.inner(q_source, vT) * x[0] * dx + h_cf * vT * T_f * x[0] * ds_sf + h_cout * vT * T_out * x[0] * ds_out + \
    h_cbottom * vT * T_bottom * x[0] * ds_bottom - ufl.inner(q_top, vT) * x[0] * ds_top

uT_func.x.array[:] = 350.
uT_func.x.scatter_forward()
problem = NonlinearProblem(a_T - l_T, uT_func, bcs=[])

bc_list_geometric = list()


def bc_1_geometric(x):
    indices_0 = np.where((x[0] >= 4.875) & (x[0] <= 5.5188))[0]
    indices_1 = np.where((x[0] >= 5.5188) & (x[0] <= 5.9501))[0]
    y = (0. * x[0], 0. * x[1])
    y[0][indices_0] = (x[0][indices_0] - 4.875) * (mu[0] - mu_ref[0]) / (5.5188 - 4.875)
    y[0][indices_1] = \
        (x[0][indices_1] - 5.5188) * (mu[1] - mu_ref[1]) / (5.9501 - 5.5188) + \
        (mu[0] - mu_ref[0]) * (x[0][indices_1] / x[0][indices_1])
    return y


bc_list_geometric.append(bc_1_geometric)


def bc_2_geometric(x):
    return ((mu[1] - mu_ref[1] + mu[0] - mu_ref[0]) * (x[0] / x[0]), (x[1] - 0.) * (mu[2] - mu_ref[2]) / (1. - 0.))


bc_list_geometric.append(bc_2_geometric)


def bc_3_geometric(x):
    indices_0 = np.where((x[0] >= 4.875) & (x[0] <= 5.5188))[0]
    indices_1 = np.where((x[0] >= 5.5188) & (x[0] <= 5.9501))[0]
    y = (0. * x[0], 0. * x[1])
    y[0][indices_0] = (x[0][indices_0] - 4.875) * (mu[0] - mu_ref[0]) / (5.5188 - 4.875)
    y[0][indices_1] = \
        (x[0][indices_1] - 5.5188) * (mu[1] - mu_ref[1]) / (5.9501 - 5.5188) + \
        (mu[0] - mu_ref[0]) * (x[0][indices_1] / x[0][indices_1])
    y[1][:] = (mu[2] - mu_ref[2]) * x[1]
    return y


bc_list_geometric.append(bc_3_geometric)


def bc_4_geometric(x):
    return (0. * x[0], (mu[2] - mu_ref[2]) * x[1])


bc_list_geometric.append(bc_4_geometric)


def bc_5_geometric(x):
    return (0. * x[0], (mu[2] - mu_ref[2]) * x[1])


bc_list_geometric.append(bc_5_geometric)


def bc_6_geometric(x):
    return (0. * x[0], (mu[2] - mu_ref[2]) * (x[1] / x[1]))


bc_list_geometric.append(bc_6_geometric)


def bc_7_geometric(x):
    return (0. * x[0], (mu[2] - mu_ref[2]) * (x[1] / x[1]))


bc_list_geometric.append(bc_7_geometric)


def bc_8_geometric(x):
    return (0. * x[0], (mu[2] - mu_ref[2]) * (x[1] / x[1]))


bc_list_geometric.append(bc_8_geometric)


def bc_9_geometric(x):
    return (0. * x[0], (mu[2] - mu_ref[2]) * (x[1] / x[1]))


bc_list_geometric.append(bc_9_geometric)


def bc_10_geometric(x):
    return (0. * x[0], (x[1] - 1.6) * (mu[3] - mu_ref[3]) / (2.1 - 1.6) + (mu[2] - mu_ref[2]) * (x[1] / x[1]))


bc_list_geometric.append(bc_10_geometric)


def bc_11_geometric(x):
    return (0. * x[0], (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * (x[1] / x[1]))


bc_list_geometric.append(bc_11_geometric)


def bc_12_geometric(x):
    return (0. * x[0], (x[1] - 1.6) * (mu[3] - mu_ref[3]) / (2.1 - 1.6) + (mu[2] - mu_ref[2]) * (x[1] / x[1]))


bc_list_geometric.append(bc_12_geometric)


def bc_13_geometric(x):
    return (0. * x[0], (mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_13_geometric)


def bc_14_geometric(x):
    indices_0 = np.where((x[1] >= 1.6) & (x[1] <= 2.1))[0]
    indices_1 = np.where(x[1] > 2.1)[0]
    y = (0. * x[0], 0. * x[1])
    y[1][indices_0] = (x[1][indices_0] - 1.6) * (mu[3] - mu_ref[3]) / (2.1 - 1.6) + \
        (mu[2] - mu_ref[2]) * x[1][indices_0] / x[1][indices_0]
    y[1][indices_1] = (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1][indices_1] / x[1][indices_1]
    return y


bc_list_geometric.append(bc_14_geometric)


def bc_15_geometric(x):
    return (0. * x[0], (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_15_geometric)


def bc_16_geometric(x):
    return ((x[0] - 4.875) * (mu[0] - mu_ref[0]) / (5.5188 - 4.875),
            (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_16_geometric)


def bc_17_geometric(x):
    return ((x[0] - 4.875) * (mu[0] - mu_ref[0]) / (5.5188 - 4.875),
            (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_17_geometric)


def bc_18_geometric(x):
    return ((x[0] - 4.875) * (mu[0] - mu_ref[0]) / (5.5188 - 4.875),
            (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_18_geometric)


def bc_19_geometric(x):
    return ((x[0] - 4.875) * (mu[0] - mu_ref[0]) / (5.5188 - 4.875),
            (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_19_geometric)


def bc_20_geometric(x):
    return ((x[0] - 4.875) * (mu[0] - mu_ref[0]) / (5.5188 - 4.875),
            (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_20_geometric)


def bc_21_geometric(x):
    return (0. * x[0], (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_21_geometric)


def bc_22_geometric(x):
    return (0. * x[0], (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_22_geometric)


def bc_23_geometric(x):
    return (0. * x[0], (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_23_geometric)


def bc_24_geometric(x):
    indices_0 = np.where(x[1] < 1.6)[0]
    indices_1 = np.where((x[1] >= 1.6) & (x[1] <= 2.1))[0]
    indices_2 = np.where(x[1] > 2.1)[0]
    y = (0. * x[0], 0. * x[1])
    y[1][indices_0] = (mu[2] - mu_ref[2]) * (x[1][indices_0] / x[1][indices_0])
    y[1][indices_1] = (mu[3] - mu_ref[3]) * (x[1][indices_1] - 1.6) / (2.1 - 1.6) + \
        (mu[2] - mu_ref[2]) * x[1][indices_1] / x[1][indices_1]
    y[1][indices_2] = (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1][indices_2] / x[1][indices_2]
    y[0][:] = (mu[1] - mu_ref[1] + mu[0] - mu_ref[0]) * (x[0] / x[0])
    return y


bc_list_geometric.append(bc_24_geometric)


def bc_25_geometric(x):
    indices_0 = np.where((x[0] >= 4.875) & (x[0] <= 5.5188))[0]
    indices_1 = np.where((x[0] >= 5.5188) & (x[0] <= 5.9501))[0]
    y = (0. * x[0], 0. * x[1])
    y[0][indices_0] = (x[0][indices_0] - 4.875) * (mu[0] - mu_ref[0]) / (5.5188 - 4.875)
    y[0][indices_1] = \
        (x[0][indices_1] - 5.5188) * (mu[1] - mu_ref[1]) / (5.9501 - 5.5188) + \
        (mu[0] - mu_ref[0]) * (x[1][indices_1] / x[1][indices_1])
    y[1][:] = (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1]
    return y


bc_list_geometric.append(bc_25_geometric)


def bc_26_geometric(x):
    return ((mu[1] - mu_ref[1] + mu[0] - mu_ref[0]) * x[0] / x[0],
            (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_26_geometric)


def bc_27_geometric(x):
    return ((x[0] - 5.5188) * (mu[1] - mu_ref[1]) / (5.9501 - 5.5188) + (mu[0] - mu_ref[0]) * (x[0] / x[0]),
            (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_27_geometric)


def bc_28_geometric(x):
    return ((mu[1] - mu_ref[1] + mu[0] - mu_ref[0]) * x[0] / x[0],
            (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_28_geometric)


def bc_29_geometric(x):
    return ((mu[1] - mu_ref[1] + mu[0] - mu_ref[0]) * x[0] / x[0],
            (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_29_geometric)


def bc_30_geometric(x):
    indices_0 = np.where(x[1] < 1.)[0]
    indices_1 = np.where((x[1] >= 1.) & (x[1] <= 1.6))[0]
    indices_2 = np.where((x[1] > 1.6) & (x[1] <= 2.1))[0]
    indices_3 = np.where(x[1] > 2.1)[0]
    y = (0 * x[0], 0. * x[1])
    y[0][:] = (mu[1] - mu_ref[1] + mu[0] - mu_ref[0]) * x[0] / x[0]
    y[1][indices_0] = (x[1][indices_0] - 0.) * (mu[2] - mu_ref[2]) / (1. - 0.)
    y[1][indices_1] = (mu[2] - mu_ref[2]) * (x[1][indices_1] / x[1][indices_1])
    y[1][indices_2] = (x[1][indices_2] - 1.6) * (mu[3] - mu_ref[3]) / (2.1 - 1.6) + \
        (mu[2] - mu_ref[2]) * x[1][indices_2] / x[1][indices_2]
    y[1][indices_3] = (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1][indices_3] / x[1][indices_3]
    return y


bc_list_geometric.append(bc_30_geometric)


def bc_31_geometric(x):
    return ((mu[1] - mu_ref[1] + mu[0] - mu_ref[0]) * x[0] / x[0], 0. * x[1])


bc_list_geometric.append(bc_31_geometric)

bc_markers_list = list(np.arange(1, 32))

# Thermal solve method
with HarmonicMeshMotion(mesh, boundaries, bc_markers_list,
                        bc_list_geometric, reset_reference=True,
                        is_deformation=True):

    solver = NewtonSolver(mesh.comm, problem)
    solver.convergence_criterion = "incremental"

    solver.rtol = 1e-10
    solver.report = True
    ksp = solver.krylov_solver
    ksp.setFromOptions()
    # dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

    n, converged = solver.solve(uT_func)
    # print(f"Computed solution array: {uT_func.x.array}")
    assert (converged)
    print(f"Number of interations: {n:d}")

    uT_func_plot.interpolate(uT_func)
    uT_func_plot.x.scatter_forward()
    print(f"Temperature field norm: {mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(uT_func, uT_func) * x[0] * ufl.dx + ufl.inner(ufl.grad(uT_func), ufl.grad(uT_func)) * x[0] * ufl.dx)))}")
    computed_file = "solution_nonlinear_thermomechanical_thermal/solution_computed.xdmf"
    with dolfinx.io.XDMFFile(mesh.comm, computed_file, "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(uT_func_plot)

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

def young_modulus_4(sym_T):
    conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 823.)), ufl.And(ufl.ge(sym_T, 823.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
    interps = [-781.855955678702*sym_T**2 + 927087.257617759*sym_T + 1645484985.45706, -781.855955678702*sym_T**2 + 927087.257617759*sym_T + 1645484985.45706, 656.925207756334*sym_T**2 - 1441146.53739632*sym_T + 2620013192.10535, 656.925207756334*sym_T**2 - 1441146.53739632*sym_T + 2620013192.10535]
    assert len(conditions) == len(interps)
    d_func = ufl.conditional(conditions[0], interps[0], interps[0])
    for i in range(1, len(conditions)):
        d_func = ufl.conditional(conditions[i], interps[i], d_func)
    return d_func

def young_modulus_5(sym_T):
    conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 823.)), ufl.And(ufl.ge(sym_T, 823.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
    interps = [1781.75702413907*sym_T**2 + 242712.70280987*sym_T + 14275923119.3114, 1781.75702413907*sym_T**2 + 242712.70280987*sym_T + 14275923119.3114, 1781.75702413907*sym_T**2 + 242712.70280987*sym_T + 14275923119.3114, 1781.75702413907*sym_T**2 + 242712.70280987*sym_T + 14275923119.3114]
    assert len(conditions) == len(interps)
    d_func = ufl.conditional(conditions[0], interps[0], interps[0])
    for i in range(1, len(conditions)):
        d_func = ufl.conditional(conditions[i], interps[i], d_func)
    return d_func

def young_modulus_7(sym_T):
    conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 823.)), ufl.And(ufl.ge(sym_T, 823.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
    interps = [-547.091412742593*sym_T**2 - 2026218.83656489*sym_T + 16040649369.8061, -547.091412742593*sym_T**2 - 2026218.83656489*sym_T + 16040649369.8061, 8466.75900277084*sym_T**2 - 16863016.6205001*sym_T + 22145991657.8954, 8466.75900277084*sym_T**2 - 16863016.6205001*sym_T + 22145991657.8954]
    assert len(conditions) == len(interps)
    d_func = ufl.conditional(conditions[0], interps[0], interps[0])
    for i in range(1, len(conditions)):
        d_func = ufl.conditional(conditions[i], interps[i], d_func)
    return d_func

young_modulus_6 = 1.9E11

poisson_ratio_1 = 0.3
poisson_ratio_2 = 0.2
poisson_ratio_3 = 0.1
poisson_ratio_4 = 0.1
poisson_ratio_5 = 0.2
poisson_ratio_6 = 0.3
poisson_ratio_7 = 0.2

thermal_expansion_coefficient_1 = 2.3e-6
thermal_expansion_coefficient_2 = 4.6e-6
thermal_expansion_coefficient_3 = 4.7e-6
thermal_expansion_coefficient_4 = 4.6e-6
thermal_expansion_coefficient_5 = 6.e-6
thermal_expansion_coefficient_6 = 1.2e-5
thermal_expansion_coefficient_7 = 4.6e-6

ymax = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.))

def epsilon(u, x):
    return ufl.as_tensor([[u[0].dx(0), 0.5*(u[0].dx(1)+u[1].dx(0)), 0.],[0.5*(u[0].dx(1)+u[1].dx(0)), u[1].dx(1), 0.],[0., 0., u[0]/x[0]]])

aM = \
    ufl.inner(young_modulus_1(uT_func) * poisson_ratio_1 / ((1 - 2 * poisson_ratio_1) * (1 + poisson_ratio_1)) * (uM[0].dx(0) + uM[1].dx(1) + uM[0]/x[0]) * ufl.Identity(3) + 2 * young_modulus_1(uT_func) / (2 * (1 + poisson_ratio_1)) * epsilon(uM, x), epsilon(vM, x)) * x[0] * dx(1) + \
    ufl.inner(young_modulus_2(uT_func) * poisson_ratio_2 / ((1 - 2 * poisson_ratio_2) * (1 + poisson_ratio_2)) * (uM[0].dx(0) + uM[1].dx(1) + uM[0]/x[0]) * ufl.Identity(3) + 2 * young_modulus_2(uT_func) / (2 * (1 + poisson_ratio_2)) * epsilon(uM, x), epsilon(vM, x)) * x[0] * dx(2) + \
    ufl.inner(young_modulus_3(uT_func) * poisson_ratio_3 / ((1 - 2 * poisson_ratio_3) * (1 + poisson_ratio_3)) * (uM[0].dx(0) + uM[1].dx(1) + uM[0]/x[0]) * ufl.Identity(3) + 2 * young_modulus_3(uT_func) / (2 * (1 + poisson_ratio_3)) * epsilon(uM, x), epsilon(vM, x)) * x[0] * dx(3) + \
    ufl.inner(young_modulus_4(uT_func) * poisson_ratio_4 / ((1 - 2 * poisson_ratio_4) * (1 + poisson_ratio_4)) * (uM[0].dx(0) + uM[1].dx(1) + uM[0]/x[0]) * ufl.Identity(3) + 2 * young_modulus_4(uT_func) / (2 * (1 + poisson_ratio_4)) * epsilon(uM, x), epsilon(vM, x)) * x[0] * dx(4) + \
    ufl.inner(young_modulus_5(uT_func) * poisson_ratio_5 / ((1 - 2 * poisson_ratio_5) * (1 + poisson_ratio_5)) * (uM[0].dx(0) + uM[1].dx(1) + uM[0]/x[0]) * ufl.Identity(3) + 2 * young_modulus_5(uT_func) / (2 * (1 + poisson_ratio_5)) * epsilon(uM, x), epsilon(vM, x)) * x[0] * dx(5) + \
    ufl.inner(young_modulus_6 * poisson_ratio_6 / ((1 - 2 * poisson_ratio_6) * (1 + poisson_ratio_6)) * (uM[0].dx(0) + uM[1].dx(1) + uM[0]/x[0]) * ufl.Identity(3) + 2 * young_modulus_6 / (2 * (1 + poisson_ratio_6)) * epsilon(uM, x), epsilon(vM, x)) * x[0] * dx(6) + \
    ufl.inner(young_modulus_7(uT_func) * poisson_ratio_7 / ((1 - 2 * poisson_ratio_7) * (1 + poisson_ratio_7)) * (uM[0].dx(0) + uM[1].dx(1) + uM[0]/x[0]) * ufl.Identity(3) + 2 * young_modulus_7(uT_func) / (2 * (1 + poisson_ratio_7)) * epsilon(uM, x), epsilon(vM, x)) * x[0] * dx(7)

lM = \
    (uT_func - T0) * young_modulus_1(uT_func) /( 1 - 2 * poisson_ratio_1) * thermal_expansion_coefficient_1 * (vM[0].dx(0) + vM[1].dx(1) + vM[0]/x[0]) * x[0] * dx(1) + \
    (uT_func - T0) * young_modulus_2(uT_func) /( 1 - 2 * poisson_ratio_2) * thermal_expansion_coefficient_2 * (vM[0].dx(0) + vM[1].dx(1) + vM[0]/x[0]) * x[0] * dx(2) + \
    (uT_func - T0) * young_modulus_3(uT_func) /( 1 - 2 * poisson_ratio_3) * thermal_expansion_coefficient_3 * (vM[0].dx(0) + vM[1].dx(1) + vM[0]/x[0]) * x[0] * dx(3) + \
    (uT_func - T0) * young_modulus_4(uT_func) /( 1 - 2 * poisson_ratio_4) * thermal_expansion_coefficient_4 * (vM[0].dx(0) + vM[1].dx(1) + vM[0]/x[0]) * x[0] * dx(4) + \
    (uT_func - T0) * young_modulus_5(uT_func) /( 1 - 2 * poisson_ratio_5) * thermal_expansion_coefficient_5 * (vM[0].dx(0) + vM[1].dx(1) + vM[0]/x[0]) * x[0] * dx(5) + \
    (uT_func - T0) * young_modulus_6 /( 1 - 2 * poisson_ratio_6) * thermal_expansion_coefficient_6 * (vM[0].dx(0) + vM[1].dx(1) + vM[0]/x[0]) * x[0] * dx(6) + \
    (uT_func - T0) * young_modulus_7(uT_func) /( 1 - 2 * poisson_ratio_7) * thermal_expansion_coefficient_7 * (vM[0].dx(0) + vM[1].dx(1) + vM[0]/x[0]) * x[0] * dx(7) - \
    rho * g * (ymax - x[1]) * ufl.dot(vM, n_vec) * x[0] * ds_sf

aM_cpp = dolfinx.fem.form(aM)
lM_cpp = dolfinx.fem.form(lM)

dofs_bottom_1 = dolfinx.fem.locate_dofs_topological(VM.sub(1), gdim-1, boundaries.find(1))
dofs_bottom_31 = dolfinx.fem.locate_dofs_topological(VM.sub(1), gdim-1, boundaries.find(31))
dofs_sym_5 = dolfinx.fem.locate_dofs_topological(VM.sub(0), gdim-1, boundaries.find(5))
dofs_sym_9 = dolfinx.fem.locate_dofs_topological(VM.sub(0), gdim-1, boundaries.find(9))
dofs_sym_12 = dolfinx.fem.locate_dofs_topological(VM.sub(0), gdim-1, boundaries.find(12))
dofs_bottom_1_2 = dolfinx.fem.locate_dofs_topological(VM.sub(0), gdim-1, boundaries.find(1))
dofs_bottom_31_2 = dolfinx.fem.locate_dofs_topological(VM.sub(0), gdim-1, boundaries.find(31))
dofs_top_18 = dolfinx.fem.locate_dofs_topological(VM.sub(0), mesh.geometry.dim-1, boundaries.find(18))
dofs_top_28 = dolfinx.fem.locate_dofs_topological(VM.sub(0), mesh.geometry.dim-1, boundaries.find(28))
dofs_top_19 = dolfinx.fem.locate_dofs_topological(VM.sub(1), mesh.geometry.dim-1, boundaries.find(19))
dofs_top_27 = dolfinx.fem.locate_dofs_topological(VM.sub(1), mesh.geometry.dim-1, boundaries.find(27))
dofs_top_29 = dolfinx.fem.locate_dofs_topological(VM.sub(1), mesh.geometry.dim-1, boundaries.find(29))

bc_bottom_1 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_bottom_1, VM.sub(1))
bc_bottom_31 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_bottom_31, VM.sub(1))
bc_sym_5 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_sym_5, VM.sub(0))
bc_sym_9 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_sym_9, VM.sub(0))
bc_sym_12 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_sym_12, VM.sub(0))
bc_bottom_1_2 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_bottom_1_2, VM.sub(0))
bc_bottom_31_2 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_bottom_31_2, VM.sub(0))
bc_top_18 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_top_18, VM.sub(0))
bc_top_28 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_top_28, VM.sub(0))
bc_top_19 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_top_19, VM.sub(1))
bc_top_27 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_top_27, VM.sub(1))
bc_top_29 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_top_29, VM.sub(1))

bcsM = [bc_bottom_1, bc_bottom_31, bc_sym_5, bc_sym_9, bc_sym_12, bc_bottom_1_2, bc_bottom_31_2,
        bc_top_18, bc_top_28, bc_top_19, bc_top_27, bc_top_29]

# Mechanical solve method
with HarmonicMeshMotion(mesh, boundaries, bc_markers_list,
                        bc_list_geometric, reset_reference=True,
                        is_deformation=True):

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
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setFromOptions()
    ksp.solve(L, uM_func.vector)
    uM_func.x.scatter_forward()
    # print(displacement_field.x.array)
    print(f"Displacement field norm: {mesh_comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(uM_func, uM_func) * x[0] * ufl.dx + ufl.inner(epsilon(uM_func, x), epsilon(uM_func, x)) * x[0] * ufl.dx)))}")

    uM_func_plot.interpolate(uM_func)
    uM_func_plot.x.scatter_forward()
    computed_file = "solution_nonlinear_thermomechanical_mechanical/solution_computed.xdmf"
    with dolfinx.io.XDMFFile(mesh.comm, computed_file, "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(uM_func_plot)

print(f"Displacement field norm: {mesh_comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(uM_func, uM_func) * x[0] * ufl.dx + ufl.inner(epsilon(uM_func, x), epsilon(uM_func, x)) * x[0] * ufl.dx)))}")

'''
# TODO
1. dlrbnicx multiprocess format (serial, cpu parallel, gpu parallel)
2. time measurements
3. paper update
'''
