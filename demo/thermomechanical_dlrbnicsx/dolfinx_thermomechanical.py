from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import sympy

import matplotlib.pyplot as plt
import time

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

VT = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
uT, vT = ufl.TrialFunction(VT), ufl.TestFunction(VT)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=boundaries)

ds_sf = ds(11) + ds(20) + ds(21) + ds(22) + ds(23)
ds_bottom = ds(1) + ds(31)
ds_out = ds(30)
ds_sym = ds(5) + ds(9) + ds(12)
ds_top = ds(18) + ds(19) + ds(27) + ds(28) + ds(29)

omega_1_cells = subdomains.find(1)
omega_2_cells = subdomains.find(2)
omega_3_cells = subdomains.find(3)
omega_4_cells = subdomains.find(4)
omega_5_cells = subdomains.find(5)
omega_6_cells = subdomains.find(6)
omega_7_cells = subdomains.find(7)

x = ufl.SpatialCoordinate(mesh)
n_vec = ufl.FacetNormal(mesh)

sym_T = sympy.Symbol("sym_T") #  Sympy symbol for spline interpolation

# Geometric deformation boundary condition w.r.t. reference domain
# i.e. set reset_reference=True and is_deformation=True
# Parameter tuple (D_0, D_1, t_0, t_1)
mu_ref = [0.6438, 0.4313, 1., 0.5]  # reference geometry
mu = [0.45, 0.56, 0.9, 0.7] # [0.8, 0.55, 0.8, 0.4]  # Parametric geometry

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

# ### Thermal model ###
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

# Temperature_field t previous iteration
temperature_field = dolfinx.fem.Function(VT,name="Temperature")
temperature_field.x.array[:] = 350. # Inital guess 350 K
# temperature field at current iteration
solution_field = dolfinx.fem.Function(VT)

# Material properties
Q = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))
thermal_conductivity_func = dolfinx.fem.Function(Q)
thermal_conductivity_func_diff = dolfinx.fem.Function(Q)
thermal_conductivity_func_1 = dolfinx.fem.Function(Q)
thermal_conductivity_func_diff_1 = dolfinx.fem.Function(Q)
thermal_conductivity_func_2 = dolfinx.fem.Function(Q)
thermal_conductivity_func_diff_2 = dolfinx.fem.Function(Q)
thermal_conductivity_func_5 = dolfinx.fem.Function(Q)
thermal_conductivity_func_diff_5 = dolfinx.fem.Function(Q)
thermal_conductivity_func_7 = dolfinx.fem.Function(Q)
thermal_conductivity_func_diff_7 = dolfinx.fem.Function(Q)

# Weak form
JaT = ufl.inner(thermal_conductivity_func * ufl.grad(uT), ufl.grad(vT)) * x[0] * ufl.dx + \
    ufl.inner(uT * thermal_conductivity_func_diff * ufl.grad(temperature_field), ufl.grad(vT)) * x[0] * ufl.dx
cT = ufl.inner(h_cf * uT, vT) * x[0] * ds_sf + ufl.inner(h_cout * uT, vT) * x[0] * ds_out + \
    ufl.inner(h_cbottom * uT, vT) * x[0] * ds_bottom
JlT = ufl.inner(temperature_field * thermal_conductivity_func_diff * ufl.grad(temperature_field), ufl.grad(vT)) * x[0] * ufl.dx
lT = ufl.inner(q_source, vT) * x[0] * ufl.dx + h_cf * vT * T_f * x[0] * ds_sf + \
    h_cout * vT * T_out * x[0] * ds_out + h_cbottom * vT * T_bottom * x[0] * ds_bottom - ufl.inner(q_top, vT) * x[0] * ds_top
aT_cpp = dolfinx.fem.form(JaT + cT)
lT_cpp = dolfinx.fem.form(JlT + lT)
bcsT = []

# ### Material property interpolation ###
# Conductivity for subdomain 1 ==================

thermal_conductivity_sym_1 = sympy.interpolating_spline(2, sym_T, [293., 473., 873., 1273.], [16.07, 15.53, 15.97, 17.23])
thermal_conductivity_sym_1 = sympy.Piecewise(
    thermal_conductivity_sym_1.args[0], (thermal_conductivity_sym_1.args[1][0], True))
thermal_conductivity_sym_1_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_1)
thermal_conductivity_sym_diff_1 = sympy.diff(thermal_conductivity_sym_1, sym_T)
thermal_conductivity_sym_diff_1_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_diff_1)


def conductivity_eval_1(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return thermal_conductivity_sym_1_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])


def conductivity_eval_diff_1(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return thermal_conductivity_sym_diff_1_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

temp_vec = np.linspace(293., 1800., 152)
thermal_conductivity_sym_diff_1_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_diff_1)
k_1_vec = np.empty_like(temp_vec)
k_1_diff_vec = np.empty_like(temp_vec)
for i in range(len(temp_vec)):
    k_1_vec[i] = thermal_conductivity_sym_1_lambdified(temp_vec[i])
    k_1_diff_vec[i] = thermal_conductivity_sym_diff_1_lambdified(temp_vec[i])

plt.figure(figsize=[10, 10])
plt.plot(temp_vec, k_1_vec, "*r", label="Thermal conductivity 1")
plt.plot(temp_vec, k_1_diff_vec, "-b", label="Thermal conductivity Diff 1")
plt.legend(loc="best")
plt.savefig("k_1")

# Conductivity for subdomain 2 ==================

thermal_conductivity_sym_2 = sympy.interpolating_spline(2, sym_T, [293., 473., 873., 1273.], [49.35, 24.75, 27.06, 38.24])
thermal_conductivity_sym_2 = sympy.Piecewise(
    thermal_conductivity_sym_2.args[0], (thermal_conductivity_sym_2.args[1][0], True))
thermal_conductivity_sym_2_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_2)
thermal_conductivity_sym_diff_2 = sympy.diff(thermal_conductivity_sym_2, sym_T)
thermal_conductivity_sym_diff_2_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_diff_2)


def conductivity_eval_2(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return thermal_conductivity_sym_2_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])


def conductivity_eval_diff_2(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return thermal_conductivity_sym_diff_2_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

temp_vec = np.linspace(293., 1800., 152)
thermal_conductivity_sym_diff_2_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_diff_2)
k_2_vec = np.empty_like(temp_vec)
k_2_diff_vec = np.empty_like(temp_vec)
for i in range(len(temp_vec)):
    k_2_vec[i] = thermal_conductivity_sym_2_lambdified(temp_vec[i])
    k_2_diff_vec[i] = thermal_conductivity_sym_diff_2_lambdified(temp_vec[i])

plt.figure(figsize=[10, 10])
plt.plot(temp_vec, k_2_vec, "*r", label="Thermal conductivity 2")
plt.plot(temp_vec, k_2_diff_vec, "-b", label="Thermal conductivity Diff 2")
plt.legend(loc="best")
plt.savefig("k_2")

# Conductivity for subdomain 5  ==================

thermal_conductivity_sym_5 = sympy.interpolating_spline(2, sym_T, [293., 473., 873., 1273.], [23.34, 20.81, 20.99, 21.62])
thermal_conductivity_sym_5 = sympy.Piecewise(
    thermal_conductivity_sym_5.args[0], (thermal_conductivity_sym_5.args[1][0], True))
thermal_conductivity_sym_5_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_5)
thermal_conductivity_sym_diff_5 = sympy.diff(thermal_conductivity_sym_5, sym_T)
thermal_conductivity_sym_diff_5_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_diff_5)


def conductivity_eval_5(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return thermal_conductivity_sym_5_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])


def conductivity_eval_diff_5(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return thermal_conductivity_sym_diff_5_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

temp_vec = np.linspace(293., 1800., 152)
k_5_vec = np.empty_like(temp_vec)
k_5_diff_vec = np.empty_like(temp_vec)
for i in range(len(temp_vec)):
    k_5_vec[i] = thermal_conductivity_sym_5_lambdified(temp_vec[i])
    k_5_diff_vec[i] = thermal_conductivity_sym_diff_5_lambdified(temp_vec[i])

plt.figure(figsize=[10, 10])
plt.plot(temp_vec, k_5_vec, "*r", label="Thermal conductivity 5")
plt.plot(temp_vec, k_5_diff_vec, "-b", label="Thermal conductivity Diff 5")
plt.legend(loc="best")
plt.savefig("k_5")

# Conductivity for subdomain 7  ==================
thermal_conductivity_sym_7 = sympy.interpolating_spline(2, sym_T, [293., 473., 873., 1273.], [49.35, 24.75, 27.06, 38.24])
thermal_conductivity_sym_7 = sympy.Piecewise(
    thermal_conductivity_sym_7.args[0], (thermal_conductivity_sym_7.args[1][0], True))
thermal_conductivity_sym_7_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_7)
thermal_conductivity_sym_diff_7 = sympy.diff(thermal_conductivity_sym_7, sym_T)
thermal_conductivity_sym_diff_7_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_diff_7)


def conductivity_eval_7(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return thermal_conductivity_sym_7_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])


def conductivity_eval_diff_7(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return thermal_conductivity_sym_diff_7_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

temp_vec = np.linspace(293., 1800., 152)
k_7_vec = np.empty_like(temp_vec)
k_7_diff_vec = np.empty_like(temp_vec)
for i in range(len(temp_vec)):
    k_7_vec[i] = thermal_conductivity_sym_7_lambdified(temp_vec[i])
    k_7_diff_vec[i] = thermal_conductivity_sym_diff_7_lambdified(temp_vec[i])

plt.figure(figsize=[10, 10])
plt.plot(temp_vec, k_7_vec, "*r", label="Thermal conductivity 7")
plt.plot(temp_vec, k_7_diff_vec, "-b", label="Thermal conductivity Diff 7")
plt.legend(loc="best")
plt.savefig("k_7")

# Conductivity for subdomains 3, 4, 6  ==================
thermal_conductivity_func.x.array[omega_3_cells] = 5.3
thermal_conductivity_func.x.array[omega_4_cells] = 4.75
thermal_conductivity_func.x.array[omega_6_cells] = 45.6

# Solver values
max_iteration = 10
rtol = 1.e-4
atol = 1.e-12

# Thermal solve method
with HarmonicMeshMotion(mesh, boundaries, bc_markers_list,
                        bc_list_geometric, reset_reference=True,
                        is_deformation=True):

    for iteration in range(max_iteration):

        print(f"Iteration {iteration + 1}/{max_iteration}")

        thermal_conductivity_func_1.interpolate(conductivity_eval_1)
        thermal_conductivity_func_diff_1.interpolate(conductivity_eval_diff_1)

        thermal_conductivity_func_2.interpolate(conductivity_eval_2)
        thermal_conductivity_func_diff_2.interpolate(conductivity_eval_diff_2)

        thermal_conductivity_func_5.interpolate(conductivity_eval_5)
        thermal_conductivity_func_diff_5.interpolate(conductivity_eval_diff_5)

        thermal_conductivity_func_7.interpolate(conductivity_eval_7)
        thermal_conductivity_func_diff_7.interpolate(conductivity_eval_diff_7)

        thermal_conductivity_func.x.array[omega_1_cells] = thermal_conductivity_func_1.x.array[omega_1_cells]
        thermal_conductivity_func_diff.x.array[omega_1_cells] = thermal_conductivity_func_diff_1.x.array[omega_1_cells]

        thermal_conductivity_func.x.array[omega_2_cells] = thermal_conductivity_func_2.x.array[omega_2_cells]
        thermal_conductivity_func_diff.x.array[omega_2_cells] = thermal_conductivity_func_diff_2.x.array[omega_2_cells]

        thermal_conductivity_func.x.array[omega_5_cells] = thermal_conductivity_func_5.x.array[omega_5_cells]
        thermal_conductivity_func_diff.x.array[omega_5_cells] = thermal_conductivity_func_diff_5.x.array[omega_5_cells]

        thermal_conductivity_func.x.array[omega_7_cells] = thermal_conductivity_func_7.x.array[omega_7_cells]
        thermal_conductivity_func_diff.x.array[omega_7_cells] = thermal_conductivity_func_diff_7.x.array[omega_7_cells]

        residual = abs(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(thermal_conductivity_func * ufl.grad(temperature_field),
                                                                                                    ufl.grad(vT)) * x[0] * ufl.dx +
        h_cf * ufl.inner(temperature_field - T_f, vT) * x[0] * ds_sf + h_cbottom * ufl.inner(temperature_field - T_bottom, vT) * x[0] * ds_bottom +
        h_cout * ufl.inner(temperature_field - T_out, vT) * x[0] * ds_out)), op=MPI.SUM))

        if iteration == 0:
            initial_residual = residual

        # print(f"Residual: {residual/initial_residual}")
        print(f"Residual: {residual}")
        
        '''
        if residual/initial_residual < rtol:
            print(f"Residual tolerance {rtol} reached in iterations {iteration}")
            break
        '''

        if residual < rtol:
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

        temperature_field.x.array[:] = solution_field.x.array.copy()

        if update_abs < atol:
            print(f"Solver tolerance {atol} reached in iterations {iteration + 1}")
            break

    computed_file = "solution_nonlinear_thermomechanical_thermal/solution_computed.xdmf"
    with dolfinx.io.XDMFFile(mesh.comm, computed_file, "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(temperature_field)

# ### Mechanical model ###
VM = dolfinx.fem.VectorFunctionSpace(mesh, ("CG", 1))
uM = ufl.TrialFunction(VM)
vM = ufl.TestFunction(VM)
displacement_field = dolfinx.fem.Function(VM, name="Displacement")
rho = 77106.
g = 9.8
ymax = mesh_comm.allreduce(np.max(mesh.geometry.x), op=MPI.MAX)
T0 = 300.

# ### Material property interpolation
# Young's modulus for subdomain 1 ==================
young_modulus_sym_1 = sympy.interpolating_spline(2, sym_T, [293., 573., 1073., 1273.], [10.5e9, 10.3e9, 10.4e9, 10.3e9])
young_modulus_sym_1 = sympy.Piecewise(
    young_modulus_sym_1.args[0], (young_modulus_sym_1.args[1][0], True))
young_modulus_sym_1_lambdified = sympy.lambdify(sym_T, young_modulus_sym_1)

def young_modulus_eval_1(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return young_modulus_sym_1_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

temp_vec = np.linspace(293., 1800., 152)
E_1_vec = np.empty_like(temp_vec)
for i in range(len(temp_vec)):
    E_1_vec[i] = young_modulus_sym_1_lambdified(temp_vec[i])

plt.figure(figsize=[10, 10])
plt.plot(temp_vec, E_1_vec, "*r", label="Young modulus 1")
plt.legend(loc="best")
plt.savefig("E_1")

# Young's modulus for subdomain 2 ==================
young_modulus_sym_2 = sympy.interpolating_spline(2, sym_T, [293., 573., 1073., 1273.], [15.4e9, 14.7e9, 13.8e9, 14.4e9])
young_modulus_sym_2 = sympy.Piecewise(
    young_modulus_sym_2.args[0], (young_modulus_sym_2.args[1][0], True))
young_modulus_sym_2_lambdified = sympy.lambdify(sym_T, young_modulus_sym_2)

def young_modulus_eval_2(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return young_modulus_sym_2_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

temp_vec = np.linspace(293., 1800., 152)
E_2_vec = np.empty_like(temp_vec)
for i in range(len(temp_vec)):
    E_2_vec[i] = young_modulus_sym_2_lambdified(temp_vec[i])

plt.figure(figsize=[10, 10])
plt.plot(temp_vec, E_2_vec, "*r", label="Young modulus 2")
plt.legend(loc="best")
plt.savefig("E_2")

# Young's modulus for subdomain 3 ==================
young_modulus_sym_3 = sympy.interpolating_spline(2, sym_T, [293., 573., 1073., 1273.], [58.2e9, 67.3e9, 52.9e9, 51.6e9])
young_modulus_sym_3 = sympy.Piecewise(
    young_modulus_sym_3.args[0], (young_modulus_sym_3.args[1][0], True))
young_modulus_sym_3_lambdified = sympy.lambdify(sym_T, young_modulus_sym_3)

def young_modulus_eval_3(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return young_modulus_sym_3_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

temp_vec = np.linspace(293., 1800., 152)
E_3_vec = np.empty_like(temp_vec)
for i in range(len(temp_vec)):
    E_3_vec[i] = young_modulus_sym_3_lambdified(temp_vec[i])

plt.figure(figsize=[10, 10])
plt.plot(temp_vec, E_3_vec, "*r", label="Young modulus 3")
plt.legend(loc="best")
plt.savefig("E_3")

# Young's modulus for subdomain 4 ==================
young_modulus_sym_4 = sympy.interpolating_spline(2, sym_T, [293., 573., 1073., 1273.], [1.85e9, 1.92e9, 1.83e9, 1.85e9])
young_modulus_sym_4 = sympy.Piecewise(
    young_modulus_sym_4.args[0], (young_modulus_sym_4.args[1][0], True))
young_modulus_sym_4_lambdified = sympy.lambdify(sym_T, young_modulus_sym_4)

def young_modulus_eval_4(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return young_modulus_sym_4_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

temp_vec = np.linspace(293., 1800., 152)
E_4_vec = np.empty_like(temp_vec)
for i in range(len(temp_vec)):
    E_4_vec[i] = young_modulus_sym_4_lambdified(temp_vec[i])

plt.figure(figsize=[10, 10])
plt.plot(temp_vec, E_4_vec, "*r", label="Young modulus 4")
plt.legend(loc="best")
plt.savefig("E_4")

# Young's modulus for subdomain 5 ==================
young_modulus_sym_5 = sympy.interpolating_spline(2, sym_T, [293., 573., 1073., 1273.], [14.5e9, 15.0e9, 15.3e9, 13.3e9])
young_modulus_sym_5 = sympy.Piecewise(
    young_modulus_sym_5.args[0], (young_modulus_sym_5.args[1][0], True))
young_modulus_sym_5_lambdified = sympy.lambdify(sym_T, young_modulus_sym_5)

def young_modulus_eval_5(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return young_modulus_sym_5_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

temp_vec = np.linspace(293., 1800., 152)
E_5_vec = np.empty_like(temp_vec)
for i in range(len(temp_vec)):
    E_5_vec[i] = young_modulus_sym_5_lambdified(temp_vec[i])

plt.figure(figsize=[10, 10])
plt.plot(temp_vec, E_5_vec, "*r", label="Young modulus 5")
plt.legend(loc="best")
plt.savefig("E_5")

# Young's modulus for subdomain 7 ==================
young_modulus_sym_7 = sympy.interpolating_spline(2, sym_T, [293., 573., 1073., 1273.], [15.4e9, 14.7e9, 13.8e9, 14.4e9])
young_modulus_sym_7 = sympy.Piecewise(
    young_modulus_sym_7.args[0], (young_modulus_sym_7.args[1][0], True))
young_modulus_sym_7_lambdified = sympy.lambdify(sym_T, young_modulus_sym_7)

def young_modulus_eval_7(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return young_modulus_sym_7_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

temp_vec = np.linspace(293., 1800., 152)
E_7_vec = np.empty_like(temp_vec)
for i in range(len(temp_vec)):
    E_7_vec[i] = young_modulus_sym_7_lambdified(temp_vec[i])

plt.figure(figsize=[10, 10])
plt.plot(temp_vec, E_7_vec, "*r", label="Young modulus 7")
plt.legend(loc="best")
plt.savefig("E_7")

# Young modulus
young_modulus_func = dolfinx.fem.Function(Q)
young_modulus_func_1 = dolfinx.fem.Function(Q)
young_modulus_func_2 = dolfinx.fem.Function(Q)
young_modulus_func_3 = dolfinx.fem.Function(Q)
young_modulus_func_4 = dolfinx.fem.Function(Q)
young_modulus_func_5 = dolfinx.fem.Function(Q)
young_modulus_func_7 = dolfinx.fem.Function(Q)

young_modulus_func.x.array[omega_6_cells] = 1.9E11

# Poisson ratio
poisson_ratio_func = dolfinx.fem.Function(Q)
poisson_ratio_func.x.array[omega_1_cells] = 0.3
poisson_ratio_func.x.array[omega_2_cells] = 0.2
poisson_ratio_func.x.array[omega_3_cells] = 0.1
poisson_ratio_func.x.array[omega_4_cells] = 0.1
poisson_ratio_func.x.array[omega_5_cells] = 0.2
poisson_ratio_func.x.array[omega_6_cells] = 0.3
poisson_ratio_func.x.array[omega_7_cells] = 0.2

# Thermal expansion coefficient
thermal_expansion_coefficient_func = dolfinx.fem.Function(Q)
thermal_expansion_coefficient_func.x.array[omega_1_cells] = 2.3e-6
thermal_expansion_coefficient_func.x.array[omega_2_cells] = 4.6e-6
thermal_expansion_coefficient_func.x.array[omega_3_cells] = 4.7e-6
thermal_expansion_coefficient_func.x.array[omega_4_cells] = 4.6e-6
thermal_expansion_coefficient_func.x.array[omega_5_cells] = 6.e-6
thermal_expansion_coefficient_func.x.array[omega_6_cells] = 1.2e-5
thermal_expansion_coefficient_func.x.array[omega_7_cells] = 4.6e-6

# Weak form
def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u, E=young_modulus_func, nu=poisson_ratio_func, epsilon=epsilon):
    lambda_ = E * nu / ((1 - 2 * nu) * (1 + nu))
    mu = E / (2 * (1 + nu))
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

aM = ufl.inner(sigma(uM, E=young_modulus_func, nu=poisson_ratio_func), epsilon(vM)) * x[0] * ufl.dx
lM = (temperature_field - T0) * thermal_expansion_coefficient_func * ufl.div(vM) * x[0] * ufl.dx - rho * g * (ymax - x[1]) * ufl.dot(vM, n_vec) * x[0] * ds_sf

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

with HarmonicMeshMotion(mesh, boundaries, bc_markers_list,
                        bc_list_geometric, reset_reference=True,
                        is_deformation=True):

    young_modulus_func_1.interpolate(young_modulus_eval_1)
    young_modulus_func_2.interpolate(young_modulus_eval_2)
    young_modulus_func_3.interpolate(young_modulus_eval_3)
    young_modulus_func_4.interpolate(young_modulus_eval_4)
    young_modulus_func_5.interpolate(young_modulus_eval_5)
    young_modulus_func_7.interpolate(young_modulus_eval_7)

    young_modulus_func.x.array[omega_1_cells] = young_modulus_func_1.x.array[omega_1_cells]
    young_modulus_func.x.array[omega_2_cells] = young_modulus_func_2.x.array[omega_2_cells]
    young_modulus_func.x.array[omega_3_cells] = young_modulus_func_3.x.array[omega_3_cells]
    young_modulus_func.x.array[omega_4_cells] = young_modulus_func_4.x.array[omega_4_cells]
    young_modulus_func.x.array[omega_5_cells] = young_modulus_func_5.x.array[omega_5_cells]
    young_modulus_func.x.array[omega_7_cells] = young_modulus_func_7.x.array[omega_7_cells]

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
    # print(displacement_field.x.array)
    print(f"Displacement field norm: {mesh_comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(displacement_field, displacement_field)*ufl.dx)))}")

    computed_file = "solution_nonlinear_thermomechanical_mechanical/solution_computed.xdmf"
    with dolfinx.io.XDMFFile(mesh.comm, computed_file, "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(displacement_field)
