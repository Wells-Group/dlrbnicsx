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
ds = ufl.Measure("ds", domain=mesh, subdomain_data=boundaries)
ds_sf = ds(11) + ds(20) + ds(21) + ds(22) + ds(23)
ds_bottom = ds(1) + ds(31)
ds_out = ds(30)
x = ufl.SpatialCoordinate(mesh)

V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

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
temperature_field = dolfinx.fem.Function(V)
temperature_field.x.array[:] = 300.  # Initial guess for Newton solver 300 K
# mesh_comm.Barrier()

sym_T = sympy.Symbol("sym_T")

h_cf, h_bottom, h_out = 2000., 200., 200.
T_f, T_bottom, T_out = 1773., 300., 300.

# Conductivity for subdomain 1 ==================

thermal_conductivity_sym_1 = sympy.interpolating_spline(2, sym_T, [293., 473., 873., 1273.], [15., 15.2, 16.2, 15.1])
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


omega_1_cells = subdomains.find(1)

# Conductivity for subdomain 2 ==================

thermal_conductivity_sym_2 = sympy.interpolating_spline(2, sym_T, [293., 473., 873., 1273.], [35.8, 37.3, 42.7, 47.2])
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


omega_2_cells = subdomains.find(2)

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

thermal_conductivity_sym_5 = sympy.interpolating_spline(2, sym_T, [293., 473., 873., 1273.], [19.2, 19.6, 20.7, 21.3])
# thermal_conductivity_sym_5 = sympy.interpolating_spline(2, sym_T, [293., 473., 873., 1273.], [19.2,18.6,20.7,21.3])
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


omega_5_cells = subdomains.find(5)

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

thermal_conductivity_sym_7 = sympy.interpolating_spline(2, sym_T, [293., 473., 873., 1273.], [35.8, 37.3, 42.7, 47.2])
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


omega_7_cells = subdomains.find(7)

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
omega_3_cells = subdomains.find(3)
omega_4_cells = subdomains.find(4)
omega_6_cells = subdomains.find(6)
thermal_conductivity_func.x.array[omega_3_cells] = 5.5
thermal_conductivity_func.x.array[omega_4_cells] = 5.
thermal_conductivity_func.x.array[omega_6_cells] = 48.


bcs = []  # No Dirichlet BCs


a_T = ufl.inner(thermal_conductivity_func * ufl.grad(u), ufl.grad(v)) * x[0] * ufl.dx + \
    ufl.inner(u * thermal_conductivity_func_diff * ufl.grad(temperature_field), ufl.grad(v)) * x[0] * ufl.dx + \
    ufl.inner(h_cf * u, v) * x[0] * ds_sf + ufl.inner(h_bottom * u, v) * x[0] * ds_bottom + \
    ufl.inner(h_out * u, v) * x[0] * ds_out
l_T = h_cf * T_f * v * x[0] * ds_sf + h_bottom * T_bottom * v * x[0] * ds_bottom + h_out * T_out * v * x[0] * ds_out
a_T_cpp = dolfinx.fem.form(a_T)
l_T_cpp = dolfinx.fem.form(l_T)


# Parameter tuple (D_0, D_1, t_0, t_1)
mu_ref = [0.6438, 0.4313, 1., 0.5]  # reference geometry
mu = [0.8, 0.55, 0.8, 0.4]  # Parametric geometry

# Geometric deformation boundary condition w.r.t. reference domain
# i.e. set reset_reference=True and is_deformation=True

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


with HarmonicMeshMotion(mesh, boundaries, bc_markers_list,
                        bc_list_geometric, reset_reference=True,
                        is_deformation=True):

    residual_list = list()
    update_tol_list = list()


    max_iterations = 20
    for iteration in range(max_iterations):

        print("\n =======================================================")
        print(f"\n Iteration {iteration + 1} / {max_iterations} \n")

        thermal_conductivity_func_1.interpolate(conductivity_eval_1)
        thermal_conductivity_func.x.array[omega_1_cells] = thermal_conductivity_func_1.x.array[omega_1_cells]

        thermal_conductivity_func_diff_1.interpolate(conductivity_eval_diff_1)
        thermal_conductivity_func_diff.x.array[omega_1_cells] = thermal_conductivity_func_diff_1.x.array[omega_1_cells]

        thermal_conductivity_func_2.interpolate(conductivity_eval_2)
        thermal_conductivity_func.x.array[omega_2_cells] = thermal_conductivity_func_2.x.array[omega_2_cells]

        thermal_conductivity_func_diff_2.interpolate(conductivity_eval_diff_2)
        thermal_conductivity_func_diff.x.array[omega_2_cells] = thermal_conductivity_func_diff_2.x.array[omega_2_cells]

        thermal_conductivity_func_5.interpolate(conductivity_eval_5)
        thermal_conductivity_func.x.array[omega_5_cells] = thermal_conductivity_func_5.x.array[omega_5_cells]

        thermal_conductivity_func_diff_5.interpolate(conductivity_eval_diff_5)
        thermal_conductivity_func_diff.x.array[omega_5_cells] = thermal_conductivity_func_diff_5.x.array[omega_5_cells]

        thermal_conductivity_func_7.interpolate(conductivity_eval_7)
        thermal_conductivity_func.x.array[omega_7_cells] = thermal_conductivity_func_7.x.array[omega_7_cells]

        thermal_conductivity_func_diff_7.interpolate(conductivity_eval_diff_7)
        thermal_conductivity_func_diff.x.array[omega_7_cells] = thermal_conductivity_func_diff_7.x.array[omega_7_cells]

        mesh_comm.Barrier()

        if iteration == 0:
            residual = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(thermal_conductivity_func * ufl.grad(temperature_field), ufl.grad(v)) * x[0] * ufl.dx +
                                                                    ufl.inner(h_cf * (temperature_field - T_f), v) * x[0] * ds_sf + ufl.inner(h_bottom * (temperature_field - T_bottom), v) * x[0] * ds_bottom + ufl.inner(h_out * (temperature_field - T_out), v) * x[0] * ds_out))
            residual = mesh_comm.allreduce(residual, op=MPI.SUM)
            initial_residual = residual
        else:
            if residual / initial_residual < 1.e-4:
                print(f"Residual rtol reached")
                break

        print(f"Relative residual: {residual/initial_residual}")

        # Bilinear side assembly
        A = dolfinx.fem.petsc.assemble_matrix(a_T_cpp, bcs=bcs)
        A.assemble()

        # Linear side assembly
        L = dolfinx.fem.petsc.assemble_vector(l_T_cpp)
        dolfinx.fem.petsc.apply_lifting(L, [a_T_cpp], [bcs])
        L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(L, bcs)

        # Solver setup
        ksp = PETSc.KSP()
        ksp.create(mesh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        ksp.setFromOptions()
        solution = dolfinx.fem.Function(V)
        ksp.solve(L, solution.vector)
        solution.x.scatter_forward()

        update_function = dolfinx.fem.Function(V)
        update_function.x.array[:] = solution.x.array[:] - temperature_field.x.array[:]
        solution_update = \
            mesh_comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(update_function, update_function) * x[0] * ufl.dx + ufl.inner(ufl.grad(update_function), ufl.grad(update_function)) * x[0] * ufl.dx)), op=MPI.SUM) / \
            mesh_comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(temperature_field, temperature_field) *
                                x[0] * ufl.dx + ufl.inner(ufl.grad(temperature_field), ufl.grad(temperature_field)) * x[0] * ufl.dx)), op=MPI.SUM)

        print(f"Relative update (in norm): {solution_update}")
        if solution_update < 1.e-12:
            print(f"Relative update tolerance reached")
            break

        temperature_field.x.array[:] = solution.x.array.copy()

        computed_file = "solution_nonlinear_thermomechanical_thermal/solution_computed.xdmf"
        with dolfinx.io.XDMFFile(mesh.comm, computed_file, "w") as solution_file:
            solution_file.write_mesh(mesh)
            solution_file.write_function(temperature_field, iteration)


        residual = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(thermal_conductivity_func * ufl.grad(temperature_field), ufl.grad(v)) * x[0] * ufl.dx +
                                                                ufl.inner(h_cf * (temperature_field - T_f), v) * x[0] * ds_sf + ufl.inner(h_bottom * (temperature_field - T_bottom), v) * x[0] * ds_bottom + ufl.inner(h_out * (temperature_field - T_out), v) * x[0] * ds_out))
        residual = mesh_comm.allreduce(residual, op=MPI.SUM)

        print(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(solution, solution) *
            x[0] * ufl.dx + ufl.inner(ufl.grad(solution), ufl.grad(solution)) * x[0] * ufl.dx)), op=MPI.SUM))
