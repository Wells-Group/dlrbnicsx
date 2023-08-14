from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import sympy
import matplotlib.pyplot as plt  # noqa: F401

import ufl
import dolfinx

from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion  # noqa: F401

# Read mesh
mesh_comm = MPI.COMM_WORLD
gmsh_model_rank = 0
gdim = 2
mesh, subdomains, boundaries = dolfinx.io.gmshio.read_from_msh(
    "mesh_data/mesh.msh", mesh_comm, gmsh_model_rank, gdim=gdim)
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=boundaries)
ds_sf = ds(11) + ds(20) + ds(21) + ds(22) + ds(23)
ds_bottom = ds(1) + ds(31)
ds_out = ds(30)
ds_sym = ds(5) + ds(9) + ds(12)
ds_top = ds(18) + ds(19) + ds(27) + ds(28) + ds(29)
x = ufl.SpatialCoordinate(mesh)

y_max = mesh.comm.allreduce(max(mesh.geometry.x[:, 1]), op=MPI.MAX)

normal_vec = ufl.FacetNormal(mesh)

V = dolfinx.fem.VectorFunctionSpace(mesh, ("CG", 1))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

V_T = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
temperature_field = dolfinx.fem.Function(V_T)
temperature_field.x.array[:] = 1000.
temperature_ref = dolfinx.fem.Function(V_T)
temperature_ref.x.array[:] = 300.


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma(u):
    return (youngs_modulus_func * poisson_ratio_func)/((1 - 2*youngs_modulus_func) * (1 + poisson_ratio_func)) * ufl.nabla_div(u) * ufl.Identity(len(u)) + youngs_modulus_func / (1 + poisson_ratio_func) * epsilon(u)


sym_T = sympy.Symbol("sym_T")

youngs_modulus_sym_1 = sympy.interpolating_spline(
    2, sym_T, [293., 573., 1073., 1273.], [9.88E9, 9.79E9, 9.72E9, 9.98E9])
youngs_modulus_sym_1 = sympy.Piecewise(youngs_modulus_sym_1.args[0], (youngs_modulus_sym_1.args[1][0], True))
youngs_modulus_sym_1_lambdified = sympy.lambdify(sym_T, youngs_modulus_sym_1)


def youngs_modulus_eval_1(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return youngs_modulus_sym_1_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])


youngs_modulus_sym_2 = sympy.interpolating_spline(
    2, sym_T, [293., 573., 1073., 1273.], [13.3E9, 13.6E9, 14.7E9, 15.4E9])
youngs_modulus_sym_2 = sympy.Piecewise(youngs_modulus_sym_2.args[0], (youngs_modulus_sym_2.args[1][0], True))
youngs_modulus_sym_2_lambdified = sympy.lambdify(sym_T, youngs_modulus_sym_2)


def youngs_modulus_eval_2(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return youngs_modulus_sym_2_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])


youngs_modulus_sym_3 = sympy.interpolating_spline(
    2, sym_T, [293., 573., 1073., 1273.], [31.9E9, 53.5E9, 74.4E9, 82.3E9])
youngs_modulus_sym_3 = sympy.Piecewise(youngs_modulus_sym_3.args[0], (youngs_modulus_sym_3.args[1][0], True))
youngs_modulus_sym_3_lambdified = sympy.lambdify(sym_T, youngs_modulus_sym_3)


def youngs_modulus_eval_3(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return youngs_modulus_sym_3_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])


youngs_modulus_sym_5 = sympy.interpolating_spline(
    2, sym_T, [293., 573., 1073., 1273.], [13.7E9, 13.1E9, 14.4E9, 15.3E9])
youngs_modulus_sym_5 = sympy.Piecewise(youngs_modulus_sym_5.args[0], (youngs_modulus_sym_5.args[1][0], True))
youngs_modulus_sym_5_lambdified = sympy.lambdify(sym_T, youngs_modulus_sym_5)


def youngs_modulus_eval_5(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return youngs_modulus_sym_5_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])


youngs_modulus_sym_7 = sympy.interpolating_spline(
    2, sym_T, [293., 573., 1073., 1273.], [13.3E9, 13.6E9, 14.7E9, 15.4E9])
youngs_modulus_sym_7 = sympy.Piecewise(youngs_modulus_sym_7.args[0], (youngs_modulus_sym_7.args[1][0], True))
youngs_modulus_sym_7_lambdified = sympy.lambdify(sym_T, youngs_modulus_sym_7)


def youngs_modulus_eval_7(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return youngs_modulus_sym_7_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])


V_E_nu_alpha = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))
youngs_modulus_func = dolfinx.fem.Function(V_E_nu_alpha)
poisson_ratio_func = dolfinx.fem.Function(V_E_nu_alpha)
thermal_expansion_coefficient_func = dolfinx.fem.Function(V_E_nu_alpha)

omega_1_cells = subdomains.find(1)
youngs_modulus_1_func = dolfinx.fem.Function(V_E_nu_alpha)

poisson_ratio_func.x.array[omega_1_cells] = 0.3
thermal_expansion_coefficient_func.x.array[omega_1_cells] = 2.5e-6

omega_2_cells = subdomains.find(2)
youngs_modulus_2_func = dolfinx.fem.Function(V_E_nu_alpha)

poisson_ratio_func.x.array[omega_2_cells] = 0.2
thermal_expansion_coefficient_func.x.array[omega_2_cells] = 4.56e-6

omega_3_cells = subdomains.find(3)
youngs_modulus_3_func = dolfinx.fem.Function(V_E_nu_alpha)

poisson_ratio_func.x.array[omega_3_cells] = 0.08
thermal_expansion_coefficient_func.x.array[omega_3_cells] = 4.66e-6

omega_4_cells = subdomains.find(4)
youngs_modulus_func.x.array[omega_4_cells] = 0.092E9  # TODO ceramic cup properties
poisson_ratio_func.x.array[omega_4_cells] = 0.2  # TODO ceramic cup properties
thermal_expansion_coefficient_func.x.array[omega_4_cells] = 4.58e-6  # TODO ceramic cup properties

omega_5_cells = subdomains.find(5)
youngs_modulus_5_func = dolfinx.fem.Function(V_E_nu_alpha)

poisson_ratio_func.x.array[omega_5_cells] = 0.2
thermal_expansion_coefficient_func.x.array[omega_5_cells] = 6.04e-6

omega_6_cells = subdomains.find(6)
youngs_modulus_func.x.array[omega_6_cells] = 200.E9
poisson_ratio_func.x.array[omega_6_cells] = 0.25
thermal_expansion_coefficient_func.x.array[omega_6_cells] = 11.55e-6

omega_7_cells = subdomains.find(7)
youngs_modulus_7_func = dolfinx.fem.Function(V_E_nu_alpha)

poisson_ratio_func.x.array[omega_7_cells] = 0.2
thermal_expansion_coefficient_func.x.array[omega_7_cells] = 4.56e-6

boundary_facets = np.concatenate(
    (boundaries.indices[boundaries.values == 5], boundaries.indices[boundaries.values == 9], boundaries.indices[boundaries.values == 12]))
boundary_dofs_x = dolfinx.fem.locate_dofs_topological(V.sub(0), gdim-1, boundary_facets)
bcx = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), boundary_dofs_x, V.sub(0))

boundary_facets = np.concatenate(
    (boundaries.indices[boundaries.values == 1], boundaries.indices[boundaries.values == 31]))
boundary_dofs_y = dolfinx.fem.locate_dofs_topological(V.sub(1), gdim-1, boundary_facets)
bcy = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), boundary_dofs_y, V.sub(1))

bcs = [bcx, bcy]

a_M = dolfinx.fem.form(ufl.inner(sigma(u), epsilon(v)) * x[0] * dx)
l_M = dolfinx.fem.form((v[0].dx(0) + v[1].dx(1) + v[0]/x[0]) * youngs_modulus_func * thermal_expansion_coefficient_func * (
    temperature_field - temperature_ref) * dx - 77106 * 9.81 * (y_max - x[1]) * ufl.inner(normal_vec, v) * x[0] * ds_sf)
# TODO check formulation after inserting anisotropy


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

    youngs_modulus_1_func.interpolate(youngs_modulus_eval_1)
    youngs_modulus_func.x.array[omega_1_cells] = youngs_modulus_1_func.x.array[omega_1_cells]

    youngs_modulus_2_func.interpolate(youngs_modulus_eval_2)
    youngs_modulus_func.x.array[omega_2_cells] = youngs_modulus_2_func.x.array[omega_2_cells]

    youngs_modulus_3_func.interpolate(youngs_modulus_eval_3)
    youngs_modulus_func.x.array[omega_3_cells] = youngs_modulus_3_func.x.array[omega_3_cells]

    youngs_modulus_5_func.interpolate(youngs_modulus_eval_5)
    youngs_modulus_func.x.array[omega_5_cells] = youngs_modulus_5_func.x.array[omega_5_cells]

    youngs_modulus_7_func.interpolate(youngs_modulus_eval_7)
    youngs_modulus_func.x.array[omega_7_cells] = youngs_modulus_7_func.x.array[omega_7_cells]

    # Bilinear side assembly
    A = dolfinx.fem.petsc.assemble_matrix(a_M, bcs=bcs)
    A.assemble()

    # Linear side assembly
    L = dolfinx.fem.petsc.assemble_vector(l_M)
    dolfinx.fem.petsc.apply_lifting(L, [a_M], [bcs])
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

    computed_file = "solution_nonlinear_thermomechanical_mechanical/solution_computed.xdmf"
    with dolfinx.io.XDMFFile(mesh.comm, computed_file, "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(solution)


    print(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(ufl.inner(solution[0], solution[0]) * ds_sym)), op=MPI.SUM))
    print(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(
        ufl.inner(solution[1], solution[1]) * ds_bottom)), op=MPI.SUM))
    print(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(ufl.inner(solution[1], solution[1]) * ds_sym)), op=MPI.SUM))
    print(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(
        ufl.inner(solution[0], solution[0]) * ds_bottom)), op=MPI.SUM))
