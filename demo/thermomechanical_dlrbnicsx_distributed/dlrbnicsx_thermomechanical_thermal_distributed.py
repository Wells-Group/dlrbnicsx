import dolfinx
import ufl

from mpi4py import MPI
from petsc4py import PETSc

from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion

import numpy as np
import sympy
import itertools
import abc
import matplotlib.pyplot as plt

import rbnicsx
import rbnicsx.backends
import rbnicsx.online

from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory import Sigmoid
from dlrbnicsx.dataset.custom_partitioned_dataset \
    import CustomPartitionedDataset
from dlrbnicsx.interface.wrappers import DataLoader, model_synchronise
from dlrbnicsx.train_validate_test.train_validate_test import \
    train_nn, validate_nn, online_nn, error_analysis
from dlrbnicsx.train_validate_test.train_validate_test_distributed \
    import train_nn, validate_nn, online_nn, error_analysis

class ThermalProblemOnDeformedDomain(abc.ABC):
    def __init__(self, mesh, subdomains, boundaries):
        self._mesh = mesh
        self._subdomains = subdomains
        self._boundaries = boundaries
        self._boundary_markers = list(np.arange(1, 32))
        # For function space
        self._V = dolfinx.fem.FunctionSpace(self._mesh, ("CG", 1))
        # For material properties
        self._Q = dolfinx.fem.FunctionSpace(self._mesh, ("DG", 0))
        u, v = ufl.TrialFunction(self._V), ufl.TestFunction(self._V)
        self._trial, self._test = u, v
        x = ufl.SpatialCoordinate(self._mesh)
        self._inner_product = ufl.inner(u, v) * x[0] * ufl.dx + \
            ufl.inner(ufl.grad(u), ufl.grad(v)) * x[0] * ufl.dx
        self.inner_product_action = \
            rbnicsx.backends.bilinear_form_action(self._inner_product,
                                                  part="real")
        self._solution = dolfinx.fem.Function(self._V)
        self._thermal_conductivity_func = dolfinx.fem.Function(self._Q)
        self._thermal_conductivity_func_diff = dolfinx.fem.Function(self._Q)
        self._max_iterations = 20
        sym_T = sympy.Symbol("sym_T")
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=boundaries)
        self._ds_sf = ds(11) + ds(20) + ds(21) + ds(22) + ds(23)
        self._ds_bottom = ds(1) + ds(31)
        self._ds_out = ds(30)
        self._h_cf, self._h_bottom, self._h_out = 2000., 200., 200.
        self._T_f, self._T_bottom, self._T_out = 1773., 300., 300.
        self.mu_ref = [0.6438, 0.4313, 1., 0.5]

        thermal_conductivity_sym_1 = sympy.interpolating_spline(2, sym_T, [293., 473., 873., 1273.], [15., 15.2, 16.2, 15.1])
        thermal_conductivity_sym_1 = sympy.Piecewise(
            thermal_conductivity_sym_1.args[0], (thermal_conductivity_sym_1.args[1][0], True))
        self._thermal_conductivity_sym_1_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_1)
        thermal_conductivity_sym_diff_1 = sympy.diff(thermal_conductivity_sym_1, sym_T)
        self._thermal_conductivity_sym_diff_1_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_diff_1)

        thermal_conductivity_sym_2 = sympy.interpolating_spline(2, sym_T, [293., 473., 873., 1273.], [35.8, 37.3, 42.7, 47.2])
        thermal_conductivity_sym_2 = sympy.Piecewise(
            thermal_conductivity_sym_2.args[0], (thermal_conductivity_sym_2.args[1][0], True))
        self._thermal_conductivity_sym_2_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_2)
        thermal_conductivity_sym_diff_2 = sympy.diff(thermal_conductivity_sym_2, sym_T)
        self._thermal_conductivity_sym_diff_2_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_diff_2)

        thermal_conductivity_sym_5 = sympy.interpolating_spline(2, sym_T, [293., 473., 873., 1273.], [19.2, 19.6, 20.7, 21.3])
        thermal_conductivity_sym_5 = sympy.Piecewise(
            thermal_conductivity_sym_5.args[0], (thermal_conductivity_sym_5.args[1][0], True))
        self._thermal_conductivity_sym_5_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_5)
        thermal_conductivity_sym_diff_5 = sympy.diff(thermal_conductivity_sym_5, sym_T)
        self._thermal_conductivity_sym_diff_5_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_diff_5)

        thermal_conductivity_sym_7 = sympy.interpolating_spline(2, sym_T, [293., 473., 873., 1273.], [35.8, 37.3, 42.7, 47.2])
        thermal_conductivity_sym_7 = sympy.Piecewise(
            thermal_conductivity_sym_7.args[0], (thermal_conductivity_sym_7.args[1][0], True))
        self._thermal_conductivity_sym_7_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_7)
        thermal_conductivity_sym_diff_7 = sympy.diff(thermal_conductivity_sym_7, sym_T)
        self._thermal_conductivity_sym_diff_7_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_diff_7)

    def conductivity_eval_1(self, x):
        tree = dolfinx.geometry.bb_tree(self._mesh, self._mesh.geometry.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self._mesh, cell_candidates, x.T)
        return self._thermal_conductivity_sym_1_lambdified(self._solution.eval(x.T, colliding_cells.array)[:, 0])

    def conductivity_eval_diff_1(self, x):
        tree = dolfinx.geometry.bb_tree(self._mesh, self._mesh.geometry.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self._mesh, cell_candidates, x.T)
        return self._thermal_conductivity_sym_diff_1_lambdified(self._solution.eval(x.T, colliding_cells.array)[:, 0])

    def conductivity_eval_2(self, x):
        tree = dolfinx.geometry.bb_tree(self._mesh, self._mesh.geometry.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self._mesh, cell_candidates, x.T)
        return self._thermal_conductivity_sym_2_lambdified(self._solution.eval(x.T, colliding_cells.array)[:, 0])

    def conductivity_eval_diff_2(self, x):
        tree = dolfinx.geometry.bb_tree(self._mesh, self._mesh.geometry.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self._mesh, cell_candidates, x.T)
        return self._thermal_conductivity_sym_diff_2_lambdified(self._solution.eval(x.T, colliding_cells.array)[:, 0])

    def conductivity_eval_5(self, x):
        tree = dolfinx.geometry.bb_tree(self._mesh, self._mesh.geometry.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self._mesh, cell_candidates, x.T)
        return self._thermal_conductivity_sym_5_lambdified(self._solution.eval(x.T, colliding_cells.array)[:, 0])

    def conductivity_eval_diff_5(self, x):
        tree = dolfinx.geometry.bb_tree(self._mesh, self._mesh.geometry.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self._mesh, cell_candidates, x.T)
        return self._thermal_conductivity_sym_diff_5_lambdified(self._solution.eval(x.T, colliding_cells.array)[:, 0])

    def conductivity_eval_7(self, x):
        tree = dolfinx.geometry.bb_tree(self._mesh, self._mesh.geometry.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self._mesh, cell_candidates, x.T)
        return self._thermal_conductivity_sym_7_lambdified(self._solution.eval(x.T, colliding_cells.array)[:, 0])

    def conductivity_eval_diff_7(self, x):
        tree = dolfinx.geometry.bb_tree(self._mesh, self._mesh.geometry.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self._mesh, cell_candidates, x.T)
        return self._thermal_conductivity_sym_diff_7_lambdified(self._solution.eval(x.T, colliding_cells.array)[:, 0])

    def thermal_conductivity_func_assemble(self):
        omega_1_cells = self._subdomains.find(1)
        thermal_conductivity_func_placeholder = dolfinx.fem.Function(self._Q)
        thermal_conductivity_func_placeholder.interpolate(self.conductivity_eval_1)
        self._thermal_conductivity_func.x.array[omega_1_cells] = thermal_conductivity_func_placeholder.x.array[omega_1_cells]

        omega_2_cells = self._subdomains.find(2)
        thermal_conductivity_func_placeholder = dolfinx.fem.Function(self._Q)
        thermal_conductivity_func_placeholder.interpolate(self.conductivity_eval_2)
        self._thermal_conductivity_func.x.array[omega_2_cells] = thermal_conductivity_func_placeholder.x.array[omega_2_cells]

        omega_5_cells = self._subdomains.find(5)
        thermal_conductivity_func_placeholder = dolfinx.fem.Function(self._Q)
        thermal_conductivity_func_placeholder.interpolate(self.conductivity_eval_5)
        self._thermal_conductivity_func.x.array[omega_5_cells] = thermal_conductivity_func_placeholder.x.array[omega_5_cells]

        omega_7_cells = self._subdomains.find(7)
        thermal_conductivity_func_placeholder = dolfinx.fem.Function(self._Q)
        thermal_conductivity_func_placeholder.interpolate(self.conductivity_eval_7)
        self._thermal_conductivity_func.x.array[omega_7_cells] = thermal_conductivity_func_placeholder.x.array[omega_7_cells]

        omega_3_cells = self._subdomains.find(3)
        self._thermal_conductivity_func.x.array[omega_3_cells] = 5.5
        omega_4_cells = self._subdomains.find(4)
        self._thermal_conductivity_func.x.array[omega_4_cells] = 5.
        omega_6_cells = self._subdomains.find(6)
        self._thermal_conductivity_func.x.array[omega_6_cells] = 48.

    def thermal_conductivity_func_diff_assemble(self):
        omega_1_cells = self._subdomains.find(1)
        thermal_conductivity_func_diff_placeholder = dolfinx.fem.Function(self._Q)
        thermal_conductivity_func_diff_placeholder.interpolate(self.conductivity_eval_diff_1)
        self._thermal_conductivity_func_diff.x.array[omega_1_cells] = thermal_conductivity_func_diff_placeholder.x.array[omega_1_cells]

        omega_2_cells = self._subdomains.find(2)
        thermal_conductivity_func_diff_placeholder = dolfinx.fem.Function(self._Q)
        thermal_conductivity_func_diff_placeholder.interpolate(self.conductivity_eval_diff_2)
        self._thermal_conductivity_func_diff.x.array[omega_2_cells] = thermal_conductivity_func_diff_placeholder.x.array[omega_2_cells]

        omega_5_cells = self._subdomains.find(5)
        thermal_conductivity_func_diff_placeholder = dolfinx.fem.Function(self._Q)
        thermal_conductivity_func_diff_placeholder.interpolate(self.conductivity_eval_diff_5)
        self._thermal_conductivity_func_diff.x.array[omega_5_cells] = thermal_conductivity_func_diff_placeholder.x.array[omega_5_cells]

        omega_7_cells = self._subdomains.find(7)
        thermal_conductivity_func_diff_placeholder = dolfinx.fem.Function(self._Q)
        thermal_conductivity_func_diff_placeholder.interpolate(self.conductivity_eval_diff_7)
        self._thermal_conductivity_func_diff.x.array[omega_7_cells] = thermal_conductivity_func_diff_placeholder.x.array[omega_7_cells]

    @property
    def bilinear_form(self):
        u, v = self._trial, self._test
        x = ufl.SpatialCoordinate(self._mesh)
        return dolfinx.fem.form(ufl.inner(self._thermal_conductivity_func * ufl.grad(u), ufl.grad(v)) * x[0] * ufl.dx + \
            ufl.inner(u * self._thermal_conductivity_func_diff * ufl.grad(self._solution), ufl.grad(v)) * x[0] * ufl.dx + \
            ufl.inner(self._h_cf * u, v) * x[0] * self._ds_sf + ufl.inner(self._h_bottom * u, v) * x[0] * self._ds_bottom + \
            ufl.inner(self._h_out * u, v) * x[0] * self._ds_out)

    @property
    def linear_form(self):
        v = self._test
        x = ufl.SpatialCoordinate(self._mesh)
        return dolfinx.fem.form(self._h_cf * self._T_f * v * x[0] * self._ds_sf + \
            self._h_bottom * self._T_bottom * v * x[0] * self._ds_bottom + \
                self._h_out * self._T_out * v * x[0] * self._ds_out)
    @property
    def residual_form(self):
        u, v = self._trial, self._test
        x = ufl.SpatialCoordinate(self._mesh)
        return dolfinx.fem.form(ufl.inner(self._thermal_conductivity_func * ufl.grad(self._solution), ufl.grad(v)) * x[0] * ufl.dx +
                                ufl.inner(self._h_cf * (self._solution - self._T_f), v) * x[0] * self._ds_sf +
                                ufl.inner(self._h_bottom * (self._solution - self._T_bottom), v) * x[0] * self._ds_bottom +
                                ufl.inner(self._h_out * (self._solution - self._T_out), v) * x[0] * self._ds_out)

    def bc_1_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        indices_0 = np.where((x[0] >= 4.875) & (x[0] <= 5.5188))[0]
        indices_1 = np.where((x[0] >= 5.5188) & (x[0] <= 5.9501))[0]
        y = (0. * x[0], 0. * x[1])
        y[0][indices_0] = (x[0][indices_0] - 4.875) * (mu[0] - mu_ref[0]) / (5.5188 - 4.875)
        y[0][indices_1] = \
            (x[0][indices_1] - 5.5188) * (mu[1] - mu_ref[1]) / (5.9501 - 5.5188) + \
            (mu[0] - mu_ref[0]) * (x[0][indices_1] / x[0][indices_1])
        return y

    def bc_2_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return ((mu[1] - mu_ref[1] + mu[0] - mu_ref[0]) * (x[0] / x[0]), (x[1] - 0.) * (mu[2] - mu_ref[2]) / (1. - 0.))

    def bc_3_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        indices_0 = np.where((x[0] >= 4.875) & (x[0] <= 5.5188))[0]
        indices_1 = np.where((x[0] >= 5.5188) & (x[0] <= 5.9501))[0]
        y = (0. * x[0], 0. * x[1])
        y[0][indices_0] = (x[0][indices_0] - 4.875) * (mu[0] - mu_ref[0]) / (5.5188 - 4.875)
        y[0][indices_1] = \
            (x[0][indices_1] - 5.5188) * (mu[1] - mu_ref[1]) / (5.9501 - 5.5188) + \
            (mu[0] - mu_ref[0]) * (x[0][indices_1] / x[0][indices_1])
        y[1][:] = (mu[2] - mu_ref[2]) * x[1]
        return y

    def bc_4_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return (0. * x[0], (mu[2] - mu_ref[2]) * x[1])

    def bc_5_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return (0. * x[0], (mu[2] - mu_ref[2]) * x[1])

    def bc_6_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return (0. * x[0], (mu[2] - mu_ref[2]) * (x[1] / x[1]))

    def bc_7_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return (0. * x[0], (mu[2] - mu_ref[2]) * (x[1] / x[1]))

    def bc_8_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return (0. * x[0], (mu[2] - mu_ref[2]) * (x[1] / x[1]))

    def bc_9_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return (0. * x[0], (mu[2] - mu_ref[2]) * (x[1] / x[1]))

    def bc_10_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return (0. * x[0], (x[1] - 1.6) * (mu[3] - mu_ref[3]) / (2.1 - 1.6) + (mu[2] - mu_ref[2]) * (x[1] / x[1]))

    def bc_11_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return (0. * x[0], (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * (x[1] / x[1]))

    def bc_12_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return (0. * x[0], (x[1] - 1.6) * (mu[3] - mu_ref[3]) / (2.1 - 1.6) + (mu[2] - mu_ref[2]) * (x[1] / x[1]))

    def bc_13_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return (0. * x[0], (mu[2] - mu_ref[2]) * x[1] / x[1])

    def bc_14_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        indices_0 = np.where((x[1] >= 1.6) & (x[1] <= 2.1))[0]
        indices_1 = np.where(x[1] > 2.1)[0]
        y = (0. * x[0], 0. * x[1])
        y[1][indices_0] = (x[1][indices_0] - 1.6) * (mu[3] - mu_ref[3]) / (2.1 - 1.6) + \
            (mu[2] - mu_ref[2]) * x[1][indices_0] / x[1][indices_0]
        y[1][indices_1] = (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1][indices_1] / x[1][indices_1]
        return y

    def bc_15_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return (0. * x[0], (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])

    def bc_16_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return ((x[0] - 4.875) * (mu[0] - mu_ref[0]) / (5.5188 - 4.875),
                (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])

    def bc_17_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return ((x[0] - 4.875) * (mu[0] - mu_ref[0]) / (5.5188 - 4.875),
                (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])

    def bc_18_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return ((x[0] - 4.875) * (mu[0] - mu_ref[0]) / (5.5188 - 4.875),
                (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])

    def bc_19_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return ((x[0] - 4.875) * (mu[0] - mu_ref[0]) / (5.5188 - 4.875),
                (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])

    def bc_20_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return ((x[0] - 4.875) * (mu[0] - mu_ref[0]) / (5.5188 - 4.875),
                (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])

    def bc_21_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return (0. * x[0], (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])

    def bc_22_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return (0. * x[0], (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])

    def bc_23_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return (0. * x[0], (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])

    def bc_24_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
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

    def bc_25_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        indices_0 = np.where((x[0] >= 4.875) & (x[0] <= 5.5188))[0]
        indices_1 = np.where((x[0] >= 5.5188) & (x[0] <= 5.9501))[0]
        y = (0. * x[0], 0. * x[1])
        y[0][indices_0] = (x[0][indices_0] - 4.875) * (mu[0] - mu_ref[0]) / (5.5188 - 4.875)
        y[0][indices_1] = \
            (x[0][indices_1] - 5.5188) * (mu[1] - mu_ref[1]) / (5.9501 - 5.5188) + \
            (mu[0] - mu_ref[0]) * (x[1][indices_1] / x[1][indices_1])
        y[1][:] = (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1]
        return y

    def bc_26_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return ((mu[1] - mu_ref[1] + mu[0] - mu_ref[0]) * x[0] / x[0],
                (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])

    def bc_27_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return ((x[0] - 5.5188) * (mu[1] - mu_ref[1]) / (5.9501 - 5.5188) + (mu[0] - mu_ref[0]) * (x[0] / x[0]),
                (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])

    def bc_28_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return ((mu[1] - mu_ref[1] + mu[0] - mu_ref[0]) * x[0] / x[0],
                (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])

    def bc_29_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return ((mu[1] - mu_ref[1] + mu[0] - mu_ref[0]) * x[0] / x[0],
                (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])

    def bc_30_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
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

    def bc_31_geometric(self, x):
        mu_ref = self.mu_ref
        mu = self.mu
        return ((mu[1] - mu_ref[1] + mu[0] - mu_ref[0]) * x[0] / x[0], 0. * x[1])

    def assemble_bc_list_geometric(self):
        bc_list_geometric = [self.bc_1_geometric, self.bc_2_geometric,
                             self.bc_3_geometric, self.bc_4_geometric,
                             self.bc_5_geometric, self.bc_6_geometric,
                             self.bc_7_geometric, self.bc_8_geometric,
                             self.bc_9_geometric, self.bc_10_geometric,
                             self.bc_11_geometric, self.bc_12_geometric,
                             self.bc_13_geometric, self.bc_14_geometric,
                             self.bc_15_geometric, self.bc_16_geometric,
                             self.bc_17_geometric, self.bc_18_geometric,
                             self.bc_19_geometric, self.bc_20_geometric,
                             self.bc_21_geometric, self.bc_22_geometric,
                             self.bc_23_geometric, self.bc_24_geometric,
                             self.bc_25_geometric, self.bc_26_geometric,
                             self.bc_27_geometric, self.bc_28_geometric,
                             self.bc_29_geometric, self.bc_30_geometric,
                             self.bc_31_geometric]
        return bc_list_geometric

    def solve(self, mu):
        self.mu = mu
        bc_list_geometric = self.assemble_bc_list_geometric()
        bc_markers_list = self._boundary_markers
        self._solution.x.array[:] = 300.  # Initial solution guess
        update_function = dolfinx.fem.Function(self._V)
        with HarmonicMeshMotion(self._mesh, self._boundaries, bc_markers_list,
                                bc_list_geometric, reset_reference=True,
                                is_deformation=True):
            for iteration in range(self._max_iterations):

                print("\n =======================================================")
                print(f"\n Iteration {iteration + 1} / {self._max_iterations} \n")

                self.thermal_conductivity_func_assemble()
                self.thermal_conductivity_func_diff_assemble()
                residual = dolfinx.fem.assemble_scalar(self.residual_form)
                residual = mesh.comm.allreduce(residual, op=MPI.SUM)
                if iteration == 0:
                    initial_residual = residual
                else:
                    if residual / initial_residual < 1.e-4:
                        print(f"Residual rtol reached")
                        break

                a_T_cpp = self.bilinear_form
                l_T_cpp = self.linear_form
                print(f"Relative residual: {residual/initial_residual}")

                # Bilinear side assembly
                A = dolfinx.fem.petsc.assemble_matrix(a_T_cpp, bcs=[])
                A.assemble()

                # Linear side assembly
                L = dolfinx.fem.petsc.assemble_vector(l_T_cpp)
                dolfinx.fem.petsc.apply_lifting(L, [a_T_cpp], [[]])
                L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                dolfinx.fem.petsc.set_bc(L, [])

                # Solver setup
                ksp = PETSc.KSP()
                ksp.create(self._mesh.comm)
                ksp.setOperators(A)
                ksp.setType("preonly")
                ksp.getPC().setType("lu")
                ksp.getPC().setFactorSolverType("mumps")
                ksp.setFromOptions()
                current_solution = dolfinx.fem.Function(self._V)
                ksp.solve(L, current_solution.vector)
                current_solution.x.scatter_forward()
                print(current_solution.x.array)

                update_function.x.array[:] = current_solution.x.array[:].copy() -self._solution.x.array[:].copy()

                x = ufl.SpatialCoordinate(self._mesh)
                solution_update = \
                    mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(update_function, update_function) * x[0] * ufl.dx +
                                                                                    ufl.inner(ufl.grad(update_function), ufl.grad(update_function)) * x[0] * ufl.dx)), op=MPI.SUM) / \
                    mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(self._solution, self._solution) * x[0] * ufl.dx +
                                                                                    ufl.inner(ufl.grad(self._solution), ufl.grad(self._solution)) * x[0] * ufl.dx)), op=MPI.SUM)

                print(f"Relative update (in norm): {solution_update}")
                if solution_update < 1.e-12:
                    print(f"Relative update tolerance reached")
                    break

                self._solution.x.array[:] = current_solution.x.array.copy()

            return current_solution


class PODANNReducedProblem(abc.ABC):
    '''
    TODO
    Mesh deformation at reconstruct_solution,
    compute_norm, project_snapshot (??)
    '''
    """Define a linear projection-based problem, and solve it with KSP."""

    def __init__(self, problem) -> None:
        self._basis_functions = rbnicsx.backends.FunctionsList(problem._V)
        u, v = ufl.TrialFunction(problem._V), ufl.TestFunction(problem._V)
        x = ufl.SpatialCoordinate(problem._mesh)
        self._inner_product = ufl.inner(u, v) * x[0] * ufl.dx +\
            ufl.inner(ufl.grad(u), ufl.grad(v)) * x[0] * ufl.dx
        self._inner_product_action = \
            rbnicsx.backends.bilinear_form_action(self._inner_product,
                                                  part="real")
        self.input_scaling_range = [0., 1.]
        self.output_scaling_range = [0., 1.]
        self.input_range = \
            np.array([[0.55, 0.35, 0.8, 0.4], [0.75, 0.55, 1.2, 0.6]])
        self.output_range = [-6., 3.]
        self.loss_fn = "MSE"
        self.learning_rate = 1e-3
        self.optimizer = "Adam"
        self.regularisation = "EarlyStopping"

    def reconstruct_solution(self, reduced_solution):
        """Reconstructed reduced solution on the high fidelity space."""
        return self._basis_functions[:reduced_solution.size] * \
            reduced_solution

    def compute_norm(self, function):
        """Compute the norm of a function inner product
        on the reference domain."""
        return np.sqrt(self._inner_product_action(function)(function))

    def project_snapshot(self, solution, N):
        return self._project_snapshot(solution, N)

    def _project_snapshot(self, solution, N):
        projected_snapshot = rbnicsx.online.create_vector(N)
        A = rbnicsx.backends.\
            project_matrix(self._inner_product_action,
                           self._basis_functions[:N])
        F = rbnicsx.backends.\
            project_vector(self._inner_product_action(solution),
                           self._basis_functions[:N])
        ksp = PETSc.KSP()
        ksp.create(projected_snapshot.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
        ksp.solve(F, projected_snapshot)
        return projected_snapshot

    def norm_error(self, u, v):
        return self.compute_norm(u-v)/self.compute_norm(u)


# MPI.COMM_WORLD variables
world_comm = MPI.COMM_WORLD
world_rank = world_comm.rank
world_size = world_comm.size

# Read mesh
mesh_comm = MPI.COMM_SELF
gdim = 2
gmsh_model_rank = 0
mesh, cell_tags, facet_tags = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh",
                                    mesh_comm, gmsh_model_rank, gdim=gdim)

# Mesh deformation parameters
mu_ref = [0.6438, 0.4313, 1., 0.5]  # reference geometry
mu = [0.8, 0.55, 0.8, 0.4]  # Parametric geometry

# FEM solve
thermal_problem_parametric = \
    ThermalProblemOnDeformedDomain(mesh, cell_tags, facet_tags)

solution_mu = thermal_problem_parametric.solve(mu)
print(f"Solution norm at mu:{mu}: {thermal_problem_parametric.inner_product_action(solution_mu)(solution_mu)}")

solution_mu = thermal_problem_parametric.solve(mu_ref)
print(f"Solution norm at mu:{mu_ref}: {thermal_problem_parametric.inner_product_action(solution_mu)(solution_mu)}")


itemsize = MPI.DOUBLE.Get_size()
num_snapshots = 3 * 2 * 2 * 3 # 5 * 4 * 5 * 4
num_dofs = solution_mu.x.array.shape[0]
para_dim = 4

if world_comm.rank == 0:
    nbytes_para = num_snapshots * num_dofs * para_dim
    nbytes_dofs = num_snapshots * num_dofs * itemsize
else:
    nbytes_dofs = 0
    nbytes_para = 0

# POD starts ###

def generate_training_set(samples=[3, 2, 2, 3]):#(samples=[5, 4, 5, 4]):
    # Parameter tuple (D_0, D_1, t_0, t_1)
    training_set_0 = np.linspace(0.55, 0.75, samples[0])
    training_set_1 = np.linspace(0.35, 0.55, samples[1])
    training_set_2 = np.linspace(0.8, 1.2, samples[2])
    training_set_3 = np.linspace(0.4, 0.6, samples[3])
    training_set = np.array(list(itertools.product(training_set_0,
                                                   training_set_1,
                                                   training_set_2,
                                                   training_set_3)))
    return training_set


win0 = MPI.Win.Allocate_shared(nbytes_para, itemsize, comm=MPI.COMM_WORLD)
buf0, itemsize = win0.Shared_query(0)
thermal_training_set = np.ndarray(buffer=buf0, dtype="d", shape=(num_snapshots, para_dim))

if world_comm.rank == 0:
    thermal_training_set[:, :] = generate_training_set()

world_comm.Barrier()

thermal_training_set_indices = \
    np.arange(world_comm.rank, thermal_training_set.shape[0], world_comm.size)

win1 = MPI.Win.Allocate_shared(nbytes_dofs, itemsize, comm=MPI.COMM_WORLD)
buf1, itemsize = win1.Shared_query(0)
thermal_training_set_solutions = np.ndarray(buffer=buf1, dtype="d", shape=(num_snapshots, num_dofs))

world_comm.Barrier()

for mu_index in thermal_training_set_indices:
    print(rbnicsx.io.TextLine
          (f"{mu_index+1}/{thermal_training_set_indices.shape[0]}",
           fill="#"))
    thermal_solution_snapshot = \
        thermal_problem_parametric.solve(thermal_training_set[mu_index, :])
    thermal_training_set_solutions[mu_index, :] = thermal_solution_snapshot.x.array

world_comm.Barrier()

Nmax = 30

print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
print("")

print("set up thermal snapshots matrix")
thermal_snapshots_matrix = rbnicsx.backends.FunctionsList(thermal_problem_parametric._V)

print("set up reduced problem")
thermal_reduced_problem = PODANNReducedProblem(thermal_problem_parametric)
print("")

for (mu_index, mu) in enumerate(thermal_training_set_solutions):
    print(rbnicsx.io.TextLine(
        f"{mu_index+1}/{thermal_training_set_solutions.shape[0]}",
        fill="#"))
    thermal_snapshot = dolfinx.fem.Function(thermal_problem_parametric._V)
    thermal_snapshot.x.array[:] = thermal_training_set_solutions[mu_index, :]

    print("Update snapshots matrix")
    thermal_snapshots_matrix.append(thermal_snapshot)

    print("")

print(rbnicsx.io.TextLine("perform POD", fill="#"))
eigenvalues, modes, _ = \
    rbnicsx.backends.\
    proper_orthogonal_decomposition(thermal_snapshots_matrix,
                                    thermal_reduced_problem._inner_product_action,
                                    N=Nmax, tol=1.e-6)
thermal_reduced_problem._basis_functions.extend(modes)
thermal_reduced_size = len(thermal_reduced_problem._basis_functions)
print("")

print(rbnicsx.io.TextBox("POD-Galerkin offline phase ends", fill="="))

positive_eigenvalues = np.where(eigenvalues > 0., eigenvalues, np.nan)
singular_values = np.sqrt(positive_eigenvalues)

if world_comm.rank == 0:
    plt.figure(figsize=[8, 10])
    xint = list()
    yval = list()

    for x, y in enumerate(eigenvalues[:len(thermal_reduced_problem._basis_functions)]):
        yval.append(y)
        xint.append(x+1)

    plt.plot(xint, yval, "*-", color="orange")
    plt.xlabel("Eigenvalue number", fontsize=18)
    plt.ylabel("Eigenvalue", fontsize=18)
    plt.xticks(xint)
    plt.yscale("log")
    plt.title("Eigenvalue decay", fontsize=24)
    plt.tight_layout()
    plt.savefig("eigenvalue_thermal")

# POD Ends ###

# ### ANN implementation ###

def generate_ann_input_set(samples=[2, 3, 3, 2]):#(samples=[4, 5, 4, 5]):
    # Parameter tuple (D_0, D_1, t_0, t_1)
    training_set_0 = np.linspace(0.55, 0.75, samples[0])
    training_set_1 = np.linspace(0.35, 0.55, samples[1])
    training_set_2 = np.linspace(0.8, 1.2, samples[2])
    training_set_3 = np.linspace(0.4, 0.6, samples[3])
    training_set = np.array(list(itertools.product(training_set_0,
                                                   training_set_1,
                                                   training_set_2,
                                                   training_set_3)))
    return training_set


def generate_ann_output_set(problem, reduced_problem, input_set,
                            output_set, indices, mode=None):
    # Compute output set for ANN based on input set
    rb_size = len(reduced_problem._basis_functions)
    for i in indices:
        if mode is None:
            print(f"Parameter number {i+1} of ")
            print(f"{input_set.shape[0]}: {input_set[i, :]}")
        else:
            print(f"{mode} parameter number {i+1} of ")
            print(f"{input_set.shape[0]}: {input_set[i, :]}")
        solution = problem.solve(input_set[i, :])
        output_set[i, :] = reduced_problem.project_snapshot(solution,
                                                            rb_size).array
    # return output_set


num_ann_input_samples = 2 * 3 * 3 * 2# 4 * 5 * 4 * 5
num_ann_input_samples_training = int(0.5 * num_ann_input_samples) #TODO int(0.7 * num_ann_input_samples)
num_ann_input_samples_validation = \
    num_ann_input_samples - int(0.5 * num_ann_input_samples) #TODO num_ann_input_samples - int(0.7 * num_ann_input_samples)
itemsize = MPI.DOUBLE.Get_size()


if world_comm.rank == 0:
    ann_input_samples = generate_ann_input_set()
    # np.random.shuffle(ann_input_samples)
    nbytes_para_ann_training = num_ann_input_samples_training * \
        itemsize * para_dim
    nbytes_dofs_ann_training = num_ann_input_samples_training * itemsize * \
        len(thermal_reduced_problem._basis_functions)
    nbytes_para_ann_validation = num_ann_input_samples_validation * \
        itemsize * para_dim
    nbytes_dofs_ann_validation = num_ann_input_samples_validation * \
        itemsize * len(thermal_reduced_problem._basis_functions)
else:
    nbytes_para_ann_training = 0
    nbytes_dofs_ann_training = 0
    nbytes_para_ann_validation = 0
    nbytes_dofs_ann_validation = 0

world_comm.barrier()

win2 = MPI.Win.Allocate_shared(nbytes_para_ann_training, itemsize,
                               comm=MPI.COMM_WORLD)
buf2, itemsize = win2.Shared_query(0)
ann_input_samples_training = \
    np.ndarray(buffer=buf2, dtype="d",
               shape=(num_ann_input_samples_training,
                      para_dim))

win3 = MPI.Win.Allocate_shared(nbytes_para_ann_validation, itemsize,
                               comm=MPI.COMM_WORLD)
buf3, itemsize = win3.Shared_query(0)
ann_input_samples_validation = \
    np.ndarray(buffer=buf3, dtype="d",
               shape=(num_ann_input_samples_validation,
                      para_dim))

win4 = MPI.Win.Allocate_shared(nbytes_dofs_ann_training, itemsize,
                               comm=MPI.COMM_WORLD)
buf4, itemsize = win4.Shared_query(0)
ann_output_samples_training = \
    np.ndarray(buffer=buf4, dtype="d",
               shape=(num_ann_input_samples_training,
                      len(thermal_reduced_problem._basis_functions)))

win5 = MPI.Win.Allocate_shared(nbytes_dofs_ann_validation, itemsize,
                               comm=MPI.COMM_WORLD)
buf5, itemsize = win5.Shared_query(0)
ann_output_samples_validation = \
    np.ndarray(buffer=buf5, dtype="d",
               shape=(num_ann_input_samples_validation,
                      len(thermal_reduced_problem._basis_functions)))

if world_comm.rank == 0:
    ann_input_samples_training[:, :] = \
        ann_input_samples[:num_ann_input_samples_training, :]
    ann_input_samples_validation[:, :] = \
        ann_input_samples[num_ann_input_samples_training:, :]
    ann_output_samples_training[:, :] = \
        np.zeros([num_ann_input_samples_training,
                  len(thermal_reduced_problem._basis_functions)])
    ann_output_samples_validation[:, :] = \
        np.zeros([num_ann_input_samples_validation,
                  len(thermal_reduced_problem._basis_functions)])

world_comm.Barrier()


training_set_indices = np.arange(world_comm.rank,
                                 ann_input_samples_training.shape[0],
                                 world_comm.size)

validation_set_indices = np.arange(world_comm.rank,
                                   ann_input_samples_validation.shape[0],
                                   world_comm.size)

world_comm.Barrier()


# Training dataset
generate_ann_output_set(thermal_problem_parametric, thermal_reduced_problem,
                        ann_input_samples_training,
                        ann_output_samples_training, training_set_indices,
                        mode="Training")

generate_ann_output_set(thermal_problem_parametric, thermal_reduced_problem,
                        ann_input_samples_validation,
                        ann_output_samples_validation, validation_set_indices,
                        mode="Validation")

world_comm.Barrier()

thermal_reduced_problem.output_range[0] = \
    np.min(np.vstack([ann_output_samples_training,
                      ann_output_samples_validation]))
thermal_reduced_problem.output_range[1] = \
    np.max(np.vstack([ann_output_samples_training,
                      ann_output_samples_validation]))

# NOTE Output_range based on the computed values instead of user guess.

thermal_customDataset = \
    CustomPartitionedDataset(thermal_problem_parametric,
                             thermal_reduced_problem,
                             len(thermal_reduced_problem._basis_functions),
                             ann_input_samples_training,
                             ann_output_samples_training)

thermal_train_dataloader = DataLoader(thermal_customDataset, batch_size=25,
                                      shuffle=False)

thermal_customDataset = \
    CustomPartitionedDataset(thermal_problem_parametric,
                             thermal_reduced_problem,
                             len(thermal_reduced_problem._basis_functions),
                             ann_input_samples_validation,
                             ann_output_samples_validation)
thermal_valid_dataloader = DataLoader(thermal_customDataset, shuffle=False)


# ANN model
thermal_model = HiddenLayersNet(ann_input_samples_training.shape[1], [40, 40, 40],
                        len(thermal_reduced_problem._basis_functions), Sigmoid())

import torch
# torch.save(thermal_model.state_dict(), "initial_thermal_model_state_dict.pth")
thermal_model.load_state_dict(torch.load('initial_thermal_model_state_dict.pth'))

model_synchronise(thermal_model, verbose=True)

# Training of ANN
training_loss = list()
validation_loss = list()

max_epochs = 20000
min_validation_loss = None
for epochs in range(max_epochs):
    print(f"Epoch: {epochs+1}/{max_epochs}")
    current_training_loss = train_nn(thermal_reduced_problem,
                                     thermal_train_dataloader,
                                     thermal_model)
    training_loss.append(current_training_loss)
    current_validation_loss = validate_nn(thermal_reduced_problem,
                                          thermal_valid_dataloader,
                                          thermal_model)
    validation_loss.append(current_validation_loss)
    if epochs > 0 and current_validation_loss > 1.01 * min_validation_loss \
       and thermal_reduced_problem.regularisation == "EarlyStopping":
        # 1% safety margin against min_validation_loss
        # before invoking early stopping criteria
        print(f"Early stopping criteria invoked at epoch: {epochs+1}")
        break
    min_validation_loss = min(validation_loss)

exit()

# Error analysis dataset
print("\n")
print("Generating error analysis (only input/parameters) dataset")
print("\n")

error_analysis_num_para = 2 * 2 * 2 * 2

itemsize = MPI.DOUBLE.Get_size()

if world_comm.rank == 0:
    nbytes_para = error_analysis_num_para * itemsize * para_dim
    nbytes_error = error_analysis_num_para * itemsize
else:
    nbytes_para = 0
    nbytes_error = 0

win6 = MPI.Win.Allocate_shared(nbytes_para, itemsize,
                                comm=world_comm)
buf6, itemsize = win6.Shared_query(0)
thermal_error_analysis_set = \
    np.ndarray(buffer=buf6, dtype="d",
               shape=(error_analysis_num_para,
                      para_dim))

win7 = MPI.Win.Allocate_shared(nbytes_error, itemsize,
                               comm=world_comm)
buf7, itemsize = win7.Shared_query(0)
thermal_relative_error = np.ndarray(buffer=buf7, dtype="d",
                                    shape=(error_analysis_num_para))

if world_comm.rank == 0:
    thermal_error_analysis_set[:, :] = generate_ann_input_set(samples=[2, 2, 2, 2])

world_comm.Barrier()

error_analysis_indices = np.arange(world_comm.rank,
                                   thermal_error_analysis_set.shape[0],
                                   world_comm.size)

for i in error_analysis_indices:
    print(f"Error analysis parameter number {i+1} of ")
    print(f"{thermal_error_analysis_set.shape[0]}: {thermal_error_analysis_set[i, :]}")
    thermal_relative_error[i] = error_analysis(thermal_reduced_problem,
                                               thermal_problem_parametric,
                                               thermal_error_analysis_set[i, :],
                                               thermal_model,
                                               len(thermal_reduced_problem._basis_functions),
                                               online_nn, device=None)
    print(f"Error: {thermal_relative_error[i]}")

world_comm.Barrier()

# TODO Online phase at parameter online_mu

if world_comm.rank == 0:
    online_mu = np.array([0.65, 0.45, 1., 0.5])
    thermal_fem_solution = thermal_problem_parametric.solve(online_mu)
    thermal_rb_solution = \
        thermal_reduced_problem.reconstruct_solution(
            online_nn(thermal_reduced_problem, thermal_problem_parametric,
                    online_mu, thermal_model,
                    len(thermal_reduced_problem._basis_functions), device=None))

    thermal_fem_online_file \
        = "dlrbnicsx_solution_thermomechanical_thermal/thermal_fem_online_mu_computed.xdmf"

    bc_markers_list = thermal_problem_parametric._boundary_markers
    bc_list_geometric = thermal_problem_parametric.assemble_bc_list_geometric()
    with HarmonicMeshMotion(mesh, facet_tags, bc_markers_list,
                            bc_list_geometric, reset_reference=True,
                            is_deformation=True):
        with dolfinx.io.XDMFFile(mesh.comm, thermal_fem_online_file,
                                "w") as solution_file:
            solution_file.write_mesh(mesh)
            solution_file.write_function(thermal_fem_solution)

    thermal_rb_online_file \
        = "dlrbnicsx_solution_thermomechanical_thermal/thermal_rb_online_mu_computed.xdmf"
    with HarmonicMeshMotion(mesh, facet_tags, bc_markers_list,
                            bc_list_geometric, reset_reference=True,
                            is_deformation=True):
        with dolfinx.io.XDMFFile(mesh.comm, thermal_rb_online_file,
                                "w") as solution_file:
            # NOTE scatter_forward not considered for online solution
            solution_file.write_mesh(mesh)
            solution_file.write_function(thermal_rb_solution)

    thermal_error_function = dolfinx.fem.Function(thermal_problem_parametric._V)
    thermal_error_function.x.array[:] = \
        thermal_fem_solution.x.array - thermal_rb_solution.x.array
    thermal_fem_rb_error_file \
        = "dlrbnicsx_solution_thermomechanical_thermal/thermal_fem_rb_error_computed.xdmf"
    with HarmonicMeshMotion(mesh, facet_tags, bc_markers_list,
                            bc_list_geometric, reset_reference=True,
                            is_deformation=True):
        with dolfinx.io.XDMFFile(mesh.comm, thermal_fem_rb_error_file,
                                "w") as solution_file:
            solution_file.write_mesh(mesh)
            solution_file.write_function(thermal_error_function)

    with HarmonicMeshMotion(mesh, facet_tags, bc_markers_list,
                            bc_list_geometric, reset_reference=True,
                            is_deformation=True):
        print(thermal_reduced_problem.norm_error(thermal_fem_solution, thermal_rb_solution))
        print(thermal_reduced_problem.compute_norm(thermal_error_function))

    print(thermal_reduced_problem.norm_error(thermal_fem_solution, thermal_rb_solution))
    print(thermal_reduced_problem.compute_norm(thermal_error_function))

    print(f"Error array: {thermal_relative_error}")
    print(f"Eigenvalues: {eigenvalues}")
