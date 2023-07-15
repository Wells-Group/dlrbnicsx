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

from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory import Tanh
from dlrbnicsx.dataset.custom_dataset import CustomDataset
from dlrbnicsx.interface.wrappers import DataLoader
from dlrbnicsx.train_validate_test.train_validate_test import \
    train_nn, validate_nn, online_nn, error_analysis

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
        '''
        self.input_scaling_range = [-1., 1.]
        self.output_scaling_range = [-1., 1.]
        self.input_range = \
            np.array([[0.2, -0.2, 1.], [0.3, -0.4, 4.]])
        self.output_range = [-6., 3.]
        self.loss_fn = "MSE"
        self.learning_rate = 1e-5
        self.optimizer = "Adam"
        self.regularisation = "EarlyStopping"
        '''

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


# Read mesh
world_comm = MPI.COMM_WORLD
gdim = 2
gmsh_model_rank = 0
mesh, cell_tags, facet_tags = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh",
                                    world_comm, gmsh_model_rank, gdim=gdim)

# Mesh deformation parameters
mu_ref = [0.6438, 0.4313, 1., 0.5]  # reference geometry
mu = [0.8, 0.55, 0.8, 0.4]  # Parametric geometry

# FEM solve
thermal_problem_parametric = \
    ThermalProblemOnDeformedDomain(mesh, cell_tags, facet_tags)

'''
solution_mu = thermal_problem_parametric.solve(mu)
print(f"Solution norm at mu:{mu}: {thermal_problem_parametric.inner_product_action(solution_mu)(solution_mu)}")

solution_mu = thermal_problem_parametric.solve(mu_ref)
print(f"Solution norm at mu:{mu_ref}: {thermal_problem_parametric.inner_product_action(solution_mu)(solution_mu)}")
'''

# POD starts ###

def generate_training_set(samples=[2, 2, 2, 2]):#(samples=[5, 4, 5, 4]):
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

thermal_training_set = rbnicsx.io.on_rank_zero(mesh.comm, generate_training_set)

Nmax = 30

print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
print("")

print("set up thermal snapshots matrix")
thermal_snapshots_matrix = rbnicsx.backends.FunctionsList(thermal_problem_parametric._V)

print("set up reduced problem")
thermal_reduced_problem = PODANNReducedProblem(thermal_problem_parametric)

print("")

for (mu_index, mu) in enumerate(thermal_training_set):
    print(rbnicsx.io.TextLine(str(mu_index+1), fill="#"))

    print("Parameter number ", (mu_index+1), "of", thermal_training_set.shape[0])
    print("high fidelity solve for mu =", mu)
    thermal_snapshot = thermal_problem_parametric.solve(mu)

    print("update snapshots matrix")
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
# plt.show()

# POD Ends ###

# ### ANN implementation ###

def generate_ann_input_set(samples=[2, 2, 2, 2]):
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


def generate_ann_output_set(problem, reduced_problem, N,
                            input_set, indices, mode=None):
    # Solve the FE problem at given input_sets and
    # project on the RB space
    output_set = np.zeros([input_set.shape[0], N])
    for i in indices:
        if mode is None:
            print(f"Parameter number {i+1} of {input_set.shape[0]}")
            print(f"Parameter: {input_set[i,:]}")
        else:
            print(f"{mode} parameter number {i+1} of {input_set.shape[0]}")
            print(f"Parameter: {input_set[i,:]}")
        output_set[i, :] = \
            reduced_problem.project_snapshot(problem.solve(input_set[i, :]),
                                             N).array.astype("f")
    return output_set


# Training dataset
thermal_ann_input_set = generate_ann_input_set()
np.random.shuffle(thermal_ann_input_set)
thermal_ann_output_set = \
    generate_ann_output_set(thermal_problem_parametric,
                            thermal_reduced_problem,
                            len(thermal_reduced_problem._basis_functions),
                            thermal_ann_input_set, mode="Training")

num_training_samples = int(0.7 * thermal_ann_input_set.shape[0])
num_validation_samples = thermal_ann_input_set.shape[0] - num_training_samples

thermal_reduced_problem.output_range[0] = np.min(thermal_ann_output_set)
thermal_reduced_problem.output_range[1] = np.max(thermal_ann_output_set)
# NOTE Output_range based on the computed values instead of user guess.

thermal_input_training_set = \
    thermal_ann_input_set[:num_training_samples, :]
thermal_output_training_set = \
    thermal_ann_output_set[:num_training_samples, :]

thermal_input_validation_set = \
    thermal_ann_input_set[num_training_samples:, :]
thermal_output_validation_set = \
    thermal_ann_output_set[num_training_samples:, :]

thermal_customDataset = \
    CustomDataset(thermal_problem_parametric, thermal_reduced_problem,
                  len(thermal_reduced_problem._basis_functions),
                  thermal_input_training_set, thermal_output_training_set)
thermal_train_dataloader = DataLoader(thermal_customDataset, batch_size=30,
                                      shuffle=True)

thermal_customDataset = \
    CustomDataset(thermal_problem_parametric, thermal_reduced_problem,
                  len(thermal_reduced_problem._basis_functions),
                  thermal_input_validation_set, thermal_output_validation_set)
thermal_valid_dataloader = DataLoader(thermal_customDataset, shuffle=False)
