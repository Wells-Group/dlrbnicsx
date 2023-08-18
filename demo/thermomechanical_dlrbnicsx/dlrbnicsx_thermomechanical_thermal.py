import dolfinx
import ufl

from mpi4py import MPI
from petsc4py import PETSc

from dlrbnicsx_thermomechanical_geometric_deformation import MeshDeformationWrapperClass

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
from dlrbnicsx.dataset.custom_dataset import CustomDataset
from dlrbnicsx.interface.wrappers import DataLoader
from dlrbnicsx.train_validate_test.train_validate_test import \
    train_nn, validate_nn, online_nn, error_analysis

class ThermalProblemOnDeformedDomain(abc.ABC):
    def __init__(self, mesh, subdomains, boundaries):
        self._mesh = mesh
        self._subdomains = subdomains
        self._boundaries = boundaries
        ds = ufl.Measure("ds", domain=self._mesh, subdomain_data=self._boundaries)
        self._ds_sf = ds(11) + ds(20) + ds(21) + ds(22) + ds(23)
        self._ds_bottom = ds(1) + ds(31)
        self._ds_out = ds(30)
        self._ds_sym = ds(5) + ds(9) + ds(12)
        self._ds_top = ds(18) + ds(19) + ds(27) + ds(28) + ds(29)
        self._omega_1_cells = self._subdomains.find(1)
        self._omega_2_cells = self._subdomains.find(2)
        self._omega_3_cells = self._subdomains.find(3)
        self._omega_4_cells = self._subdomains.find(4)
        self._omega_5_cells = self._subdomains.find(5)
        self._omega_6_cells = self._subdomains.find(6)
        self._omega_7_cells = self._subdomains.find(7)
        x = ufl.SpatialCoordinate(self._mesh)
        self._VT = dolfinx.fem.FunctionSpace(self._mesh, ("CG", 1))
        uT, vT = ufl.TrialFunction(self._VT), ufl.TestFunction(self._VT)
        self._trial, self._test = uT, vT
        self._inner_product = ufl.inner(uT, vT) * x[0] * ufl.dx + \
            ufl.inner(ufl.grad(uT), ufl.grad(vT)) * x[0] * ufl.dx
        self.inner_product_action = \
            rbnicsx.backends.bilinear_form_action(self._inner_product,
                                                  part="real")
        self._T_f, self._T_out, self._T_bottom = 1773., 300., 300.
        self._h_cf, self._h_cout, self._h_cbottom = 2000., 200., 200.
        self._q_source = dolfinx.fem.Function(self._VT)
        self._q_source.x.array[:] = 0.
        self._q_top = dolfinx.fem.Function(self._VT)
        self._q_top.x.array[:] = 0.
        self.temperature_field = dolfinx.fem.Function(self._VT)
        self.mu_ref = [0.6438, 0.4313, 1., 0.5]

        self._Q = dolfinx.fem.FunctionSpace(self._mesh, ("DG", 0))
        self._thermal_conductivity_func = dolfinx.fem.Function(self._Q)
        self._thermal_conductivity_func_diff = dolfinx.fem.Function(self._Q)
        self._thermal_conductivity_func_1 = dolfinx.fem.Function(self._Q)
        self._thermal_conductivity_func_diff_1 = dolfinx.fem.Function(self._Q)
        self._thermal_conductivity_func_2 = dolfinx.fem.Function(self._Q)
        self._thermal_conductivity_func_diff_2 = dolfinx.fem.Function(self._Q)
        self._thermal_conductivity_func_5 = dolfinx.fem.Function(self._Q)
        self._thermal_conductivity_func_diff_5 = dolfinx.fem.Function(self._Q)
        self._thermal_conductivity_func_7 = dolfinx.fem.Function(self._Q)
        self._thermal_conductivity_func_diff_7 = dolfinx.fem.Function(self._Q)

        self._max_iterations = 20
        self.rtol = 1.e-4
        self.atol = 1.e-12
        
        sym_T = sympy.Symbol("sym_T")
        
        self._thermal_conductivity_func.x.array[self._omega_3_cells] = 5.3
        self._thermal_conductivity_func.x.array[self._omega_4_cells] = 4.75
        self._thermal_conductivity_func.x.array[self._omega_6_cells] = 45.6

        thermal_conductivity_sym_1 = sympy.interpolating_spline(2, sym_T, [293., 473., 873., 1273.], [16.07, 15.53, 15.97, 17.23])
        thermal_conductivity_sym_1 = sympy.Piecewise(
            thermal_conductivity_sym_1.args[0], (thermal_conductivity_sym_1.args[1][0], True))
        self._thermal_conductivity_sym_1_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_1)
        thermal_conductivity_sym_diff_1 = sympy.diff(thermal_conductivity_sym_1, sym_T)
        self._thermal_conductivity_sym_diff_1_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_diff_1)

        thermal_conductivity_sym_2 = sympy.interpolating_spline(2, sym_T, [293., 473., 873., 1273.], [49.35, 24.75, 27.06, 38.24])
        thermal_conductivity_sym_2 = sympy.Piecewise(
            thermal_conductivity_sym_2.args[0], (thermal_conductivity_sym_2.args[1][0], True))
        self._thermal_conductivity_sym_2_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_2)
        thermal_conductivity_sym_diff_2 = sympy.diff(thermal_conductivity_sym_2, sym_T)
        self._thermal_conductivity_sym_diff_2_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_diff_2)

        thermal_conductivity_sym_5 = sympy.interpolating_spline(2, sym_T, [293., 473., 873., 1273.], [23.34, 20.81, 20.99, 21.62])
        thermal_conductivity_sym_5 = sympy.Piecewise(
            thermal_conductivity_sym_5.args[0], (thermal_conductivity_sym_5.args[1][0], True))
        self._thermal_conductivity_sym_5_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_5)
        thermal_conductivity_sym_diff_5 = sympy.diff(thermal_conductivity_sym_5, sym_T)
        self._thermal_conductivity_sym_diff_5_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_diff_5)

        thermal_conductivity_sym_7 = sympy.interpolating_spline(2, sym_T, [293., 473., 873., 1273.], [49.35, 24.75, 27.06, 38.24])
        thermal_conductivity_sym_7 = sympy.Piecewise(
            thermal_conductivity_sym_7.args[0], (thermal_conductivity_sym_7.args[1][0], True))
        self._thermal_conductivity_sym_7_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_7)
        thermal_conductivity_sym_diff_7 = sympy.diff(thermal_conductivity_sym_7, sym_T)
        self._thermal_conductivity_sym_diff_7_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_diff_7)
        
    def conductivity_eval_1(self, x):
        temperature_field = self.temperature_field
        tree = dolfinx.geometry.bb_tree(self._mesh, self._mesh.geometry.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self._mesh, cell_candidates, x.T)
        return self._thermal_conductivity_sym_1_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

    def conductivity_eval_diff_1(self, x):
        temperature_field = self.temperature_field
        tree = dolfinx.geometry.bb_tree(self._mesh, self._mesh.geometry.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self._mesh, cell_candidates, x.T)
        return self._thermal_conductivity_sym_diff_1_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

    def conductivity_eval_2(self, x):
        temperature_field = self.temperature_field
        tree = dolfinx.geometry.bb_tree(self._mesh, self._mesh.geometry.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self._mesh, cell_candidates, x.T)
        return self._thermal_conductivity_sym_2_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

    def conductivity_eval_diff_2(self, x):
        temperature_field = self.temperature_field
        tree = dolfinx.geometry.bb_tree(self._mesh, self._mesh.geometry.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self._mesh, cell_candidates, x.T)
        return self._thermal_conductivity_sym_diff_2_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

    def conductivity_eval_5(self, x):
        temperature_field = self.temperature_field
        tree = dolfinx.geometry.bb_tree(self._mesh, self._mesh.geometry.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self._mesh, cell_candidates, x.T)
        return self._thermal_conductivity_sym_5_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

    def conductivity_eval_diff_5(self, x):
        temperature_field = self.temperature_field
        tree = dolfinx.geometry.bb_tree(self._mesh, self._mesh.geometry.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self._mesh, cell_candidates, x.T)
        return self._thermal_conductivity_sym_diff_5_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

    def conductivity_eval_7(self, x):
        temperature_field = self.temperature_field
        tree = dolfinx.geometry.bb_tree(self._mesh, self._mesh.geometry.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self._mesh, cell_candidates, x.T)
        return self._thermal_conductivity_sym_7_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

    def conductivity_eval_diff_7(self, x):
        temperature_field = self.temperature_field
        tree = dolfinx.geometry.bb_tree(self._mesh, self._mesh.geometry.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self._mesh, cell_candidates, x.T)
        return self._thermal_conductivity_sym_diff_7_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])
    
    @property
    def bilinear_form(self):
        uT, vT = self._trial, self._test
        x = ufl.SpatialCoordinate(self._mesh)
        JaT = ufl.inner(self._thermal_conductivity_func * ufl.grad(uT), ufl.grad(vT)) * x[0] * ufl.dx + \
            ufl.inner(uT * self._thermal_conductivity_func_diff * ufl.grad(self.temperature_field), ufl.grad(vT)) * x[0] * ufl.dx
        cT = ufl.inner(self._h_cf * uT, vT) * x[0] * self._ds_sf + ufl.inner(self._h_cout * uT, vT) * x[0] * self._ds_out + \
            ufl.inner(self._h_cbottom * uT, vT) * x[0] * self._ds_bottom
        return dolfinx.fem.form(JaT + cT)
    
    @property
    def linear_form(self):
        vT = self._test
        x = ufl.SpatialCoordinate(self._mesh)
        JlT = ufl.inner(self.temperature_field * self._thermal_conductivity_func_diff * \
            ufl.grad(self.temperature_field), ufl.grad(vT)) * x[0] * ufl.dx
        lT = ufl.inner(self._q_source, vT) * x[0] * ufl.dx + self._h_cf * vT * self._T_f * x[0] * self._ds_sf + \
            self._h_cout * vT * self._T_out * x[0] * self._ds_out + \
            self._h_cbottom * vT * self._T_bottom * x[0] * self._ds_bottom - \
            ufl.inner(self._q_top, vT) * x[0] * self._ds_top
        return dolfinx.fem.form(JlT + lT)
    
    def solve(self, mu):
        vT = self._test
        self.mu = mu
        self.temperature_field.x.array[:] = 350.
        with MeshDeformationWrapperClass(self._mesh, self._boundaries,
                                         self.mu_ref, self.mu):
            x = ufl.SpatialCoordinate(self._mesh)
            for iteration in range(self._max_iterations):
                print(f"Iteration {iteration + 1}/{self._max_iterations}")

                self._thermal_conductivity_func_1.interpolate(self.conductivity_eval_1)
                self._thermal_conductivity_func_diff_1.interpolate(self.conductivity_eval_diff_1)

                self._thermal_conductivity_func_2.interpolate(self.conductivity_eval_2)
                self._thermal_conductivity_func_diff_2.interpolate(self.conductivity_eval_diff_2)

                self._thermal_conductivity_func_5.interpolate(self.conductivity_eval_5)
                self._thermal_conductivity_func_diff_5 .interpolate(self.conductivity_eval_diff_5)

                self._thermal_conductivity_func_7.interpolate(self.conductivity_eval_7)
                self._thermal_conductivity_func_diff_7.interpolate(self.conductivity_eval_diff_7)

                self._thermal_conductivity_func.x.array[self._omega_1_cells] = self._thermal_conductivity_func_1.x.array[self._omega_1_cells]
                self._thermal_conductivity_func_diff.x.array[self._omega_1_cells] = self._thermal_conductivity_func_diff_1.x.array[self._omega_1_cells]

                self._thermal_conductivity_func.x.array[self._omega_2_cells] = self._thermal_conductivity_func_2.x.array[self._omega_2_cells]
                self._thermal_conductivity_func_diff.x.array[self._omega_2_cells] = self._thermal_conductivity_func_diff_2.x.array[self._omega_2_cells]

                self._thermal_conductivity_func.x.array[self._omega_5_cells] = self._thermal_conductivity_func_5.x.array[self._omega_5_cells]
                self._thermal_conductivity_func_diff.x.array[self._omega_5_cells] = self._thermal_conductivity_func_diff_5.x.array[self._omega_5_cells]

                self._thermal_conductivity_func.x.array[self._omega_7_cells] = self._thermal_conductivity_func_7.x.array[self._omega_7_cells]
                self._thermal_conductivity_func_diff.x.array[self._omega_7_cells] = self._thermal_conductivity_func_diff_7.x.array[self._omega_7_cells]

                residual = abs(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(self._thermal_conductivity_func * ufl.grad(self.temperature_field),
                                                                                                            ufl.grad(vT)) * x[0] * ufl.dx +
                self._h_cf * ufl.inner(self.temperature_field - self._T_f, vT) * x[0] * self._ds_sf + self._h_cbottom * ufl.inner(self.temperature_field - self._T_bottom, vT) * x[0] * self._ds_bottom + self._h_cout * ufl.inner(self.temperature_field - self._T_out, vT) * x[0] * self._ds_out)), op=MPI.SUM))

                if iteration == 0:
                    initial_residual = residual

                # print(f"Residual: {residual/initial_residual}")
                print(f"Residual: {residual}")
                
                '''
                if residual/initial_residual < rtol:
                    print(f"Residual tolerance {rtol} reached in iterations {iteration}")
                    break
                '''

                if residual < self.rtol:
                    print(f"Residual tolerance {self.rtol} reached in iterations {iteration}")
                    break
                
                aT_cpp = self.bilinear_form
                lT_cpp = self.linear_form

                # Bilinear side assembly
                A = dolfinx.fem.petsc.assemble_matrix(aT_cpp, bcs=[])
                A.assemble()

                # Linear side assembly
                L = dolfinx.fem.petsc.assemble_vector(lT_cpp)
                dolfinx.fem.petsc.apply_lifting(L, [aT_cpp], [[]])
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
                solution_field = dolfinx.fem.Function(self._VT)
                ksp.solve(L, solution_field.vector)
                solution_field.x.scatter_forward()
                # print(solution_field.x.array)

                update_abs = \
                    (np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(solution_field - self.temperature_field,
                                                                                            solution_field - self.temperature_field) * x[0] * ufl.dx)), op=MPI.SUM)))/\
                    (np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(self.temperature_field, self.temperature_field) * x[0] * ufl.dx)), op=MPI.SUM)))

                print(f"Absolute update: {update_abs}")

                self.temperature_field.x.array[:] = solution_field.x.array.copy()

                if update_abs < self.atol:
                    print(f"Solver tolerance {self.atol} reached in iterations {iteration + 1}")
                    break
        
        return solution_field

class MechanicalProblemOnDeformedDomain(abc.ABC):
    def __init__(self, mesh, subdomains, boundaries, thermalproblem):
        self._mesh = mesh
        self._subdomains = subdomains
        self._boundaries = boundaries
        self._thermalproblem = thermalproblem
        ds = ufl.Measure("ds", domain=self._mesh, subdomain_data=self._boundaries)
        self._ds_sf = ds(11) + ds(20) + ds(21) + ds(22) + ds(23)
        self._ds_bottom = ds(1) + ds(31)
        self._ds_out = ds(30)
        self._ds_sym = ds(5) + ds(9) + ds(12)
        self._ds_top = ds(18) + ds(19) + ds(27) + ds(28) + ds(29)
        self._omega_1_cells = self._subdomains.find(1)
        self._omega_2_cells = self._subdomains.find(2)
        self._omega_3_cells = self._subdomains.find(3)
        self._omega_4_cells = self._subdomains.find(4)
        self._omega_5_cells = self._subdomains.find(5)
        self._omega_6_cells = self._subdomains.find(6)
        self._omega_7_cells = self._subdomains.find(7)
        x = ufl.SpatialCoordinate(self._mesh)
        # NOTE Placeholder for ymax, Updated at each new parameter
        self._ymax = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.))
        self._VM = dolfinx.fem.VectorFunctionSpace(self._mesh, ("CG", 1))
        uM, vM = ufl.TrialFunction(self._VM), ufl.TestFunction(self._VM)
        self._trial, self._test = uM, vM
        self._inner_product = ufl.inner(uM, vM) * x[0] * ufl.dx + \
            ufl.inner(self.epsilon(uM), self.epsilon(vM)) * x[0] * ufl.dx
        self.inner_product_action = \
            rbnicsx.backends.bilinear_form_action(self._inner_product,
                                                  part="real")
        self._rho = 77106.
        self._g = 9.8
        self._T0 = 300.
        self.mu_ref = [0.6438, 0.4313, 1., 0.5]

        sym_T = sympy.Symbol("sym_T")

        young_modulus_sym_1 = sympy.interpolating_spline(2, sym_T, [293., 573., 1073., 1273.], [10.5e9, 10.3e9, 10.4e9, 10.3e9])
        young_modulus_sym_1 = sympy.Piecewise(
            young_modulus_sym_1.args[0], (young_modulus_sym_1.args[1][0], True))
        self._young_modulus_sym_1_lambdified = sympy.lambdify(sym_T, young_modulus_sym_1)

        young_modulus_sym_2 = sympy.interpolating_spline(2, sym_T, [293., 573., 1073., 1273.], [15.4e9, 14.7e9, 13.8e9, 14.4e9])
        young_modulus_sym_2 = sympy.Piecewise(
            young_modulus_sym_2.args[0], (young_modulus_sym_2.args[1][0], True))
        self._young_modulus_sym_2_lambdified = sympy.lambdify(sym_T, young_modulus_sym_2)

        young_modulus_sym_3 = sympy.interpolating_spline(2, sym_T, [293., 573., 1073., 1273.], [58.2e9, 67.3e9, 52.9e9, 51.6e9])
        young_modulus_sym_3 = sympy.Piecewise(
            young_modulus_sym_3.args[0], (young_modulus_sym_3.args[1][0], True))
        self._young_modulus_sym_3_lambdified = sympy.lambdify(sym_T, young_modulus_sym_3)

        young_modulus_sym_4 = sympy.interpolating_spline(2, sym_T, [293., 573., 1073., 1273.], [1.85e9, 1.92e9, 1.83e9, 1.85e9])
        young_modulus_sym_4 = sympy.Piecewise(
            young_modulus_sym_4.args[0], (young_modulus_sym_4.args[1][0], True))
        self._young_modulus_sym_4_lambdified = sympy.lambdify(sym_T, young_modulus_sym_4)

        young_modulus_sym_5 = sympy.interpolating_spline(2, sym_T, [293., 573., 1073., 1273.], [14.5e9, 15.0e9, 15.3e9, 13.3e9])
        young_modulus_sym_5 = sympy.Piecewise(
            young_modulus_sym_5.args[0], (young_modulus_sym_5.args[1][0], True))
        self._young_modulus_sym_5_lambdified = sympy.lambdify(sym_T, young_modulus_sym_5)

        young_modulus_sym_7 = sympy.interpolating_spline(2, sym_T, [293., 573., 1073., 1273.], [15.4e9, 14.7e9, 13.8e9, 14.4e9])
        young_modulus_sym_7 = sympy.Piecewise(
            young_modulus_sym_7.args[0], (young_modulus_sym_7.args[1][0], True))
        self._young_modulus_sym_7_lambdified = sympy.lambdify(sym_T, young_modulus_sym_7)

        self._Q = dolfinx.fem.FunctionSpace(self._mesh, ("DG", 0))
        self._poisson_ratio_func = dolfinx.fem.Function(self._Q)
        self._poisson_ratio_func.x.array[self._omega_1_cells] = 0.3
        self._poisson_ratio_func.x.array[self._omega_2_cells] = 0.2
        self._poisson_ratio_func.x.array[self._omega_3_cells] = 0.1
        self._poisson_ratio_func.x.array[self._omega_4_cells] = 0.1
        self._poisson_ratio_func.x.array[self._omega_5_cells] = 0.2
        self._poisson_ratio_func.x.array[self._omega_6_cells] = 0.3
        self._poisson_ratio_func.x.array[self._omega_7_cells] = 0.2

        self._thermal_expansion_coefficient_func = dolfinx.fem.Function(self._Q)
        self._thermal_expansion_coefficient_func.x.array[self._omega_1_cells] = 2.3e-6
        self._thermal_expansion_coefficient_func.x.array[self._omega_2_cells] = 4.6e-6
        self._thermal_expansion_coefficient_func.x.array[self._omega_3_cells] = 4.7e-6
        self._thermal_expansion_coefficient_func.x.array[self._omega_4_cells] = 4.6e-6
        self._thermal_expansion_coefficient_func.x.array[self._omega_5_cells] = 6.e-6
        self._thermal_expansion_coefficient_func.x.array[self._omega_6_cells] = 1.2e-5
        self._thermal_expansion_coefficient_func.x.array[self._omega_7_cells] = 4.6e-6

        self._young_modulus_func = dolfinx.fem.Function(self._Q)
        self._young_modulus_func_1 = dolfinx.fem.Function(self._Q)
        self._young_modulus_func_2 = dolfinx.fem.Function(self._Q)
        self._young_modulus_func_3 = dolfinx.fem.Function(self._Q)
        self._young_modulus_func_4 = dolfinx.fem.Function(self._Q)
        self._young_modulus_func_5 = dolfinx.fem.Function(self._Q)
        self._young_modulus_func_7 = dolfinx.fem.Function(self._Q)
        self._young_modulus_func.x.array[self._omega_6_cells] = 1.9E11

        dofs_bottom_1 = dolfinx.fem.locate_dofs_topological(self._VM.sub(1), self._mesh.geometry.dim-1, self._boundaries.find(1))
        dofs_bottom_31 = dolfinx.fem.locate_dofs_topological(self._VM.sub(1), self._mesh.geometry.dim-1, self._boundaries.find(31))
        dofs_sym_5 = dolfinx.fem.locate_dofs_topological(self._VM.sub(0), self._mesh.geometry.dim-1, self._boundaries.find(5))
        dofs_sym_9 = dolfinx.fem.locate_dofs_topological(self._VM.sub(0), self._mesh.geometry.dim-1, self._boundaries.find(9))
        dofs_sym_12 = dolfinx.fem.locate_dofs_topological(self._VM.sub(0), self._mesh.geometry.dim-1, self._boundaries.find(12))

        bc_bottom_1 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_bottom_1, self._VM.sub(1))
        bc_bottom_31 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_bottom_31, self._VM.sub(1))
        bc_sym_5 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_sym_5, self._VM.sub(0))
        bc_sym_9 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_sym_9, self._VM.sub(0))
        bc_sym_12 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_sym_12, self._VM.sub(0))

        self._bcsM = [bc_bottom_1, bc_bottom_31, bc_sym_5, bc_sym_9, bc_sym_12]

    def young_modulus_eval_1(self, x):
        tree = dolfinx.geometry.bb_tree(self._mesh, self._mesh.geometry.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self._mesh, cell_candidates, x.T)
        return self._young_modulus_sym_1_lambdified(self.temperature_field.eval(x.T, colliding_cells.array)[:, 0])

    def young_modulus_eval_2(self, x):
        tree = dolfinx.geometry.bb_tree(self._mesh, self._mesh.geometry.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self._mesh, cell_candidates, x.T)
        return self._young_modulus_sym_2_lambdified(self.temperature_field.eval(x.T, colliding_cells.array)[:, 0])

    def young_modulus_eval_3(self, x):
        tree = dolfinx.geometry.bb_tree(self._mesh, self._mesh.geometry.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self._mesh, cell_candidates, x.T)
        return self._young_modulus_sym_3_lambdified(self.temperature_field.eval(x.T, colliding_cells.array)[:, 0])

    def young_modulus_eval_4(self, x):
        tree = dolfinx.geometry.bb_tree(self._mesh, self._mesh.geometry.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self._mesh, cell_candidates, x.T)
        return self._young_modulus_sym_4_lambdified(self.temperature_field.eval(x.T, colliding_cells.array)[:, 0])

    def young_modulus_eval_5(self, x):
        tree = dolfinx.geometry.bb_tree(self._mesh, self._mesh.geometry.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self._mesh, cell_candidates, x.T)
        return self._young_modulus_sym_5_lambdified(self.temperature_field.eval(x.T, colliding_cells.array)[:, 0])

    def young_modulus_eval_7(self, x):
        tree = dolfinx.geometry.bb_tree(self._mesh, self._mesh.geometry.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(tree, x.T)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self._mesh, cell_candidates, x.T)
        return self._young_modulus_sym_7_lambdified(self.temperature_field.eval(x.T, colliding_cells.array)[:, 0])

    def epsilon(self, u):
        x = ufl.SpatialCoordinate(self._mesh)
        return ufl.as_tensor([[u[0].dx(0), 0.5*(u[0].dx(1)+u[1].dx(0)), 0.],[0.5*(u[0].dx(1)+u[1].dx(0)), u[1].dx(1), 0.],[0., 0., u[0]/x[0]]]) # ufl.sym(ufl.grad(u))

    def sigma(self, u):
        E = self._young_modulus_func
        nu = self._poisson_ratio_func
        epsilon = self.epsilon
        lambda_ = E * nu / ((1 - 2 * nu) * (1 + nu))
        mu = E / (2 * (1 + nu))
        x = ufl.SpatialCoordinate(self._mesh)
        return lambda_ * (u[0].dx(0) + u[1].dx(1) + u[0]/x[0]) * ufl.Identity(3) + 2 * mu * epsilon(u) # lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

    @property
    def bilinear_form(self):
        x = ufl.SpatialCoordinate(self._mesh)
        uM, vM = self._trial, self._test
        aM = ufl.inner(self.sigma(uM), self.epsilon(vM)) * x[0] * ufl.dx
        return dolfinx.fem.form(aM)

    @property
    def linear_form(self):
        x = ufl.SpatialCoordinate(self._mesh)
        vM = self._test
        n_vec = ufl.FacetNormal(self._mesh)
        lM = (self.temperature_field - self._T0) * self._young_modulus_func/(1 - 2 * self._poisson_ratio_func) * self._thermal_expansion_coefficient_func * (vM[0].dx(0) + vM[1].dx(1) + vM[0]/x[0]) * x[0] * ufl.dx - self._rho * self._g * (self._ymax - x[1]) * ufl.dot(vM, n_vec) * x[0] * self._ds_sf
        return dolfinx.fem.form(lM)

    def solve(self, mu):
        self.mu = mu
        # NOTE VVIP, make sure temperature_field is solved before geometric deformation of mechanical problem else
        # the mesh gets deformed twice for thermal problem
        # 1. One in solve of thermal problem
        # 2. Other in solve of mehcanical problem
        self.temperature_field = self._thermalproblem.solve(self.mu)
        print(f"Temperature field norm: {self._thermalproblem.inner_product_action(self.temperature_field)(self.temperature_field)}")
        with MeshDeformationWrapperClass(self._mesh, self._boundaries,
                                         self.mu_ref, self.mu):

            self._ymax.value = self._mesh.comm.allreduce(np.max(self._mesh.geometry.x[:, 1]), op=MPI.MAX)

            self._young_modulus_func_1.interpolate(self.young_modulus_eval_1)
            self._young_modulus_func_2.interpolate(self.young_modulus_eval_2)
            self._young_modulus_func_3.interpolate(self.young_modulus_eval_3)
            self._young_modulus_func_4.interpolate(self.young_modulus_eval_4)
            self._young_modulus_func_5.interpolate(self.young_modulus_eval_5)
            self._young_modulus_func_7.interpolate(self.young_modulus_eval_7)

            self._young_modulus_func.x.array[self._omega_1_cells] = self._young_modulus_func_1.x.array[self._omega_1_cells]
            self._young_modulus_func.x.array[self._omega_2_cells] = self._young_modulus_func_2.x.array[self._omega_2_cells]
            self._young_modulus_func.x.array[self._omega_3_cells] = self._young_modulus_func_3.x.array[self._omega_3_cells]
            self._young_modulus_func.x.array[self._omega_4_cells] = self._young_modulus_func_4.x.array[self._omega_4_cells]
            self._young_modulus_func.x.array[self._omega_5_cells] = self._young_modulus_func_5.x.array[self._omega_5_cells]
            self._young_modulus_func.x.array[self._omega_7_cells] = self._young_modulus_func_7.x.array[self._omega_7_cells]

            # Bilinear side assembly
            aM_cpp = self.bilinear_form
            A = dolfinx.fem.petsc.assemble_matrix(aM_cpp, bcs=self._bcsM)
            A.assemble()

            # Linear side assembly
            lM_cpp = self.linear_form
            L = dolfinx.fem.petsc.assemble_vector(lM_cpp)
            dolfinx.fem.petsc.apply_lifting(L, [aM_cpp], [self._bcsM])
            L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            dolfinx.fem.petsc.set_bc(L, self._bcsM)

            # Solver setup
            ksp = PETSc.KSP()
            ksp.create(self._mesh.comm)
            ksp.setOperators(A)
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType("mumps")
            ksp.setFromOptions()
            displacement_field = dolfinx.fem.Function(self._VM)
            ksp.solve(L, displacement_field.vector)
            displacement_field.x.scatter_forward()
        return displacement_field

class ThermalPODANNReducedProblem(abc.ABC):
    def __init__(self, thermal_problem) -> None:
        self._basis_functions = rbnicsx.backends.FunctionsList(thermal_problem._VT)
        uT, vT = ufl.TrialFunction(thermal_problem._VT), ufl.TestFunction(thermal_problem._VT)
        x = ufl.SpatialCoordinate(thermal_problem._mesh)
        self._inner_product = ufl.inner(uT, vT) * x[0] * ufl.dx + \
            ufl.inner(ufl.grad(uT), ufl.grad(vT)) * x[0] * ufl.dx
        self._inner_product_action = \
            rbnicsx.backends.bilinear_form_action(self._inner_product,
                                                  part="real")
        self.input_scaling_range = [-1., 1.]
        self.output_scaling_range = [-1., 1.]
        self.input_range = \
            np.array([[0.55, 0.35, 0.8, 0.4], [0.75, 0.55, 1.2, 0.6]])
        self.output_range = [-6., 3.]
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


class MechanicalPODANNReducedProblem(abc.ABC):
    def __init__(self, mechanical_problem) -> None:
        self._basis_functions = rbnicsx.backends.FunctionsList(mechanical_problem._VM)
        uM, vM = ufl.TrialFunction(mechanical_problem._VM), ufl.TestFunction(mechanical_problem._VM)
        x = ufl.SpatialCoordinate(mechanical_problem._mesh)
        self._inner_product = ufl.inner(uM, vM) * x[0] * ufl.dx + \
            ufl.inner(mechanical_problem.epsilon(uM), mechanical_problem.epsilon(vM)) * x[0] * ufl.dx
        self._inner_product_action = \
            rbnicsx.backends.bilinear_form_action(self._inner_product,
                                                  part="real")
        self.input_scaling_range = [-1., 1.]
        self.output_scaling_range = [-1., 1.]
        self.input_range = \
            np.array([[0.55, 0.35, 0.8, 0.4], [0.75, 0.55, 1.2, 0.6]])
        self.output_range = [-6., 3.]
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

# Read mesh
world_comm = MPI.COMM_WORLD
gdim = 2
gmsh_model_rank = 0
mesh, cell_tags, facet_tags = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh",
                                    world_comm, gmsh_model_rank, gdim=gdim)

# Mesh deformation parameters
mu_ref = [0.6438, 0.4313, 1., 0.5]  # reference geometry
mu = [0.45, 0.56, 0.9, 0.7] # [0.8, 0.55, 0.8, 0.4]  # Parametric geometry

pod_ann_samples = [3, 4, 3, 4]


# FEM solve
thermal_problem_parametric = \
    ThermalProblemOnDeformedDomain(mesh, cell_tags, facet_tags)

# solution_mu = thermal_problem_parametric.solve(mu_ref)
# print(f"Solution norm at mu:{mu_ref}: {thermal_problem_parametric.inner_product_action(solution_mu)(solution_mu)}")

solution_mu = thermal_problem_parametric.solve(mu)
print(f"Solution norm at mu:{mu}: {thermal_problem_parametric.inner_product_action(solution_mu)(solution_mu)}")

computed_file = "solution_nonlinear_thermomechanical_thermal/solution_computed.xdmf"

with MeshDeformationWrapperClass(mesh, facet_tags, mu_ref, mu):
    with dolfinx.io.XDMFFile(mesh.comm, computed_file, "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(solution_mu)

# Thermal POD Starts ###

def generate_training_set(samples=pod_ann_samples):
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

print("set up snapshots matrix")
thermal_snapshots_matrix = rbnicsx.backends.FunctionsList(thermal_problem_parametric._VT)

print("set up reduced problem")
thermal_reduced_problem = ThermalPODANNReducedProblem(thermal_problem_parametric)

print("")

for (mu_index, mu) in enumerate(thermal_training_set):
    print(rbnicsx.io.TextLine(str(mu_index+1), fill="#"))

    print("Parameter number ", (mu_index+1), "of", thermal_training_set.shape[0])
    print("high fidelity solve for mu =", mu)
    snapshot = thermal_problem_parametric.solve(mu)

    print("update snapshots matrix")
    thermal_snapshots_matrix.append(snapshot)

    print("")

print(rbnicsx.io.TextLine("perform POD", fill="#"))
thermal_eigenvalues, thermal_modes, _ = \
    rbnicsx.backends.\
    proper_orthogonal_decomposition(thermal_snapshots_matrix,
                                    thermal_reduced_problem._inner_product_action,
                                    N=Nmax, tol=1.e-6)
thermal_reduced_problem._basis_functions.extend(thermal_modes)
thermal_reduced_size = len(thermal_reduced_problem._basis_functions)
print("")

print(rbnicsx.io.TextBox("POD-Galerkin offline phase ends", fill="="))

thermal_positive_eigenvalues = np.where(thermal_eigenvalues > 0., thermal_eigenvalues, np.nan)
thermal_singular_values = np.sqrt(thermal_positive_eigenvalues)

plt.figure(figsize=[8, 10])
xint = list()
yval = list()

for x, y in enumerate(thermal_eigenvalues[:Nmax]):
    yval.append(y)
    xint.append(x+1)

plt.plot(xint, yval, "*-", color="orange")
plt.xlabel("Eigenvalue number", fontsize=18)
plt.ylabel("Eigenvalue", fontsize=18)
plt.xticks(xint)
plt.yscale("log")
plt.title("Eigenvalue decay (Thermal)", fontsize=24)
plt.tight_layout()
plt.savefig("thermal_eigenvalues.png")

print(f"Eigenvalues (Thermal): {thermal_positive_eigenvalues}")

# Thermal POD Ends ###

# Mechanical POD starts ###
mechanical_problem_parametric = \
    MechanicalProblemOnDeformedDomain(mesh, cell_tags, facet_tags,
                                      thermal_problem_parametric)

# solution_mu = mechanical_problem_parametric.solve(mu_ref)
# print(f"Solution norm at mu:{mu_ref}: {mechanical_problem_parametric.inner_product_action(solution_mu)(solution_mu)}")

solution_mu = mechanical_problem_parametric.solve(mu)
print(f"Solution norm at mu:{mu}: {mechanical_problem_parametric.inner_product_action(solution_mu)(solution_mu)}")

computed_file = "solution_nonlinear_thermomechanical_mechanical/solution_computed.xdmf"
with MeshDeformationWrapperClass(mesh, facet_tags, mu_ref, mu):
    with dolfinx.io.XDMFFile(mesh.comm, computed_file, "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(solution_mu)

# Mechanical POD starts ###

# NOTE using same generate_training_set as Thermal problem

mechanical_training_set = rbnicsx.io.on_rank_zero(mesh.comm, generate_training_set)

Nmax = 30

print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
print("")

print("set up snapshots matrix")
mechanical_snapshots_matrix = rbnicsx.backends.FunctionsList(mechanical_problem_parametric._VM)

print("set up reduced problem")
mechanical_reduced_problem = MechanicalPODANNReducedProblem(mechanical_problem_parametric)

print("")

for (mu_index, mu) in enumerate(mechanical_training_set):
    print(rbnicsx.io.TextLine(str(mu_index+1), fill="#"))

    print("Parameter number ", (mu_index+1), "of", mechanical_training_set.shape[0])
    print("high fidelity solve for mu =", mu)
    snapshot = mechanical_problem_parametric.solve(mu)

    print("update snapshots matrix")
    mechanical_snapshots_matrix.append(snapshot)

    print("")

print(rbnicsx.io.TextLine("perform POD", fill="#"))
mechanical_eigenvalues, mechanical_modes, _ = \
    rbnicsx.backends.\
    proper_orthogonal_decomposition(mechanical_snapshots_matrix,
                                    mechanical_reduced_problem._inner_product_action,
                                    N=Nmax, tol=1.e-6)
mechanical_reduced_problem._basis_functions.extend(mechanical_modes)
mechanical_reduced_size = len(mechanical_reduced_problem._basis_functions)
print("")

print(rbnicsx.io.TextBox("POD-Galerkin offline phase ends", fill="="))

mechanical_positive_eigenvalues = np.where(mechanical_eigenvalues > 0., mechanical_eigenvalues, np.nan)
mechanical_singular_values = np.sqrt(mechanical_positive_eigenvalues)

plt.figure(figsize=[8, 10])
xint = list()
yval = list()

for x, y in enumerate(mechanical_eigenvalues[:Nmax]):
    yval.append(y)
    xint.append(x+1)

plt.plot(xint, yval, "*-", color="orange")
plt.xlabel("Eigenvalue number", fontsize=18)
plt.ylabel("Eigenvalue", fontsize=18)
plt.xticks(xint)
plt.yscale("log")
plt.title("Eigenvalue decay (Mechanical)", fontsize=24)
plt.tight_layout()
plt.savefig("mechanical_eigenvalues.png")

print(f"Eigenvalues (Mechanical): {mechanical_positive_eigenvalues}")

# Mechanical POD Ends ###

# Thermal ANN starts ###



# Thermal ANN ends ###
