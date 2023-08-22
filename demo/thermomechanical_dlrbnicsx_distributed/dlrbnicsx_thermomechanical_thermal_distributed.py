import dolfinx
import ufl

import rbnicsx
import rbnicsx.online
import rbnicsx.backends

from dlrbnicsx_thermomechanical_geometric_deformation import MeshDeformationWrapperClass

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import sympy
import itertools
import abc
import matplotlib.pyplot as plt
import os

from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory \
    import Tanh
from dlrbnicsx.dataset.custom_partitioned_dataset \
    import CustomPartitionedDataset
from dlrbnicsx.interface.wrappers import DataLoader, save_model, \
    load_model, save_checkpoint, load_checkpoint, model_synchronise, \
    init_cpu_process_group, get_optimiser, get_loss_func, share_model
from dlrbnicsx.train_validate_test.train_validate_test_distributed \
    import train_nn, validate_nn, online_nn, error_analysis

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


# MPI communicator variables
world_comm = MPI.COMM_WORLD
rank = world_comm.Get_rank()
size = world_comm.Get_size()

# Read mesh
mesh_comm = MPI.COMM_SELF  # NOTE
gdim = 2
gmsh_model_rank = 0
mesh, cell_tags, facet_tags = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh",
                                    mesh_comm, gmsh_model_rank, gdim=gdim)


# Mesh deformation parameters
mu_ref = [0.6438, 0.4313, 1., 0.5]  # reference geometry
mu = [0.45, 0.56, 0.9, 0.7] # [0.8, 0.55, 0.8, 0.4]  # Parametric geometry

thermal_problem_parametric = ThermalProblemOnDeformedDomain(mesh, cell_tags,
                                                    facet_tags)
solution_mu = thermal_problem_parametric.solve(mu)

itemsize = MPI.DOUBLE.Get_size()
para_dim = 4
pod_samples = [3, 4, 3, 4]
ann_samples = [4, 3, 4, 3]
error_analysis_samples = [2, 2, 2, 2]
thermal_num_dofs = solution_mu.x.array.shape[0]
num_snapshots = np.product(pod_samples)
nbytes_para = itemsize * num_snapshots * para_dim
nbytes_dofs = itemsize * num_snapshots * thermal_num_dofs

# Thermal POD Starts ###

def generate_training_set(samples=pod_samples):
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
    thermal_training_set[:, :] = generate_training_set(samples=pod_samples)


world_comm.Barrier()

win1 = MPI.Win.Allocate_shared(nbytes_dofs, itemsize, comm=MPI.COMM_WORLD)
buf1, itemsize = win1.Shared_query(0)
thermal_training_set_solution = np.ndarray(buffer=buf1, dtype="d", shape=(num_snapshots, thermal_num_dofs))

# Solution manifold
indices = np.arange(world_comm.rank, num_snapshots, world_comm.size)

for i in indices:
    print(f"Solving FEM problem {i+1}/{num_snapshots}")
    thermal_training_set_solution[i, :] = (thermal_problem_parametric.solve(thermal_training_set[i, :])).x.array

world_comm.Barrier()

# Maximum RB size
Nmax = 30

print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
print("")

print("Set up snapshots matrix")
snapshots_matrix = rbnicsx.backends.FunctionsList(thermal_problem_parametric._VT)

for i in range(num_snapshots):
    snapshot = dolfinx.fem.Function(thermal_problem_parametric._VT)
    snapshot.x.array[:] = thermal_training_set_solution[i, :]

    print(f"Update snapshots matrix: {i+1}/{num_snapshots}")
    snapshots_matrix.append(snapshot)

print("Set up reduced problem")
thermal_reduced_problem = ThermalPODANNReducedProblem(thermal_problem_parametric)

print("")

print(rbnicsx.io.TextLine("Perform POD", fill="#"))
thermal_eigenvalues, thermal_modes, _ = \
    rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_matrix,
                                    thermal_reduced_problem._inner_product_action,
                                    N=Nmax, tol=1e-10)
thermal_reduced_problem._basis_functions.extend(thermal_modes)
thermal_reduced_size = len(thermal_reduced_problem._basis_functions)
print("")

print(rbnicsx.io.TextBox("POD-Galerkin offline phase ends", fill="="))

thermal_positive_eigenvalues = np.where(thermal_eigenvalues > 0., thermal_eigenvalues, np.nan)
thermal_singular_values = np.sqrt(thermal_positive_eigenvalues)

if world_comm.rank == 0:
    plt.figure(figsize=[8, 10])
    xint = list()
    yval = list()

    for x, y in enumerate(thermal_eigenvalues[:len(thermal_reduced_problem._basis_functions)]):
        yval.append(y)
        xint.append(x+1)

    plt.plot(xint, yval, "*-", color="orange")
    plt.xlabel("Eigenvalue number", fontsize=18)
    plt.ylabel("Eigenvalue", fontsize=18)
    plt.xticks(xint)
    plt.yscale("log")
    plt.title("Eigenvalue decay", fontsize=24)
    plt.tight_layout()
    plt.savefig("eigenvalue_decay")

print(f"Thermal eigenvalues: {thermal_positive_eigenvalues}")

# ### # Thermal POD Ends ###

# ### Thermal ANN starts ###

def generate_ann_input_set(samples=ann_samples):
    # Select samples from the parameter space for ANN
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
    # Solve the FE problem at given input_sets and
    # project on the RB space
    rb_size = len(reduced_problem._basis_functions)
    for i in indices:
        if mode is None:
            print(f"Parameter {i+1}/{input_set.shape[0]}")
        else:
            print(f"{mode} parameter number {i+1}/{input_set.shape[0]}")
        solution = problem.solve(input_set[i, :])
        output_set[i, :] = reduced_problem.project_snapshot(solution,
                                                            rb_size).array

thermal_num_ann_input_samples = np.product(ann_samples)
thermal_num_training_samples = int(0.7 * thermal_num_ann_input_samples)
thermal_num_validation_samples = \
    thermal_num_ann_input_samples - int(0.7 * thermal_num_ann_input_samples)
itemsize = MPI.DOUBLE.Get_size()

if world_comm.rank == 0:
    thermal_ann_input_set = generate_ann_input_set(samples=ann_samples)
    # np.random.shuffle(thermal_ann_input_set)
    thermal_nbytes_para_ann_training = thermal_num_training_samples * itemsize * para_dim
    thermal_nbytes_dofs_ann_training = thermal_num_training_samples * itemsize * \
        len(thermal_reduced_problem._basis_functions)
    thermal_nbytes_para_ann_validation = thermal_num_validation_samples * itemsize * para_dim
    thermal_nbytes_dofs_ann_validation = thermal_num_validation_samples * itemsize * \
        len(thermal_reduced_problem._basis_functions)
else:
    thermal_nbytes_para_ann_training = 0
    thermal_nbytes_dofs_ann_training = 0
    thermal_nbytes_para_ann_validation = 0
    thermal_nbytes_dofs_ann_validation = 0

world_comm.barrier()

win2 = MPI.Win.Allocate_shared(thermal_nbytes_para_ann_training, itemsize,
                               comm=MPI.COMM_WORLD)
buf2, itemsize = win2.Shared_query(0)
thermal_input_training_set = \
    np.ndarray(buffer=buf2, dtype="d",
               shape=(thermal_num_training_samples, para_dim))

win3 = MPI.Win.Allocate_shared(thermal_nbytes_para_ann_validation, itemsize,
                               comm=MPI.COMM_WORLD)
buf3, itemsize = win3.Shared_query(0)
thermal_input_validation_set = \
    np.ndarray(buffer=buf3, dtype="d",
               shape=(thermal_num_validation_samples, para_dim))

win4 = MPI.Win.Allocate_shared(thermal_nbytes_dofs_ann_training, itemsize,
                               comm=MPI.COMM_WORLD)
buf4, itemsize = win4.Shared_query(0)
thermal_output_training_set = \
    np.ndarray(buffer=buf4, dtype="d",
               shape=(thermal_num_training_samples,
                      len(thermal_reduced_problem._basis_functions)))

win5 = MPI.Win.Allocate_shared(thermal_nbytes_dofs_ann_validation, itemsize,
                               comm=MPI.COMM_WORLD)
buf5, itemsize = win5.Shared_query(0)
thermal_output_validation_set = \
    np.ndarray(buffer=buf5, dtype="d",
               shape=(thermal_num_validation_samples,
                      len(thermal_reduced_problem._basis_functions)))

if world_comm.rank == 0:
    thermal_input_training_set[:, :] = \
        thermal_ann_input_set[:thermal_num_training_samples, :]
    thermal_input_validation_set[:, :] = \
        thermal_ann_input_set[thermal_num_training_samples:, :]
    thermal_output_training_set[:, :] = \
        np.zeros([thermal_num_training_samples,
                  len(thermal_reduced_problem._basis_functions)])
    thermal_output_validation_set[:, :] = \
        np.zeros([thermal_num_validation_samples,
                  len(thermal_reduced_problem._basis_functions)])

world_comm.Barrier()

thermal_training_set_indices = \
    np.arange(world_comm.rank, thermal_input_training_set.shape[0],
              world_comm.size)

thermal_validation_set_indices = \
    np.arange(world_comm.rank, thermal_input_validation_set.shape[0],
              world_comm.size)

world_comm.Barrier()

# Training dataset
generate_ann_output_set(thermal_problem_parametric, thermal_reduced_problem,
                        thermal_input_training_set, thermal_output_training_set,
                        thermal_training_set_indices, mode="Training")

generate_ann_output_set(thermal_problem_parametric, thermal_reduced_problem,
                        thermal_input_validation_set, thermal_output_validation_set,
                        thermal_validation_set_indices, mode="Validation")

world_comm.Barrier()

thermal_reduced_problem.output_range[0] = \
    min(np.min(thermal_output_training_set), np.min(thermal_output_validation_set))
thermal_reduced_problem.output_range[1] = \
    max(np.max(thermal_output_training_set), np.max(thermal_output_validation_set))

print("\n")

thermal_cpu_group0_procs = world_comm.group.Incl([0, 1, 2, 3])
thermal_cpu_group0_comm = world_comm.Create_group(thermal_cpu_group0_procs)

# ANN model
thermal_model = HiddenLayersNet(thermal_training_set.shape[1], [35, 35],
                        len(thermal_reduced_problem._basis_functions), Tanh())

if thermal_cpu_group0_comm != MPI.COMM_NULL:
    init_cpu_process_group(thermal_cpu_group0_comm)

    thermal_training_set_indices_cpu = np.arange(thermal_cpu_group0_comm.rank,
                                         thermal_input_training_set.shape[0],
                                         thermal_cpu_group0_comm.size)
    thermal_validation_set_indices_cpu = np.arange(thermal_cpu_group0_comm.rank,
                                           thermal_input_validation_set.shape[0],
                                           thermal_cpu_group0_comm.size)

    customDataset = CustomPartitionedDataset(thermal_reduced_problem, thermal_input_training_set,
                                             thermal_output_training_set, thermal_training_set_indices_cpu)
    thermal_train_dataloader = DataLoader(customDataset, batch_size=10, shuffle=False)# shuffle=True)

    customDataset = CustomPartitionedDataset(thermal_reduced_problem, thermal_input_validation_set,
                                            thermal_output_validation_set, thermal_validation_set_indices_cpu)
    thermal_valid_dataloader = DataLoader(customDataset, shuffle=False)

    thermal_path = "thermal_model.pth"
    # save_model(thermal_model, thermal_path)
    load_model(thermal_model, thermal_path)

    model_synchronise(thermal_model, verbose=True)

    # Training of ANN
    thermal_training_loss = list()
    thermal_validation_loss = list()

    thermal_max_epochs = 20 #000
    thermal_min_validation_loss = None
    thermal_start_epoch = 0
    thermal_checkpoint_path = "thermal_checkpoint"
    thermal_checkpoint_epoch = 10

    thermal_learning_rate = 1e-4
    thermal_optimiser = get_optimiser(thermal_model, "Adam", thermal_learning_rate)
    thermal_loss_fn = get_loss_func("MSE", reduction="sum")

    if os.path.exists(thermal_checkpoint_path):
        thermal_start_epoch, thermal_min_validation_loss = \
            load_checkpoint(thermal_checkpoint_path, thermal_model, thermal_optimiser)

    import time
    start_time = time.time()
    for thermal_epochs in range(thermal_start_epoch, thermal_max_epochs):
        if thermal_epochs > 0 and thermal_epochs % thermal_checkpoint_epoch == 0:
            save_checkpoint(thermal_checkpoint_path, thermal_epochs,
                            thermal_model, thermal_optimiser,
                            thermal_min_validation_loss)
        print(f"Epoch: {thermal_epochs+1}/{thermal_max_epochs}")
        thermal_current_training_loss = train_nn(thermal_reduced_problem,
                                         thermal_train_dataloader,
                                         thermal_model, thermal_loss_fn, thermal_optimiser)
        thermal_training_loss.append(thermal_current_training_loss)
        thermal_current_validation_loss = validate_nn(thermal_reduced_problem,
                                              thermal_valid_dataloader,
                                              thermal_model, thermal_loss_fn)
        thermal_validation_loss.append(thermal_current_validation_loss)
        if thermal_epochs > 0 and thermal_current_validation_loss > thermal_min_validation_loss \
        and thermal_reduced_problem.regularisation == "EarlyStopping":
            # 1% safety margin against min_validation_loss
            # before invoking early stopping criteria
            print(f"Early stopping criteria invoked at epoch: {thermal_epochs+1}")
            break
        thermal_min_validation_loss = min(thermal_validation_loss)
    end_time = time.time()
    thermal_elapsed_time = end_time - start_time

    os.system(f"rm {thermal_checkpoint_path}")

thermal_model_root_process = 0
share_model(thermal_model, world_comm, thermal_model_root_process)
world_comm.Barrier()

# ### Thermal ANN ends ###

# ### Thermal Error analysis starts ###

print("\n")
print("Generating error analysis (only input/parameters) dataset")
print("\n")

thermal_error_analysis_num_para = np.product(error_analysis_samples)
itemsize = MPI.DOUBLE.Get_size()

if world_comm.rank == 0:
    thermal_nbytes_para = thermal_error_analysis_num_para * itemsize * para_dim
    thermal_nbytes_error = thermal_error_analysis_num_para * itemsize
else:
    thermal_nbytes_para = 0
    thermal_nbytes_error = 0

win6 = MPI.Win.Allocate_shared(thermal_nbytes_para, itemsize,
                               comm=world_comm)
buf6, itemsize = win6.Shared_query(0)
thermal_error_analysis_set = \
    np.ndarray(buffer=buf6, dtype="d",
               shape=(thermal_error_analysis_num_para,
                      para_dim))

win7 = MPI.Win.Allocate_shared(thermal_nbytes_error, itemsize,
                               comm=world_comm)
buf7, itemsize = win7.Shared_query(0)
thermal_error_numpy = np.ndarray(buffer=buf7, dtype="d",
                         shape=(thermal_error_analysis_num_para))

if world_comm.rank == 0:
    thermal_error_analysis_set[:, :] = generate_ann_input_set(samples=error_analysis_samples)

world_comm.Barrier()

thermal_error_analysis_indices = np.arange(world_comm.rank,
                                   thermal_error_analysis_set.shape[0],
                                   world_comm.size)
for i in thermal_error_analysis_indices:
    thermal_error_numpy[i] = error_analysis(thermal_reduced_problem, thermal_problem_parametric,
                                    thermal_error_analysis_set[i, :], thermal_model,
                                    len(thermal_reduced_problem._basis_functions),
                                    online_nn)
    print(f"Error analysis {i+1} of {thermal_error_analysis_set.shape[0]}, Error: {thermal_error_numpy[i]}")

world_comm.Barrier()

# ### Thermal Error analysis ends ###

mechanical_problem_parametric = \
    MechanicalProblemOnDeformedDomain(mesh, cell_tags, facet_tags,
                                      thermal_problem_parametric)

# solution_mu = mechanical_problem_parametric.solve(mu_ref)
# print(f"Solution norm at mu:{mu_ref}: {mechanical_problem_parametric.inner_product_action(solution_mu)(solution_mu)}")

solution_mu = mechanical_problem_parametric.solve(mu)
print(f"Solution norm at mu:{mu}: {mechanical_problem_parametric.inner_product_action(solution_mu)(solution_mu)}")

if world_comm.rank == 0:
    computed_file = "solution_nonlinear_thermomechanical_mechanical/solution_computed.xdmf"
    with MeshDeformationWrapperClass(mesh, facet_tags, mu_ref, mu):
        with dolfinx.io.XDMFFile(mesh.comm, computed_file, "w") as solution_file:
            solution_file.write_mesh(mesh)
            solution_file.write_function(solution_mu)

# ### # Mechanical POD starts ###

mechanical_num_dofs = solution_mu.x.array.shape[0]
num_snapshots = np.product(pod_samples)
nbytes_para = itemsize * num_snapshots * para_dim
nbytes_dofs = itemsize * num_snapshots * mechanical_num_dofs

# NOTE using same generate_training_set as Thermal problem

win2 = MPI.Win.Allocate_shared(nbytes_para, itemsize, comm=MPI.COMM_WORLD)
buf2, itemsize = win2.Shared_query(0)
mechanical_training_set = np.ndarray(buffer=buf2, dtype="d", shape=(num_snapshots, para_dim))

if world_comm.rank == 0:
    mechanical_training_set[:, :] = generate_training_set(samples=pod_samples)


world_comm.Barrier()

win3 = MPI.Win.Allocate_shared(nbytes_dofs, itemsize, comm=MPI.COMM_WORLD)
buf3, itemsize = win3.Shared_query(0)
mechanical_training_set_solution = np.ndarray(buffer=buf3, dtype="d", shape=(num_snapshots, mechanical_num_dofs))

# Solution manifold
indices = np.arange(world_comm.rank, num_snapshots, world_comm.size)

for i in indices:
    print(f"Solving FEM problem {i+1}/{num_snapshots}")
    mechanical_training_set_solution[i, :] = (mechanical_problem_parametric.solve(mechanical_training_set[i, :])).x.array

world_comm.Barrier()

# Maximum RB size
Nmax = 30

print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
print("")

print("Set up snapshots matrix")
snapshots_matrix = rbnicsx.backends.FunctionsList(mechanical_problem_parametric._VM)

for i in range(num_snapshots):
    snapshot = dolfinx.fem.Function(mechanical_problem_parametric._VM)
    snapshot.x.array[:] = mechanical_training_set_solution[i, :]

    print(f"Update snapshots matrix: {i+1}/{num_snapshots}")
    snapshots_matrix.append(snapshot)

print("Set up reduced problem")
mechanical_reduced_problem = MechanicalPODANNReducedProblem(mechanical_problem_parametric)

print("")

print(rbnicsx.io.TextLine("Perform POD", fill="#"))
mechanical_eigenvalues, mechanical_modes, _ = \
    rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_matrix,
                                    mechanical_reduced_problem._inner_product_action,
                                    N=Nmax, tol=1e-10)
mechanical_reduced_problem._basis_functions.extend(mechanical_modes)
mechanical_reduced_size = len(mechanical_reduced_problem._basis_functions)
print("")

print(rbnicsx.io.TextBox("POD-Galerkin offline phase ends", fill="="))

mechanical_positive_eigenvalues = np.where(mechanical_eigenvalues > 0., mechanical_eigenvalues, np.nan)
mechanical_singular_values = np.sqrt(mechanical_positive_eigenvalues)

if world_comm.rank == 0:
    plt.figure(figsize=[8, 10])
    xint = list()
    yval = list()

    for x, y in enumerate(mechanical_eigenvalues[:len(mechanical_reduced_problem._basis_functions)]):
        yval.append(y)
        xint.append(x+1)

    plt.plot(xint, yval, "*-", color="orange")
    plt.xlabel("Eigenvalue number", fontsize=18)
    plt.ylabel("Eigenvalue", fontsize=18)
    plt.xticks(xint)
    plt.yscale("log")
    plt.title("Eigenvalue decay", fontsize=24)
    plt.tight_layout()
    plt.savefig("eigenvalue_decay")

print(f"Mechanical eigenvalues: {mechanical_positive_eigenvalues}")

# ### # Mechanical POD Ends ###

# ### Mechanical ANN starts ###
mechanical_num_ann_input_samples = np.product(ann_samples)
mechanical_num_training_samples = int(0.7 * mechanical_num_ann_input_samples)
mechanical_num_validation_samples = \
    mechanical_num_ann_input_samples - int(0.7 * mechanical_num_ann_input_samples)
itemsize = MPI.DOUBLE.Get_size()

if world_comm.rank == 0:
    mechanical_ann_input_set = generate_ann_input_set(samples=ann_samples)
    # np.random.shuffle(mechanical_ann_input_set)
    mechanical_nbytes_para_ann_training = mechanical_num_training_samples * itemsize * para_dim
    mechanical_nbytes_dofs_ann_training = mechanical_num_training_samples * itemsize * \
        len(mechanical_reduced_problem._basis_functions)
    mechanical_nbytes_para_ann_validation = mechanical_num_validation_samples * itemsize * para_dim
    mechanical_nbytes_dofs_ann_validation = mechanical_num_validation_samples * itemsize * \
        len(mechanical_reduced_problem._basis_functions)
else:
    mechanical_nbytes_para_ann_training = 0
    mechanical_nbytes_dofs_ann_training = 0
    mechanical_nbytes_para_ann_validation = 0
    mechanical_nbytes_dofs_ann_validation = 0

world_comm.barrier()

win2 = MPI.Win.Allocate_shared(mechanical_nbytes_para_ann_training, itemsize,
                               comm=MPI.COMM_WORLD)
buf2, itemsize = win2.Shared_query(0)
mechanical_input_training_set = \
    np.ndarray(buffer=buf2, dtype="d",
               shape=(mechanical_num_training_samples, para_dim))

win3 = MPI.Win.Allocate_shared(mechanical_nbytes_para_ann_validation, itemsize,
                               comm=MPI.COMM_WORLD)
buf3, itemsize = win3.Shared_query(0)
mechanical_input_validation_set = \
    np.ndarray(buffer=buf3, dtype="d",
               shape=(mechanical_num_validation_samples, para_dim))

win4 = MPI.Win.Allocate_shared(mechanical_nbytes_dofs_ann_training, itemsize,
                               comm=MPI.COMM_WORLD)
buf4, itemsize = win4.Shared_query(0)
mechanical_output_training_set = \
    np.ndarray(buffer=buf4, dtype="d",
               shape=(mechanical_num_training_samples,
                      len(mechanical_reduced_problem._basis_functions)))

win5 = MPI.Win.Allocate_shared(mechanical_nbytes_dofs_ann_validation, itemsize,
                               comm=MPI.COMM_WORLD)
buf5, itemsize = win5.Shared_query(0)
mechanical_output_validation_set = \
    np.ndarray(buffer=buf5, dtype="d",
               shape=(mechanical_num_validation_samples,
                      len(mechanical_reduced_problem._basis_functions)))

if world_comm.rank == 0:
    mechanical_input_training_set[:, :] = \
        mechanical_ann_input_set[:mechanical_num_training_samples, :]
    mechanical_input_validation_set[:, :] = \
        mechanical_ann_input_set[mechanical_num_training_samples:, :]
    mechanical_output_training_set[:, :] = \
        np.zeros([mechanical_num_training_samples,
                  len(mechanical_reduced_problem._basis_functions)])
    mechanical_output_validation_set[:, :] = \
        np.zeros([mechanical_num_validation_samples,
                  len(mechanical_reduced_problem._basis_functions)])

world_comm.Barrier()

mechanical_training_set_indices = \
    np.arange(world_comm.rank, mechanical_input_training_set.shape[0],
              world_comm.size)

mechanical_validation_set_indices = \
    np.arange(world_comm.rank, mechanical_input_validation_set.shape[0],
              world_comm.size)

world_comm.Barrier()

# Training dataset
generate_ann_output_set(mechanical_problem_parametric, mechanical_reduced_problem,
                        mechanical_input_training_set, mechanical_output_training_set,
                        mechanical_training_set_indices, mode="Training")

generate_ann_output_set(mechanical_problem_parametric, mechanical_reduced_problem,
                        mechanical_input_validation_set, mechanical_output_validation_set,
                        mechanical_validation_set_indices, mode="Validation")

world_comm.Barrier()

mechanical_reduced_problem.output_range[0] = \
    min(np.min(mechanical_output_training_set), np.min(mechanical_output_validation_set))
mechanical_reduced_problem.output_range[1] = \
    max(np.max(mechanical_output_training_set), np.max(mechanical_output_validation_set))

print("\n")

mechanical_cpu_group0_procs = world_comm.group.Incl([4]) # ([4, 5, 6, 7])
mechanical_cpu_group0_comm = world_comm.Create_group(mechanical_cpu_group0_procs)

# ANN model
mechanical_model = HiddenLayersNet(mechanical_training_set.shape[1], [35, 35],
                        len(mechanical_reduced_problem._basis_functions), Tanh())

if mechanical_cpu_group0_comm != MPI.COMM_NULL:
    init_cpu_process_group(mechanical_cpu_group0_comm)

    mechanical_training_set_indices_cpu = np.arange(mechanical_cpu_group0_comm.rank,
                                         mechanical_input_training_set.shape[0],
                                         mechanical_cpu_group0_comm.size)
    mechanical_validation_set_indices_cpu = np.arange(mechanical_cpu_group0_comm.rank,
                                           mechanical_input_validation_set.shape[0],
                                           mechanical_cpu_group0_comm.size)

    customDataset = CustomPartitionedDataset(mechanical_reduced_problem, mechanical_input_training_set,
                                             mechanical_output_training_set, mechanical_training_set_indices_cpu)
    mechanical_train_dataloader = DataLoader(customDataset, batch_size=40, shuffle=False)# shuffle=True)

    customDataset = CustomPartitionedDataset(mechanical_reduced_problem, mechanical_input_validation_set,
                                            mechanical_output_validation_set, mechanical_validation_set_indices_cpu)
    mechanical_valid_dataloader = DataLoader(customDataset, shuffle=False)

    mechanical_path = "mechanical_model.pth"
    # save_model(mechanical_model, mechanical_path)
    load_model(mechanical_model, mechanical_path)

    model_synchronise(mechanical_model, verbose=True)

    # Training of ANN
    mechanical_training_loss = list()
    mechanical_validation_loss = list()

    mechanical_max_epochs = 20 # 20000
    mechanical_min_validation_loss = None
    mechanical_start_epoch = 0
    mechanical_checkpoint_path = "mechanical_checkpoint"
    mechanical_checkpoint_epoch = 10

    mechanical_learning_rate = 1e-4
    mechanical_optimiser = get_optimiser(mechanical_model, "Adam", mechanical_learning_rate)
    mechanical_loss_fn = get_loss_func("MSE", reduction="sum")

    if os.path.exists(mechanical_checkpoint_path):
        mechanical_start_epoch, mechanical_min_validation_loss = \
            load_checkpoint(mechanical_checkpoint_path, mechanical_model, mechanical_optimiser)

    import time
    start_time = time.time()
    for mechanical_epochs in range(mechanical_start_epoch, mechanical_max_epochs):
        if mechanical_epochs > 0 and mechanical_epochs % mechanical_checkpoint_epoch == 0:
            save_checkpoint(mechanical_checkpoint_path, mechanical_epochs,
                            mechanical_model, mechanical_optimiser,
                            mechanical_min_validation_loss)
        print(f"Epoch: {mechanical_epochs+1}/{mechanical_max_epochs}")
        mechanical_current_training_loss = train_nn(mechanical_reduced_problem,
                                         mechanical_train_dataloader,
                                         mechanical_model, mechanical_loss_fn, mechanical_optimiser)
        mechanical_training_loss.append(mechanical_current_training_loss)
        mechanical_current_validation_loss = validate_nn(mechanical_reduced_problem,
                                              mechanical_valid_dataloader,
                                              mechanical_model, mechanical_loss_fn)
        mechanical_validation_loss.append(mechanical_current_validation_loss)
        if mechanical_epochs > 0 and mechanical_current_validation_loss > mechanical_min_validation_loss \
        and mechanical_reduced_problem.regularisation == "EarlyStopping":
            # 1% safety margin against min_validation_loss
            # before invoking early stopping criteria
            print(f"Early stopping criteria invoked at epoch: {mechanical_epochs+1}")
            break
        mechanical_min_validation_loss = min(mechanical_validation_loss)
    end_time = time.time()
    mechanical_elapsed_time = end_time - start_time

    os.system(f"rm {mechanical_checkpoint_path}")

mechanical_model_root_process = 4
share_model(mechanical_model, world_comm, mechanical_model_root_process)
world_comm.Barrier()
# ### Mechanical ANN ends ###

# ### Mechanical Error analysis starts ###

print("\n")
print("Generating error analysis (only input/parameters) dataset")
print("\n")

mechanical_error_analysis_num_para = np.product(error_analysis_samples)
itemsize = MPI.DOUBLE.Get_size()

if world_comm.rank == 0:
    mechanical_nbytes_para = mechanical_error_analysis_num_para * itemsize * para_dim
    mechanical_nbytes_error = mechanical_error_analysis_num_para * itemsize
else:
    mechanical_nbytes_para = 0
    mechanical_nbytes_error = 0

win6 = MPI.Win.Allocate_shared(mechanical_nbytes_para, itemsize,
                               comm=world_comm)
buf6, itemsize = win6.Shared_query(0)
mechanical_error_analysis_set = \
    np.ndarray(buffer=buf6, dtype="d",
               shape=(mechanical_error_analysis_num_para,
                      para_dim))

win7 = MPI.Win.Allocate_shared(mechanical_nbytes_error, itemsize,
                               comm=world_comm)
buf7, itemsize = win7.Shared_query(0)
mechanical_error_numpy = np.ndarray(buffer=buf7, dtype="d",
                         shape=(mechanical_error_analysis_num_para))

if world_comm.rank == 0:
    mechanical_error_analysis_set[:, :] = generate_ann_input_set(samples=error_analysis_samples)

world_comm.Barrier()

mechanical_error_analysis_indices = np.arange(world_comm.rank,
                                   mechanical_error_analysis_set.shape[0],
                                   world_comm.size)
for i in mechanical_error_analysis_indices:
    mechanical_error_numpy[i] = error_analysis(mechanical_reduced_problem, mechanical_problem_parametric,
                                    mechanical_error_analysis_set[i, :], mechanical_model,
                                    len(mechanical_reduced_problem._basis_functions),
                                    online_nn)
    print(f"Error analysis (Mechanical) {i+1} of {mechanical_error_analysis_set.shape[0]}, Error: {mechanical_error_numpy[i]}")

world_comm.Barrier()
# ### Mechanical Error analysis ends ###

# ### Online phase ###
# Online phase at parameter online_mu

if world_comm.rank == 0:
    online_mu = np.array([0.45, 0.56, 0.9, 0.7])
    thermal_fem_solution = thermal_problem_parametric.solve(online_mu)
    thermal_rb_solution = \
        thermal_reduced_problem.reconstruct_solution(
            online_nn(thermal_reduced_problem, thermal_problem_parametric,
                    online_mu, thermal_model,
                    len(thermal_reduced_problem._basis_functions)))


    thermal_fem_online_file \
        = "dlrbnicsx_solution_thermomechanical/thermal_fem_online_mu_computed.xdmf"
    with MeshDeformationWrapperClass(mesh, facet_tags, mu_ref,
                            online_mu):
        with dolfinx.io.XDMFFile(mesh.comm, thermal_fem_online_file,
                                "w") as thermal_solution_file:
            thermal_solution_file.write_mesh(mesh)
            thermal_solution_file.write_function(thermal_fem_solution)

    thermal_rb_online_file \
        = "dlrbnicsx_solution_thermomechanical/thermal_rb_online_mu_computed.xdmf"
    with MeshDeformationWrapperClass(mesh, facet_tags, mu_ref,
                            online_mu):
        with dolfinx.io.XDMFFile(mesh.comm, thermal_rb_online_file,
                                "w") as thermal_solution_file:
            # NOTE scatter_forward not considered for online solution
            thermal_solution_file.write_mesh(mesh)
            thermal_solution_file.write_function(thermal_rb_solution)

    thermal_error_function = dolfinx.fem.Function(thermal_problem_parametric._VT)
    thermal_error_function.x.array[:] = \
        thermal_fem_solution.x.array - thermal_rb_solution.x.array
    thermal_fem_rb_error_file \
        = "dlrbnicsx_solution_thermomechanical/thermal_fem_rb_error_computed.xdmf"
    with MeshDeformationWrapperClass(mesh, facet_tags, mu_ref,
                            online_mu):
        with dolfinx.io.XDMFFile(mesh.comm, thermal_fem_rb_error_file,
                                "w") as thermal_solution_file:
            thermal_solution_file.write_mesh(mesh)
            thermal_solution_file.write_function(thermal_error_function)

    with MeshDeformationWrapperClass(mesh, facet_tags, mu_ref,
                            online_mu):
        print(thermal_reduced_problem.norm_error(thermal_fem_solution, thermal_rb_solution))
        print(thermal_reduced_problem.compute_norm(thermal_error_function))

    print(thermal_reduced_problem.norm_error(thermal_fem_solution, thermal_rb_solution))
    print(thermal_reduced_problem.compute_norm(thermal_error_function))

    mechanical_fem_solution = mechanical_problem_parametric.solve(online_mu)
    mechanical_rb_solution = \
        mechanical_reduced_problem.reconstruct_solution(
            online_nn(mechanical_reduced_problem, mechanical_problem_parametric,
                    online_mu, mechanical_model,
                    len(mechanical_reduced_problem._basis_functions)))


    mechanical_fem_online_file \
        = "dlrbnicsx_solution_thermomechanical/mechanical_fem_online_mu_computed.xdmf"
    with MeshDeformationWrapperClass(mesh, facet_tags, mu_ref,
                            online_mu):
        with dolfinx.io.XDMFFile(mesh.comm, mechanical_fem_online_file,
                                "w") as mechanical_solution_file:
            mechanical_solution_file.write_mesh(mesh)
            mechanical_solution_file.write_function(mechanical_fem_solution)

    mechanical_rb_online_file \
        = "dlrbnicsx_solution_thermomechanical/mechanical_rb_online_mu_computed.xdmf"
    with MeshDeformationWrapperClass(mesh, facet_tags, mu_ref,
                            online_mu):
        with dolfinx.io.XDMFFile(mesh.comm, mechanical_rb_online_file,
                                "w") as mechanical_solution_file:
            # NOTE scatter_forward not considered for online solution
            mechanical_solution_file.write_mesh(mesh)
            mechanical_solution_file.write_function(mechanical_rb_solution)

    mechanical_error_function = dolfinx.fem.Function(mechanical_problem_parametric._VM)
    mechanical_error_function.x.array[:] = \
        mechanical_fem_solution.x.array - mechanical_rb_solution.x.array
    mechanical_fem_rb_error_file \
        = "dlrbnicsx_solution_thermomechanical/mechanical_fem_rb_error_computed.xdmf"
    with MeshDeformationWrapperClass(mesh, facet_tags, mu_ref,
                            online_mu):
        with dolfinx.io.XDMFFile(mesh.comm, mechanical_fem_rb_error_file,
                                "w") as mechanical_solution_file:
            mechanical_solution_file.write_mesh(mesh)
            mechanical_solution_file.write_function(mechanical_error_function)

    with MeshDeformationWrapperClass(mesh, facet_tags, mu_ref,
                            online_mu):
        print(mechanical_reduced_problem.norm_error(mechanical_fem_solution, mechanical_rb_solution))
        print(mechanical_reduced_problem.compute_norm(mechanical_error_function))

    print(mechanical_reduced_problem.norm_error(mechanical_fem_solution, mechanical_rb_solution))
    print(mechanical_reduced_problem.compute_norm(mechanical_error_function))

if thermal_cpu_group0_comm != MPI.COMM_NULL:
    print(f"Training time (Thermal): {thermal_elapsed_time}")

if mechanical_cpu_group0_comm != MPI.COMM_NULL:
    print(f"Training time (Mechanical): {mechanical_elapsed_time}")
