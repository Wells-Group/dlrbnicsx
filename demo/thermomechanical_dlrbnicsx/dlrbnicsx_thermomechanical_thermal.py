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
        x = ufl.SpatialCoordinate(mesh)
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
                print(solution_field.x.array)

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

# FEM solve
thermal_problem_parametric = \
    ThermalProblemOnDeformedDomain(mesh, cell_tags, facet_tags)

solution_mu = thermal_problem_parametric.solve(mu_ref)
print(f"Solution norm at mu:{mu_ref}: {thermal_problem_parametric.inner_product_action(solution_mu)(solution_mu)}")

solution_mu = thermal_problem_parametric.solve(mu)
print(f"Solution norm at mu:{mu}: {thermal_problem_parametric.inner_product_action(solution_mu)(solution_mu)}")

computed_file = "solution_nonlinear_thermomechanical_thermal/solution_computed.xdmf"

with MeshDeformationWrapperClass(mesh, facet_tags, mu_ref, mu):
    with dolfinx.io.XDMFFile(mesh.comm, computed_file, "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(solution_mu)
