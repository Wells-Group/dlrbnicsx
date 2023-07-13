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

class ThermalProblemOnDeformedDomain(abc.ABC):
    def __init__(self, mesh, subdomains, boundaries, meshDeformationContext):
        self._mesh = mesh
        self._subdomains = subdomains
        self._boundaries = boundaries
        self._boundary_markers = list(np.arange(1, 32))
        self._meshDeformationContext = meshDeformationContext
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
        self._solution.x.array[:] = 300.  # Initial solution
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

    def solve(self, mu):
        update_function = dolfinx.fem.Function(self._V)

        for iteration in range(self._max_iterations):

            print("\n =======================================================")
            print(f"\n Iteration {iteration + 1} / {self._max_iterations} \n")

            self.thermal_conductivity_func_assemble()
            self.thermal_conductivity_func_diff_assemble()
            residual = dolfinx.fem.assemble_scalar(self.residual_form)
            residual = mesh_comm.allreduce(residual, op=MPI.SUM)
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
                mesh_comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(update_function, update_function) * x[0] * ufl.dx +
                                                                                 ufl.inner(ufl.grad(update_function), ufl.grad(update_function)) * x[0] * ufl.dx)), op=MPI.SUM) / \
                mesh_comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(self._solution, self._solution) * x[0] * ufl.dx +
                                                                                 ufl.inner(ufl.grad(self._solution), ufl.grad(self._solution)) * x[0] * ufl.dx)), op=MPI.SUM)

            print(f"Relative update (in norm): {solution_update}")
            if solution_update < 1.e-12:
                print(f"Relative update tolerance reached")
                break

            self._solution.x.array[:] = current_solution.x.array.copy()

        return current_solution




# Read mesh
mesh_comm = MPI.COMM_WORLD
gdim = 2
gmsh_model_rank = 0
mesh, cell_tags, facet_tags = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh",
                                    mesh_comm, gmsh_model_rank, gdim=gdim)

# Mesh deformation parameters
mu_ref = [0.6438, 0.4313, 1., 0.5]  # reference geometry
mu = [0.8, 0.55, 0.8, 0.4]  # Parametric geometry

# FEM solve
problem_parametric = ThermalProblemOnDeformedDomain(mesh, cell_tags,
                                             facet_tags,
                                             HarmonicMeshMotion)
solution_mu = problem_parametric.solve(mu_ref)
