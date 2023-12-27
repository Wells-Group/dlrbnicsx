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
from smt.sampling_methods import LHS
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
        dx = ufl.Measure("dx", domain=self._mesh, subdomain_data=self._subdomains)
        ds = ufl.Measure("ds", domain=self._mesh, subdomain_data=self._boundaries)
        slef._dx = dx
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
        self.uT_func = dolfinx.fem.Function(self._VT)
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
        self.temperature_field.x.array[:] = 300.  # TODO 300 or 350??
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

                residual = abs(self._mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(self._thermal_conductivity_func * ufl.grad(self.temperature_field),
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
                    (np.sqrt(self._mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(solution_field - self.temperature_field,
                                                                                            solution_field - self.temperature_field) * x[0] * ufl.dx)), op=MPI.SUM)))/\
                    (np.sqrt(self._mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(self.temperature_field, self.temperature_field) * x[0] * ufl.dx)), op=MPI.SUM)))

                print(f"Absolute update: {update_abs}")

                self.temperature_field.x.array[:] = solution_field.x.array.copy()

                if update_abs < self.atol:
                    print(f"Solver tolerance {self.atol} reached in iterations {iteration + 1}")
                    break
        
        return solution_field

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

if __name__ == "__main__":
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
    thermal_ann_input_samples_num = 420
    thermal_error_analysis_samples_num = 144
    num_snapshots = 400
    thermal_num_dofs = solution_mu.x.array.shape[0]
    nbytes_para = itemsize * num_snapshots * para_dim
    nbytes_dofs = itemsize * num_snapshots * thermal_num_dofs

    # Thermal POD Starts ###
    def generate_training_set(num_samples, para_dim):
        training_set = np.random.uniform(size=(num_samples, para_dim))
        training_set[:, 0] = (0.75 - 0.55) * training_set[:, 0] + 0.55
        training_set[:, 1] = (0.55 - 0.35) * training_set[:, 1] + 0.35
        training_set[:, 2] = (1.20 - 0.80) * training_set[:, 2] + 0.80
        training_set[:, 3] = (0.60 - 0.40) * training_set[:, 3] + 0.40
        return training_set

    win0 = MPI.Win.Allocate_shared(nbytes_para, itemsize, comm=MPI.COMM_WORLD)
    buf0, itemsize = win0.Shared_query(0)
    thermal_training_set = np.ndarray(buffer=buf0, dtype="d", shape=(num_snapshots, para_dim))

    if world_comm.rank == 0:
        thermal_training_set[:, :] = generate_training_set(num_snapshots, para_dim)


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
                                        N=Nmax, tol=1e-6)
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

        for x, y in enumerate(thermal_eigenvalues[:20]):
            yval.append(y)
            xint.append(x+1)

        plt.plot(xint, yval, "*-", color="orange")
        plt.xlabel(r"$i$", fontsize=18)
        plt.ylabel(r"$\theta_T^i$", fontsize=18)
        plt.xticks(xint)
        plt.yscale("log")
        # plt.title("Eigenvalue decay", fontsize=24)
        plt.tight_layout()
        plt.savefig("thermal_eigenvalue_decay")

    print(f"Thermal eigenvalues: {thermal_positive_eigenvalues}")

    # ### # Thermal POD Ends ###

    # ### Thermal ANN starts ###
    def generate_ann_input_set(num_ann_samples):
        xlimits = np.array([[0.55, 0.75], [0.35, 0.55],
                            [0.8, 1.2], [0.4, 0.6]])
        sampling = LHS(xlimits=xlimits)
        training_set = sampling(num_ann_samples)
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

    thermal_num_training_samples = int(0.7 * thermal_ann_input_samples_num)
    thermal_num_validation_samples = \
        thermal_ann_input_samples_num - int(0.7 * thermal_ann_input_samples_num)
    itemsize = MPI.DOUBLE.Get_size()

    if world_comm.rank == 0:
        thermal_ann_input_set = generate_ann_input_set(thermal_ann_input_samples_num)
        np.random.shuffle(thermal_ann_input_set)
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
    thermal_model = HiddenLayersNet(thermal_training_set.shape[1], [25, 25, 25],
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
        thermal_train_dataloader = DataLoader(customDataset, batch_size=10, shuffle=True)

        customDataset = CustomPartitionedDataset(thermal_reduced_problem, thermal_input_validation_set,
                                                 thermal_output_validation_set, thermal_validation_set_indices_cpu)
        thermal_valid_dataloader = DataLoader(customDataset, shuffle=False)

        thermal_path = "thermal_model.pth"
        # save_model(thermal_model, thermal_path)
        # load_model(thermal_model, thermal_path)

        model_synchronise(thermal_model, verbose=False)

        # Training of ANN
        thermal_training_loss = list()
        thermal_validation_loss = list()

        thermal_max_epochs = 20000
        thermal_min_validation_loss = None
        thermal_start_epoch = 0
        thermal_checkpoint_path = "thermal_checkpoint"
        thermal_checkpoint_epoch = 10

        thermal_learning_rate = 1e-6
        thermal_optimiser = get_optimiser(thermal_model, "Adam", thermal_learning_rate)
        thermal_loss_fn = get_loss_func("MSE", reduction="sum")

        if os.path.exists(thermal_checkpoint_path):
            thermal_start_epoch, thermal_min_validation_loss = \
                load_checkpoint(thermal_checkpoint_path, thermal_model, thermal_optimiser)

        import time
        start_time = MPI.Wtime()
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
        end_time = MPI.Wtime()
        thermal_elapsed_time = end_time - start_time

        os.system(f"rm {thermal_checkpoint_path}")

    thermal_model_root_process = 0
    share_model(thermal_model, world_comm, thermal_model_root_process)
    world_comm.Barrier()

    # Thermal Error analysis

    print("\n")
    print("Generating error analysis (only input/parameters) dataset")
    print("\n")

    itemsize = MPI.DOUBLE.Get_size()

    if world_comm.rank == 0:
        thermal_nbytes_para = thermal_error_analysis_samples_num * itemsize * para_dim
        thermal_nbytes_error = thermal_error_analysis_samples_num * itemsize
    else:
        thermal_nbytes_para = 0
        thermal_nbytes_error = 0

    win6 = MPI.Win.Allocate_shared(thermal_nbytes_para, itemsize,
                                   comm=world_comm)
    buf6, itemsize = win6.Shared_query(0)
    thermal_error_analysis_set = \
        np.ndarray(buffer=buf6, dtype="d",
                shape=(thermal_error_analysis_samples_num,
                       para_dim))

    win7 = MPI.Win.Allocate_shared(thermal_nbytes_error, itemsize,
                                   comm=world_comm)
    buf7, itemsize = win7.Shared_query(0)
    thermal_error_numpy = np.ndarray(buffer=buf7, dtype="d",
                            shape=(thermal_error_analysis_samples_num))


    win8 = MPI.Win.Allocate_shared(thermal_nbytes_error, itemsize,
                                   comm=world_comm)
    buf8, itemsize = win8.Shared_query(0)
    thermal_projection_error_numpy = np.ndarray(buffer=buf8, dtype="d",
                                                shape=(thermal_error_analysis_samples_num))

    if world_comm.rank == 0:
        thermal_error_analysis_set[:, :] = generate_ann_input_set(thermal_error_analysis_samples_num)

    world_comm.Barrier()

    thermal_error_analysis_indices = np.arange(world_comm.rank,
                                    thermal_error_analysis_set.shape[0],
                                    world_comm.size)
    for i in thermal_error_analysis_indices:
        thermal_error_numpy[i] = error_analysis(thermal_reduced_problem, thermal_problem_parametric,
                                        thermal_error_analysis_set[i, :], thermal_model,
                                        len(thermal_reduced_problem._basis_functions),
                                        online_nn)
        thermal_fem_solution = thermal_problem_parametric.solve(thermal_error_analysis_set[i, :])
        thermal_projected_solution = \
            thermal_reduced_problem.project_snapshot(thermal_fem_solution,
                                                     len(thermal_reduced_problem._basis_functions))
        thermal_reconstructed_solution = \
            thermal_reduced_problem.reconstruct_solution(thermal_projected_solution)
        thermal_projection_error_numpy[i] = \
            thermal_reduced_problem.norm_error(thermal_fem_solution,
                                               thermal_reconstructed_solution)
        print(f"Error analysis {i+1} of {thermal_error_analysis_set.shape[0]}, " +
              f"RB error: {thermal_error_numpy[i]}, " +
              f"Projection error: {thermal_projection_error_numpy[i]}")

    world_comm.Barrier()

    # ### Thermal ANN starts ###

    # Online phase
    if world_comm.rank == 0:
        online_mu = np.array([0.45, 0.56, 0.9, 0.7])
        thermal_fem_solution = thermal_problem_parametric.solve(online_mu)
        thermal_rb_solution = \
            thermal_reduced_problem.reconstruct_solution(
                online_nn(thermal_reduced_problem, thermal_problem_parametric,
                        online_mu, thermal_model,
                        len(thermal_reduced_problem._basis_functions)))


        thermal_fem_online_file \
            = "dlrbnicsx_solution_thermal/thermal_fem_online_mu_computed.xdmf"
        with MeshDeformationWrapperClass(mesh, facet_tags, mu_ref,
                                         online_mu):
            with dolfinx.io.XDMFFile(mesh.comm, thermal_fem_online_file,
                                    "w") as thermal_solution_file:
                thermal_solution_file.write_mesh(mesh)
                thermal_solution_file.write_function(thermal_fem_solution)

        thermal_rb_online_file \
            = "dlrbnicsx_solution_thermal/thermal_rb_online_mu_computed.xdmf"
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
            = "dlrbnicsx_solution_thermal/thermal_fem_rb_error_computed.xdmf"
        with MeshDeformationWrapperClass(mesh, facet_tags, mu_ref,
                                         online_mu):
            with dolfinx.io.XDMFFile(mesh.comm, thermal_fem_rb_error_file,
                                    "w") as thermal_solution_file:
                thermal_solution_file.write_mesh(mesh)
                thermal_solution_file.write_function(thermal_error_function)

        thermal_projection_error_function = dolfinx.fem.Function(thermal_problem_parametric._VT)
        thermal_reconstructed_solution = \
            thermal_reduced_problem.reconstruct_solution(
                thermal_reduced_problem.project_snapshot(thermal_problem_parametric.solve(online_mu),
                                                         len(thermal_reduced_problem._basis_functions)))
        thermal_projection_error_function.x.array[:] = \
            thermal_fem_solution.x.array - thermal_reconstructed_solution.x.array
        thermal_fem_rb_projection_error_file \
            = "dlrbnicsx_solution_thermal/thermal_fem_rb_projection_error_computed.xdmf"
        with MeshDeformationWrapperClass(mesh, facet_tags, mu_ref,
                                         online_mu):
            with dolfinx.io.XDMFFile(mesh.comm, thermal_fem_rb_projection_error_file,
                                    "w") as thermal_solution_file:
                thermal_solution_file.write_mesh(mesh)
                thermal_solution_file.write_function(thermal_projection_error_function)


    if thermal_cpu_group0_comm != MPI.COMM_NULL:
        print(f"Training time (Thermal): {thermal_elapsed_time}")

    if world_comm.rank == 0:
        np.save("thermal_error_analysis_set.npy", thermal_error_analysis_set)
        np.save("thermal_rb_error.npy", thermal_error_numpy)
        np.save("thermal_projection_error.npy", thermal_projection_error_numpy)
