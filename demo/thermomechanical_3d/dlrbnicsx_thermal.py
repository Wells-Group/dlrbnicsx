import dolfinx
import ufl
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

import rbnicsx
import rbnicsx.backends
import rbnicsx.online

from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import sympy
from smt.sampling_methods import LHS
import itertools
import abc
import matplotlib.pyplot as plt
import os
import time

from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory import Tanh, Sigmoid
from dlrbnicsx.dataset.custom_dataset import CustomDataset
from dlrbnicsx.interface.wrappers import DataLoader, save_model, load_model, \
    save_checkpoint, load_checkpoint, get_optimiser, get_loss_func
from dlrbnicsx.train_validate_test.train_validate_test import \
    train_nn, validate_nn, online_nn, error_analysis


class ProblemOnDeformedDomain(abc.ABC):
    # Define FEM problem on the reference problem
    def __init__(self, mesh, cell_tags, facet_tags, MeshMotion):
        # Mesh, Subdomians and Boundaries, Mesh deformation
        self._mesh = mesh
        self.gdim = self._mesh.geometry.dim
        self._cell_tags = cell_tags
        self._facet_tags = facet_tags
        self.meshDeformationContext = HarmonicMeshMotion

        # Define function space, Trial and Test Function
        self._VT = dolfinx.fem.FunctionSpace(self._mesh, ("CG", 1))
        self._uT, self._vT = ufl.TrialFunction(self._VT), ufl.TestFunction(self._VT)
        self._uT_func = dolfinx.fem.Function(self._VT)

        self._dx = ufl.Measure("dx", domain=self._mesh, subdomain_data=self._cell_tags)
        self._ds = ufl.Measure("ds", domain=self._mesh, subdomain_data=self._facet_tags)
        
        self._ds_bottom = self._ds(4) + self._ds(10)
        self._ds_outer_bottom = self._ds(5) + self._ds(11)
        self._ds_outer_top = self._ds(15) + self._ds(19)
        self._ds_1top = self._ds(8) + self._ds(14)
        self._ds_2top = self._ds(16) + self._ds(20)
        self._ds_3top = self._ds(23) + self._ds(26)
        self._ds_inner = self._ds(22) + self._ds(25)

        self._dx_sub_1 = self._dx(1) + self._dx(2)
        self._dx_sub_2 = self._dx(3) + self._dx(4)
        self._dx_sub_3 = self._dx(5) + self._dx(6)

        self._T_f = 1773.
        self._T_out = 300.
        self._T_bottom = 300.
        
        self._h_cf = 2000.
        self._h_cout = 200.
        self._h_cbottom = 200.

        self._q_source = dolfinx.fem.Function(self._VT)
        self._q_source.x.array[:] = 0.
        self._q_top = dolfinx.fem.Function(self._VT)
        self._q_top.x.array[:] = 0.

        self._x = ufl.SpatialCoordinate(self._mesh)
        self._n_vec = ufl.FacetNormal(self._mesh)
        self._sym_T = sympy.Symbol("sym_T")

        # Velocity and Pressure inner product (To be used in POD)
        self._inner_product_uT = ufl.inner(self._uT, self._vT) * self._dx + \
            ufl.inner(ufl.grad(self._uT), ufl.grad(self._vT)) * self._dx
        self._inner_product_action_uT = \
            rbnicsx.backends.bilinear_form_action(self._inner_product_uT,
                                                  part="real")
    
    def thermal_diffusivity_1(self, sym_T):
        conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 673.)), ufl.And(ufl.ge(sym_T, 673.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
        interps = [8.17484662576687e-6*sym_T**2 - 0.00926193251533741*sym_T + 18.0819438190184, 8.17484662576687e-6*sym_T**2 - 0.00926193251533741*sym_T + 18.0819438190184, 1.76073619631904e-6*sym_T**2 - 0.000628539877300632*sym_T + 15.176807196319, 1.76073619631904e-6*sym_T**2 - 0.000628539877300632*sym_T + 15.176807196319]
        assert len(conditions) == len(interps)
        d_func = ufl.conditional(conditions[0], interps[0], interps[0])
        for i in range(1, len(conditions)):
            d_func = ufl.conditional(conditions[i], interps[i], d_func)
        return d_func

    def thermal_diffusivity_2(self, sym_T):
        conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 693.)), ufl.And(ufl.ge(sym_T, 673.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
        interps = [0.000299054192229039*sym_T**2 - 0.36574217791411*sym_T + 130.838954780164, 0.000299054192229039*sym_T**2 - 0.36574217791411*sym_T + 130.838954780164, -1.10434560327197e-5*sym_T**2 + 0.0516492566462166*sym_T - 9.61326294938646, -1.10434560327197e-5*sym_T**2 + 0.0516492566462166*sym_T - 9.61326294938646]
        assert len(conditions) == len(interps)
        d_func = ufl.conditional(conditions[0], interps[0], interps[0])
        for i in range(1, len(conditions)):
            d_func = ufl.conditional(conditions[i], interps[i], d_func)
        return d_func

    def thermal_diffusivity_3(self, sym_T):
        conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 693.)), ufl.And(ufl.ge(sym_T, 673.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
        interps = [8.17484662576687e-6*sym_T**2 - 0.00926193251533741*sym_T + 18.0819438190184, 8.17484662576687e-6*sym_T**2 - 0.00926193251533741*sym_T + 18.0819438190184, 1.76073619631904e-6*sym_T**2 - 0.000628539877300632*sym_T + 15.176807196319, 1.76073619631904e-6*sym_T**2 - 0.000628539877300632*sym_T + 15.176807196319]
        assert len(conditions) == len(interps)
        d_func = ufl.conditional(conditions[0], interps[0], interps[0])
        for i in range(1, len(conditions)):
            d_func = ufl.conditional(conditions[i], interps[i], d_func)
        return d_func

    @property
    def bilinear_form(self):
        a_T = \
            ufl.inner(self.thermal_diffusivity_1(self._uT_func) *
                      ufl.grad(self._uT_func), ufl.grad(self._vT)) * \
            self._dx_sub_1 + \
            ufl.inner(self.thermal_diffusivity_2(self._uT_func) *
                      ufl.grad(self._uT_func), ufl.grad(self._vT)) * \
            self._dx_sub_2 + \
            ufl.inner(self.thermal_diffusivity_3(self._uT_func) *
                      ufl.grad(self._uT_func), ufl.grad(self._vT)) * \
            self._dx_sub_3 + \
            ufl.inner(self._h_cf * self._uT_func, self._vT) * \
            (self._ds_inner + self._ds_1top) + \
            ufl.inner(self._h_cout * self._uT_func, self._vT) * \
            (self._ds_outer_bottom + self._ds_outer_top) + \
            ufl.inner(self._h_cbottom * self._uT_func, self._vT) * \
            self._ds_bottom
        return a_T

    @property
    def linear_form(self):
        l_T = \
            ufl.inner(self._q_source, self._vT) * \
            (self._dx_sub_1 + self._dx_sub_2 + self._dx_sub_3) + \
            self._h_cf * self._vT * self._T_f * \
            (self._ds_inner + self._ds_1top) + \
            self._h_cout * self._vT * self._T_out * \
            (self._ds_outer_bottom + self._ds_outer_top) + \
            self._h_cbottom * self._vT * self._T_bottom * \
            self._ds_bottom - \
            ufl.inner(self._q_top, self._vT) * \
            (self._ds_2top + self._ds_3top)
        return l_T

    def bc_internal(self, x):
        return (self.mu[0] * np.sin(x[1] * 2. * np.pi), 0. * x[1], 0. * x[2])

    def bc_external(self, x):
        return (0. * x[0], 0. * x[1], 0. * x[2])

    @property
    def set_problem(self):
        a_T = self.bilinear_form
        l_T = self.linear_form
        problem = NonlinearProblem(a_T - l_T,
                                   self._uT_func, bcs=[])
        return problem

    def solve(self, mu):
        # Solve the problem at given parameter mu
        self.mu = mu
        self._uT_func.x.array[:] = 350.
        self._uT_func.x.scatter_forward()
        problem = self.set_problem
        # Mesh deformation (Harmonic mesh motion)
        with HarmonicMeshMotion(self._mesh, self._facet_tags,
                                [4, 10, 5, 11, 15, 19,
                                 8, 14, 16, 20, 23, 26,
                                 22, 25, 17, 21, 6, 12,
                                 7, 13],
                                [self.bc_external, self.bc_external,
                                 self.bc_external, self.bc_external,
                                 self.bc_external, self.bc_external,
                                 self.bc_external, self.bc_external,
                                 self.bc_external, self.bc_external,
                                 self.bc_external, self.bc_external,
                                 self.bc_external, self.bc_external,
                                 self.bc_internal, self.bc_internal,
                                 self.bc_external, self.bc_external,
                                 self.bc_external, self.bc_external],
                                reset_reference=True,
                                is_deformation=True):
            solver = NewtonSolver(self._mesh.comm, problem)
            solver.convergence_criterion = "incremental"

            solver.rtol = 1e-10
            solver.report = True
            ksp = solver.krylov_solver
            ksp.setFromOptions()
            # dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

            thermal_start_time = time.process_time()
            n, converged = solver.solve(self._uT_func)
            thermal_end_time = time.process_time()
            # print(f"Computed solution array: {uT_func.x.array}")
            assert (converged)
            print(f"Number of iterations: {n:d}")
            print(f"Time to solve (Thermal): {thermal_end_time - thermal_start_time}")
            solution = dolfinx.fem.Function(self._VT)
            solution.x.array[:] = self._uT_func.x.array.copy()

            sol_norm_local = self._inner_product_action_uT(self._uT_func)(self._uT_func)
            print(f"Temperature field norm: {sol_norm_local}")
            return solution


class PODANNReducedProblem(abc.ABC):
    def __init__(self, problem) -> None:
        self._basis_functions = rbnicsx.backends.FunctionsList(problem._VT)
        uT, vT = ufl.TrialFunction(problem._VT), ufl.TestFunction(problem._VT)
        x = ufl.SpatialCoordinate(problem._mesh)
        self._inner_product = problem._inner_product_uT
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

if __name__ == '__main__':

    # Read unit square mesh with Triangular elements
    mesh_comm = MPI.COMM_WORLD
    gdim = 3
    gmsh_model_rank = 0
    mesh, cell_tags, facet_tags = \
        dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh",
                                        mesh_comm, gmsh_model_rank, gdim=gdim)

    # Parameter tuple (D_0, ??, ??, ??)
    mu_ref = [0., 0.55, 0.8, 0.4]  # reference geometry
    mu = [0.23, 0.55, 0.8, 0.4] # Parametrised geometry

    para_dim = 4
    thermal_ann_input_samples_num = 13 # 420
    thermal_error_analysis_samples_num = 23 # 144
    num_snapshots = 17 # 400
    projection_error_samples_num = [7, 1, 1, 1]

    problem_parametric = ProblemOnDeformedDomain(mesh, cell_tags,
                                                 facet_tags,
                                                 HarmonicMeshMotion)
    uT_func = problem_parametric.solve(mu)

    VT_plot = dolfinx.fem.FunctionSpace(mesh, ("CG", 2))
    uT_func_plot = dolfinx.fem.Function(VT_plot)
    uT_func_plot.interpolate(uT_func)

    computed_file = "dlrbnicsx_solution_nonlinear_thermomechanical_thermal/solution_computed.xdmf"

    with dolfinx.io.XDMFFile(mesh.comm, computed_file, "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(uT_func_plot)


    # POD Starts ###
    def generate_training_set(samples=[5, 1, 1, 1]):
        training_set_0 = np.linspace(0., 0.2, samples[0])
        training_set_1 = np.linspace(0., 0., samples[1])
        training_set_2 = np.linspace(0., 0., samples[2])
        training_set_3 = np.linspace(0., 0., samples[2])
        training_set = np.array(list(itertools.product(training_set_0,
                                                    training_set_1,
                                                    training_set_2,
                                                    training_set_3)))
        return training_set


    training_set = rbnicsx.io.on_rank_zero(mesh.comm, generate_training_set)

    Nmax = 30

    print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
    print("")

    print("set up snapshots matrix")
    snapshots_matrix = rbnicsx.backends.FunctionsList(problem_parametric._VT)

    print("set up reduced problem")
    reduced_problem = PODANNReducedProblem(problem_parametric)

    print("")

    for (mu_index, mu) in enumerate(training_set):
        print(rbnicsx.io.TextLine(str(mu_index+1), fill="#"))

        print("Parameter number ", (mu_index+1), "of", training_set.shape[0])
        print("high fidelity solve for mu =", mu)
        snapshot = problem_parametric.solve(mu)

        print("update snapshots matrix")
        snapshots_matrix.append(snapshot)

        print("")

    print(rbnicsx.io.TextLine("perform POD", fill="#"))
    eigenvalues_thermal, modes_thermal, _ = \
        rbnicsx.backends.\
        proper_orthogonal_decomposition(snapshots_matrix,
                                        problem_parametric._inner_product_action_uT,
                                        N=Nmax, tol=1.e-6)
    reduced_problem._basis_functions.extend(modes_thermal)
    reduced_size = len(reduced_problem._basis_functions)
    print("")

    print(rbnicsx.io.TextBox("POD-Galerkin offline phase ends", fill="="))

    positive_eigenvalues = np.where(eigenvalues_thermal > 0., eigenvalues_thermal, np.nan)
    singular_values = np.sqrt(positive_eigenvalues)


    plt.figure(figsize=[8, 10])
    xint = list()
    yval = list()

    for x, y in enumerate(eigenvalues_thermal[:reduced_size]):
        yval.append(y)
        xint.append(x+1)

    plt.plot(xint, yval, "*-", color="orange")
    plt.xlabel("Eigenvalue number", fontsize=18)
    plt.ylabel("Eigenvalue", fontsize=18)
    plt.xticks(xint)
    plt.yscale("log")
    plt.title("Eigenvalue decay", fontsize=24)
    plt.tight_layout()
    plt.savefig("eigenvalue_thermal.png")
    # plt.show()

    # Projection errors #

    projection_error_samples = \
        generate_training_set(samples=projection_error_samples_num)
    projection_error_array = np.zeros(projection_error_samples.shape[0])

    for i in range(projection_error_samples.shape[0]):
        fem_sol = problem_parametric.solve(projection_error_samples[i, :])
        projected_sol = reduced_problem.project_snapshot(fem_sol, reduced_size)
        rb_sol = reduced_problem.reconstruct_solution(projected_sol)
        projection_error_array[i] = \
            reduced_problem.norm_error(fem_sol, rb_sol)
        print(f"Projection error {projection_error_array[i]}, Parameter: {i}, {projection_error_samples[i, :]}")
        fem_sol_norm = reduced_problem.compute_norm(fem_sol)
        rb_sol_norm = reduced_problem.compute_norm(rb_sol)
        print(f"FEM solution norm: {fem_sol_norm}, RB solution norm: {rb_sol_norm}")

    # POD Ends ###

    # 5. ANN implementation

    def generate_ann_input_set(num_ann_samples):
        xlimits = np.array([[0.05, 0.55], [0.35, 0.35],
                            [0.8, 0.8], [0.4, 0.4]])
        sampling = LHS(xlimits=xlimits)
        training_set = sampling(num_ann_samples)
        return training_set

    def generate_ann_output_set(problem, reduced_problem,
                                input_set, mode=None):
        output_set = np.zeros([input_set.shape[0],
                              len(reduced_problem._basis_functions)])
        for i in range(input_set.shape[0]):
            if mode is None:
                print(f"Parameter number {i+1} of {input_set.shape[0]}")
                print(f"Parameter: {input_set[i,:]}")
            else:
                print(f"{mode} parameter number {i+1} of {input_set.shape[0]}")
                print(f"Parameter: {input_set[i,:]}")
            output_set[i, :] = \
                reduced_problem.project_snapshot(problem.solve(input_set[i, :]),
                                                 len(reduced_problem._basis_functions)).array.astype("f")
        return output_set


    # Training dataset
    thermal_ann_input_set = generate_ann_input_set(thermal_ann_input_samples_num)
    np.random.shuffle(thermal_ann_input_set)
    thermal_ann_output_set = \
        generate_ann_output_set(problem_parametric,
                                reduced_problem,
                                thermal_ann_input_set, mode="Training")

    thermal_num_training_samples = int(0.7 * thermal_ann_input_set.shape[0])
    thermal_num_validation_samples = \
        thermal_ann_input_set.shape[0] - thermal_num_training_samples

    reduced_problem.output_range[0] = np.min(thermal_ann_output_set)
    reduced_problem.output_range[1] = np.max(thermal_ann_output_set)
    # NOTE Output_range based on the computed values instead of user guess.

    thermal_input_training_set = thermal_ann_input_set[:thermal_num_training_samples, :]
    thermal_output_training_set = thermal_ann_output_set[:thermal_num_training_samples, :]

    thermal_input_validation_set = thermal_ann_input_set[thermal_num_training_samples:, :]
    thermal_output_validation_set = thermal_ann_output_set[thermal_num_training_samples:, :]

    customDataset = CustomDataset(reduced_problem,
                                  thermal_input_training_set,
                                  thermal_output_training_set)
    thermal_train_dataloader = DataLoader(customDataset, batch_size=40, shuffle=True)

    customDataset = CustomDataset(reduced_problem,
                                  thermal_input_validation_set,
                                  thermal_output_validation_set)
    thermal_valid_dataloader = DataLoader(customDataset, shuffle=False)

    # ANN model
    thermal_model = HiddenLayersNet(training_set.shape[1],
                                    [35, 35],
                                    len(reduced_problem._basis_functions),
                                    Tanh())

    for params in thermal_model.parameters():
        print(params.shape)

    thermal_path = "thermal_model.pth"
    save_model(thermal_model, thermal_path)
    # load_model(thermal_model, thermal_path)


    # Training of ANN
    thermal_training_loss = list()
    thermal_validation_loss = list()

    thermal_max_epochs = 20 # 20000
    thermal_min_validation_loss = None
    thermal_start_epoch = 0
    thermal_checkpoint_path = "thermal_checkpoint"
    thermal_checkpoint_epoch = 10

    thermal_learning_rate = 1.e-6
    thermal_optimiser = get_optimiser(thermal_model, "Adam", thermal_learning_rate)
    thermal_loss_fn = get_loss_func("MSE", reduction="sum")

    if os.path.exists(thermal_checkpoint_path):
        thermal_start_epoch, thermal_min_validation_loss = \
            load_checkpoint(thermal_checkpoint_path, thermal_model, thermal_optimiser)

    import time
    start_time = time.time()
    for thermal_epochs in range(thermal_start_epoch, thermal_max_epochs):
        if thermal_epochs > 0 and thermal_epochs % thermal_checkpoint_epoch == 0:
            save_checkpoint(thermal_checkpoint_path, thermal_epochs, thermal_model, thermal_optimiser,
                            thermal_min_validation_loss)
        print(f"Epoch: {thermal_epochs+1}/{thermal_max_epochs}")
        thermal_current_training_loss = train_nn(reduced_problem, thermal_train_dataloader,
                                                 thermal_model, thermal_loss_fn, thermal_optimiser)
        thermal_training_loss.append(thermal_current_training_loss)
        thermal_current_validation_loss = validate_nn(reduced_problem, thermal_valid_dataloader,
                                                      thermal_model, thermal_loss_fn)
        thermal_validation_loss.append(thermal_current_validation_loss)
        if thermal_epochs > 0 and thermal_current_validation_loss > 1.01 * thermal_min_validation_loss \
        and reduced_problem.regularisation == "EarlyStopping":
            # 1% safety margin against min_validation_loss
            # before invoking early stopping criteria
            print(f"Early stopping criteria invoked at epoch: {thermal_epochs+1}")
            break
        thermal_min_validation_loss = min(thermal_validation_loss)
    end_time = time.time()
    thermal_elapsed_time = end_time - start_time

    os.system(f"rm {thermal_checkpoint_path}")

    # Error analysis dataset
    print("\n")
    print("Generating error analysis (only input/parameters) dataset")
    print("\n")
    thermal_error_analysis_set = generate_ann_input_set(thermal_error_analysis_samples_num)
    thermal_error_numpy = np.zeros(thermal_error_analysis_set.shape[0])

    for i in range(thermal_error_analysis_set.shape[0]):
        print(f"Error analysis {i+1} of {thermal_error_analysis_set.shape[0]}")
        print(f"Parameter: : {thermal_error_analysis_set[i,:]}")
        thermal_error_numpy[i] = error_analysis(reduced_problem, problem_parametric,
                                        thermal_error_analysis_set[i, :], thermal_model,
                                        len(reduced_problem._basis_functions),
                                        online_nn)
        print(f"Error: {thermal_error_numpy[i]}")

    # ### Thermal ANN ends

    # ### Online phase ###
    online_mu = np.array([0.45, 0.56, 0.9, 0.7])
    thermal_fem_solution = problem_parametric.solve(online_mu)
    thermal_rb_solution = \
        reduced_problem.reconstruct_solution(
            online_nn(reduced_problem, problem_parametric,
                    online_mu, thermal_model,
                    len(reduced_problem._basis_functions)))

    thermal_fem_solution_plot = dolfinx.fem.Function(VT_plot)
    thermal_fem_solution_plot.interpolate(thermal_fem_solution)

    thermal_rb_solution_plot = dolfinx.fem.Function(VT_plot)
    thermal_rb_solution_plot.interpolate(thermal_rb_solution)

    thermal_fem_online_file \
        = "dlrbnicsx_solution_thermomechanical/thermal_fem_online_mu_computed.xdmf"
    with HarmonicMeshMotion(problem_parametric._mesh,
                            problem_parametric._facet_tags,
                            [4, 10, 5, 11, 15, 19,
                             8, 14, 16, 20, 23, 26,
                             22, 25, 17, 21, 6, 12,
                             7, 13],
                            [problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_internal,
                             problem_parametric.bc_internal,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external],
                            reset_reference=True,
                            is_deformation=True):
        with dolfinx.io.XDMFFile(mesh.comm, thermal_fem_online_file,
                                 "w") as thermal_solution_file:
            thermal_solution_file.write_mesh(mesh)
            thermal_solution_file.write_function(thermal_fem_solution_plot)

    thermal_rb_online_file \
        = "dlrbnicsx_solution_thermomechanical/thermal_rb_online_mu_computed.xdmf"
    with HarmonicMeshMotion(problem_parametric._mesh,
                            problem_parametric._facet_tags,
                            [4, 10, 5, 11, 15, 19,
                             8, 14, 16, 20, 23, 26,
                             22, 25, 17, 21, 6, 12,
                             7, 13],
                            [problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_internal,
                             problem_parametric.bc_internal,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external],
                            reset_reference=True,
                            is_deformation=True):
        with dolfinx.io.XDMFFile(mesh.comm, thermal_rb_online_file,
                                 "w") as thermal_solution_file:
            # NOTE scatter_forward not considered for online solution
            thermal_solution_file.write_mesh(mesh)
            thermal_solution_file.write_function(thermal_rb_solution_plot)

    thermal_error_function = dolfinx.fem.Function(problem_parametric._VT)
    thermal_error_function.x.array[:] = \
        thermal_fem_solution.x.array - thermal_rb_solution.x.array

    thermal_error_function_plot = dolfinx.fem.Function(VT_plot)
    thermal_error_function_plot.interpolate(thermal_error_function)

    thermal_fem_rb_error_file \
        = "dlrbnicsx_solution_thermomechanical/thermal_fem_rb_error_computed.xdmf"
    with HarmonicMeshMotion(problem_parametric._mesh,
                            problem_parametric._facet_tags,
                            [4, 10, 5, 11, 15, 19,
                             8, 14, 16, 20, 23, 26,
                             22, 25, 17, 21, 6, 12,
                             7, 13],
                            [problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_internal,
                             problem_parametric.bc_internal,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external],
                            reset_reference=True,
                            is_deformation=True):
        with dolfinx.io.XDMFFile(mesh.comm, thermal_fem_rb_error_file,
                                 "w") as thermal_solution_file:
            thermal_solution_file.write_mesh(mesh)
            thermal_solution_file.write_function(thermal_error_function_plot)

    with HarmonicMeshMotion(problem_parametric._mesh,
                            problem_parametric._facet_tags,
                            [4, 10, 5, 11, 15, 19,
                             8, 14, 16, 20, 23, 26,
                             22, 25, 17, 21, 6, 12,
                             7, 13],
                            [problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_internal,
                             problem_parametric.bc_internal,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external,
                             problem_parametric.bc_external],
                            reset_reference=True,
                            is_deformation=True):
        print(reduced_problem.norm_error(thermal_fem_solution, thermal_rb_solution))
        print(reduced_problem.compute_norm(thermal_error_function))

    print(reduced_problem.norm_error(thermal_fem_solution, thermal_rb_solution))
    print(reduced_problem.compute_norm(thermal_error_function))

    print(f"Training time (Thermal): {thermal_elapsed_time}")