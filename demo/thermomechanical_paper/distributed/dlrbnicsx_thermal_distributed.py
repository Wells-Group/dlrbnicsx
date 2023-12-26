import dolfinx
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
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
        dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)
        ds = ufl.Measure("ds", domain=self._mesh, subdomain_data=self._boundaries)
        self.dx = dx
        self._ds_sf = ds(11) + ds(20) + ds(21) + ds(22) + ds(23)
        self._ds_bottom = ds(1) + ds(31)
        self._ds_out = ds(30)
        self._ds_sym = ds(5) + ds(9) + ds(12)
        self._ds_top = ds(18) + ds(19) + ds(27) + ds(28) + ds(29)
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

        self._max_iterations = 20
        self.rtol = 1.e-4
        self.atol = 1.e-12
        
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

    def thermal_diffusivity_4(self, sym_T):
        conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 693.)), ufl.And(ufl.ge(sym_T, 673.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
        interps = [8.17484662576687e-6*sym_T**2 - 0.00926193251533741*sym_T + 18.0819438190184, 8.17484662576687e-6*sym_T**2 - 0.00926193251533741*sym_T + 18.0819438190184, 1.76073619631904e-6*sym_T**2 - 0.000628539877300632*sym_T + 15.176807196319, 1.76073619631904e-6*sym_T**2 - 0.000628539877300632*sym_T + 15.176807196319]
        assert len(conditions) == len(interps)
        d_func = ufl.conditional(conditions[0], interps[0], interps[0])
        for i in range(1, len(conditions)):
            d_func = ufl.conditional(conditions[i], interps[i], d_func)
        return d_func

    def thermal_diffusivity_5(self, sym_T):
        conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 693.)), ufl.And(ufl.ge(sym_T, 673.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
        interps = [3.08018064076346e-5*sym_T**2 - 0.0376497392638036*sym_T + 31.7270693260054, 3.08018064076346e-5*sym_T**2 - 0.0376497392638036*sym_T + 31.7270693260054, -2.79311520109062e-6*sym_T**2 + 0.00756902522154049*sym_T + 16.5109550766871, -2.79311520109062e-6*sym_T**2 + 0.00756902522154049*sym_T + 16.5109550766871]
        assert len(conditions) == len(interps)
        d_func = ufl.conditional(conditions[0], interps[0], interps[0])
        for i in range(1, len(conditions)):
            d_func = ufl.conditional(conditions[i], interps[i], d_func)
        return d_func

    def thermal_diffusivity_6(self, sym_T):
        conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 693.)), ufl.And(ufl.ge(sym_T, 673.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
        interps = [8.17484662576687e-6*sym_T**2 - 0.00926193251533741*sym_T + 18.0819438190184, 8.17484662576687e-6*sym_T**2 - 0.00926193251533741*sym_T + 18.0819438190184, 1.76073619631904e-6*sym_T**2 - 0.000628539877300632*sym_T + 15.176807196319, 1.76073619631904e-6*sym_T**2 - 0.000628539877300632*sym_T + 15.176807196319]
        assert len(conditions) == len(interps)
        d_func = ufl.conditional(conditions[0], interps[0], interps[0])
        for i in range(1, len(conditions)):
            d_func = ufl.conditional(conditions[i], interps[i], d_func)
        return d_func

    def thermal_diffusivity_7(self, sym_T):
        conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 693.)), ufl.And(ufl.ge(sym_T, 673.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
        interps = [0.000299054192229039*sym_T**2 - 0.36574217791411*sym_T + 130.838954780164, 0.000299054192229039*sym_T**2 - 0.36574217791411*sym_T + 130.838954780164, -1.10434560327197e-5*sym_T**2 + 0.0516492566462166*sym_T - 9.61326294938646, -1.10434560327197e-5*sym_T**2 + 0.0516492566462166*sym_T - 9.61326294938646]
        assert len(conditions) == len(interps)
        d_func = ufl.conditional(conditions[0], interps[0], interps[0])
        for i in range(1, len(conditions)):
            d_func = ufl.conditional(conditions[i], interps[i], d_func)
        return d_func        
    
    @property
    def lhs_form(self):
        uT_func, vT = self.uT_func, self._test
        x = ufl.SpatialCoordinate(self._mesh)
        dx = self.dx
        a_T = \
            ufl.inner(self.thermal_diffusivity_1(uT_func) * ufl.grad(uT_func), ufl.grad(vT))  * x[0] * dx(1) + \
            ufl.inner(self.thermal_diffusivity_2(uT_func) * ufl.grad(uT_func), ufl.grad(vT))  * x[0] * dx(2) + \
            ufl.inner(5.3 * ufl.grad(uT_func), ufl.grad(vT))  * x[0] * dx(3) + \
            ufl.inner(4.75 * ufl.grad(uT_func), ufl.grad(vT))  * x[0] * dx(4) + \
            ufl.inner(self.thermal_diffusivity_5(uT_func) * ufl.grad(uT_func), ufl.grad(vT))  * x[0] * dx(5) + \
            ufl.inner(45.6 * ufl.grad(uT_func), ufl.grad(vT))  * x[0] * dx(6) + \
            ufl.inner(self.thermal_diffusivity_7(uT_func) * ufl.grad(uT_func), ufl.grad(vT))  * x[0] * dx(7) + \
            ufl.inner(self._h_cf * uT_func, vT) * x[0] * self._ds_sf + \
            ufl.inner(self._h_cout * uT_func, vT) * x[0] * self._ds_out + \
            ufl.inner(self._h_cbottom * uT_func, vT) * x[0] * self._ds_bottom
        return a_T
    
    @property
    def rhs_form(self):
        vT = self._test
        x = ufl.SpatialCoordinate(self._mesh)
        dx = self.dx
        l_T = \
            ufl.inner(self._q_source, vT) * x[0] * dx + \
            self._h_cf * vT * self._T_f * x[0] * self._ds_sf + \
            self._h_cout * vT * self._T_out * x[0] * self._ds_out + \
            self._h_cbottom * vT * self._T_bottom * x[0] * self._ds_bottom - \
            ufl.inner(self._q_top, vT) * x[0] * self._ds_top
        return l_T
    
    @property
    def set_problem(self):
        problemNonlinear = \
            NonlinearProblem(self.lhs_form - self.rhs_form,
                             self.uT_func, bcs=[])
        return problemNonlinear

    def solve(self, mu):
        vT = self._test
        self.mu = mu
        self.uT_func.x.array[:] = 350.
        self.uT_func.x.scatter_forward()
        problemNonlinear = self.set_problem
        solution = dolfinx.fem.Function(self._VT)
        with MeshDeformationWrapperClass(self._mesh, self._boundaries,
                                         self.mu_ref, self.mu):
            solver = NewtonSolver(self._mesh.comm, problemNonlinear)
            solver.convergence_criterion = "incremental"

            solver.rtol = 1e-10
            solver.report = True
            ksp = solver.krylov_solver
            ksp.setFromOptions()
            # dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

            n, converged = solver.solve(self.uT_func)
            # print(f"Computed solution array: {uT_func.x.array}")
            assert (converged)
            print(f"Number of interations: {n:d}")
            
            solution.x.array[:] = self.uT_func.x.array.copy()
            solution.x.scatter_forward()
            x = ufl.SpatialCoordinate(self._mesh)
            print(f"Temperature field norm: {self._mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(solution, solution) * x[0] * ufl.dx + ufl.inner(ufl.grad(solution), ufl.grad(solution)) * x[0] * ufl.dx)))}")
        return solution

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

    VT_plot = dolfinx.fem.FunctionSpace(mesh, ("CG", mesh.geometry.cmaps[0].degree))

    itemsize = MPI.DOUBLE.Get_size()
    para_dim = 4
    thermal_ann_input_samples_num = 460
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
    world_comm.Barrier()
    pod_start_time = MPI.Wtime()
    thermal_eigenvalues, thermal_modes, _ = \
        rbnicsx.backends.\
        proper_orthogonal_decomposition(snapshots_matrix,
                                        thermal_reduced_problem._inner_product_action,
                                        N=Nmax, tol=1e-6)
    pod_end_time = MPI.Wtime()
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
    hidden_H = 20
    thermal_model_0 = HiddenLayersNet(thermal_training_set.shape[1], [hidden_H, hidden_H, hidden_H],
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

        thermal_path = "thermal_model_0.pth"
        # save_model(thermal_model, thermal_path)
        # load_model(thermal_model, thermal_path)

        model_synchronise(thermal_model_0, verbose=False)

        # Training of ANN
        thermal_training_loss = list()
        thermal_validation_loss = list()

        thermal_max_epochs = 20000
        thermal_min_validation_loss = None
        thermal_start_epoch = 0
        thermal_checkpoint_path = "thermal_checkpoint_0"
        thermal_checkpoint_epoch = 10

        thermal_learning_rate = 1e-6
        thermal_optimiser = get_optimiser(thermal_model_0, "Adam", thermal_learning_rate)
        thermal_loss_fn = get_loss_func("MSE", reduction="sum")

        if os.path.exists(thermal_checkpoint_path):
            thermal_start_epoch, thermal_min_validation_loss = \
                load_checkpoint(thermal_checkpoint_path, thermal_model_0, thermal_optimiser)

        import time
        start_time = MPI.Wtime()
        for thermal_epochs in range(thermal_start_epoch, thermal_max_epochs):
            if thermal_epochs > 0 and thermal_epochs % thermal_checkpoint_epoch == 0:
                save_checkpoint(thermal_checkpoint_path, thermal_epochs,
                                thermal_model_0, thermal_optimiser,
                                thermal_min_validation_loss)
            print(f"Epoch: {thermal_epochs+1}/{thermal_max_epochs}")
            thermal_current_training_loss = train_nn(thermal_reduced_problem,
                                            thermal_train_dataloader,
                                            thermal_model_0, thermal_loss_fn, thermal_optimiser)
            thermal_training_loss.append(thermal_current_training_loss)
            thermal_current_validation_loss = validate_nn(thermal_reduced_problem,
                                                thermal_valid_dataloader,
                                                thermal_model_0, thermal_loss_fn)
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

    print("\n")

    thermal_cpu_group1_procs = world_comm.group.Incl([4, 5, 6, 7])
    thermal_cpu_group1_comm = world_comm.Create_group(thermal_cpu_group1_procs)

    # ANN model
    hidden_H = 40
    thermal_model_1 = HiddenLayersNet(thermal_training_set.shape[1], [hidden_H, hidden_H, hidden_H],
                                    len(thermal_reduced_problem._basis_functions), Tanh())

    if thermal_cpu_group1_comm != MPI.COMM_NULL:
        init_cpu_process_group(thermal_cpu_group1_comm)

        thermal_training_set_indices_cpu = np.arange(thermal_cpu_group1_comm.rank,
                                            thermal_input_training_set.shape[0],
                                            thermal_cpu_group1_comm.size)
        thermal_validation_set_indices_cpu = np.arange(thermal_cpu_group1_comm.rank,
                                            thermal_input_validation_set.shape[0],
                                            thermal_cpu_group1_comm.size)

        customDataset = CustomPartitionedDataset(thermal_reduced_problem, thermal_input_training_set,
                                                 thermal_output_training_set, thermal_training_set_indices_cpu)
        thermal_train_dataloader = DataLoader(customDataset, batch_size=10, shuffle=True)

        customDataset = CustomPartitionedDataset(thermal_reduced_problem, thermal_input_validation_set,
                                                 thermal_output_validation_set, thermal_validation_set_indices_cpu)
        thermal_valid_dataloader = DataLoader(customDataset, shuffle=False)

        thermal_path = "thermal_model_1.pth"
        # save_model(thermal_model, thermal_path)
        # load_model(thermal_model, thermal_path)

        model_synchronise(thermal_model_1, verbose=False)

        # Training of ANN
        thermal_training_loss = list()
        thermal_validation_loss = list()

        thermal_max_epochs = 20000
        thermal_min_validation_loss = None
        thermal_start_epoch = 0
        thermal_checkpoint_path = "thermal_checkpoint_1"
        thermal_checkpoint_epoch = 10

        thermal_learning_rate = 1e-6
        thermal_optimiser = get_optimiser(thermal_model_1, "Adam", thermal_learning_rate)
        thermal_loss_fn = get_loss_func("MSE", reduction="sum")

        if os.path.exists(thermal_checkpoint_path):
            thermal_start_epoch, thermal_min_validation_loss = \
                load_checkpoint(thermal_checkpoint_path, thermal_model_1, thermal_optimiser)

        import time
        start_time = MPI.Wtime()
        for thermal_epochs in range(thermal_start_epoch, thermal_max_epochs):
            if thermal_epochs > 0 and thermal_epochs % thermal_checkpoint_epoch == 0:
                save_checkpoint(thermal_checkpoint_path, thermal_epochs,
                                thermal_model_1, thermal_optimiser,
                                thermal_min_validation_loss)
            print(f"Epoch: {thermal_epochs+1}/{thermal_max_epochs}")
            thermal_current_training_loss = train_nn(thermal_reduced_problem,
                                            thermal_train_dataloader,
                                            thermal_model_1, thermal_loss_fn, thermal_optimiser)
            thermal_training_loss.append(thermal_current_training_loss)
            thermal_current_validation_loss = validate_nn(thermal_reduced_problem,
                                                thermal_valid_dataloader,
                                                thermal_model_1, thermal_loss_fn)
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

    print("\n")

    thermal_cpu_group2_procs = world_comm.group.Incl([8, 9, 10, 11])
    thermal_cpu_group2_comm = world_comm.Create_group(thermal_cpu_group2_procs)

    # ANN model
    hidden_H = 60
    thermal_model_2 = HiddenLayersNet(thermal_training_set.shape[1], [hidden_H, hidden_H, hidden_H],
                                    len(thermal_reduced_problem._basis_functions), Tanh())

    if thermal_cpu_group2_comm != MPI.COMM_NULL:
        init_cpu_process_group(thermal_cpu_group2_comm)

        thermal_training_set_indices_cpu = np.arange(thermal_cpu_group2_comm.rank,
                                            thermal_input_training_set.shape[0],
                                            thermal_cpu_group2_comm.size)
        thermal_validation_set_indices_cpu = np.arange(thermal_cpu_group2_comm.rank,
                                            thermal_input_validation_set.shape[0],
                                            thermal_cpu_group2_comm.size)

        customDataset = CustomPartitionedDataset(thermal_reduced_problem, thermal_input_training_set,
                                                 thermal_output_training_set, thermal_training_set_indices_cpu)
        thermal_train_dataloader = DataLoader(customDataset, batch_size=10, shuffle=True)

        customDataset = CustomPartitionedDataset(thermal_reduced_problem, thermal_input_validation_set,
                                                 thermal_output_validation_set, thermal_validation_set_indices_cpu)
        thermal_valid_dataloader = DataLoader(customDataset, shuffle=False)

        thermal_path = "thermal_model_2.pth"
        # save_model(thermal_model, thermal_path)
        # load_model(thermal_model, thermal_path)

        model_synchronise(thermal_model_2, verbose=False)

        # Training of ANN
        thermal_training_loss = list()
        thermal_validation_loss = list()

        thermal_max_epochs = 20000
        thermal_min_validation_loss = None
        thermal_start_epoch = 0
        thermal_checkpoint_path = "thermal_checkpoint_2"
        thermal_checkpoint_epoch = 10

        thermal_learning_rate = 1e-6
        thermal_optimiser = get_optimiser(thermal_model_2, "Adam", thermal_learning_rate)
        thermal_loss_fn = get_loss_func("MSE", reduction="sum")

        if os.path.exists(thermal_checkpoint_path):
            thermal_start_epoch, thermal_min_validation_loss = \
                load_checkpoint(thermal_checkpoint_path, thermal_model_2, thermal_optimiser)

        import time
        start_time = MPI.Wtime()
        for thermal_epochs in range(thermal_start_epoch, thermal_max_epochs):
            if thermal_epochs > 0 and thermal_epochs % thermal_checkpoint_epoch == 0:
                save_checkpoint(thermal_checkpoint_path, thermal_epochs,
                                thermal_model_2, thermal_optimiser,
                                thermal_min_validation_loss)
            print(f"Epoch: {thermal_epochs+1}/{thermal_max_epochs}")
            thermal_current_training_loss = train_nn(thermal_reduced_problem,
                                            thermal_train_dataloader,
                                            thermal_model_2, thermal_loss_fn, thermal_optimiser)
            thermal_training_loss.append(thermal_current_training_loss)
            thermal_current_validation_loss = validate_nn(thermal_reduced_problem,
                                                thermal_valid_dataloader,
                                                thermal_model_2, thermal_loss_fn)
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


    world_comm.Barrier()
    thermal_model_root_process = 0
    share_model(thermal_model_0, world_comm, thermal_model_root_process)
    thermal_model_root_process = 4
    share_model(thermal_model_1, world_comm, thermal_model_root_process)
    thermal_model_root_process = 8
    share_model(thermal_model_2, world_comm, thermal_model_root_process)
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
    thermal_error_numpy_0 = np.ndarray(buffer=buf7, dtype="d",
                            shape=(thermal_error_analysis_samples_num))


    win8 = MPI.Win.Allocate_shared(thermal_nbytes_error, itemsize,
                                   comm=world_comm)
    buf8, itemsize = win8.Shared_query(0)
    thermal_projection_error_numpy = np.ndarray(buffer=buf8, dtype="d",
                                                shape=(thermal_error_analysis_samples_num))

    win9 = MPI.Win.Allocate_shared(thermal_nbytes_error, itemsize,
                                   comm=world_comm)
    buf9, itemsize = win9.Shared_query(0)
    thermal_error_numpy_1 = np.ndarray(buffer=buf9, dtype="d",
                            shape=(thermal_error_analysis_samples_num))

    win10 = MPI.Win.Allocate_shared(thermal_nbytes_error, itemsize,
                                   comm=world_comm)
    buf10, itemsize = win10.Shared_query(0)
    thermal_error_numpy_2 = np.ndarray(buffer=buf10, dtype="d",
                            shape=(thermal_error_analysis_samples_num))

    if world_comm.rank == 0:
        thermal_error_analysis_set[:, :] = generate_ann_input_set(thermal_error_analysis_samples_num)

    world_comm.Barrier()

    thermal_error_analysis_indices = np.arange(world_comm.rank,
                                    thermal_error_analysis_set.shape[0],
                                    world_comm.size)
    for i in thermal_error_analysis_indices:
        thermal_error_numpy_0[i] = error_analysis(thermal_reduced_problem, thermal_problem_parametric,
                                        thermal_error_analysis_set[i, :], thermal_model_0,
                                        len(thermal_reduced_problem._basis_functions),
                                        online_nn)
        thermal_error_numpy_1[i] = error_analysis(thermal_reduced_problem, thermal_problem_parametric,
                                        thermal_error_analysis_set[i, :], thermal_model_1,
                                        len(thermal_reduced_problem._basis_functions),
                                        online_nn)
        thermal_error_numpy_2[i] = error_analysis(thermal_reduced_problem, thermal_problem_parametric,
                                        thermal_error_analysis_set[i, :], thermal_model_2,
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
              f"RB error 0: {thermal_error_numpy_0[i]}, " +
              f"RB error 1: {thermal_error_numpy_1[i]}, " +
              f"RB error 2: {thermal_error_numpy_2[i]}, " +
              f"Projection error: {thermal_projection_error_numpy[i]}")

    world_comm.Barrier()

    # ### Thermal ANN starts ###

    '''
    # Online phase
    if world_comm.rank == 0:
        online_mu = np.array([0.45, 0.56, 0.9, 0.7])
        thermal_fem_solution = thermal_problem_parametric.solve(online_mu)
        thermal_rb_solution = \
            thermal_reduced_problem.reconstruct_solution(
                online_nn(thermal_reduced_problem, thermal_problem_parametric,
                        online_mu, thermal_model,
                        len(thermal_reduced_problem._basis_functions)))

        thermal_fem_solution_plot = dolfinx.fem.Function(VT_plot)
        thermal_fem_solution_plot.interpolate(thermal_fem_solution)
        thermal_rb_solution_plot = dolfinx.fem.Function(VT_plot)
        thermal_rb_solution_plot.interpolate(thermal_rb_solution)

        thermal_fem_online_file \
            = "dlrbnicsx_solution_thermal/thermal_fem_online_mu_computed.xdmf"
        with MeshDeformationWrapperClass(mesh, facet_tags, mu_ref,
                                         online_mu):
            with dolfinx.io.XDMFFile(mesh.comm, thermal_fem_online_file,
                                    "w") as thermal_solution_file:
                thermal_solution_file.write_mesh(mesh)
                thermal_solution_file.write_function(thermal_fem_solution_plot)

        thermal_rb_online_file \
            = "dlrbnicsx_solution_thermal/thermal_rb_online_mu_computed.xdmf"
        with MeshDeformationWrapperClass(mesh, facet_tags, mu_ref,
                                online_mu):
            with dolfinx.io.XDMFFile(mesh.comm, thermal_rb_online_file,
                                    "w") as thermal_solution_file:
                # NOTE scatter_forward not considered for online solution
                thermal_solution_file.write_mesh(mesh)
                thermal_solution_file.write_function(thermal_rb_solution_plot)

        thermal_error_function = dolfinx.fem.Function(thermal_problem_parametric._VT)
        thermal_error_function.x.array[:] = \
            thermal_fem_solution.x.array - thermal_rb_solution.x.array

        thermal_error_function_plot = dolfinx.fem.Function(VT_plot)
        thermal_error_function_plot.interpolate(thermal_error_function)

        thermal_fem_rb_error_file \
            = "dlrbnicsx_solution_thermal/thermal_fem_rb_error_computed.xdmf"
        with MeshDeformationWrapperClass(mesh, facet_tags, mu_ref,
                                         online_mu):
            with dolfinx.io.XDMFFile(mesh.comm, thermal_fem_rb_error_file,
                                    "w") as thermal_solution_file:
                thermal_solution_file.write_mesh(mesh)
                thermal_solution_file.write_function(thermal_error_function_plot)

        thermal_projection_error_function = dolfinx.fem.Function(thermal_problem_parametric._VT)
        thermal_reconstructed_solution = \
            thermal_reduced_problem.reconstruct_solution(
                thermal_reduced_problem.project_snapshot(thermal_problem_parametric.solve(online_mu),
                                                         len(thermal_reduced_problem._basis_functions)))
        thermal_projection_error_function.x.array[:] = \
            thermal_fem_solution.x.array - thermal_reconstructed_solution.x.array

        thermal_projection_error_function_plot = dolfinx.fem.Function(VT_plot)
        thermal_projection_error_function_plot.interpolate(thermal_projection_error_function)

        thermal_fem_rb_projection_error_file \
            = "dlrbnicsx_solution_thermal/thermal_fem_rb_projection_error_computed.xdmf"
        with MeshDeformationWrapperClass(mesh, facet_tags, mu_ref,
                                         online_mu):
            with dolfinx.io.XDMFFile(mesh.comm, thermal_fem_rb_projection_error_file,
                                    "w") as thermal_solution_file:
                thermal_solution_file.write_mesh(mesh)
                thermal_solution_file.write_function(thermal_projection_error_function_plot)
    '''

    if thermal_cpu_group0_comm != MPI.COMM_NULL:
        print(f"Training time (Thermal 0): {thermal_elapsed_time}")

    if thermal_cpu_group1_comm != MPI.COMM_NULL:
        print(f"Training time (Thermal 1): {thermal_elapsed_time}")

    if thermal_cpu_group2_comm != MPI.COMM_NULL:
        print(f"Training time (Thermal 2): {thermal_elapsed_time}")

    if world_comm.rank == 0:
        np.save("thermal_error_analysis_set.npy", thermal_error_analysis_set)
        np.save("thermal_rb_error_0.npy", thermal_error_numpy_0)
        np.save("thermal_rb_error_1.npy", thermal_error_numpy_1)
        np.save("thermal_rb_error_2.npy", thermal_error_numpy_2)
        np.save("thermal_projection_error.npy", thermal_projection_error_numpy)
    
    world_comm.Barrier()
    
    print(f"Basis size: {thermal_reduced_size}, hidden_H: {20}, Training samples: {thermal_ann_input_samples_num}, Error: {np.mean(thermal_error_numpy_0)}, Projection error: {np.mean(thermal_projection_error_numpy)}, Rank: {world_comm.rank}, POD time: {pod_end_time - pod_start_time}")
    print(f"Basis size: {thermal_reduced_size}, hidden_H: {40}, Training samples: {thermal_ann_input_samples_num}, Error: {np.mean(thermal_error_numpy_1)}, Projection error: {np.mean(thermal_projection_error_numpy)}, Rank: {world_comm.rank}, POD time: {pod_end_time - pod_start_time}")
    print(f"Basis size: {thermal_reduced_size}, hidden_H: {60}, Training samples: {thermal_ann_input_samples_num}, Error: {np.mean(thermal_error_numpy_2)}, Projection error: {np.mean(thermal_projection_error_numpy)}, Rank: {world_comm.rank}, POD time: {pod_end_time - pod_start_time}")
