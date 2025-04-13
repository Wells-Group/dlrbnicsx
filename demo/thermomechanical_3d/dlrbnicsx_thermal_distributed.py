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
import argparse
import os
import time

from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory \
    import Tanh
from dlrbnicsx.dataset.custom_partitioned_dataset_gpu \
    import CustomPartitionedDatasetGpu
from dlrbnicsx.interface.wrappers import DataLoader, save_model, \
    load_model, model_synchronise, init_gpu_process_group, model_to_gpu, \
    model_to_cpu, save_checkpoint, load_checkpoint, get_optimiser, \
    get_loss_func, share_model
from dlrbnicsx.train_validate_test.train_validate_test_multigpu \
    import train_nn, validate_nn
from dlrbnicsx.train_validate_test.train_validate_test_distributed \
    import online_nn, error_analysis


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

            solver.rtol = 1.e-10
            solver.report = True
            ksp = solver.krylov_solver
            ksp.setFromOptions()
            # dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

            thermal_start_time = time.process_time()
            n, converged = solver.solve(self._uT_func)
            self._uT_func.x.scatter_forward()
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
            np.array([[0.55, 0.35, 0.8, 0.4],
                      [0.75, 0.55, 1.2, 0.6]])
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

    parser = argparse.ArgumentParser("Thermomechanical 3D case (Thermal) ")
    parser.add_argument("--num_fem_comms", nargs='?', const=1, type=int,
                        help="Number of subcommunicators for parallel FEM snapshot computations")
    args = parser.parse_args()

    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.rank
    world_size = world_comm.size
    num_comms = args.num_fem_comms
    fem_comm_list = list()

    world_comm.Barrier()

    for i in range(num_comms):
        fem_procs = \
            world_comm.group.Incl(np.arange(i, world_size,
                                            num_comms))
        fem_procs_comm = world_comm.Create_group(fem_procs)
        fem_comm_list.append(fem_procs_comm)

    for comm_i in fem_comm_list:
        if comm_i != MPI.COMM_NULL:
            mesh_comm = comm_i

    # Read unit square mesh with Triangular elements
    # mesh_comm = comm_i
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

    computed_file = "dlrbnicsx_solution_nonlinear_thermomechanical_thermal_distributed/solution_computed.xdmf"

    with dolfinx.io.XDMFFile(mesh.comm, computed_file, "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(uT_func_plot)
    
    itemsize = MPI.DOUBLE.Get_size()
    para_dim = 4
    rstart, rend = uT_func.vector.getOwnershipRange()
    num_dofs = mesh.comm.allreduce(rend, op=MPI.MAX) - \
        mesh.comm.allreduce(rstart, op=MPI.MIN)

    pod_samples = [9, 1, 1, 1]
    ann_samples = [2, 1, 1, 1]
    error_analysis_samples = [6, 1, 1, 1]
    num_snapshots = np.product(pod_samples)
    nbytes_para = itemsize * num_snapshots * para_dim
    nbytes_dofs = itemsize * num_snapshots * num_dofs

    thermal_ann_input_samples_num = 12 # 420
    thermal_error_analysis_samples_num = 14 # 144

    # Thermal POD Starts ###
    def generate_training_set(num_samples, para_dim):
        np.random.seed(32)
        training_set = np.random.uniform(size=(num_samples, para_dim))
        training_set[:, 0] = (0.75 - 0.55) * training_set[:, 0] + 0.55
        training_set[:, 1] = (0.55 - 0.35) * training_set[:, 1] + 0.35
        training_set[:, 2] = (1.20 - 0.80) * training_set[:, 2] + 0.80
        training_set[:, 3] = (0.60 - 0.40) * training_set[:, 3] + 0.40
        return training_set

    win0 = MPI.Win.Allocate_shared(nbytes_para, itemsize, comm=MPI.COMM_WORLD)
    buf0, itemsize = win0.Shared_query(0)
    para_matrix = np.ndarray(buffer=buf0, dtype="d",
                             shape=(num_snapshots, para_dim))

    if world_comm.rank == 0:
        para_matrix[:, :] = \
            generate_training_set(num_snapshots, para_dim)

    world_comm.Barrier()

    for i in range(len(fem_comm_list)):
        if fem_comm_list[i] != MPI.COMM_NULL:
            cpu_indices = np.arange(i, para_matrix.shape[0],
                                    len(fem_comm_list))
    
    if world_comm.rank == 0:
        nbytes = num_snapshots * num_dofs * itemsize
    else:
        nbytes = 0

    world_comm.Barrier()

    win1 = MPI.Win.Allocate_shared(nbytes, itemsize, comm=MPI.COMM_WORLD)
    buf1, itemsize = win1.Shared_query(0)
    snapshot_arrays = np.ndarray(buffer=buf1, dtype="d", shape=(num_snapshots, num_dofs))
    snapshots_matrix = rbnicsx.backends.FunctionsList(problem_parametric._VT)
    Nmax = 10

    for i in range(len(fem_comm_list)):
        if fem_comm_list[i] != MPI.COMM_NULL:
            for mu_index in cpu_indices:
                solution = problem_parametric.solve(para_matrix[mu_index, :])
                rstart, rend = solution.vector.getOwnershipRange()
                solution.x.scatter_forward()
                solution.vector.assemble()
                snapshot_arrays[mu_index, rstart:rend] = solution.vector[rstart:rend]

    world_comm.Barrier()

    for j in range(len(fem_comm_list)):
        if fem_comm_list[j] != MPI.COMM_NULL:
            for i in range(snapshot_arrays.shape[0]):
                solution_empty = dolfinx.fem.Function(problem_parametric._VT)
                # rstart, rend = solution_empty.vector.getOwnershipRange()
                solution_empty.vector[rstart:rend] = snapshot_arrays[i, rstart:rend]
                solution_empty.x.scatter_forward()
                solution_empty.vector.assemble()
                snapshots_matrix.append(solution_empty)

    world_comm.Barrier()

    print("Set up reduced problem")
    reduced_problem = PODANNReducedProblem(problem_parametric)

    print("")

    print(rbnicsx.io.TextLine("Perform POD", fill="#"))

    eigenvalues_thermal, modes_thermal, _ = \
        rbnicsx.backends.proper_orthogonal_decomposition(
            snapshots_matrix, reduced_problem._inner_product_action,
            N=Nmax, tol=1.e-12)

    reduced_problem._basis_functions.extend(modes_thermal)

    reduced_size = len(reduced_problem._basis_functions)
    print("")

    print(rbnicsx.io.TextBox("POD-Galerkin offline phase ends", fill="="))

    positive_eigenvalues = np.where(eigenvalues_thermal > 0.,
                                    eigenvalues_thermal, np.nan)
    singular_values = np.sqrt(positive_eigenvalues)

    if world_comm.rank == 0:
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
        plt.savefig("eigenvalue_decay")

    print(f"Eigenvalues: {positive_eigenvalues[:reduced_size]}")

    # ### POD Ends ###

    # ### ANN implementation ###
    def generate_ann_input_set(samples=[4, 4, 4]):
        # Select samples from the parameter space for ANN
        training_set_0 = np.linspace(0.2, 0.3, samples[0])
        training_set_1 = np.linspace(-0.2, -0.4, samples[1])
        training_set_2 = np.linspace(1., 4., samples[2])
        training_set_3 = np.linspace(1., 4., samples[3])
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

    num_ann_input_samples = np.product(ann_samples)
    num_training_samples = int(0.7 * num_ann_input_samples)
    num_validation_samples = num_ann_input_samples - int(0.7 * num_ann_input_samples)
    itemsize = MPI.DOUBLE.Get_size()

    if world_comm.rank == 0:
        ann_input_set = generate_ann_input_set(samples=ann_samples)
        # np.random.shuffle(ann_input_set)
        nbytes_para_ann_training = num_training_samples * itemsize * para_dim
        nbytes_dofs_ann_training = num_training_samples * itemsize * \
            len(reduced_problem._basis_functions)
        nbytes_para_ann_validation = num_validation_samples * itemsize * para_dim
        nbytes_dofs_ann_validation = num_validation_samples * itemsize * \
            len(reduced_problem._basis_functions)
    else:
        nbytes_para_ann_training = 0
        nbytes_dofs_ann_training = 0
        nbytes_para_ann_validation = 0
        nbytes_dofs_ann_validation = 0

    world_comm.barrier()

    win2 = MPI.Win.Allocate_shared(nbytes_para_ann_training, itemsize,
                                   comm=MPI.COMM_WORLD)
    buf2, itemsize = win2.Shared_query(0)
    input_training_set = \
        np.ndarray(buffer=buf2, dtype="d",
                   shape=(num_training_samples, para_dim))

    win3 = MPI.Win.Allocate_shared(nbytes_para_ann_validation, itemsize,
                                   comm=MPI.COMM_WORLD)
    buf3, itemsize = win3.Shared_query(0)
    input_validation_set = \
        np.ndarray(buffer=buf3, dtype="d",
                   shape=(num_validation_samples, para_dim))

    win4 = MPI.Win.Allocate_shared(nbytes_dofs_ann_training,
                                   itemsize, comm=MPI.COMM_WORLD)
    buf4, itemsize = win4.Shared_query(0)
    output_training_set = \
        np.ndarray(buffer=buf4, dtype="d",
                shape=(num_training_samples,
                       len(reduced_problem._basis_functions)))

    win5 = MPI.Win.Allocate_shared(nbytes_dofs_ann_validation,
                                   itemsize, comm=MPI.COMM_WORLD)
    buf5, itemsize = win5.Shared_query(0)
    output_validation_set = \
        np.ndarray(buffer=buf5, dtype="d",
                shape=(num_validation_samples,
                        len(reduced_problem._basis_functions)))

    if world_comm.rank == 0:
        input_training_set[:, :] = \
            ann_input_set[:num_training_samples, :]
        input_validation_set[:, :] = \
            ann_input_set[num_training_samples:, :]
        output_training_set[:, :] = \
            np.zeros([num_training_samples,
                    len(reduced_problem._basis_functions)])
        output_validation_set[:, :] = \
            np.zeros([num_validation_samples,
                    len(reduced_problem._basis_functions)])

    world_comm.Barrier()

    training_set_indices = np.arange(world_comm.rank,
                                    input_training_set.shape[0],
                                    world_comm.size)

    validation_set_indices = np.arange(world_comm.rank,
                                    input_validation_set.shape[0],
                                    world_comm.size)

    world_comm.Barrier()

    # Training dataset
    generate_ann_output_set(problem_parametric, reduced_problem,
                            input_training_set, output_training_set,
                            training_set_indices, mode="Training")

    generate_ann_output_set(problem_parametric, reduced_problem,
                            input_validation_set, output_validation_set,
                            validation_set_indices, mode="Validation")

    world_comm.Barrier()

    reduced_problem.output_range[0] = min(np.min(output_training_set),
                                          np.min(output_validation_set))
    reduced_problem.output_range[1] = max(np.max(output_training_set),
                                          np.max(output_validation_set))

    print("\n")

    # gpu_group0_procs = world_comm.group.Incl([0, 1, 2, 3])
    gpu_group0_procs = world_comm.group.Incl([0])
    gpu_group0_comm = world_comm.Create_group(gpu_group0_procs)

    # ANN model
    model = HiddenLayersNet(para_matrix.shape[1], [35, 35],
                            len(reduced_problem._basis_functions),
                            Tanh())

    if gpu_group0_comm != MPI.COMM_NULL:

        cuda_rank_list = [0, 1, 2, 3]
        init_gpu_process_group(gpu_group0_comm)

        training_set_indices_gpu = np.arange(gpu_group0_comm.rank,
                                            input_training_set.shape[0],
                                            gpu_group0_comm.size)
        validation_set_indices_gpu = np.arange(gpu_group0_comm.rank,
                                            input_validation_set.shape[0],
                                            gpu_group0_comm.size)


        customDataset = \
            CustomPartitionedDatasetGpu(reduced_problem,
                                        input_training_set,
                                        output_training_set,
                                        training_set_indices_gpu,
                                        cuda_rank_list[gpu_group0_comm.rank])
        train_dataloader = DataLoader(customDataset, batch_size=40,
                                      shuffle=False)#shuffle=True)

        customDataset = \
            CustomPartitionedDatasetGpu(reduced_problem,
                                        input_validation_set,
                                        output_validation_set,
                                        validation_set_indices_gpu,
                                        cuda_rank_list[gpu_group0_comm.rank])
        valid_dataloader = DataLoader(customDataset, shuffle=False)

        # ANN model
        path = "model.pth"
        save_model(model, path)
        load_model(model, path)

        model_to_gpu(model, cuda_rank=cuda_rank_list[gpu_group0_comm.rank])

        model_synchronise(model, verbose=True)

        # Training of ANN
        training_loss = list()
        validation_loss = list()

        max_epochs = 20000
        min_validation_loss = None
        start_epoch = 0
        checkpoint_path = "checkpoint"
        checkpoint_epoch = 10

        learning_rate = 1e-4
        optimiser = get_optimiser(model, "Adam", learning_rate)
        loss_fn = get_loss_func("MSE", reduction="sum")

        if os.path.exists(checkpoint_path):
            start_epoch, min_validation_loss = \
                load_checkpoint(checkpoint_path, model, optimiser)

        import time
        start_time = time.time()

        for epochs in range(start_epoch, max_epochs):
            if epochs > 0 and epochs % checkpoint_epoch == 0:
                save_checkpoint(checkpoint_path, epochs,
                                model, optimiser,
                                min_validation_loss)
            print(f"Epoch: {epochs+1}/{max_epochs}")
            current_training_loss = \
                train_nn(reduced_problem, train_dataloader,
                         model, loss_fn, optimiser)
            training_loss.append(current_training_loss)
            current_validation_loss = \
                validate_nn(reduced_problem, valid_dataloader,
                            model, cuda_rank_list[gpu_group0_comm.rank],
                            loss_fn)
            validation_loss.append(current_validation_loss)
            if epochs > 0 and current_validation_loss > min_validation_loss \
            and reduced_problem.regularisation == "EarlyStopping":
                # 1% safety margin against min_validation_loss
                # before invoking early stopping criteria
                print(f"Early stopping criteria invoked at epoch: {epochs+1}")
                break
            min_validation_loss = min(validation_loss)

        end_time = time.time()
        elapsed_time = end_time - start_time
        model_to_cpu(model)
        os.system(f"rm {checkpoint_path}")

    model_root_process = 0
    share_model(model, world_comm, model_root_process)
    world_comm.Barrier()

    # Error analysis dataset
    print("\n")
    print("Generating error analysis (only input/parameters) dataset")
    print("\n")

    error_analysis_num_para = np.product(error_analysis_samples)
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
    error_analysis_set = \
        np.ndarray(buffer=buf6, dtype="d",
                shape=(error_analysis_num_para,
                        para_dim))

    win7 = MPI.Win.Allocate_shared(nbytes_error, itemsize,
                                comm=world_comm)
    buf7, itemsize = win7.Shared_query(0)
    error_numpy = np.ndarray(buffer=buf7, dtype="d",
                            shape=(error_analysis_num_para))

    if world_comm.rank == 0:
        error_analysis_set[:, :] = \
            generate_ann_input_set(samples=error_analysis_samples)

    world_comm.Barrier()

    error_analysis_indices = \
        np.arange(world_comm.rank,
                  error_analysis_set.shape[0],
                  world_comm.size)
    for i in error_analysis_indices:
        error_numpy[i] = \
            error_analysis(reduced_problem, problem_parametric,
                           error_analysis_set[i, :], model,
                           len(reduced_problem._basis_functions),
                           online_nn)
        print(f"Error analysis {i+1} of {error_analysis_set.shape[0]}, Error: {error_numpy[i]}")

    world_comm.Barrier()

    if gpu_group0_comm != MPI.COMM_NULL:
        print(f"Rank {gpu_group0_comm.rank}, Training time: {elapsed_time}")

    # Online phase at parameter online_mu TODO