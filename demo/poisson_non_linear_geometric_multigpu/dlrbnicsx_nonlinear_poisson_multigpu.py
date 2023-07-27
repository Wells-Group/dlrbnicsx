import abc
import numpy as np
import itertools
import matplotlib.pyplot as plt

import dolfinx
import ufl

from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion

from mpi4py import MPI
from petsc4py import PETSc
import os

import rbnicsx
import rbnicsx.backends
import rbnicsx.online

import torch
import torch.distributed as dist

from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory \
    import Tanh
from dlrbnicsx.dataset.custom_partitioned_dataset_gpu import CustomPartitionedDatasetGpu, init_gpu_process_group
from dlrbnicsx.interface.wrappers import DataLoader, model_to_gpu, model_synchronise, load_model, save_model
from dlrbnicsx.train_validate_test.train_validate_test_multigpu import train_nn, validate_nn, error_analysis, online_nn


class ProblemOnDeformedDomain(abc.ABC):
    def __init__(self, mesh, subdomains, boundaries,
                 meshDeformationContext):
        self._mesh = mesh
        self._subdomains = subdomains
        self._boundaries = boundaries
        self._meshDeformationContext = meshDeformationContext
        self._V = dolfinx.fem.FunctionSpace(self._mesh, ("CG", 2))
        self._trial = ufl.TrialFunction(self._V)
        self._test = ufl.TestFunction(self._V)
        # u, v = self._trial, self._test
        self._solution = dolfinx.fem.Function(self._V)
        self._dirichletFunc = dolfinx.fem.Function(self._V)
        self._boundary_markers = [1, 2, 3, 4]
        self.gdim = self._mesh.geometry.dim

    @property
    def assemble_bcs(self):
        bcs = list()
        for i in self._boundary_markers:
            dofs = dolfinx.fem.locate_dofs_topological(self._V,
                                                       self.gdim-1,
                                                       self._boundaries.find(i))
            bcs.append(dolfinx.fem.dirichletbc(self._dirichletFunc, dofs))
        return bcs

    @property
    def source_term(self):
        return -ufl.div(ufl.exp(self._dirichletFunc) *
                        ufl.grad(self._dirichletFunc))

    @property
    def residual_term(self):
        return ufl.inner(ufl.exp(self._solution) * ufl.grad(self._solution),
                         ufl.grad(self._test)) * ufl.dx - \
            ufl.inner(self.source_term, self._test) * ufl.dx

    @property
    def set_problem(self):
        problemNonLinear = \
            dolfinx.fem.petsc.NonlinearProblem(self.residual_term,
                                               self._solution,
                                               bcs=self.assemble_bcs)
        return problemNonLinear

    def solve(self, mu):
        self._bcs_geometric = \
            [lambda x: (0. * x[1], mu[0]*np.sin(x[0]*np.pi)),
             lambda x: (0. * x[1], 0. * x[0]),
             lambda x: (0. * x[1], -mu[1]*np.sin(x[0]*np.pi)),
             lambda x: (0. * x[1], 0. * x[0])]
        problemNonLinear = self.set_problem
        solution = dolfinx.fem.Function(self._V)
        with self._meshDeformationContext(self._mesh, self._boundaries,
                                          self._boundary_markers,
                                          self._bcs_geometric,
                                          reset_reference=True,
                                          is_deformation=True) as mesh_class:
            solver = dolfinx.nls.petsc.NewtonSolver(mesh_class._mesh.comm,
                                                    problemNonLinear)
            solver.convergence_criterion = "incremental"
            solver.rtol = 1.e-6
            solver.report = True
            ksp = solver.krylov_solver
            ksp.setFromOptions()
            self._dirichletFunc.interpolate(lambda x:
                                            x[1] * np.sin(x[0] * np.pi) *
                                            np.cos(x[1] * np.pi))
            n, converged = solver.solve(self._solution)
            assert (converged)
            solution.x.array[:] = self._solution.x.array.copy()
            return solution


class PODANNReducedProblem(abc.ABC):
    def __init__(self, problem):
        self._basis_functions = \
            rbnicsx.backends.FunctionsList(problem._V)
        u, v = ufl.TrialFunction(problem._V), ufl.TestFunction(problem._V)
        self._inner_product = ufl.inner(u, v) * ufl.dx\
            + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        self._inner_product_action = \
            rbnicsx.backends.bilinear_form_action(self._inner_product,
                                                  part="real")
        self.input_scaling_range = [-1., 1.]
        self.output_scaling_range = [-1., 1.]
        self.input_range = np.array([[0.2, -0.2], [0.3, -0.4]])
        self.output_range = [None, None]
        self.loss_fn = "MSE"
        self.learning_rate = 1e-5
        self.optimizer = "Adam"
        self.regularisation = "EarlyStopping"

    def reconstruct_solution(self, reduced_solution):
        return self._basis_functions[:reduced_solution.size] * \
            reduced_solution

    def compute_norm(self, function):
        return np.sqrt(self._inner_product_action(function)(function))

    def project_snapshot(self, solution, N):
        return self._project_snapshot(solution, N)

    def _project_snapshot(self, solution, N):
        projected_snapshot = rbnicsx.online.create_vector(N)
        A = rbnicsx.backends.project_matrix(
            self._inner_product_action,
            self._basis_functions[:N])
        F = rbnicsx.backends.project_vector(
            self._inner_product_action(solution),
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
        return self.compute_norm(u-v) / self.compute_norm(u)


# Read from mesh
world_comm = MPI.COMM_WORLD
rank = world_comm.rank
size = world_comm.size

mesh_comm = MPI.COMM_SELF
gdim = 2
gmsh_model_rank = 0
mesh, subdomains, boundaries = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh", mesh_comm,
                                    gmsh_model_rank, gdim=gdim)

mu = np.array([0.3, -0.413])

problem_parametric = ProblemOnDeformedDomain(mesh, subdomains,
                                             boundaries, HarmonicMeshMotion)

solution_mu = problem_parametric.solve(mu)
itemsize = MPI.DOUBLE.Get_size()
num_snapshots = 8 * 8
num_dofs = solution_mu.x.array.shape[0]

para_dim = 2
print(f"Rank: {rank}, Solution dofs:{solution_mu.x.array}")

if world_comm.rank == 0:
    nbytes_para = num_snapshots * itemsize * para_dim
    nbytes_dofs = num_snapshots * itemsize * num_dofs
else:
    nbytes_para = 0
    nbytes_dofs = 0


def generate_training_set(samples=[8, 8]):
    training_set_0 = np.linspace(0.2, 0.3, samples[0])
    training_set_1 = np.linspace(-0.2, -0.4, samples[1])
    training_set = np.array(list(itertools.product(training_set_0,
                                                   training_set_1)))
    return training_set


win0 = MPI.Win.Allocate_shared(nbytes_para, itemsize, comm=MPI.COMM_WORLD)
buf0, itemsize = win0.Shared_query(0)
training_set = np.ndarray(buffer=buf0, dtype="d", shape=(num_snapshots, para_dim))

if world_comm.rank == 0:
    training_set[:, :] = generate_training_set()

world_comm.Barrier()

training_set_indices = \
    np.arange(world_comm.rank, training_set.shape[0], world_comm.size)

win1 = MPI.Win.Allocate_shared(nbytes_dofs, itemsize, comm=MPI.COMM_WORLD)
buf1, itemsize = win1.Shared_query(0)
training_set_solutions = \
    np.ndarray(buffer=buf1, dtype="d", shape=(num_snapshots, int(num_dofs)))

world_comm.Barrier()

for mu_index in training_set_indices:
    print(rbnicsx.io.TextLine(f"{mu_index+1}/{training_set_indices.shape[0]}",
                              fill="#"))
    solution_snapshot = \
        problem_parametric.solve(training_set[mu_index, :])
    training_set_solutions[mu_index, :] = solution_snapshot.x.array

world_comm.Barrier()

Nmax = 30

print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
print("")

print("Set up snapshots matrix")
snapshots_matrix = rbnicsx.backends.FunctionsList(problem_parametric._V)

print("Setup reduced problem")
reduced_problem = PODANNReducedProblem(problem_parametric)
print("")

for (mu_index, mu) in enumerate(training_set_solutions):
    print(rbnicsx.io.TextLine(
        f"{mu_index+1}/{training_set_solutions.shape[0]}",
        fill="#"))
    snapshot = dolfinx.fem.Function(problem_parametric._V)
    snapshot.x.array[:] = training_set_solutions[mu_index, :]

    print("Update snapshots matrix")
    snapshots_matrix.append(snapshot)

    print("")

print(rbnicsx.io.TextLine("perform POD", fill="#"))
eigenvalues, modes, _ = \
    rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_matrix,
                                    reduced_problem._inner_product_action,
                                    N=Nmax, tol=1.e-10)
reduced_problem._basis_functions.extend(modes)
reduced_size = len(reduced_problem._basis_functions)
print("")

print(rbnicsx.io.TextBox("POD-Galerkin offline phase ends", fill="="))

positive_eigenvalues = np.where(eigenvalues > 0., eigenvalues, np.nan)
singular_values = np.sqrt(positive_eigenvalues)

print(f"Rank {rank}, Positive eigenvalues: {positive_eigenvalues[:len(reduced_problem._basis_functions)]}")

if world_comm.rank == 0:
    plt.figure(figsize=[8, 10])
    xint = list()
    yval = list()

    for x, y in enumerate(eigenvalues[:reduced_size]):
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


num_ann_input_samples = 7 * 7
num_ann_input_samples_training = int(0.7 * num_ann_input_samples)
num_ann_input_samples_validation = \
    num_ann_input_samples - int(0.7 * num_ann_input_samples)
itemsize = MPI.DOUBLE.Get_size()


def generate_ann_input_set(samples=[7, 7]):
    input_set_0 = np.linspace(0.2, 0.3, samples[0])
    input_set_1 = np.linspace(-0.2, -0.4, samples[1])
    input_set = np.array(list(itertools.product(input_set_0,
                                                input_set_1)))
    return input_set


if world_comm.rank == 0:
    ann_input_samples = generate_ann_input_set()
    # np.random.shuffle(ann_input_samples)
    nbytes_para_ann_training = num_ann_input_samples_training * \
        itemsize * para_dim
    nbytes_dofs_ann_training = num_ann_input_samples_training * itemsize * \
        len(reduced_problem._basis_functions)
    nbytes_para_ann_validation = num_ann_input_samples_validation * \
        itemsize * para_dim
    nbytes_dofs_ann_validation = num_ann_input_samples_validation * \
        itemsize * len(reduced_problem._basis_functions)
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
                      len(reduced_problem._basis_functions)))

win5 = MPI.Win.Allocate_shared(nbytes_dofs_ann_validation, itemsize,
                               comm=MPI.COMM_WORLD)
buf5, itemsize = win5.Shared_query(0)
ann_output_samples_validation = \
    np.ndarray(buffer=buf5, dtype="d",
               shape=(num_ann_input_samples_validation,
                      len(reduced_problem._basis_functions)))

if world_comm.rank == 0:
    ann_input_samples_training[:, :] = \
        ann_input_samples[:num_ann_input_samples_training, :]
    ann_input_samples_validation[:, :] = \
        ann_input_samples[num_ann_input_samples_training:, :]
    ann_output_samples_training[:, :] = \
        np.zeros([num_ann_input_samples_training,
                  len(reduced_problem._basis_functions)])
    ann_output_samples_validation[:, :] = \
        np.zeros([num_ann_input_samples_validation,
                  len(reduced_problem._basis_functions)])

world_comm.Barrier()

group0_procs = world_comm.group.Incl([0, 1, 2, 3])
gpu_group0_comm = world_comm.Create_group(group0_procs)

group1_procs = world_comm.group.Incl([4, 5, 6, 7])
gpu_group1_comm = world_comm.Create_group(group1_procs)

group2_procs = world_comm.group.Incl([8, 9, 10, 11])
gpu_group2_comm = world_comm.Create_group(group2_procs)

group3_procs = world_comm.group.Incl([12, 13, 14, 15])
gpu_group3_comm = world_comm.Create_group(group3_procs)

comm_list = [gpu_group0_comm, gpu_group1_comm, gpu_group2_comm, gpu_group3_comm]

for i in range(len(comm_list)):
    if comm_list[i] != MPI.COMM_NULL:
        cuda_rank = i
        training_set_indices = \
            np.arange(i, num_ann_input_samples_training, len(comm_list))
        validation_set_indices = \
            np.arange(i, num_ann_input_samples_validation, len(comm_list))
        training_set_indices_local = \
            np.arange((comm_list[i]).rank, training_set_indices.shape[0],
                      (comm_list[i]).size)
        validation_set_indices_local = \
            np.arange((comm_list[i]).rank, validation_set_indices.shape[0],
                      (comm_list[i]).size)
        for k in training_set_indices_local:
            fem_solution = \
                problem_parametric.solve(
                    ann_input_samples_training[training_set_indices[k],
                                               :])
            ann_output_samples_training[training_set_indices[k], :] = \
                reduced_problem.project_snapshot(
                    fem_solution, len(reduced_problem._basis_functions))
        for k in validation_set_indices_local:
            fem_solution = \
                problem_parametric.solve(
                    ann_input_samples_validation[validation_set_indices[k],
                                                 :])
            ann_output_samples_validation[validation_set_indices[k], :] = \
                reduced_problem.project_snapshot(
                    fem_solution, len(reduced_problem._basis_functions))

world_comm.Barrier()

reduced_problem.output_range[0] = np.min(ann_output_samples_training)
reduced_problem.output_range[1] = np.max(ann_output_samples_training)

training_communicator_procs = world_comm.group.Incl([0])
# TODO add more processes in Incl, Currently only 1 GPU is considered
training_communicator_comm = \
    world_comm.Create_group(training_communicator_procs)

if training_communicator_comm != MPI.COMM_NULL:

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    init_gpu_process_group(training_communicator_comm)
    print(f"World rank: {world_comm.rank}, Gpu comm rank: {training_communicator_comm.rank}")

    training_set_indices_gpu = \
        np.arange(training_communicator_comm.rank,
                  num_ann_input_samples_training,
                  training_communicator_comm.size)

    validation_set_indices_gpu = \
        np.arange(training_communicator_comm.rank,
                  num_ann_input_samples_validation,
                  training_communicator_comm.size)

    cuda_rank = [0, 1, 2, 3]

    customDataset = \
        CustomPartitionedDatasetGpu(problem_parametric, reduced_problem,
                                    len(reduced_problem._basis_functions),
                                    ann_input_samples_training[training_set_indices_gpu,
                                                 :],
                                    ann_output_samples_training[training_set_indices_gpu,
                                                  :],
                                    training_set_indices_gpu,
                                    cuda_rank[training_communicator_comm.rank])
    train_dataloader = DataLoader(customDataset, batch_size=30,
                                  shuffle=False)#True)

    customDataset = \
        CustomPartitionedDatasetGpu(problem_parametric, reduced_problem,
                                    len(reduced_problem._basis_functions),
                                    ann_input_samples_validation[
                                    validation_set_indices_gpu, :],
                                    ann_output_samples_validation[
                                    validation_set_indices_gpu, :],
                                    validation_set_indices_gpu,
                                    cuda_rank[training_communicator_comm.rank])
    valid_dataloader = DataLoader(customDataset, shuffle=False)

    model = \
        HiddenLayersNet(ann_input_samples_training.shape[1], [30, 30],
                        len(reduced_problem._basis_functions), Tanh()
                        )

    path = "model.pth"
    #save_model(model, path)

    load_model(model, path)

    for param in model.parameters():
        print(f"after loading: {param.data}")

    model_to_gpu(model, cuda_rank=cuda_rank[training_communicator_comm.rank])
    model_synchronise(model)

    training_loss = list()
    validation_loss = list()

    max_epochs = 20#20000
    min_validation_loss = None
    for epochs in range(max_epochs):
        print(f"Epoch: {epochs+1}/{max_epochs}")
        current_training_loss = train_nn(reduced_problem, train_dataloader,
                                         model)
        training_loss.append(current_training_loss)
        current_validation_loss = validate_nn(reduced_problem,
                                              valid_dataloader,
                                              model,
                                              cuda_rank[training_communicator_comm.rank])
        validation_loss.append(current_validation_loss)
        # 1% safety margin against min_validation_loss before invoking
        # early stopping criteria
        if epochs > 0 and current_validation_loss > 1.01 * min_validation_loss \
        and reduced_problem.regularisation == "EarlyStopping":
            print(f"Early stopping criteria invoked at epoch: {epochs+1}")
            break
        min_validation_loss = min(validation_loss)

error_analysis_num_para = 5 * 5 # 15 * 15

itemsize = MPI.DOUBLE.Get_size()

if training_communicator_comm != MPI.COMM_NULL and training_communicator_comm.rank == 0:
    nbytes_para = error_analysis_num_para * itemsize * para_dim
    nbytes_error = error_analysis_num_para * itemsize
else:
    nbytes_para = 0
    nbytes_error = 0

win6 = MPI.Win.Allocate_shared(nbytes_para, itemsize,
                               comm=world_comm)
buf6, itemsize = win6.Shared_query(0)
error_analysis_set = np.ndarray(buffer=buf6, dtype="d",
                                shape=(error_analysis_num_para,
                                        para_dim))

win7 = MPI.Win.Allocate_shared(nbytes_error, itemsize,
                                comm=world_comm)
buf7, itemsize = win7.Shared_query(0)
relative_error = np.ndarray(buffer=buf7, dtype="d",
                            shape=(error_analysis_num_para))

if training_communicator_comm != MPI.COMM_NULL and training_communicator_comm.rank == 0:
    error_analysis_set[:, :] = \
        generate_ann_input_set(samples=[5, 5]) # (samples=[15, 15])
    relative_error[:] = np.zeros([error_analysis_num_para])

world_comm.Barrier()

for i in range(len(comm_list)):
    if comm_list[i] !=  MPI.COMM_NULL:
        error_analysis_indices_cpu = np.arange((comm_list[i]).rank,
                                               error_analysis_set.shape[0],
                                               (comm_list[i]).size)

print(f"world comm rank: {world_comm.rank}, {error_analysis_indices_cpu}")

if training_communicator_comm == MPI.COMM_NULL:
    model = None
    cuda_num = [None]
else:
    cuda_num = cuda_rank[training_communicator_comm.rank]

relative_error = error_analysis(reduced_problem, problem_parametric,
                                error_analysis_set, model,
                                len(reduced_problem._basis_functions),
                                online_nn, cuda_num,
                                comm_list, training_communicator_comm,
                                MPI.COMM_WORLD, error_analysis_indices_cpu)

online_mu = np.array([0.28, -0.32])

if training_communicator_comm != MPI.COMM_NULL and training_communicator_comm.rank == 0:
    solution = problem_parametric.solve(online_mu)
    solution_reduced = \
        online_nn(reduced_problem, problem_parametric,
                  online_mu, model,
                  len(reduced_problem._basis_functions),
                  cuda_rank[training_communicator_comm.rank],
                  input_scaling_range=reduced_problem.input_scaling_range,
                  output_scaling_range=reduced_problem.output_scaling_range,
                  input_range=reduced_problem.input_range,
                  output_range=reduced_problem.output_range)
    ann_reconstructed_solution = \
        reduced_problem.reconstruct_solution(solution_reduced)

    fem_online_file \
        = "dlrbnicsx_solution_nonlinear_poisson/fem_online_mu_computed.xdmf"
    with HarmonicMeshMotion(mesh, problem_parametric._boundaries,
                            problem_parametric._boundary_markers,
                            problem_parametric._bcs_geometric,
                            reset_reference=True,
                            is_deformation=True) as mesh_class:
        with dolfinx.io.XDMFFile(mesh.comm, fem_online_file,
                                "w") as solution_file:
            solution_file.write_mesh(mesh)
            solution_file.write_function(fem_solution)

    rb_online_file \
        = "dlrbnicsx_solution_nonlinear_poisson/rb_online_mu_computed.xdmf"
    with HarmonicMeshMotion(mesh, problem_parametric._boundaries,
                            problem_parametric._boundary_markers,
                            problem_parametric._bcs_geometric,
                            reset_reference=True,
                            is_deformation=True) as mesh_class:
        with dolfinx.io.XDMFFile(mesh.comm, rb_online_file,
                                "w") as solution_file:
            # NOTE scatter_forward not considered for online solution
            solution_file.write_mesh(mesh)
            solution_file.write_function(ann_reconstructed_solution)

    error_function = dolfinx.fem.Function(problem_parametric._V)
    error_function.x.array[:] = \
        solution.x.array - ann_reconstructed_solution.x.array
    fem_rb_error_file \
        = "dlrbnicsx_solution_nonlinear_poisson/fem_rb_error_computed.xdmf"
    with HarmonicMeshMotion(mesh, problem_parametric._boundaries,
                            problem_parametric._boundary_markers,
                            problem_parametric._bcs_geometric,
                            reset_reference=True,
                            is_deformation=True) as mesh_class:
        with dolfinx.io.XDMFFile(mesh.comm, fem_rb_error_file,
                                "w") as solution_file:
            solution_file.write_mesh(mesh)
            solution_file.write_function(error_function)

    print(f"Relative error: {relative_error}")

    print(f"Done! cuda rank {cuda_rank[training_communicator_comm.rank]}")


'''
# TODO
1. Data transfer during every epoch or only once before customDataset? --> NOTE Only Once (See 2. as well)
2. If before customDataset, write new customDataset in dlrbnicsx.
3. NOTE dist needs to be initialised only on training_communicator_comm.
4. comm.Free() at appropriate places
'''
