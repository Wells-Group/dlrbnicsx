import abc
import itertools
# import typing
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

import ufl
import dolfinx
# import mdfenicsx

from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion

import rbnicsx
import rbnicsx.backends
import rbnicsx.online

# import dlrbnicsx

from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory import Tanh
from dlrbnicsx.dataset.custom_partitioned_dataset \
    import CustomPartitionedDataset
from dlrbnicsx.interface.wrappers import DataLoader
from dlrbnicsx.train_validate_test.train_validate_test_distributed \
    import train_nn, validate_nn, online_nn, error_analysis

import matplotlib.pyplot as plt
import torch.distributed as dist  # TODO

# Mesh deformation class (from MDFEniCSx)


class CustomMeshDeformation(HarmonicMeshMotion):
    def __init__(self, mesh, boundaries, bc_markers_list, bc_function_list,
                 mu, reset_reference=True, is_deformation=True):
        super().__init__(mesh, boundaries, bc_markers_list,
                         bc_function_list, reset_reference, is_deformation)
        self.mu = mu

    def __enter__(self):
        gdim = self._mesh.geometry.dim
        mu = self.mu
        # Compute shape parametrization
        self.shape_parametrization = self.solve()
        self._mesh.geometry.x[:, 0] \
            += (mu[2] - 1.) * (self._mesh.geometry.x[:, 0])
        self._mesh.geometry.x[:, :gdim] += \
            self.shape_parametrization.x.array.\
            reshape(self._reference_coordinates.shape[0], gdim)
        self._mesh.geometry.x[:, 0] -= min(self._mesh.geometry.x[:, 0])
        return self


'''

'''


class ProblemOnDeformedDomain(abc.ABC):
    def __init__(self, mesh, subdomains, boundaries, meshDeformationContext):
        self._mesh = mesh
        self.gdim = self._mesh.geometry.dim
        self._subdomains = subdomains
        self._boundaries = boundaries
        self._boundary_markers = [1, 2, 3, 4]
        self._meshDeformationContext = meshDeformationContext
        self._V = dolfinx.fem.FunctionSpace(self._mesh, ("CG", 2))
        self._test = ufl.TestFunction(self._V)
        self._solution = dolfinx.fem.Function(self._V)
        self._x = ufl.SpatialCoordinate(self._mesh)
        self._dirichletFunc = dolfinx.fem.Function(self._V)
        u, v = ufl.TrialFunction(self._V), ufl.TestFunction(self._V)
        self._inner_product = ufl.inner(u, v) * ufl.dx + \
            ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        self._inner_product_action = \
            rbnicsx.backends.bilinear_form_action(self._inner_product,
                                                  part="real")

    @property
    def assemble_bcs(self):
        bcs = list()
        for i in self._boundary_markers:
            dofs = \
                dolfinx.fem.locate_dofs_topological(self._V,
                                                    self.gdim-1,
                                                    self._boundaries.find(i))
            bcs.append(dolfinx.fem.dirichletbc
                       (self._dirichletFunc, dofs))
        return bcs

    @property
    def source_term(self):
        return - ufl.div(ufl.exp(self._dirichletFunc) *
                         ufl.grad(self._dirichletFunc))

    @property
    def residual_term(self):
        return ufl.inner(ufl.exp(self._solution) * ufl.grad(self._solution),
                         ufl.grad(self._test)) * ufl.dx - \
            ufl.inner(self.source_term, self._test) * ufl.dx

    @property
    def set_problem(self):
        problemNonlinear = \
            dolfinx.fem.petsc.NonlinearProblem(self.residual_term,
                                               self._solution,
                                               bcs=self.assemble_bcs)
        return problemNonlinear

    def solve(self, mu):
        self._bcs_geometric = \
            [lambda x: (0.*x[1], mu[0]*np.sin(x[0]*np.pi)),
             lambda x: (0.*x[0], 0.*x[1]),
             lambda x: (0.*x[0], -mu[1]*np.sin(x[0]*np.pi)),
             lambda x: (0.*x[0], 0.*x[1])]
        problemNonlinear = self.set_problem
        solution = dolfinx.fem.Function(self._V)
        with self._meshDeformationContext(self._mesh, self._boundaries,
                                          self._boundary_markers, self._bcs_geometric, mu) as mesh_class:
            # solver = dolfinx.nls.petsc.NewtonSolver(self._mesh.comm,
            #                                         problemNonlinear)
            solver = dolfinx.nls.petsc.NewtonSolver(mesh_class._mesh.comm,
                                                    problemNonlinear)
            solver.convergence_criterion = "incremental"

            solver.rtol = 1e-6
            solver.report = True
            ksp = solver.krylov_solver
            ksp.setFromOptions()
            # dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

            self._dirichletFunc.interpolate(lambda x:
                                            x[1] * np.sin(x[0] * np.pi)
                                            * np.cos(x[1] * np.pi))
            n, converged = solver.solve(self._solution)
            assert (converged)
            solution.x.array[:] = self._solution.x.array.copy()
            print(f"Computed solution array: {solution.x.array}")
            print(f"Number of interations: {n:d}")
            return solution


class PODANNReducedProblem(abc.ABC):
    '''
    # TODO
    # Mesh deformation at reconstruct_solution,
    # compute_norm, project_snapshot (??)
    '''
    """Define a linear projection-based problem, and solve it with KSP."""

    def __init__(self, problem) -> None:
        self._basis_functions = rbnicsx.backends.FunctionsList(problem._V)
        u, v = ufl.TrialFunction(problem._V), ufl.TestFunction(problem._V)
        self._inner_product = ufl.inner(u, v) * ufl.dx +\
            ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        self._inner_product_action = \
            rbnicsx.backends.bilinear_form_action(self._inner_product,
                                                  part="real")
        self.input_scaling_range = [-1., 1.]
        self.output_scaling_range = [-1., 1.]
        self.input_range = \
            np.array([[0.2, -0.2, 1.], [0.3, -0.4, 4.]])
        self.output_range = [-6., 3.]
        self.loss_fn = "MSE"
        self.learning_rate = 1e-5
        self.optimizer = "Adam"
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
mu = np.array([0.3, -0.413, 4.])

problem_parametric = ProblemOnDeformedDomain(mesh, cell_tags,
                                             facet_tags,
                                             CustomMeshDeformation)
solution_mu = problem_parametric.solve(mu)


# POD Starts ###


def generate_training_set(samples=[4, 4, 4]):
    training_set_0 = np.linspace(0.2, 0.3, samples[0])
    training_set_1 = np.linspace(-0.2, -0.4, samples[1])
    training_set_2 = np.linspace(1., 4., samples[2])
    training_set = np.array(list(itertools.product(training_set_0,
                                                   training_set_1,
                                                   training_set_2)))
    return training_set


# Generate samples on rank 0 and Bcast to other processes
if rank == 0:
    training_set = generate_training_set()
else:
    training_set = np.zeros_like(generate_training_set())

world_comm.Bcast(training_set, root=0)

training_set_indices = np.arange(rank, training_set.shape[0], size)

training_set_solutions = \
    np.zeros([training_set.shape[0], solution_mu.x.array.shape[0]])

for mu_index in training_set_indices:
    print(rbnicsx.io.TextLine(str(mu_index+1) + f"/{training_set.shape[0]}",
                              fill="#"))
    print(f"High fidelity solve for mu = {training_set[mu_index,:]}")
    training_set_solutions[mu_index, :] = \
        problem_parametric.solve(training_set[mu_index, :]).x.array


training_set_solutions_recv = np.zeros_like(training_set_solutions)
world_comm.Barrier()
world_comm.Allreduce(training_set_solutions, training_set_solutions_recv,
                     op=MPI.SUM)

Nmax = 30

print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
print("")

print("Set up snapshots matrix")
snapshots_matrix = rbnicsx.backends.FunctionsList(problem_parametric._V)

print("Set up reduced problem")
reduced_problem = PODANNReducedProblem(problem_parametric)

print("")

# TODO why training_set_solutions_recv? use training set.
for (mu_index, mu) in enumerate(training_set_solutions_recv):
    print(rbnicsx.io.TextLine
          (f"{mu_index+1} / {training_set_solutions_recv.shape[0]}",
           fill="#"))
    snapshot = dolfinx.fem.Function(problem_parametric._V)
    snapshot.x.array[:] = training_set_solutions_recv[mu_index, :]

    print("Update snapshots matrix")
    snapshots_matrix.append(snapshot)

    print("")

print(rbnicsx.io.TextLine("perform POD", fill="#"))
eigenvalues, modes, _ = \
    rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_matrix,
                                    reduced_problem._inner_product_action,
                                    N=Nmax, tol=1.e-6)
reduced_problem._basis_functions.extend(modes)
reduced_size = len(reduced_problem._basis_functions)
print("")

print(rbnicsx.io.TextBox("POD-Galerkin offline phase ends", fill="="))

positive_eigenvalues = np.where(eigenvalues > 0., eigenvalues, np.nan)
singular_values = np.sqrt(positive_eigenvalues)

print(f"Rank {rank}, Positive eigenvalues: {positive_eigenvalues}")

if rank == 0:
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


# POD Ends ###

# 5. ANN implementation


def generate_ann_input_set(samples=[4, 4, 4]):
    """Generate an equispaced training set using numpy."""
    training_set_0 = np.linspace(0.2, 0.3, samples[0])
    training_set_1 = np.linspace(-0.2, -0.4, samples[1])
    training_set_2 = np.linspace(1., 4., samples[2])
    training_set = np.array(list(itertools.product(training_set_0,
                                                   training_set_1,
                                                   training_set_2)))
    training_set = training_set.astype("f")
    return training_set


def generate_ann_output_set(problem, reduced_problem, input_set,
                            indices, mode=None):
    # Compute output set for ANN based on input set
    output_set = np.zeros([input_set.shape[0],
                           len(reduced_problem._basis_functions)])
    rb_size = len(reduced_problem._basis_functions)
    for i in indices:
        if mode is None:
            print(f"Parameter number {i+1} of ")
            print(f"{input_set.shape[0]}: {input_set[i, :]}")
        else:
            print(f"{mode} parameter number {i+1} of ")
            print(f"{input_set.shape[0]}: {input_set[i, :]}")
        print(input_set[i, :])
        solution = problem.solve(input_set[i, :])
        print(solution.x.array)
        output_set[i, :] = reduced_problem.project_snapshot(solution,
                                                            rb_size).array
    return output_set


# Generate ANN input TRAINING samples on the rank 0 and Bcast to other processes
if rank == 0:
    ann_input_set = generate_ann_input_set(samples=[6, 6, 7])
    np.random.shuffle(ann_input_set)
else:
    ann_input_set = \
        np.zeros_like(generate_ann_input_set(samples=[6, 6, 7]))

world_comm.Bcast(ann_input_set, root=0)

# Indices for ANN dataset distribution
indices = np.arange(rank, ann_input_set.shape[0], size)

ann_output_set = generate_ann_output_set(problem_parametric, reduced_problem,
                                         ann_input_set, indices, mode="Training")

# Initialise zero matrix for MPI all reduce and collect output set using MPI.SUM
ann_output_set_recv = np.zeros_like(ann_output_set)
world_comm.Barrier()
world_comm.Allreduce(ann_output_set, ann_output_set_recv, op=MPI.SUM)

num_training_samples = int(0.7 * ann_input_set.shape[0])
num_validation_samples = ann_input_set.shape[0] - num_training_samples

input_training_set = ann_input_set[:num_training_samples, :]
output_training_set = ann_output_set_recv[:num_training_samples, :]

input_validation_set = ann_input_set[num_training_samples:, :]
output_validation_set = ann_output_set_recv[num_training_samples:, :]

print("\n")

reduced_problem.output_range[0] = np.min(ann_output_set_recv)
reduced_problem.output_range[1] = np.max(ann_output_set_recv)
# NOTE Output_range based on the computed values instead of user guess.

print("\n")

customDataset = \
    CustomPartitionedDataset(problem_parametric, reduced_problem,
                             len(reduced_problem._basis_functions),
                             input_training_set, output_training_set)
train_dataloader = DataLoader(customDataset, batch_size=30, shuffle=True)

customDataset = \
    CustomPartitionedDataset(problem_parametric, reduced_problem,
                             len(reduced_problem._basis_functions),
                             input_validation_set, output_validation_set)
valid_dataloader = DataLoader(customDataset, shuffle=False)

# ANN model
model = HiddenLayersNet(training_set.shape[1], [30, 30],
                        len(reduced_problem._basis_functions), Tanh())

for param in model.parameters():
    # print(f"Rank {rank} \n Params before all_reduce: {param.data}")
    # NOTE This ensures that models in all processes start with same weights and biases
    dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
    param.data /= dist.get_world_size()
    # print(f"Rank {rank} \n Params after all_reduce: {param.data}")

# Training of ANN
training_loss = list()
validation_loss = list()

max_epochs = 20000
min_validation_loss = None
for epochs in range(max_epochs):
    print(f"Epoch: {epochs+1}/{max_epochs}")
    current_training_loss = train_nn(reduced_problem, train_dataloader,
                                     model)
    training_loss.append(current_training_loss)
    current_validation_loss = validate_nn(reduced_problem, valid_dataloader,
                                          model)
    validation_loss.append(current_validation_loss)
    if epochs > 0 and current_validation_loss > 1.01 * min_validation_loss \
       and reduced_problem.regularisation == "EarlyStopping":
        # 1% safety margin against min_validation_loss
        # before invoking early stopping criteria
        print(f"Early stopping criteria invoked at epoch: {epochs+1}")
        break
    min_validation_loss = min(validation_loss)


# Error analysis dataset
print("\n")
print("Generating error analysis (only input/parameters) dataset")
print("\n")

if rank == 0:
    error_analysis_set = generate_ann_input_set(samples=[3, 3, 3])
else:
    error_analysis_set = \
        np.zeros_like(generate_ann_input_set(samples=[3, 3, 3]))

world_comm.Bcast(error_analysis_set, root=0)

world_comm.Barrier()
error_numpy = np.zeros(error_analysis_set.shape[0])
error_numpy_recv = np.zeros(error_analysis_set.shape[0])
indices = np.arange(rank, error_analysis_set.shape[0], size)

for i in indices:
    print(f"Error analysis parameter number {i+1} of ")
    print(f"{error_analysis_set.shape[0]}: {error_analysis_set[i, :]}")
    error_numpy[i] = error_analysis(reduced_problem, problem_parametric,
                                    error_analysis_set[i, :], model,
                                    len(reduced_problem._basis_functions),
                                    online_nn, device=None)
    print(f"Error: {error_numpy[i]}")

world_comm.Barrier()
world_comm.Allreduce(error_numpy, error_numpy_recv, op=MPI.SUM)

# Online phase at parameter online_mu
if rank == 0:
    online_mu = np.array([0.25, -0.3, 2.5])
    fem_solution = problem_parametric.solve(online_mu)
    rb_solution = \
        reduced_problem.reconstruct_solution(
            online_nn(reduced_problem, problem_parametric, online_mu, model,
                      len(reduced_problem._basis_functions), device=None))

    fem_online_file \
        = "dlrbnicsx_solution_nonlinear_poisson/fem_online_mu_computed.xdmf"
    with CustomMeshDeformation(mesh, facet_tags,
                               problem_parametric._boundary_markers,
                               problem_parametric._bcs_geometric,
                               online_mu, reset_reference=True,
                               is_deformation=True) as mesh_class:
        with dolfinx.io.XDMFFile(mesh.comm, fem_online_file,
                                 "w") as solution_file:
            solution_file.write_mesh(mesh)
            solution_file.write_function(fem_solution)

    rb_online_file \
        = "dlrbnicsx_solution_nonlinear_poisson/rb_online_mu_computed.xdmf"
    with CustomMeshDeformation(mesh, facet_tags,
                               problem_parametric._boundary_markers,
                               problem_parametric._bcs_geometric,
                               online_mu, reset_reference=True,
                               is_deformation=True) as mesh_class:
        with dolfinx.io.XDMFFile(mesh.comm, rb_online_file,
                                 "w") as solution_file:
            # NOTE scatter_forward not considered for online solution
            solution_file.write_mesh(mesh)
            solution_file.write_function(rb_solution)

    error_function = dolfinx.fem.Function(problem_parametric._V)
    error_function.x.array[:] = \
        fem_solution.x.array - rb_solution.x.array
    fem_rb_error_file \
        = "dlrbnicsx_solution_nonlinear_poisson/fem_rb_error_computed.xdmf"
    with CustomMeshDeformation(mesh, facet_tags,
                               problem_parametric._boundary_markers,
                               problem_parametric._bcs_geometric,
                               online_mu, reset_reference=True,
                               is_deformation=True) as mesh_class:
        with dolfinx.io.XDMFFile(mesh.comm, fem_rb_error_file,
                                 "w") as solution_file:
            solution_file.write_mesh(mesh)
            solution_file.write_function(error_function)

    with CustomMeshDeformation(mesh, facet_tags,
                               problem_parametric._boundary_markers,
                               problem_parametric._bcs_geometric,
                               online_mu, reset_reference=True,
                               is_deformation=True) as mesh_class:
        print(reduced_problem.norm_error(fem_solution, rb_solution))
        print(reduced_problem.compute_norm(error_function))

    print(reduced_problem.norm_error(fem_solution, rb_solution))
    print(reduced_problem.compute_norm(error_function))
