import torch.distributed as dist
import matplotlib.pyplot as plt
import abc
import itertools
import typing

import dolfinx.fem
import dolfinx.io
import gmsh
import mpi4py.MPI
import numpy as np
import numpy.typing
import petsc4py.PETSc
import plotly.graph_objects as go
import ufl

import rbnicsx.backends
import rbnicsx.online
import rbnicsx.test

from problems.problem_reference import ProblemBase
from problems.problem_parametrized import HarmonicExtension, ProblemOnDeformedDomain

from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory import Tanh
from dlrbnicsx.dataset.custom_partitioned_dataset import CustomPartitionedDataset
from dlrbnicsx.interface.wrappers import DataLoader
from dlrbnicsx.train_validate_test.train_validate_test_distributed import train_nn, validate_nn, online_nn, error_analysis

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Import mesh in dolfinx
gdim = 2
mesh, subdomains, boundaries = dolfinx.io.gmshio.read_from_msh(
    "mesh_data/domain_poisson.msh", mpi4py.MPI.COMM_SELF, 0, gdim=gdim)

# 4. Proper Orthogonal Decomposition


class PODANNReducedProblem(abc.ABC):
    """Define a linear projection-based problem, and solve it with KSP."""

    def __init__(self, problem: ProblemBase) -> None:
        # Define basis functions storage
        basis_functions = rbnicsx.backends.FunctionsList(problem.function_space)
        self._basis_functions = basis_functions
        # Get trial and test functions from the high fidelity problem
        u, v = problem.trial_and_test
        # Define H^1 inner product
        inner_product = ufl.inner(u, v) * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        self._inner_product = inner_product
        self._inner_product_action = rbnicsx.backends.bilinear_form_action(inner_product, part="real")
        # Store numeric and symbolic parameters from problem
        self._mu_symb = problem.mu_symb
        self._mu = np.zeros(self._mu_symb.value.shape)
        self.input_scaling_range = [-1., 1.]
        self.output_scaling_range = [-1., 1.]
        self.input_range = np.vstack((0.5*np.ones([1, 2]), np.ones([1, 2])))
        self.output_range = [-100., 1500.]
        self.loss_fn = "MSE"
        self.learning_rate = 1e-4
        self.optimizer = "Adam"
        self.regularisation = "EarlyStopping"

    @property
    def basis_functions(self) -> rbnicsx.backends.FunctionsList:
        """Return the basis functions of the reduced problem."""
        return self._basis_functions

    @property
    def inner_product_form(self) -> ufl.Form:  # type: ignore[no-any-unimported]
        """
        Return the bilinear form that defines the inner product associated to this reduced problem.

        This inner product is used to define the notion of orthogonality employed during the offline stage.
        """
        return self._inner_product

    @property
    def inner_product_action(self) -> typing.Callable[  # type: ignore[no-any-unimported]
            [dolfinx.fem.Function], typing.Callable[[dolfinx.fem.Function], petsc4py.PETSc.RealType]]:
        """
        Return the action of the bilinear form that defines the inner product associated to this reduced problem.

        This inner product is used to define the notion of orthogonality employed during the offline stage.
        """
        return self._inner_product_action

    '''@abc.abstractproperty
    def mesh_motion(self) -> rbnicsx.backends.MeshMotion:
        """Return the mesh motion for the parameter of the latest solve."""
        pass

    '''

    def reconstruct_solution(  # type: ignore[no-any-unimported]
            self, reduced_solution: petsc4py.PETSc.Vec) -> dolfinx.fem.Function:
        """Reconstructed reduced solution on the high fidelity space."""
        return self.basis_functions[:reduced_solution.size] * reduced_solution  # type: ignore[no-any-return]

    def compute_norm(  # type: ignore[no-any-unimported]
            self, function: dolfinx.fem.Function) -> petsc4py.PETSc.RealType:
        """Compute the norm of a function using the H^1 inner product on the reference domain."""
        return np.sqrt(self._inner_product_action(function)(function))

    def project_snapshot(self, solution: dolfinx.fem.Function, N: int) -> petsc4py.PETSc.Vec:
        return self._project_snapshot(solution, N)

    def _project_snapshot(self, solution, N) -> petsc4py.PETSc.Vec:  # TODO
        projected_snapshot = rbnicsx.online.create_vector(N)
        A = rbnicsx.backends.project_matrix(self._inner_product_action, self._basis_functions[:N])
        F = rbnicsx.backends.project_vector(self._inner_product_action(solution), self._basis_functions[:N])
        ksp = petsc4py.PETSc.KSP()
        ksp.create(projected_snapshot.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
        ksp.solve(F, projected_snapshot)
        return projected_snapshot

    def norm_error(self, u, v):
        return self.compute_norm(u-v)/self.compute_norm(u)


def generate_training_set() -> np.typing.NDArray[np.float64]:
    """Generate an equispaced training set using numpy."""
    training_set_0 = np.linspace(0.5, 1., 4)  # arguments: min_parameter, max_parameter, number of smaples
    training_set_1 = np.linspace(0.5, 1., 4)
    training_set = np.array(list(itertools.product(training_set_0, training_set_1)))
    return training_set


def generate_ann_input_set(samples=[4, 4]):
    """Generate an equispaced training set using numpy."""
    training_set_0 = np.linspace(0.5, 1., samples[0])  # arguments: min_parameter, max_parameter, number of smaples
    training_set_1 = np.linspace(0.5, 1., samples[1])
    training_set = np.array(list(itertools.product(training_set_0, training_set_1))).astype("f")
    return training_set


def generate_ann_output_set(problem, reduced_problem, N, input_set, indices, mode=None):
    output_set = np.zeros([input_set.shape[0], N])
    for i in indices:
        if mode == None:
            print(f"Parameter number {i+1} of {input_set.shape[0]}: {input_set[i,:]}")
        else:
            print(f"{mode} parameter number {i+1} of {input_set.shape[0]}: {input_set[i,:]}")
        output_set[i, :] = reduced_problem.project_snapshot(problem.solve(input_set[i, :]), N).array.astype("f")
    return output_set


problem = ProblemOnDeformedDomain(mesh, subdomains, boundaries)

training_set = generate_training_set()  # rbnicsx.io.on_rank_zero(mesh.comm, generate_training_set)

training_set_indices = np.arange(rank, training_set.shape[0], size)

print(f"Rank {rank}, indices: {training_set_indices}")

print(rank, problem.solve(training_set[0, :]).x.array.shape)

solution = dolfinx.fem.Function(problem.function_space)

training_set_solutions = np.zeros([training_set.shape[0], solution.x.array.shape[0]])

for mu_index in training_set_indices:
    print(rbnicsx.io.TextLine(str(mu_index+1), fill="#"))

    print(f"Parameter number {(mu_index+1)} of {training_set.shape[0]}")
    print(f"High fidelity solve for mu = {training_set[mu_index,:]}")
    training_set_solutions[mu_index, :] = problem.solve(training_set[mu_index, :]).x.array

training_set_solutions_recv = np.zeros_like(training_set_solutions)  # TODO MPI Allgather
comm.Barrier()
comm.Allreduce(training_set_solutions, training_set_solutions_recv, op=mpi4py.MPI.SUM)
print(np.max(abs(training_set_solutions_recv[training_set_indices, :] -
      training_set_solutions[training_set_indices, :])))

Nmax = 10

print(rbnicsx.io.TextBox(f"POD offline phase begins", fill="="))
print(f"")

print(f"set up snapshots matrix")
snapshots_matrix = rbnicsx.backends.FunctionsList(problem.function_space)

print(f"set up reduced problem")
reduced_problem = PODANNReducedProblem(problem)

print(f"")

for (mu_index, mu) in enumerate(training_set_solutions_recv):
    print(rbnicsx.io.TextLine(f"{mu_index+1}", fill="#"))
    print(f"Parameter number {mu_index+1} of {training_set_solutions_recv.shape[0]}")
    print(f"high fidelity solve for mu = {mu}")
    snapshot = dolfinx.fem.Function(problem.function_space)
    snapshot.x.array[:] = training_set_solutions_recv[mu_index, :]
    print(np.max(abs(snapshot.x.array-training_set_solutions_recv[mu_index, :])))

    print(f"update snapshots matrix")
    snapshots_matrix.append(snapshot)

    print(f"")

print(rbnicsx.io.TextLine(f"perform POD", fill="#"))
eigenvalues, modes, _ = rbnicsx.backends.proper_orthogonal_decomposition(
    snapshots_matrix, reduced_problem.inner_product_action, N=Nmax, tol=0)
reduced_problem.basis_functions.extend(modes)
print(f"")

print(rbnicsx.io.TextBox(f"POD-Galerkin offline phase ends", fill="="))

positive_eigenvalues = np.where(eigenvalues > 0., eigenvalues, np.nan)
singular_values = np.sqrt(positive_eigenvalues)


plt.figure(figsize=[8, 10])
xint = list()
yval = list()

for x, y in enumerate(eigenvalues):
    yval.append(y)
    xint.append(x+1)

plt.plot(xint, yval, "*-", color="orange")
plt.xlabel("Eigenvalue number", fontsize=18)
plt.ylabel("Eigenvalue", fontsize=18)
plt.xticks(xint)
plt.yscale("log")
plt.title("Eigenvalue decay", fontsize=24)
plt.tight_layout()
# plt.show()

# 5. ANN implementation

input_training_set_filepath = "ann_data/input_training_data.npy"
input_validation_set_filepath = "ann_data/input_validation_data.npy"
output_training_set_filepath = "ann_data/output_training_data.npy"
output_validation_set_filepath = "ann_data/output_validation_data.npy"
input_error_analysis_set_filepath = "ann_data/input_error_analysis_data.npy"

# Training dataset
input_training_data = generate_ann_input_set(samples=[10, 10])
indices = np.arange(rank, input_training_data.shape[0], size)
output_training_data = generate_ann_output_set(
    problem, reduced_problem, Nmax, input_training_data, indices, mode="Training")
# print(output_training_data)
output_training_data_recv = np.zeros_like(output_training_data)
comm.Barrier()
comm.Allreduce(output_training_data, output_training_data_recv, op=mpi4py.MPI.SUM)

if rank == 0:
    np.save(input_training_set_filepath, input_training_data)
    np.save(output_training_set_filepath, output_training_data_recv)
comm.Barrier()

print("\n")

reduced_problem.output_range[0], reduced_problem.output_range[1] = np.min(output_training_data_recv), np.max(
    output_training_data_recv)  # NOTE Updating output_range based on the computed values instead of user guess.

print("\n")

customDataset = CustomPartitionedDataset(problem, reduced_problem, Nmax,
                                         input_training_set_filepath, output_training_set_filepath)
train_dataloader = DataLoader(customDataset, batch_size=15, shuffle=True)

# Validation dataset
input_validation_data = generate_ann_input_set(samples=[7, 7])
indices = np.arange(rank, input_validation_data.shape[0], size)
output_validation_data = generate_ann_output_set(
    problem, reduced_problem, Nmax, input_validation_data, indices, mode="Validation")

output_validation_data_recv = np.zeros_like(output_validation_data)
comm.Barrier()
comm.Allreduce(output_validation_data, output_validation_data_recv, op=mpi4py.MPI.SUM)

if rank == 0:
    np.save(input_validation_set_filepath, input_validation_data)
    np.save(output_validation_set_filepath, output_validation_data_recv)
comm.Barrier()

customDataset = CustomPartitionedDataset(problem, reduced_problem, Nmax,
                                         input_validation_set_filepath, output_validation_set_filepath)
valid_dataloader = DataLoader(customDataset, batch_size=20, shuffle=False)

# TODO replace Nmax with with something like reduced_problem.N
model = HiddenLayersNet(training_set.shape[1], [4], Nmax, Tanh())


for param in model.parameters():
    print(f"Rank {rank} \n Params before all_reduce: {param.data}")
    # NOTE This ensures that models in all processes start with same weights and biases
    dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
    param.data /= dist.get_world_size()
    print(f"Rank {rank} \n Params after all_reduce: {param.data}")

training_loss = list()
validation_loss = list()

# train_loss = train_nn(reduced_problem, train_dataloader, model)
# valid_loss = validate_nn(reduced_problem, valid_dataloader, model)

max_epochs = 20000
for epochs in range(max_epochs):
    print(f"Epoch: {epochs+1}/{max_epochs}")
    current_training_loss = train_nn(reduced_problem, train_dataloader, model)
    training_loss.append(current_training_loss)
    current_validation_loss = validate_nn(reduced_problem, valid_dataloader, model)
    validation_loss.append(current_validation_loss)
    # 1% safety margin against min_validation_loss before invoking eraly stopping criteria
    if epochs > 0 and current_validation_loss > 1.01 * min_validation_loss and reduced_problem.regularisation == "EarlyStopping":
        print(f"Early stopping criteria invoked at epoch: {epochs+1}")
        break
    min_validation_loss = min(validation_loss)

# Error analysis dataset

print("\n")
print("Generating error analysis (only input/parameters) dataset")
print("\n")
error_analysis_set = generate_ann_input_set(samples=[5, 5])
if rank == 0:
    np.save(input_error_analysis_set_filepath, error_analysis_set)
comm.Barrier()
error_numpy = np.zeros(error_analysis_set.shape[0])
error_numpy_recv = np.zeros(error_analysis_set.shape[0])
indices = np.arange(rank, error_analysis_set.shape[0], size)

for i in indices:
    print(f"Error analysis parameter number {i+1} of {error_analysis_set.shape[0]}: {error_analysis_set[i,:]}")
    error_numpy[i] = error_analysis(reduced_problem, problem, error_analysis_set[i, :],
                                    model, Nmax, online_nn, device=None)
    print(f"Error: {error_numpy[i]}")

comm.Barrier()
comm.Allreduce(error_numpy, error_numpy_recv, op=mpi4py.MPI.SUM)

# Online solve
if rank == 0:
    online_mu = np.array([0.8, 0.9])
    print("Solving FEM")
    fem_solution = problem.solve(online_mu)
    print("Solved FEM, solving RB")
    rb_solution = reduced_problem.reconstruct_solution(
        online_nn(reduced_problem, problem, online_mu, model, Nmax, device=None))
    print("Solved RB, saving FEM")

    with problem.mesh_motion:
        with dolfinx.io.XDMFFile(mpi4py.MPI.COMM_SELF, "solution_poisson/fem_solution_online_mu.xdmf", "w") as fem_solution_file_xdmf:
            fem_solution.x.scatter_forward()
            fem_solution_file_xdmf.write_mesh(mesh)
            fem_solution_file_xdmf.write_function(fem_solution)

    print("Saved FEM, saving RB")

    with problem.mesh_motion:
        with dolfinx.io.XDMFFile(mpi4py.MPI.COMM_SELF, "solution_poisson/rb_solution_online_mu.xdmf", "w") as rb_solution_file_xdmf:
            rb_solution.x.scatter_forward()
            rb_solution_file_xdmf.write_mesh(mesh)
            rb_solution_file_xdmf.write_function(rb_solution)
