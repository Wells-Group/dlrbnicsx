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
from dlrbnicsx.dataset.custom_dataset import CustomDataset
from dlrbnicsx.interface.wrappers import DataLoader
from dlrbnicsx.train_validate_test.train_validate_test import train_nn, validate_nn, online_nn, error_analysis

# Import mesh in dolfinx
gdim = 2
mesh, subdomains, boundaries = dolfinx.io.gmshio.read_from_msh(
    "mesh_data/domain_poisson.msh", mpi4py.MPI.COMM_WORLD, 0, gdim=gdim)

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
    training_set_0 = np.linspace(0.5, 1., 6)  # arguments: min_parameter, max_parameter, number of smaples
    training_set_1 = np.linspace(0.5, 1., 6)
    training_set = np.array(list(itertools.product(training_set_0, training_set_1)))
    return training_set


def generate_ann_input_set(filepath, samples=[4, 4]):
    """Generate an equispaced training set using numpy."""
    training_set_0 = np.linspace(0.5, 1., samples[0])  # arguments: min_parameter, max_parameter, number of smaples
    training_set_1 = np.linspace(0.5, 1., samples[1])
    training_set = np.array(list(itertools.product(training_set_0, training_set_1))).astype("f")
    np.save(filepath, training_set)


def generate_ann_output_set(problem, reduced_problem, N, input_file_path, output_file_path, mode=None):
    input_set = np.load(input_file_path)
    output_set = np.empty([input_set.shape[0], N])
    for i in range(input_set.shape[0]):
        if mode == None:
            print(f"Parameter number {i+1} of {input_set.shape[0]}: {input_set[i,:]}")
        else:
            print(f"{mode} parameter number {i+1} of {input_set.shape[0]}: {input_set[i,:]}")
        output_set[i, :] = reduced_problem.project_snapshot(problem.solve(input_set[i, :]), N).array.astype("f")
    np.save(output_file_path, output_set)


problem = ProblemOnDeformedDomain(mesh, subdomains, boundaries)

training_set = rbnicsx.io.on_rank_zero(mesh.comm, generate_training_set)

Nmax = 10

print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
print("")

print("set up snapshots matrix")
snapshots_matrix = rbnicsx.backends.FunctionsList(problem.function_space)

print("set up reduced problem")
reduced_problem = PODANNReducedProblem(problem)

print("")

for (mu_index, mu) in enumerate(training_set):
    print(rbnicsx.io.TextLine(str(mu_index+1), fill="#"))

    print("Parameter number ", (mu_index+1), "of", training_set.shape[0])
    print("high fidelity solve for mu =", mu)
    snapshot = problem.solve(mu)

    print("update snapshots matrix")
    snapshots_matrix.append(snapshot)

    print("")

print(rbnicsx.io.TextLine("perform POD", fill="#"))
eigenvalues, modes, _ = rbnicsx.backends.proper_orthogonal_decomposition(
    snapshots_matrix, reduced_problem.inner_product_action, N=Nmax, tol=0)
reduced_problem.basis_functions.extend(modes)
print("")

print(rbnicsx.io.TextBox("POD-Galerkin offline phase ends", fill="="))

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
generate_ann_input_set(input_training_set_filepath, samples=[10, 10])
generate_ann_output_set(problem, reduced_problem, Nmax, input_training_set_filepath,
                        output_training_set_filepath, mode="Training")
customDataset = CustomDataset(problem, reduced_problem, Nmax, input_training_set_filepath, output_training_set_filepath)
train_dataloader = DataLoader(customDataset, batch_size=20, shuffle=True)

print("\n")

reduced_problem.output_range[0], reduced_problem.output_range[1] = np.min(np.load(output_training_set_filepath)), np.max(
    np.load(output_training_set_filepath))  # NOTE Updating output_range based on the computed values instead of user guess.

print("\n")

# Validation dataset
generate_ann_input_set(input_validation_set_filepath, samples=[7, 7])
generate_ann_output_set(problem, reduced_problem, Nmax, input_validation_set_filepath,
                        output_validation_set_filepath, mode="Validation")
customDataset = CustomDataset(problem, reduced_problem, Nmax,
                              input_validation_set_filepath, output_validation_set_filepath)
valid_dataloader = DataLoader(customDataset, batch_size=20, shuffle=False)

# TODO replace Nmax with with something like reduced_problem.N
model = HiddenLayersNet(training_set.shape[1], [4], Nmax, Tanh())

training_loss = list()
validation_loss = list()

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
generate_ann_input_set(input_error_analysis_set_filepath, samples=[5, 5])
error_analysis_set = np.load(input_error_analysis_set_filepath)
error_numpy = np.empty(error_analysis_set.shape[0])

for i in range(error_analysis_set.shape[0]):
    print(f"Error analysis parameter number {i+1} of {error_analysis_set.shape[0]}: {error_analysis_set[i,:]}")
    error_numpy[i] = error_analysis(reduced_problem, problem, error_analysis_set[i, :],
                                    model, Nmax, online_nn, device=None)
    print(f"Error: {error_numpy[i]}")

online_mu = np.array([0.8, 0.9])
fem_solution = problem.solve(online_mu)
rb_solution = reduced_problem.reconstruct_solution(
    online_nn(reduced_problem, problem, online_mu, model, Nmax, device=None))

with problem.mesh_motion:
    with dolfinx.io.XDMFFile(mpi4py.MPI.COMM_WORLD, "solution_poisson/fem_solution_online_mu.xdmf", "w") as fem_solution_file_xdmf:
        fem_solution.x.scatter_forward()
        fem_solution_file_xdmf.write_mesh(mesh)
        fem_solution_file_xdmf.write_function(fem_solution)

with problem.mesh_motion:
    with dolfinx.io.XDMFFile(mpi4py.MPI.COMM_WORLD, "solution_poisson/rb_solution_online_mu.xdmf", "w") as rb_solution_file_xdmf:
        rb_solution.x.scatter_forward()
        rb_solution_file_xdmf.write_mesh(mesh)
        rb_solution_file_xdmf.write_function(rb_solution)
