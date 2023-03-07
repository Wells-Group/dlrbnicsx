import dolfinx
import ufl

import rbnicsx
import rbnicsx.online
import rbnicsx.backends

from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import itertools
import abc
import matplotlib.pyplot as plt

from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory \
    import Tanh
from dlrbnicsx.dataset.custom_partitioned_dataset \
    import CustomPartitionedDataset
from dlrbnicsx.interface.wrappers import DataLoader
from dlrbnicsx.train_validate_test.train_validate_test_distributed \
    import train_nn, validate_nn, online_nn, error_analysis
import torch.distributed as dist  # TODO


class ProblemOnDeformedDomain(abc.ABC):
    # Define FEM problem on the reference problem
    def __init__(self, mesh, subdomains, boundaries, HarmonicMeshMotion):
        self._mesh = mesh
        self.gdim = self._mesh.geometry.dim
        self._subdomains = subdomains
        self._boundaries = boundaries
        self._boundary_markers = [1, 2, 3, 4, 5]
        self.meshDeformationContext = HarmonicMeshMotion
        # Define function space
        self._V = dolfinx.fem.FunctionSpace(mesh, ("CG", 2))
        # Define trialfunction and testfunction
        u, v = ufl.TrialFunction(self._V), ufl.TestFunction(self._V)
        self._trial, self._test = u, v
        self._inner_product = ufl.inner(u, v) * ufl.dx + \
            ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        self._inner_product_action = \
            rbnicsx.backends.bilinear_form_action(self._inner_product,
                                                  part="real")
        # Define Dirichlet BC function
        self._dirichletFunc = dolfinx.fem.Function(self._V)

    @property
    def source_term(self):
        x = ufl.SpatialCoordinate(self._mesh)
        u_ufl = 1 + x[0]**2 + 2*x[1]**2
        return -ufl.div(ufl.grad(u_ufl))

    @property
    def bilinear_form(self):
        u = self._trial
        v = self._test
        return dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)

    @property
    def linear_form(self):
        f = self.source_term
        v = self._test
        return dolfinx.fem.form(ufl.inner(f, v) * ufl.dx)

    def solve(self, mu):
        # Solve the problem at given parameter mu
        self._bcs_geometric = \
            [lambda x: (mu[0]*x[0], mu[1]*x[1]),
             lambda x: (mu[0]*x[0], mu[1]*x[1]),
             lambda x: (mu[0]*x[0], mu[1]*x[1]),
             lambda x: (mu[0]*x[0], mu[1]*x[1]),
             lambda x: (mu[0]*x[0], mu[1]*x[1])]
        with HarmonicMeshMotion(self._mesh, self._boundaries,
                                self._boundary_markers,
                                self._bcs_geometric,
                                reset_reference=True):
            # Assemble BCs on deformed mesh
            bcs = list()
            for i in self._boundary_markers:
                dofs = \
                    dolfinx.fem.locate_dofs_topological(self._V, self.gdim-1,
                                                        self._boundaries.find(i))
                self._dirichletFunc.interpolate(lambda x: 1 +
                                                x[0]**2 + 2*x[1]**2)
                bc = dolfinx.fem.dirichletbc(self._dirichletFunc, dofs)
                bcs.append(bc)

            # Bilinear side assembly
            a_cpp = self.bilinear_form
            l_cpp = self.linear_form
            A = dolfinx.fem.petsc.assemble_matrix(a_cpp, bcs=bcs)
            A.assemble()

            # Linear side assembly
            L = dolfinx.fem.petsc.assemble_vector(l_cpp)
            dolfinx.fem.petsc.apply_lifting(L, [a_cpp], [bcs])
            L.ghostUpdate(addv=PETSc.InsertMode.ADD,
                          mode=PETSc.ScatterMode.REVERSE)
            dolfinx.fem.petsc.set_bc(L, bcs)

            # Solver setup
            ksp = PETSc.KSP()
            ksp.create(mesh.comm)
            ksp.setOperators(A)
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType("mumps")
            ksp.setFromOptions()
            solution = dolfinx.fem.Function(self._V)
            ksp.solve(L, solution.vector)
            solution.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                        mode=PETSc.ScatterMode.FORWARD)
            return solution


class PODANNReducedProblem(abc.ABC):
    # Define Reduced problem class
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
            np.array([[0.8, 0.8], [1.1, 1.2]])
        self.output_range = [None, None]
        self.loss_fn = "MSE"
        self.learning_rate = 1e-4
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
        # Project FEM solution on RB space
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
        # Relative error norm
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
mu = np.array([0.8, 1.1])

# FEM solve
problem_parametric = ProblemOnDeformedDomain(mesh, cell_tags, facet_tags,
                                             HarmonicMeshMotion)
solution_mu = problem_parametric.solve(mu)

# ### POD starts ###


def generate_training_set(samples=[8, 8]):
    # Select input samples for POD
    training_set_0 = np.linspace(0.5, 1., samples[0])
    training_set_1 = np.linspace(0.5, 1., samples[1])
    training_set = \
        np.array(list(itertools.product(training_set_0, training_set_1)))
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

Nmax = 10

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

print(rbnicsx.io.TextLine("Perform POD", fill="#"))
eigenvalues, modes, _ = \
    rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_matrix,
                                    reduced_problem._inner_product_action,
                                    N=Nmax, tol=1e-10)
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

# ### POD ends ###

# ### ANN implementation ###


def generate_ann_input_set(samples=[4, 4]):
    # Select samples from the parameter space for ANN
    training_set_0 = np.linspace(0.5, 1.5, samples[0])
    training_set_1 = np.linspace(0.5, 1.5, samples[1])
    training_set = np.array(list(itertools.product(training_set_0,
                                                   training_set_1)))
    training_set = training_set.astype("f")
    return training_set


def generate_ann_output_set(problem, reduced_problem, N,
                            input_set, indices, mode=None):
    # Solve the FE problem at given input_sets and
    # project on the RB space
    output_set = np.zeros([input_set.shape[0], N])
    for i in indices:
        if mode is None:
            print(f"Parameter number {i+1} of {input_set.shape[0]}")
            print(f"Parameter: {input_set[i,:]}")
        else:
            print(f"{mode} parameter number {i+1} of {input_set.shape[0]}")
            print(f"Parameter: {input_set[i,:]}")
        output_set[i, :] = \
            reduced_problem.project_snapshot(problem.solve(input_set[i, :]),
                                             N).array.astype("f")
    return output_set


# Generate ANN input TRAINING samples on the rank 0 and Bcast to other processes
if rank == 0:
    input_training_set = generate_ann_input_set(samples=[8, 8])
else:
    input_training_set = \
        np.zeros_like(generate_ann_input_set(samples=[8, 8]))

world_comm.Bcast(input_training_set, root=0)

indices = np.arange(rank, input_training_set.shape[0], size)

# Generate ANN output samples

# ### The input data is available in all processes but output data uses
# only a chunk of the data as specified in the indices ###
output_training_set = \
    generate_ann_output_set(problem_parametric, reduced_problem,
                            len(reduced_problem._basis_functions),
                            input_training_set, indices, mode="Training")


output_training_set_recv = np.zeros_like(output_training_set)
world_comm.Barrier()
world_comm.Allreduce(output_training_set, output_training_set_recv, op=MPI.SUM)

print("\n")

reduced_problem.output_range[0] = np.min(output_training_set_recv)
reduced_problem.output_range[1] = np.max(output_training_set_recv)
# NOTE Output_range based on the computed values instead of user guess.

print("\n")

customDataset = CustomPartitionedDataset(problem_parametric, reduced_problem, reduced_size,
                                         input_training_set, output_training_set_recv)
train_dataloader = DataLoader(customDataset, batch_size=15, shuffle=True)

# Generate ANN input VALIDATION samples on the rank 0 and Bcast to other processes
if rank == 0:
    input_validation_set = generate_ann_input_set(samples=[4, 4])
else:
    input_validation_set = \
        np.zeros_like(generate_ann_input_set(samples=[4, 4]))

world_comm.Bcast(input_validation_set, root=0)

indices = np.arange(rank, input_validation_set.shape[0], size)

output_validation_set = \
    generate_ann_output_set(problem_parametric, reduced_problem,
                            len(reduced_problem._basis_functions),
                            input_validation_set, indices, mode="Training")


output_validation_set_recv = np.zeros_like(output_validation_set)
world_comm.Barrier()
world_comm.Allreduce(output_validation_set, output_validation_set_recv, op=MPI.SUM)

customDataset = CustomPartitionedDataset(problem_parametric, reduced_problem, reduced_size,
                                         input_validation_set, output_validation_set_recv)
valid_dataloader = DataLoader(customDataset, shuffle=False)

# ANN model
model = HiddenLayersNet(training_set.shape[1], [4],
                        len(reduced_problem._basis_functions), Tanh())


for param in model.parameters():
    print(f"Rank {rank} \n Params before all_reduce: {param.data}")
    # NOTE This ensures that models in all processes start with same weights and biases
    dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
    param.data /= dist.get_world_size()
    print(f"Rank {rank} \n Params after all_reduce: {param.data}")

training_loss = list()
validation_loss = list()

max_epochs = 20000
min_validation_loss = None
for epochs in range(max_epochs):
    print(f"Epoch: {epochs+1}/{max_epochs}")
    current_training_loss = train_nn(reduced_problem, train_dataloader, model)
    training_loss.append(current_training_loss)
    current_validation_loss = validate_nn(reduced_problem, valid_dataloader, model)
    validation_loss.append(current_validation_loss)
    # 1% safety margin against min_validation_loss before invoking
    # early stopping criteria
    if epochs > 0 and current_validation_loss > 1.01 * min_validation_loss \
       and reduced_problem.regularisation == "EarlyStopping":
        print(f"Early stopping criteria invoked at epoch: {epochs+1}")
        break
    min_validation_loss = min(validation_loss)


# Error analysis dataset

print("\n")
print("Generating error analysis (only input/parameters) dataset")
print("\n")
# Generate error analysis set on rank 0 and Bcast to other processes
if rank == 0:
    error_analysis_set = generate_ann_input_set(samples=[5, 5])
else:
    error_analysis_set = np.zeros_like(generate_ann_input_set(samples=[5, 5]))

world_comm.Bcast(error_analysis_set, root=0)

world_comm.Barrier()
error_numpy = np.zeros(error_analysis_set.shape[0])
error_numpy_recv = np.zeros(error_analysis_set.shape[0])
indices = np.arange(rank, error_analysis_set.shape[0], size)

for i in indices:
    print(f"Error analysis parameter number {i+1} of {error_analysis_set.shape[0]}: {error_analysis_set[i,:]}")
    error_numpy[i] = error_analysis(reduced_problem, problem_parametric, error_analysis_set[i, :],
                                    model, reduced_size, online_nn, device=None)
    print(f"Error: {error_numpy[i]}")

world_comm.Barrier()
world_comm.Allreduce(error_numpy, error_numpy_recv, op=MPI.SUM)

# ### Online phase ###

if rank == 0:
    # Online phase at parameter online_mu
    online_mu = np.array([0.7, 1.])
    fem_solution = problem_parametric.solve(online_mu)
    # First compute the RB solution using online_nn.
    # Next this solution is reconstructed on FE space
    rb_solution = \
        reduced_problem.reconstruct_solution(
            online_nn(reduced_problem, problem_parametric, online_mu, model,
                      reduced_size, device=None))

    # Post processing
    fem_online_file \
        = "dlrbnicsx_solution_linear_poisson/fem_online_mu_computed.xdmf"
    with HarmonicMeshMotion(mesh, facet_tags,
                            problem_parametric._boundary_markers,
                            problem_parametric._bcs_geometric,
                            reset_reference=True) as mesh_class:
        with dolfinx.io.XDMFFile(mesh.comm, fem_online_file,
                                 "w") as solution_file:
            solution_file.write_mesh(mesh)
            solution_file.write_function(fem_solution)

    rb_online_file \
        = "dlrbnicsx_solution_linear_poisson/rb_online_mu_computed.xdmf"
    with HarmonicMeshMotion(mesh, facet_tags,
                            problem_parametric._boundary_markers,
                            problem_parametric._bcs_geometric,
                            reset_reference=True) as mesh_class:
        with dolfinx.io.XDMFFile(mesh.comm, rb_online_file,
                                 "w") as solution_file:
            # NOTE scatter_forward not considered for online solution
            solution_file.write_mesh(mesh)
            solution_file.write_function(rb_solution)

    error_function = dolfinx.fem.Function(problem_parametric._V)
    error_function.x.array[:] = \
        abs(fem_solution.x.array - rb_solution.x.array)
    fem_rb_error_file \
        = "dlrbnicsx_solution_linear_poisson/fem_rb_error_computed.xdmf"
    with HarmonicMeshMotion(mesh, facet_tags,
                            problem_parametric._boundary_markers,
                            problem_parametric._bcs_geometric,
                            reset_reference=True) as mesh_class:
        with dolfinx.io.XDMFFile(mesh.comm, fem_rb_error_file,
                                 "w") as solution_file:
            solution_file.write_mesh(mesh)
            solution_file.write_function(error_function)
