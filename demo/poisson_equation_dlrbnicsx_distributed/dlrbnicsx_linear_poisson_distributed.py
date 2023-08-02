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
import os

from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory \
    import Tanh
from dlrbnicsx.dataset.custom_partitioned_dataset \
    import CustomPartitionedDataset
from dlrbnicsx.interface.wrappers import DataLoader, save_model, \
    load_model, save_checkpoint, load_checkpoint, model_synchronise, \
    init_cpu_process_group, get_optimiser, get_loss_func
from dlrbnicsx.train_validate_test.train_validate_test_distributed \
    import train_nn, validate_nn, online_nn, error_analysis

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

itemsize = MPI.DOUBLE.Get_size()
para_dim = 2
num_dofs = solution_mu.x.array.shape[0]
num_snapshots = 8 * 8
nbytes_para = itemsize * num_snapshots * para_dim
nbytes_dofs = itemsize * num_snapshots * num_dofs

# ### POD starts ###

def generate_training_set(samples=[8, 8]):
    # Select samples from the parameter space for ANN
    training_set_0 = np.linspace(0.5, 1.5, samples[0])
    training_set_1 = np.linspace(0.5, 1.5, samples[1])
    training_set = np.array(list(itertools.product(training_set_0,
                                                   training_set_1)))
    return training_set

win0 = MPI.Win.Allocate_shared(nbytes_para, itemsize, comm=MPI.COMM_WORLD)
buf0, itemsize = win0.Shared_query(0)
training_set = np.ndarray(buffer=buf0, dtype="d", shape=(num_snapshots, para_dim))

if world_comm.rank == 0:
    training_set[:, :] = generate_training_set()

world_comm.Barrier()

win1 = MPI.Win.Allocate_shared(nbytes_dofs, itemsize, comm=MPI.COMM_WORLD)
buf1, itemsize = win1.Shared_query(0)
training_set_solution = np.ndarray(buffer=buf1, dtype="d", shape=(num_snapshots, num_dofs))

# Solution manifold
indices = np.arange(world_comm.rank, num_snapshots, world_comm.size)

for i in indices:
    print(f"Solving FEM problem {i+1}/{num_snapshots}")
    training_set_solution[i, :] = (problem_parametric.solve(training_set[i, :])).x.array

world_comm.Barrier()

# Maximum RB size
Nmax = 30

print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
print("")

print("Set up snapshots matrix")
snapshots_matrix = rbnicsx.backends.FunctionsList(problem_parametric._V)

for i in range(num_snapshots):
    snapshot = dolfinx.fem.Function(problem_parametric._V)
    snapshot.x.array[:] = training_set_solution[i, :]

    print(f"Update snapshots matrix: {i+1}/{num_snapshots}")
    snapshots_matrix.append(snapshot)

print("Set up reduced problem")
reduced_problem = PODANNReducedProblem(problem_parametric)

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

if world_comm.rank == 0:
    plt.figure(figsize=[8, 10])
    xint = list()
    yval = list()

    for x, y in enumerate(eigenvalues[:len(reduced_problem._basis_functions)]):
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

# ### POD Ends ###

# ### ANN implementation ###
def generate_ann_input_set(samples=[4, 4]):
    # Select samples from the parameter space for ANN
    training_set_0 = np.linspace(0.5, 1.5, samples[0])
    training_set_1 = np.linspace(0.5, 1.5, samples[1])
    training_set = np.array(list(itertools.product(training_set_0,
                                                   training_set_1)))
    training_set = training_set.astype("f")
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

num_ann_input_samples = 10 * 10
num_training_samples = int(0.7 * num_ann_input_samples)
num_validation_samples = num_ann_input_samples - int(0.7 * num_ann_input_samples)
itemsize = MPI.DOUBLE.Get_size()

if world_comm.rank == 0:
    ann_input_set = generate_ann_input_set(samples=[10, 10])
    # np.random.shuffle(ann_input_samples)
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

win4 = MPI.Win.Allocate_shared(nbytes_dofs_ann_training, itemsize,
                               comm=MPI.COMM_WORLD)
buf4, itemsize = win4.Shared_query(0)
output_training_set = \
    np.ndarray(buffer=buf4, dtype="d",
               shape=(num_training_samples,
                      len(reduced_problem._basis_functions)))

win5 = MPI.Win.Allocate_shared(nbytes_dofs_ann_validation, itemsize,
                               comm=MPI.COMM_WORLD)
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

reduced_problem.output_range[0] = min(np.min(output_training_set), np.min(output_validation_set))
reduced_problem.output_range[1] = max(np.max(output_training_set), np.max(output_validation_set))

print("\n")

cpu_group0_procs = world_comm.group.Incl([0])
cpu_group0_comm = world_comm.Create_group(cpu_group0_procs)

if cpu_group0_comm != MPI.COMM_NULL:
    init_cpu_process_group(cpu_group0_comm)

    training_set_indices_cpu = np.arange(cpu_group0_comm.rank,
                                         input_training_set.shape[0],
                                         cpu_group0_comm.size)
    validation_set_indices_cpu = np.arange(cpu_group0_comm.rank,
                                           input_validation_set.shape[0],
                                           cpu_group0_comm.size)

    customDataset = CustomPartitionedDataset(reduced_problem, input_training_set,
                                             output_training_set, training_set_indices_cpu)
    train_dataloader = DataLoader(customDataset, batch_size=10, shuffle=False) # shuffle=True)

    customDataset = CustomPartitionedDataset(reduced_problem, input_validation_set,
                                            output_validation_set, validation_set_indices_cpu)
    valid_dataloader = DataLoader(customDataset, shuffle=False)

    # ANN model
    model = HiddenLayersNet(training_set.shape[1], [4],
                            len(reduced_problem._basis_functions), Tanh())

    # path = "model.pth"
    # save_model(model, path)
    # load_model(model, path)

    model_synchronise(model, verbose=True)

    # Training of ANN
    training_loss = list()
    validation_loss = list()

    max_epochs = 100 # 20000
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

    for epochs in range(start_epoch, max_epochs):
        if epochs > 0 and epochs % checkpoint_epoch == 0:
            save_checkpoint(checkpoint_path, epochs, model, optimiser,
                            min_validation_loss)
        print(f"Epoch: {epochs+1}/{max_epochs}")
        current_training_loss = train_nn(reduced_problem,
                                         train_dataloader,
                                         model, loss_fn, optimiser)
        training_loss.append(current_training_loss)
        current_validation_loss = validate_nn(reduced_problem,
                                              valid_dataloader,
                                              model, loss_fn)
        validation_loss.append(current_validation_loss)
        if epochs > 0 and current_validation_loss > min_validation_loss \
        and reduced_problem.regularisation == "EarlyStopping":
            # 1% safety margin against min_validation_loss
            # before invoking early stopping criteria
            print(f"Early stopping criteria invoked at epoch: {epochs+1}")
            break
        min_validation_loss = min(validation_loss)

    os.system(f"rm {checkpoint_path}")
exit()

# Error analysis dataset
print("\n")
print("Generating error analysis (only input/parameters) dataset")
print("\n")

error_analysis_num_para = 5 * 5
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
    error_analysis_set[:, :] = generate_ann_input_set(samples=[5, 5])

world_comm.Barrier()

error_analysis_indices = np.arange(world_comm.rank,
                                   error_analysis_set.shape[0],
                                   world_comm.size)

for i in error_analysis_indices:
    print(f"Error analysis {i+1} of {error_analysis_set.shape[0]}")
    error_numpy[i] = error_analysis(reduced_problem, problem_parametric,
                                    error_analysis_set[i, :], model,
                                    len(reduced_problem._basis_functions),
                                    online_nn, device=None)
    print(f"Error: {error_numpy[i]}")

world_comm.Barrier()

# Online phase at parameter online_mu

if world_comm.rank == 0:
    # Online phase at parameter online_mu
    online_mu = np.array([0.7, 1.])
    fem_solution = problem_parametric.solve(online_mu)
    # First compute the RB solution using online_nn.
    # Next this solution is reconstructed on FE space
    rb_solution = \
        reduced_problem.reconstruct_solution(
            online_nn(reduced_problem, problem_parametric, online_mu, model,
                      len(reduced_problem._basis_functions), device=None))


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
