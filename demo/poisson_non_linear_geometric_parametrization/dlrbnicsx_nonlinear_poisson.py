import dolfinx
import ufl

import rbnicsx
import rbnicsx.backends
import rbnicsx.online

from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import itertools
import abc
import matplotlib.pyplot as plt
import os

from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory import Tanh, Sigmoid
from dlrbnicsx.dataset.custom_dataset import CustomDataset
from dlrbnicsx.interface.wrappers import DataLoader, save_model, load_model, \
    save_checkpoint, load_checkpoint, get_optimiser, get_loss_func
from dlrbnicsx.train_validate_test.train_validate_test import \
    train_nn, validate_nn, online_nn, error_analysis


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
            # print(f"Computed solution array: {solution.x.array}")
            print(f"Number of interations: {n:d}")
            return solution


class PODANNReducedProblem(abc.ABC):
    '''
    TODO
    Mesh deformation at reconstruct_solution (No),
    compute_norm (Yes), project_snapshot (Yes) (??)
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
        """Reconstruction of reduced solution on the high fidelity space."""
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


# Read unit square mesh with Triangular elements
mesh_comm = MPI.COMM_WORLD
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

computed_file = "dlrbnicsx_solution_nonlinear_poisson/solution_computed.xdmf"

with CustomMeshDeformation(mesh, facet_tags,
                           problem_parametric._boundary_markers,
                           problem_parametric._bcs_geometric, mu,
                           reset_reference=True,
                           is_deformation=True) as mesh_class:
    with dolfinx.io.XDMFFile(mesh.comm, computed_file,
                             "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(solution_mu)


# POD Starts ###
def generate_training_set(samples=[3, 3, 3]):#(samples=[4, 4, 4]):
    training_set_0 = np.linspace(0.2, 0.3, samples[0])
    training_set_1 = np.linspace(-0.2, -0.4, samples[1])
    training_set_2 = np.linspace(1., 4., samples[2])
    training_set = np.array(list(itertools.product(training_set_0,
                                                   training_set_1,
                                                   training_set_2)))
    return training_set


training_set = rbnicsx.io.on_rank_zero(mesh.comm, generate_training_set)

Nmax = 30

print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
print("")

print("set up snapshots matrix")
snapshots_matrix = rbnicsx.backends.FunctionsList(problem_parametric._V)

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
# plt.show()

# POD Ends ###

with CustomMeshDeformation(mesh, facet_tags,
                           problem_parametric._boundary_markers,
                           problem_parametric._bcs_geometric, mu,
                           reset_reference=True,
                           is_deformation=True) as mesh_class:
    rb_test_solution = reduced_problem.project_snapshot(solution_mu, len(reduced_problem._basis_functions))
    fem_recreated_solution = reduced_problem.reconstruct_solution(rb_test_solution)
    print(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(fem_recreated_solution, fem_recreated_solution)*ufl.dx)), op=MPI.SUM))

fem_recreated_solution = reduced_problem.reconstruct_solution(rb_test_solution)
print(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(fem_recreated_solution, fem_recreated_solution)*ufl.dx)), op=MPI.SUM))

exit()

# 5. ANN implementation


def generate_ann_input_set(samples=[6, 6, 7]):
    """Generate an equispaced training set using numpy."""
    training_set_0 = np.linspace(0.2, 0.3, samples[0])
    training_set_1 = np.linspace(-0.2, -0.4, samples[1])
    training_set_2 = np.linspace(1., 4., samples[2])
    training_set = np.array(list(itertools.product(training_set_0,
                                                   training_set_1,
                                                   training_set_2)))
    training_set = training_set.astype("f")
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
ann_input_set = generate_ann_input_set(samples=[6, 6, 7])
# np.random.shuffle(ann_input_set)
ann_output_set = generate_ann_output_set(problem_parametric, reduced_problem,
                                         ann_input_set, mode="Training")

num_training_samples = int(0.7 * ann_input_set.shape[0])
num_validation_samples = ann_input_set.shape[0] - num_training_samples

reduced_problem.output_range[0] = np.min(ann_output_set)
reduced_problem.output_range[1] = np.max(ann_output_set)
# NOTE Output_range based on the computed values instead of user guess.

input_training_set = ann_input_set[:num_training_samples, :]
output_training_set = ann_output_set[:num_training_samples, :]

input_validation_set = ann_input_set[num_training_samples:, :]
output_validation_set = ann_output_set[num_training_samples:, :]

customDataset = CustomDataset(reduced_problem,
                              input_training_set, output_training_set)
train_dataloader = DataLoader(customDataset, batch_size=40, shuffle=False)# shuffle=True)

customDataset = CustomDataset(reduced_problem,
                              input_validation_set, output_validation_set)
valid_dataloader = DataLoader(customDataset, shuffle=False)

# ANN model
model = HiddenLayersNet(training_set.shape[1], [35, 35],
                        len(reduced_problem._basis_functions), Tanh())

path = "model.pth"
# save_model(model, path)
load_model(model, path)


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
        save_checkpoint(checkpoint_path, epochs, model, optimiser,
                        min_validation_loss)
    print(f"Epoch: {epochs+1}/{max_epochs}")
    current_training_loss = train_nn(reduced_problem, train_dataloader,
                                     model, loss_fn, optimiser)
    training_loss.append(current_training_loss)
    current_validation_loss = validate_nn(reduced_problem, valid_dataloader,
                                          model, loss_fn)
    validation_loss.append(current_validation_loss)
    if epochs > 0 and current_validation_loss > 1.01 * min_validation_loss \
       and reduced_problem.regularisation == "EarlyStopping":
        # 1% safety margin against min_validation_loss
        # before invoking early stopping criteria
        print(f"Early stopping criteria invoked at epoch: {epochs+1}")
        break
    min_validation_loss = min(validation_loss)
end_time = time.time()
elapsed_time = end_time - start_time

os.system(f"rm {checkpoint_path}")

# Error analysis dataset
print("\n")
print("Generating error analysis (only input/parameters) dataset")
print("\n")
error_analysis_set = generate_ann_input_set(samples=[5, 5, 5])
error_numpy = np.zeros(error_analysis_set.shape[0])

for i in range(error_analysis_set.shape[0]):
    print(f"Error analysis {i+1} of {error_analysis_set.shape[0]}")
    print(f"Parameter: : {error_analysis_set[i,:]}")
    error_numpy[i] = error_analysis(reduced_problem, problem_parametric,
                                    error_analysis_set[i, :], model,
                                    len(reduced_problem._basis_functions),
                                    online_nn)
    print(f"Error: {error_numpy[i]}")

# Online phase at parameter online_mu
online_mu = np.array([0.25, -0.3, 2.5])
fem_solution = problem_parametric.solve(online_mu)
rb_solution = \
    reduced_problem.reconstruct_solution(
        online_nn(reduced_problem, problem_parametric, online_mu, model,
                  len(reduced_problem._basis_functions)))


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

print(f"Training time: {elapsed_time}")
