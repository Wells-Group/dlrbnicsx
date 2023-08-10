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

class ProblemOnDeformedDomain(abc.ABC):
    # Define FEM problem on the reference problem
    def __init__(self, mesh, subdomains, boundaries, HarmonicMeshMotion):
        # Mesh, Subdomians and Boundaries, Mesh deformation
        self._mesh = mesh
        self.gdim = self._mesh.geometry.dim
        self._subdomains = subdomains
        self._boundaries = boundaries
        self._boundary_markers = [1, 2, 3, 4, 5, 6]
        self.meshDeformationContext = HarmonicMeshMotion

        # Define function space, Trial and Test Function
        P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        UP = P2 * P1
        self._W = dolfinx.fem.FunctionSpace(self._mesh, UP)
        (u, p) = ufl.TrialFunctions(self._W)
        (v, q) = ufl.TestFunctions(self._W)
        self._trial = (u, p)
        self._test = (v, q)

        # Velocity and Pressure inner product (To be used in POD)
        self._inner_product_u = ufl.inner(u, v) * ufl.dx + \
            ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        self._inner_product_action_u = \
            rbnicsx.backends.bilinear_form_action(self._inner_product_u,
                                                  part="real")

        self._inner_product_p = ufl.inner(p, q) * ufl.dx
        self._inner_product_action_p = \
            rbnicsx.backends.bilinear_form_action(self._inner_product_p,
                                                  part="real")

    @property
    def source_term(self):
        # Zero source term
        # Gravity not included to create symmetric solution field
        V, _ = self._W.sub(0).collapse()
        f = dolfinx.fem.Function(V)
        return f

    @property
    def bilinear_form(self):
        # Bilinear form
        (u, p) = self._trial
        (v, q) = self._test
        return dolfinx.fem.form((ufl.inner(ufl.grad(u), ufl.grad(v)) +
                                 ufl.inner(p, ufl.div(v)) +
                                 ufl.inner(ufl.div(u), q)) * ufl.dx)

    @property
    def linear_form(self):
        # Linear form
        f = self.source_term
        (v, q) = self._test
        return dolfinx.fem.form(ufl.inner(f, v) * ufl.dx)

    def no_slip(self, x):
        return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

    def inlet(self, x):
        return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))

    def free_boundary_p(self, x):
        return np.zeros(x.shape[1],)

    def solve(self, mu, plot=False):
        # Solve the problem at given parameter mu
        V, _ = self._W.sub(0).collapse()
        Q, _ = self._W.sub(1).collapse()
        self._bcs_geometric = \
            [lambda x: (x[0], x[1]),
             lambda x: (x[0], x[1]),
             lambda x: (x[0], x[1]),
             lambda x: (x[0], x[1]),
             lambda x: (mu[0] * x[0], mu[1] * x[1]),
             lambda x: (mu[0] * x[0], mu[1] * x[1])]
        with HarmonicMeshMotion(self._mesh, self._boundaries,
                                self._boundary_markers,
                                self._bcs_geometric,
                                reset_reference=True,
                                is_deformation=False):
            # Assemble BCs on deformed mesh
            bcs = list()

            for i in self._boundary_markers:
                dirichletFunc_u = dolfinx.fem.Function(V)
                dirichletFunc_p = dolfinx.fem.Function(Q)
                if i == 5 or i == 6:
                    dofs = \
                        dolfinx.fem.locate_dofs_topological((self._W.sub(0),
                                                             V), self.gdim-1,
                                                            self._boundaries.find(i))
                    dirichletFunc_u.interpolate(self.no_slip)
                    bc = dolfinx.fem.dirichletbc(dirichletFunc_u, dofs,
                                                 self._W.sub(0))
                elif i == 1 or i == 2 or i == 4:
                    dofs = dolfinx.fem.locate_dofs_topological((self._W.sub(0),
                                                                V), self.gdim-1,
                                                               self._boundaries.find(i))
                    dirichletFunc_u.interpolate(self.inlet)
                    bc = dolfinx.fem.dirichletbc(dirichletFunc_u, dofs,
                                                 self._W.sub(0))
                else:
                    dofs = dolfinx.fem.locate_dofs_topological((self._W.sub(1),
                                                                Q), self.gdim-1,
                                                               self._boundaries.find(i))
                    dirichletFunc_p.interpolate(self.free_boundary_p)
                    bc = dolfinx.fem.dirichletbc(dirichletFunc_p, dofs, self._W.sub(1))
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
            ksp.create(self._mesh.comm)
            ksp.setOperators(A)
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType("mumps")
            ksp.setFromOptions()
            solution = dolfinx.fem.Function(self._W)
            ksp.solve(L, solution.vector)
            solution.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                        mode=PETSc.ScatterMode.FORWARD)
            solution_vel, solution_pre = (solution.sub(0).collapse(),
                                          solution.sub(1).collapse())
            if plot is True:
                with dolfinx.io.XDMFFile(mesh.comm,
                                         "dlrbnicsx_solution/fem_velocity.xdmf",
                                         "w") as solution_file:
                    solution_file.write_mesh(mesh)
                    solution_file.write_function(solution_vel)
                with dolfinx.io.XDMFFile(mesh.comm,
                                         "dlrbnicsx_solution/fem_pressure.xdmf",
                                         "w") as solution_file:
                    solution_file.write_mesh(mesh)
                    solution_file.write_function(solution_pre)
            return solution_vel, solution_pre


class PODANNReducedProblem(abc.ABC):
    # Define Reduced problem class
    def __init__(self, problem) -> None:
        V, _ = problem._W.sub(0).collapse()
        Q, _ = problem._W.sub(1).collapse()
        self._basis_functions_u = rbnicsx.backends.FunctionsList(V)
        self._basis_functions_p = rbnicsx.backends.FunctionsList(Q)
        u, p = ufl.TrialFunction(V), ufl.TrialFunction(Q)
        v, q = ufl.TestFunction(V), ufl.TestFunction(Q)
        self._inner_product_u = ufl.inner(u, v) * ufl.dx + \
            ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        self._inner_product_action_u = \
            rbnicsx.backends.bilinear_form_action(self._inner_product_u,
                                                  part="real")
        self._inner_product_p = ufl.inner(p, q) * ufl.dx
        self._inner_product_action_p = \
            rbnicsx.backends.bilinear_form_action(self._inner_product_p,
                                                  part="real")
        self.input_scaling_range_u = [-1., 1.]
        self.output_scaling_range_u = [-1., 1.]
        self.input_range_u = \
            np.array([[0.5, 0.5], [1.5, 1.5]])
        self.output_range_u = [None, None]
        self.regularisation_u = "EarlyStopping"

        self.input_scaling_range_p = [-1., 1.]
        self.output_scaling_range_p = [-1., 1.]
        self.input_range_p = \
            np.array([[0.5, 0.5], [1.5, 1.5]])
        self.output_range_p = [None, None]
        self.regularisation_p = "EarlyStopping"

    def reconstruct_solution_u(self, reduced_solution):
        """Reconstructed reduced VELOCITY solution on the high fidelity space."""
        return self._basis_functions_u[:reduced_solution.size] * \
            reduced_solution

    def reconstruct_solution_p(self, reduced_solution):
        """Reconstructed reduced PRESSURE solution on the high fidelity space."""
        return self._basis_functions_p[:reduced_solution.size] * \
            reduced_solution

    def compute_norm_u(self, function):
        """Compute the norm of a VELOCITY function inner product
        on the reference domain."""
        return np.sqrt(self._inner_product_action_u(function)(function))

    def compute_norm_p(self, function):
        """Compute the norm of a PRESSURE function inner product
        on the reference domain."""
        return np.sqrt(self._inner_product_action_p(function)(function))

    def project_snapshot_u(self, solution, N):
        # Project VELOCITY FEM solution on RB space
        return self._project_snapshot_u(solution, N)

    def project_snapshot_p(self, solution, N):
        # Project PRESSURE FEM solution on RB space
        return self._project_snapshot_p(solution, N)

    def _project_snapshot_u(self, solution, N):
        projected_snapshot_u = rbnicsx.online.create_vector(N)
        A = rbnicsx.backends.\
            project_matrix(self._inner_product_action_u,
                           self._basis_functions_u[:N])
        F = rbnicsx.backends.\
            project_vector(self._inner_product_action_u(solution),
                           self._basis_functions_u[:N])
        ksp = PETSc.KSP()
        ksp.create(projected_snapshot_u.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
        ksp.solve(F, projected_snapshot_u)
        return projected_snapshot_u

    def _project_snapshot_p(self, solution, N):
        projected_snapshot_p = rbnicsx.online.create_vector(N)
        A = rbnicsx.backends.\
            project_matrix(self._inner_product_action_p,
                           self._basis_functions_p[:N])
        F = rbnicsx.backends.\
            project_vector(self._inner_product_action_p(solution),
                           self._basis_functions_p[:N])
        ksp = PETSc.KSP()
        ksp.create(projected_snapshot_p.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
        ksp.solve(F, projected_snapshot_p)
        return projected_snapshot_p

    def norm_error_u(self, u, v):
        # Relative error norm for VELOCITY
        return self.compute_norm_u(u-v)/self.compute_norm_u(u)

    def norm_error_p(self, p, q):
        # Relative error norm for PRESSURE
        return self.compute_norm_p(p-q)/self.compute_norm_p(p)


# Read mesh
comm = MPI.COMM_WORLD
gdim = 2
gmsh_model_rank = 0
mesh, cell_tags, facet_tags = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh", comm,
                                    gmsh_model_rank, gdim=gdim)

# Mesh deformation parameters
mu = np.array([0.93, 1.03])
pod_samples = [3, 3]
ann_samples = [3, 4]
error_analysis_samples = [4, 3]

# FEM solve
problem_parametric = ProblemOnDeformedDomain(mesh, cell_tags, facet_tags,
                                             HarmonicMeshMotion)
solution_vel_mu, solution_pre_mu = problem_parametric.solve(mu)

computed_file_velocity = "dlrbnicsx_solution_stokes_equation/solution_computed_velocity.xdmf"
computed_file_pressure = "dlrbnicsx_solution_stokes_equation/solution_computed_pressure.xdmf"

with HarmonicMeshMotion(mesh, facet_tags,
                        problem_parametric._boundary_markers,
                        problem_parametric._bcs_geometric,
                        reset_reference=True,
                        is_deformation=True) as mesh_class:
    with dolfinx.io.XDMFFile(mesh.comm, computed_file_velocity,
                             "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(solution_vel_mu)

with HarmonicMeshMotion(mesh, facet_tags,
                        problem_parametric._boundary_markers,
                        problem_parametric._bcs_geometric,
                        reset_reference=True,
                        is_deformation=True) as mesh_class:
    with dolfinx.io.XDMFFile(mesh.comm, computed_file_pressure,
                             "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(solution_pre_mu)

# POD Starts ###


def generate_training_set(samples=pod_samples):
    # Select input samples for POD
    training_set_0 = np.linspace(0.5, 1., samples[0])
    training_set_1 = np.linspace(0.5, 1., samples[1])
    training_set = np.array(list(itertools.product(training_set_0,
                                                   training_set_1)))
    return training_set


# POD samples
training_set = rbnicsx.io.on_rank_zero(mesh.comm, generate_training_set)

# Maximum RB size
Nmax = 30

print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
print("")

print("Set up snapshots matrix")
V, _ = problem_parametric._W.sub(0).collapse()
Q, _ = problem_parametric._W.sub(1).collapse()
snapshots_matrix_u = rbnicsx.backends.FunctionsList(V)
snapshots_matrix_p = rbnicsx.backends.FunctionsList(Q)

print("Set up reduced problem")
reduced_problem = PODANNReducedProblem(problem_parametric)

print("")

for (mu_index, mu) in enumerate(training_set):
    print(rbnicsx.io.TextLine(str(mu_index+1), fill="#"))

    print("Parameter number ", (mu_index+1), "of", training_set.shape[0])
    print("High fidelity solve for mu =", mu)
    (snapshot_u, snapshot_p) = problem_parametric.solve(mu)
    print(f"Velocity solution array: {snapshot_u.x.array}")
    print(f"Pressure solution array: {snapshot_p.x.array}")

    print("Update snapshots matrix")
    snapshots_matrix_u.append(snapshot_u)
    snapshots_matrix_p.append(snapshot_p)

    print("")

print(rbnicsx.io.TextLine("Perform POD", fill="#"))
eigenvalues_u, modes_u, _ = rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_matrix_u,
                                    reduced_problem._inner_product_action_u,
                                    N=Nmax, tol=1e-4)
eigenvalues_p, modes_p, _ = rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_matrix_p,
                                    reduced_problem._inner_product_action_p,
                                    N=Nmax, tol=1e-4)

reduced_problem._basis_functions_u.extend(modes_u)
reduced_problem._basis_functions_p.extend(modes_p)
print("")

print(rbnicsx.io.TextBox("POD-Galerkin offline phase ends", fill="="))

positive_eigenvalues_u = np.where(eigenvalues_u > 0., eigenvalues_u, np.nan)
singular_values_u = np.sqrt(positive_eigenvalues_u)

plt.figure(figsize=[8, 10])
xint = list()
yval = list()

for x, y in enumerate(eigenvalues_u[:len(reduced_problem._basis_functions_u)]):
    yval.append(y)
    xint.append(x+1)

plt.plot(xint, yval, "*-", color="orange")
plt.xlabel("Eigenvalue number", fontsize=18)
plt.ylabel("Eigenvalue", fontsize=18)
plt.xticks(xint)
plt.yscale("log")
plt.title("Eigenvalue decay", fontsize=24)
plt.tight_layout()
plt.savefig("eigenvalue_decay_u")

plt.figure(figsize=[8, 10])
xint = list()
yval = list()

for x, y in enumerate(eigenvalues_p[:len(reduced_problem._basis_functions_p)]):
    yval.append(y)
    xint.append(x+1)

plt.plot(xint, yval, "*-", color="orange")
plt.xlabel("Eigenvalue number", fontsize=18)
plt.ylabel("Eigenvalue", fontsize=18)
plt.xticks(xint)
plt.yscale("log")
plt.title("Eigenvalue decay", fontsize=24)
plt.tight_layout()
plt.savefig("eigenvalue_decay_p")

print(f"Velocity reduced basis size: {len(reduced_problem._basis_functions_u)}")
print(f"Pressure reduced basis size: {len(reduced_problem._basis_functions_p)}")

# POD Ends ###

# Creating dataset


def generate_ann_input_set(samples=ann_samples):
    # Select samples from the parameter space for ANN
    training_set_0 = np.linspace(0.5, 1., samples[0])
    training_set_1 = np.linspace(0.5, 1., samples[1])
    training_set = np.array(list(itertools.product(training_set_0, training_set_1))).astype("f")
    return training_set


def generate_ann_output_set(problem, reduced_problem, input_set, mode=None):
    # Compuet output set for ANN based on input set
    output_set_u = np.empty([input_set.shape[0], len(reduced_problem._basis_functions_u)])
    output_set_p = np.empty([input_set.shape[0], len(reduced_problem._basis_functions_p)])
    rb_size_u = len(reduced_problem._basis_functions_u)
    rb_size_p = len(reduced_problem._basis_functions_p)
    for i in range(input_set.shape[0]):
        if mode is None:
            print(f"Parameter number {i+1} of {input_set.shape[0]}: {input_set[i,:]}")
        else:
            print(f"{mode} parameter number {i+1} of {input_set.shape[0]}: {input_set[i,:]}")
        (solution_u, solution_p) = problem.solve(input_set[i, :])
        output_set_u[i, :] = reduced_problem.project_snapshot_u(solution_u, rb_size_u).array  # .astype("f")
        output_set_p[i, :] = reduced_problem.project_snapshot_p(solution_p, rb_size_p).array  # .astype("f")
    return output_set_u, output_set_p


ann_input_set = generate_ann_input_set(samples=ann_samples)
np.random.shuffle(ann_input_set)
ann_output_set_u, ann_output_set_p = \
    generate_ann_output_set(problem_parametric, reduced_problem,
                            ann_input_set, mode="Training")

num_training_samples = int(0.7 * ann_input_set.shape[0])
num_validation_samples = ann_input_set.shape[0] - num_training_samples

input_training_set = ann_input_set[:num_training_samples, :]
output_training_set_u = ann_output_set_u[:num_training_samples, :]
output_training_set_p = ann_output_set_p[:num_training_samples, :]

input_validation_set = ann_input_set[num_training_samples:, :]
output_validation_set_u = ann_output_set_u[num_training_samples:, :]
output_validation_set_p = ann_output_set_p[num_training_samples:, :]

reduced_problem.output_range_u[0] = np.min(ann_output_set_u)
reduced_problem.output_range_u[1] = np.max(ann_output_set_u)
reduced_problem.output_range_p[0] = np.min(ann_output_set_p)
reduced_problem.output_range_p[1] = np.max(ann_output_set_p)
# NOTE Output_range based on the computed values instead of user guess.

customDataset = CustomDataset(reduced_problem, input_training_set,
                              output_training_set_u,
                              input_scaling_range=reduced_problem.input_scaling_range_u,
                              output_scaling_range=reduced_problem.output_scaling_range_u,
                              input_range=reduced_problem.input_range_u,
                              output_range=reduced_problem.output_range_u, verbose=True)
train_dataloader_u = DataLoader(customDataset, batch_size=10, shuffle=True)

customDataset = CustomDataset(reduced_problem, input_validation_set,
                              output_validation_set_u,
                              input_scaling_range=reduced_problem.input_scaling_range_u,
                              output_scaling_range=reduced_problem.output_scaling_range_u,
                              input_range=reduced_problem.input_range_u,
                              output_range=reduced_problem.output_range_u, verbose=True)
valid_dataloader_u = DataLoader(customDataset, shuffle=False)

customDataset = \
    CustomDataset(reduced_problem, input_validation_set,
                  output_validation_set_p,
                  input_scaling_range=reduced_problem.input_scaling_range_p,
                  output_scaling_range=reduced_problem.output_scaling_range_p,
                  input_range=reduced_problem.input_range_p,
                  output_range=reduced_problem.output_range_p, verbose=True)
train_dataloader_p = DataLoader(customDataset, batch_size=7, shuffle=True)

customDataset = \
    CustomDataset(reduced_problem, input_validation_set,
                  output_validation_set_p,
                  input_scaling_range=reduced_problem.input_scaling_range_p,
                  output_scaling_range=reduced_problem.output_scaling_range_p,
                  input_range=reduced_problem.input_range_p,
                  output_range=reduced_problem.output_range_p, verbose=True)
valid_dataloader_p = DataLoader(customDataset, shuffle=False)

# ANN model
model_u = HiddenLayersNet(input_training_set.shape[1], [30, 30],
                          len(reduced_problem._basis_functions_u), Tanh())
model_p = HiddenLayersNet(input_training_set.shape[1], [15, 15],
                          len(reduced_problem._basis_functions_p), Tanh())

# Start of training (Velocity)

'''
path = "model_u.pth"
save_model(model_u, path)
load_model(model_u, path)
'''

training_loss_u = list()
validation_loss_u = list()

max_epochs_u = 40 # 20000
min_validation_loss_u = None
start_epoch_u = 0
checkpoint_path_u = "checkpoint_u"
checkpoint_epoch_u = 10

learning_rate_u = 5.e-6
optimiser_u = get_optimiser(model_u, "Adam", learning_rate_u)
loss_fn_u = get_loss_func("MSE", reduction="sum")

if os.path.exists(checkpoint_path_u):
    start_epoch_u, min_validation_loss_u = \
        load_checkpoint(checkpoint_path_u, model_u, optimiser_u)

import time
start_time = time.time()
for epochs in range(start_epoch_u, max_epochs_u):
    if epochs > 0 and epochs % checkpoint_epoch_u == 0:
        save_checkpoint(checkpoint_path_u, epochs, model_u, optimiser_u,
                        min_validation_loss_u)
    print(f"Epoch: {epochs+1}/{max_epochs_u}")
    current_training_loss = train_nn(reduced_problem, train_dataloader_u,
                                     model_u, loss_fn_u, optimiser_u)
    training_loss_u.append(current_training_loss)
    current_validation_loss = validate_nn(reduced_problem, valid_dataloader_u,
                                          model_u, loss_fn_u)
    validation_loss_u.append(current_validation_loss)
    if epochs > 0 and current_validation_loss > 1.01 * min_validation_loss_u \
       and reduced_problem.regularisation_u == "EarlyStopping":
        # 1% safety margin against min_validation_loss
        # before invoking early stopping criteria
        print(f"Early stopping criteria invoked at epoch: {epochs+1}")
        break
    min_validation_loss_u = min(validation_loss_u)
end_time = time.time()
elapsed_time = end_time - start_time

# Start of training (Pressure)

'''
path = "model_p.pth"
save_model(model_p, path)
load_model(model_p, path)
'''

training_loss_p = list()
validation_loss_p = list()

max_epochs_p = 40 # 20000
min_validation_loss_p = None
start_epoch_p = 0
checkpoint_path_p = "checkpoint_p"
checkpoint_epoch_p = 10

learning_rate_p = 5.e-6
optimiser_p = get_optimiser(model_p, "Adam", learning_rate_p)
loss_fn_p = get_loss_func("MSE", reduction="sum")

if os.path.exists(checkpoint_path_p):
    start_epoch_p, min_validation_loss_p = \
        load_checkpoint(checkpoint_path_p, model_p, optimiser_p)

import time
start_time = time.time()
for epochs in range(start_epoch_p, max_epochs_p):
    if epochs > 0 and epochs % checkpoint_epoch_p == 0:
        save_checkpoint(checkpoint_path_p, epochs, model_p, optimiser_p,
                        min_validation_loss_p)
    print(f"Epoch: {epochs+1}/{max_epochs_p}")
    current_training_loss = train_nn(reduced_problem, train_dataloader_p,
                                     model_p, loss_fn_p, optimiser_p)
    training_loss_p.append(current_training_loss)
    current_validation_loss = validate_nn(reduced_problem, valid_dataloader_p,
                                          model_p, loss_fn_p)
    validation_loss_p.append(current_validation_loss)
    if epochs > 0 and current_validation_loss > 1.01 * min_validation_loss_p \
       and reduced_problem.regularisation_p == "EarlyStopping":
        # 1% safety margin against min_validation_loss
        # before invoking early stopping criteria
        print(f"Early stopping criteria invoked at epoch: {epochs+1}")
        break
    min_validation_loss_p = min(validation_loss_p)
end_time = time.time()
elapsed_time = end_time - start_time


os.system(f"rm {checkpoint_path_u}")
os.system(f"rm {checkpoint_path_p}")

exit()
# TODO fix online_nn and error_analysis as N != reduced_problem._basis_functions but reduced_problem._basis_functions_p or reduced_problem._basis_functions_u

# Error analysis dataset
print("\n")
print("Generating error analysis (only input/parameters) dataset")
print("\n")
error_analysis_set_u = generate_ann_input_set(samples=error_analysis_samples)
error_numpy_u = np.zeros(error_analysis_set_u.shape[0])

for i in range(error_analysis_set_u.shape[0]):
    print(f"Error analysis parameter number {i+1} of ")
    print(f"{error_analysis_set_u.shape[0]}: {error_analysis_set_u[i,:]}")
    error_numpy_u[i] = error_analysis(reduced_problem, problem_parametric,
                                      error_analysis_set_u[i, :], model_u,
                                      online_nn,
                                      norm_error=reduced_problem.norm_error_u,
                                      reconstruct_solution=reduced_problem.reconstruct_solution_u,
                                      input_scaling_range=reduced_problem.input_scaling_range_u,
                                      output_scaling_range=reduced_problem.output_scaling_range_u,
                                      input_range=reduced_problem.input_range_u,
                                      output_range=reduced_problem.output_range_u,
                                      index=0, verbose=True)
    print(f"Error: {error_numpy_u[i]}")

# Error analysis dataset
print("\n")
print("Generating error analysis (only input/parameters) dataset")
print("\n")
error_analysis_set_p = generate_ann_input_set(samples=error_analysis_samples)
error_numpy_p = np.zeros(error_analysis_set_p.shape[0])

for i in range(error_analysis_set_p.shape[0]):
    print(f"Error analysis parameter number {i+1} of ")
    print(f"{error_analysis_set_p.shape[0]}: {error_analysis_set_p[i,:]}")
    error_numpy_p[i] = error_analysis(reduced_problem, problem_parametric,
                                      error_analysis_set_p[i, :], model_p,
                                      online_nn,
                                      norm_error=reduced_problem.norm_error_p,
                                      reconstruct_solution=reduced_problem.reconstruct_solution_p,
                                      input_scaling_range=reduced_problem.input_scaling_range_p,
                                      output_scaling_range=reduced_problem.output_scaling_range_p,
                                      input_range=reduced_problem.input_range_p,
                                      output_range=reduced_problem.output_range_p,
                                      index=1, verbose=True)
    print(f"Error: {error_numpy_p[i]}")

print(f"Velocity error: {error_numpy_u}")
print(f"Pressure error: {error_numpy_p}")

# Online phase
# Define a parameter
online_mu = np.array([0.8, 0.9])

# Compute FEM solution
(solution_u, solution_p) = problem_parametric.solve(online_mu)

# Compute RB solution
rb_solution_u = \
    reduced_problem.reconstruct_solution_u(online_nn(reduced_problem,
                                                     problem_parametric,
                                                     online_mu, model_u,
                                                     len(reduced_problem._basis_functions_u),
                                                     device=None,
                                                     input_scaling_range=reduced_problem.input_scaling_range_u,
                                                     output_scaling_range=reduced_problem.output_scaling_range_u,
                                                     input_range=reduced_problem.input_range_u,
                                                     output_range=reduced_problem.output_range_u))
rb_solution_p = \
    reduced_problem.reconstruct_solution_p(online_nn(reduced_problem,
                                                     problem_parametric,
                                                     online_mu, model_p,
                                                     len(reduced_problem._basis_functions_p),
                                                     device=None,
                                                     input_scaling_range=reduced_problem.input_scaling_range_p,
                                                     output_scaling_range=reduced_problem.output_scaling_range_p,
                                                     input_range=reduced_problem.input_range_p,
                                                     output_range=reduced_problem.output_range_p))

# Post processing of FEM and RB solution
# BCs for mesh deformation
bcs_geometric = [lambda x: (x[0], x[1]),
                 lambda x: (x[0], x[1]),
                 lambda x: (x[0], x[1]),
                 lambda x: (x[0], x[1]),
                 lambda x: (online_mu[0] * x[0], online_mu[1] * x[1]),
                 lambda x: (online_mu[0] * x[0], online_mu[1] * x[1])]

solution_velocity_error = dolfinx.fem.Function(V)
solution_pressure_error = dolfinx.fem.Function(Q)

solution_velocity_error.x.array[:] = abs(solution_u.x.array - rb_solution_u.x.array)

solution_pressure_error.x.array[:] = abs(solution_p.x.array - rb_solution_p.x.array)

with HarmonicMeshMotion(problem_parametric._mesh, problem_parametric._boundaries,
                        problem_parametric._boundary_markers, bcs_geometric,
                        reset_reference=True, is_deformation=False):

    with dolfinx.io.XDMFFile(mesh.comm, "dlrbnicsx_solution/fem_velocity_online_mu.xdmf",
                             "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(solution_u)

    with dolfinx.io.XDMFFile(mesh.comm, "dlrbnicsx_solution/fem_pressure_online_mu.xdmf",
                             "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(solution_p)

    with dolfinx.io.XDMFFile(mesh.comm, "dlrbnicsx_solution/rb_velocity_online_mu.xdmf",
                             "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(rb_solution_u)

    with dolfinx.io.XDMFFile(mesh.comm, "dlrbnicsx_solution/rb_pressure_online_mu.xdmf",
                             "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(rb_solution_p)

    with dolfinx.io.XDMFFile(mesh.comm, "dlrbnicsx_solution/error_velocity_online_mu.xdmf",
                             "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(solution_velocity_error)

    with dolfinx.io.XDMFFile(mesh.comm, "dlrbnicsx_solution/error_pressure_online_mu.xdmf",
                             "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(solution_pressure_error)
