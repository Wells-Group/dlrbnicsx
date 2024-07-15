import numpy as np
import abc
import matplotlib.pyplot as plt
import itertools
import os

from mpi4py import MPI
from petsc4py import PETSc
from smt.sampling_methods import LHS

import ufl
import basix
import dolfinx

import rbnicsx
import rbnicsx.online
import rbnicsx.backends

from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory import Tanh, Sigmoid
from dlrbnicsx.dataset.custom_dataset import CustomDataset
from dlrbnicsx.interface.wrappers import DataLoader, save_model, load_model, \
    save_checkpoint, load_checkpoint, get_optimiser, get_loss_func
from dlrbnicsx.train_validate_test.train_validate_test import \
    train_nn, validate_nn, online_nn, error_analysis

class ParametricProblem(abc.ABC):
    def __init__(self, mesh, subdomains, boundaries):
        self._mesh = mesh
        self._subdomains = subdomains
        self._boundaries = boundaries
        self.ds = ufl.Measure("ds", domain=mesh, subdomain_data=boundaries)
        self.dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)
        Q_el = basix.ufl.element("BDMCF", mesh.basix_cell(), 1)
        P_el = basix.ufl.element("DG", mesh.basix_cell(), 0)
        V_el = basix.ufl.mixed_element([Q_el, P_el])
        self._V = dolfinx.fem.FunctionSpace(mesh, V_el)
        self._V0 = self._V.sub(0)
        self._Q, _ = self._V0.collapse()
        self._U, _ = self._V.sub(1).collapse()
        sigma, tau = ufl.TrialFunction(self._Q), ufl.TestFunction(self._Q)
        u, v = ufl.TrialFunction(self._U), ufl.TestFunction(self._U)
        self._inner_product_sigma = ufl.inner(sigma, tau) * self.dx + \
            ufl.inner(ufl.div(sigma), ufl.div(tau)) * self.dx
        self._inner_product_sigma_action = \
            rbnicsx.backends.bilinear_form_action(self._inner_product_sigma,
                                                  part="real")
        self._inner_product_u = ufl.inner(u, v) * self.dx
        self._inner_product_u_action = \
            rbnicsx.backends.bilinear_form_action(self._inner_product_u,
                                                  part="real")
        self.mu_0 = dolfinx.fem.Constant(self._mesh, PETSc.ScalarType(0.))
        self.mu_1 = dolfinx.fem.Constant(self._mesh, PETSc.ScalarType(0.))
        self.mu_2 = dolfinx.fem.Constant(self._mesh, PETSc.ScalarType(0.))
        self.mu_3 = dolfinx.fem.Constant(self._mesh, PETSc.ScalarType(0.))
        self.mu_4 = dolfinx.fem.Constant(self._mesh, PETSc.ScalarType(0.))

        self.dirichlet_func_0 = dolfinx.fem.Function(self._Q)
        self.dirichlet_func_1 = dolfinx.fem.Function(self._Q)
        self.dirichlet_func_2 = dolfinx.fem.Function(self._Q)
        self.bc_dofs = self.get_bc_dofs()

    @property
    def source_term(self):
        x = ufl.SpatialCoordinate(self._mesh)
        f = 10. * ufl.exp(-self.mu_0 * ((x[0] - self.mu_1) * (x[0] - self.mu_1) +
                                        (x[1] - self.mu_2) * (x[1] - self.mu_2) +
                                        (x[2] - self.mu_3) * (x[2] - self.mu_3)))
        # NOTE Make sure that f gets updated everytime new parameter mu is given
        return f

    @property
    def bilinear_form(self):
        (sigma, u) = ufl.TrialFunctions(self._V)
        (tau, v) = ufl.TestFunctions(self._V)
        a = ufl.inner(sigma, tau) * self.dx +\
            ufl.inner(u, ufl.div(tau)) * self.dx +\
            ufl.inner(ufl.div(sigma), v) * self.dx
        return dolfinx.fem.form(a)

    @property
    def linear_form(self):
        f = self.source_term
        (_, v) = ufl.TestFunctions(self._V)
        L = -ufl.inner(f, v) * self.dx
        return dolfinx.fem.form(L)

    def get_bc_dofs(self):
        gdim = self._mesh.geometry.dim
        dofs_x0 = \
            dolfinx.fem.locate_dofs_topological((self._V0, self._Q),
                                                gdim-1, self._boundaries.find(30))
        dofs_y0 = \
            dolfinx.fem.locate_dofs_topological((self._V0, self._Q),
                                                gdim-1, self._boundaries.find(18))
        dofs_z0 = \
            dolfinx.fem.locate_dofs_topological((self._V0, self._Q),
                                                gdim-1, self._boundaries.find(1))
        return dofs_x0, dofs_y0, dofs_z0

    def dirichlet_val_0(self, x):
        values = np.zeros((3, x.shape[1]))
        values[0, :] = np.sin(self.mu_4.value * x[0])
        return values

    def dirichlet_val_1(self, x):
        values = np.zeros((3, x.shape[1]))
        values[1, :] = np.sin(self.mu_4.value * x[1])
        return values

    def dirichlet_val_2(self, x):
        values = np.zeros((3, x.shape[1]))
        values[2, :] = np.sin(self.mu_4.value * x[2])
        return values

    def solve(self, mu):
        self.mu_0.value = mu[0]
        self.mu_1.value = mu[1]
        self.mu_2.value = mu[2]
        self.mu_3.value = mu[3]
        self.mu_4.value = mu[4]

        self.dirichlet_func_0.interpolate(self.dirichlet_val_0)
        self.dirichlet_func_1.interpolate(self.dirichlet_val_1)
        self.dirichlet_func_2.interpolate(self.dirichlet_val_2)

        self.bc_0 = dolfinx.fem.dirichletbc(self.dirichlet_func_0, self.bc_dofs[0], self._V0)
        self.bc_1 = dolfinx.fem.dirichletbc(self.dirichlet_func_1, self.bc_dofs[1], self._V0)
        self.bc_2 = dolfinx.fem.dirichletbc(self.dirichlet_func_2, self.bc_dofs[2], self._V0)


        bcs = [self.bc_0, self.bc_1, self.bc_2]
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
        solution.x.scatter_forward()
        sigma_sol, u_sol = solution.split()
        sigma_sol = sigma_sol.collapse()
        u_sol = u_sol.collapse()
        return sigma_sol, u_sol

class PODANNReducedProblem(abc.ABC):
    def __init__(self, problem):
        Q, _ = problem._V.sub(0).collapse()
        W, _ = problem._V.sub(1).collapse()
        self._basis_functions_sigma = rbnicsx.backends.FunctionsList(Q)
        self._basis_functions_u = rbnicsx.backends.FunctionsList(W)
        sigma, tau = ufl.TrialFunction(Q), ufl.TestFunction(Q)
        u, v = ufl.TrialFunction(W), ufl.TestFunction(W)
        self._inner_product_sigma_action = problem._inner_product_sigma_action
        self._inner_product_u_action = problem._inner_product_u_action
        self.input_scaling_range = [-1., 1.]
        self.input_range = \
            np.array([[-5., 0.2, 0.2, 0.2, 1.],
                      [5., 0.8, 0.8, 0.8, 5.]])
        self.output_scaling_range_sigma = [-1., 1.]
        self.output_range_sigma = [None, None]
        self.output_scaling_range_u = [-1., 1.]
        self.output_range_u = [None, None]

    def reconstruct_solution_sigma(self, reduced_solution_sigma):
        return self._basis_functions_sigma[:reduced_solution_sigma.size] * \
            reduced_solution_sigma
    
    def reconstruct_solution_u(self, reduced_solution_u):
        return self._basis_functions_u[:reduced_solution_u.size] * \
            reduced_solution_u
    
    def compute_norm_sigma(self, sigma_function):
        return np.sqrt(self._inner_product_sigma_action(sigma_function)
                       (sigma_function))

    def compute_norm_u(self, u_function):
        return np.sqrt(self._inner_product_u_action(u_function)
                       (u_function))

    def project_snapshot_sigma(self, sigma_function, N_sigma):
        return self._project_snapshot_sigma(sigma_function, N_sigma)

    def project_snapshot_u(self, u_function, N_u):
        return self._project_snapshot_u(u_function, N_u)
    
    def _project_snapshot_sigma(self, sigma_function, N_sigma):
        projected_snapshot_sigma = rbnicsx.online.create_vector(N_sigma)
        A = rbnicsx.backends.\
            project_matrix(self._inner_product_sigma_action,
                           self._basis_functions_sigma[:N_sigma])
        F = rbnicsx.backends.\
            project_vector(self._inner_product_sigma_action(sigma_function),
                           self._basis_functions_sigma[:N_sigma])
        ksp = PETSc.KSP()
        ksp.create(projected_snapshot_sigma.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
        ksp.solve(F, projected_snapshot_sigma)
        return projected_snapshot_sigma

    def _project_snapshot_u(self, u_function, N_u):
        projected_snapshot_u = rbnicsx.online.create_vector(N_u)
        A = rbnicsx.backends.\
            project_matrix(self._inner_product_u_action,
                           self._basis_functions_u[:N_u])
        F = rbnicsx.backends.\
            project_vector(self._inner_product_u_action(u_function),
                           self._basis_functions_u[:N_u])
        ksp = PETSc.KSP()
        ksp.create(projected_snapshot_sigma.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
        ksp.solve(F, projected_snapshot_u)
        return projected_snapshot_u

    def norm_error_sigma(self, sigma_true, sigma_rb):
        return self.compute_norm_sigma(sigma_true - sigma_rb)/self.compute_norm_sigma(sigma_true)

    def norm_error_u(self, u_true, u_rb):
        return self.compute_norm_u(u_true - u_rb)/self.compute_norm_u(u_true)

# Import mesh in dolfinx
gdim = 3
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
mesh, subdomains, boundaries = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/3d_mesh.msh", mesh_comm,
                                    gmsh_model_rank, gdim=gdim)
# Boundary markers: x=1 is 22, x=0 is 30, y=1 is 26, y=0 is 18, z=1 is 31, z=0 is 1

num_ann_samples = 300
error_analysis_samples = 130

# Parameters
mu = np.array([-2., 0.5, 0.5, 0.5, 3.])

# FEM solve
problem_parametric = ParametricProblem(mesh, subdomains,
                                       boundaries)
sigma_sol, u_sol = problem_parametric.solve(mu)
print(sigma_sol.x.array, np.linalg.norm(sigma_sol.x.array))
print(u_sol.x.array, np.linalg.norm(u_sol.x.array))

'''
with dolfinx.io.XDMFFile(mesh.comm, "parametric_mixed_poisson/sigma.xdmf", "w") as sol_file:
    sol_file.write_mesh(mesh)
    sol_file.write_function(sigma_sol)

with dolfinx.io.XDMFFile(mesh.comm, "parametric_mixed_poisson/u.xdmf", "w") as sol_file:
    sol_file.write_mesh(mesh)
    sol_file.write_function(u_sol)
'''

# TODO more benchmarking with dolfinx implementation for correctness

# POD Starts ###


def generate_training_set(samples=[4, 4, 4, 4, 2]):
    # Select input samples for POD
    training_set_0 = np.linspace(-5., 5., samples[0])
    training_set_1 = np.linspace(0.2, 0.8, samples[1])
    training_set_2 = np.linspace(0.2, 0.8, samples[2])
    training_set_3 = np.linspace(0.2, 0.8, samples[3])
    training_set_4 = np.linspace(1., 5., samples[4])
    training_set = np.array(list(itertools.product(training_set_0,
                                                   training_set_1,
                                                   training_set_2,
                                                   training_set_3,
                                                   training_set_4)))
    return training_set


# POD samples
training_set = rbnicsx.io.on_rank_zero(mesh.comm, generate_training_set)

# Maximum RB size
Nmax = 30

print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
print("")

print("Set up snapshots matrix")
snapshots_matrix = rbnicsx.backends.FunctionsList(problem_parametric._Q)

print("Set up reduced problem")
reduced_problem = PODANNReducedProblem(problem_parametric)

print("")

for (mu_index, mu) in enumerate(training_set):
    print(rbnicsx.io.TextLine(str(mu_index+1), fill="#"))

    print("Parameter number ", (mu_index+1), "of", training_set.shape[0])
    print("High fidelity solve for mu =", mu)
    sigma_sol, u_sol = problem_parametric.solve(mu)
    print(f"Sigma solution array: {sigma_sol.x.array}")
    print(f"Sigma solution norm: {np.linalg.norm(sigma_sol.x.array)}")

    print("Update snapshots matrix")
    snapshots_matrix.append(sigma_sol)

    print("")

print(rbnicsx.io.TextLine("Perform POD", fill="#"))
eigenvalues_sigma, modes_sigma, _ = \
    rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_matrix,
                                    problem_parametric._inner_product_sigma_action,
                                    N=Nmax, tol=1e-6)

reduced_problem._basis_functions_sigma.extend(modes_sigma)
reduced_size_sigma = len(reduced_problem._basis_functions_sigma)
print("")

print(rbnicsx.io.TextBox("POD-Galerkin offline phase ends", fill="="))

positive_eigenvalues_sigma = np.where(eigenvalues_sigma > 0., eigenvalues_sigma, np.nan)
singular_values_sigma = np.sqrt(positive_eigenvalues_sigma)

plt.figure(figsize=[8, 10])
xint = list()
yval = list()

for x, y in enumerate(eigenvalues_sigma[:len(modes_sigma)]):
    yval.append(y)
    xint.append(x+1)

plt.plot(xint, yval, "*-", color="orange")
plt.xlabel("Eigenvalue number", fontsize=18)
plt.ylabel("Eigenvalue", fontsize=18)
plt.xticks(xint)
plt.yscale("log")
plt.title("Eigenvalue decay sigma", fontsize=24)
plt.tight_layout()
plt.savefig("eigenvalue_decay_sigma")

print(f"Sigma eigenvalues: {positive_eigenvalues_sigma}")

# TODO POD for u

# POD Ends ###

sigma_sol_projected = reduced_problem.project_snapshot_sigma(sigma_sol, reduced_size_sigma)
sigma_sol_reconstructed = reduced_problem.reconstruct_solution_sigma(sigma_sol_projected)
print(sigma_sol_reconstructed.x.array.shape, sigma_sol.x.array.shape)
sigma_norm = reduced_problem.compute_norm_sigma(sigma_sol_reconstructed)
sigma_error = reduced_problem.norm_error_sigma(sigma_sol, sigma_sol_reconstructed)
print(f"Norm reconstructed: {sigma_norm}, Error: {sigma_error}")

# Creating dataset
def generate_ann_input_set(num_ann_samples=243):
    xlimits = np.array([[-5., 5.], [0.2, 0.8],
                        [0.2, 0.8], [0.2, 0.8],
                        [1., 5.]])
    sampling = LHS(xlimits=xlimits)
    training_set = sampling(num_ann_samples)
    return training_set

def generate_ann_output_set(problem, reduced_problem, input_set, mode=None):
    output_set_sigma = np.zeros([input_set.shape[0], len(reduced_problem._basis_functions_sigma)])
    output_set_u = np.zeros([input_set.shape[0], len(reduced_problem._basis_functions_u)])
    for i in range(input_set.shape[0]):
        if mode is None:
            print(f"Parameter number {i+1} of {input_set.shape[0]}: {input_set[i,:]}")
        else:
            print(f"{mode} parameter number {i+1} of {input_set.shape[0]}: {input_set[i,:]}")
        solution_sigma, solution_u = problem.solve(input_set[i, :])
        output_set_sigma[i, :] = reduced_problem.project_snapshot_sigma(solution_sigma, len(reduced_problem._basis_functions_sigma)).array  # .astype("f")
        # output_set_u[i, :] = reduced_problem.project_snapshot_u(solution_u, len(reduced_problem._basis_functions_u)).array  # .astype("f")
    return output_set_sigma, output_set_u

ann_input_set = generate_ann_input_set(num_ann_samples=num_ann_samples)
# np.random.shuffle(ann_input_set)
ann_output_set_sigma, ann_output_set_u = \
    generate_ann_output_set(problem_parametric, reduced_problem,
                            ann_input_set, mode="Training")

num_training_samples = int(0.7 * ann_input_set.shape[0])
num_validation_samples = ann_input_set.shape[0] - num_training_samples

input_training_set = ann_input_set[:num_training_samples, :]
output_training_set_sigma = ann_output_set_sigma[:num_training_samples, :]
output_training_set_u = ann_output_set_u[:num_training_samples, :]

input_validation_set = ann_input_set[num_training_samples:, :]
output_validation_set_sigma = ann_output_set_sigma[num_training_samples:, :]
output_validation_set_u = ann_output_set_u[num_training_samples:, :]

reduced_problem.output_range_sigma[0] = np.min(ann_output_set_sigma)
reduced_problem.output_range_sigma[1] = np.max(ann_output_set_sigma)
# reduced_problem.output_range_u[0] = np.min(ann_output_set_u)
# reduced_problem.output_range_u[1] = np.max(ann_output_set_u)

customDataset = CustomDataset(reduced_problem, input_training_set,
                              output_training_set_sigma,
                              input_scaling_range=[-1., 1.],
                              output_scaling_range=reduced_problem.output_scaling_range_sigma,
                              input_range=reduced_problem.input_range,
                              output_range=reduced_problem.output_range_sigma, verbose=False)
train_dataloader_sigma = DataLoader(customDataset, batch_size=6, shuffle=False) # shuffle=True)

customDataset = CustomDataset(reduced_problem, input_validation_set,
                              output_validation_set_sigma,
                              input_scaling_range=[-1., 1.],
                              output_scaling_range=reduced_problem.output_scaling_range_sigma,
                              input_range=reduced_problem.input_range,
                              output_range=reduced_problem.output_range_sigma, verbose=False)
valid_dataloader_sigma = DataLoader(customDataset, shuffle=False)

'''
customDataset = \
    CustomDataset(reduced_problem, input_training_set,
                  output_training_set_u,
                  input_scaling_range=[-1., 1.],
                  output_scaling_range=reduced_problem.output_scaling_range_u,
                  input_range=reduced_problem.input_range,
                  output_range=reduced_problem.output_range_u, verbose=False)
train_dataloader_u = DataLoader(customDataset, batch_size=6, shuffle=False) # shuffle=True)

customDataset = \
    CustomDataset(reduced_problem, input_validation_set,
                  output_validation_set_u,
                  input_scaling_range=[-1., 1.],
                  output_scaling_range=reduced_problem.output_scaling_range_u,
                  input_range=reduced_problem.input_range,
                  output_range=reduced_problem.output_range_u, verbose=False)
valid_dataloader_u = DataLoader(customDataset, shuffle=False)
'''

# ANN model
model_sigma = HiddenLayersNet(input_training_set.shape[1], [30, 30, 30],
                              len(reduced_problem._basis_functions_sigma),
                              Tanh())
'''
model_u = HiddenLayersNet(input_training_set.shape[1], [15, 15],
                          len(reduced_problem._basis_functions_u),
                          Tanh())
'''

# Start of training (Velocity)

path = "model_sigma.pth"
save_model(model_sigma, path)
# load_model(model_sigma, path)

training_loss_sigma = list()
validation_loss_sigma = list()

max_epochs_sigma = 50 # 20000
min_validation_loss_sigma = None
start_epoch_sigma = 0
checkpoint_path_sigma = "checkpoint_sigma"
checkpoint_epoch_sigma = 10

learning_rate_sigma = 5.e-6
optimiser_sigma = get_optimiser(model_sigma, "Adam", learning_rate_sigma)
loss_fn_sigma = get_loss_func("MSE", reduction="sum")

if os.path.exists(checkpoint_path_sigma):
    start_epoch_sigma, min_validation_loss_sigma = \
        load_checkpoint(checkpoint_path_sigma, model_sigma, optimiser_sigma)

import time
start_time = time.time()
for epochs in range(start_epoch_sigma, max_epochs_sigma):
    if epochs > 0 and epochs % checkpoint_epoch_sigma == 0:
        save_checkpoint(checkpoint_path_sigma, epochs, model_sigma,
                        optimiser_sigma, min_validation_loss_sigma)
    print(f"Epoch: {epochs+1}/{max_epochs_sigma}")
    current_training_loss = train_nn(reduced_problem, train_dataloader_sigma,
                                     model_sigma, loss_fn_sigma,
                                     optimiser_sigma)
    training_loss_sigma.append(current_training_loss)
    current_validation_loss = validate_nn(reduced_problem, valid_dataloader_sigma,
                                          model_sigma, loss_fn_sigma)
    validation_loss_sigma.append(current_validation_loss)
    if epochs > 0 and current_validation_loss > 1.01 * min_validation_loss_sigma \
       and reduced_problem.regularisation_u == "EarlyStopping":
        # 1% safety margin against min_validation_loss
        # before invoking early stopping criteria
        print(f"Early stopping criteria invoked at epoch: {epochs+1}")
        break
    min_validation_loss_sigma = min(validation_loss_sigma)
end_time = time.time()
elapsed_time = end_time - start_time
os.system(f"rm {checkpoint_path_sigma}")

# Error analysis dataset
print("\n")
print("Generating error analysis (only input/parameters) dataset")
print("\n")
error_analysis_set_sigma = generate_ann_input_set(num_ann_samples=error_analysis_samples)
error_numpy_sigma = np.zeros(error_analysis_set_sigma.shape[0])

for i in range(error_analysis_set_sigma.shape[0]):
    print(f"Error analysis parameter number {i+1} of ")
    print(f"{error_analysis_set_sigma.shape[0]}: {error_analysis_set_sigma[i,:]}")
    error_numpy_sigma[i] = error_analysis(reduced_problem, problem_parametric,
                                      error_analysis_set_sigma[i, :], model_sigma,
                                      len(reduced_problem._basis_functions_sigma), online_nn,
                                      norm_error=reduced_problem.norm_error_sigma,
                                      reconstruct_solution=reduced_problem.reconstruct_solution_sigma,
                                      input_scaling_range=[-1., 1.],
                                      output_scaling_range=reduced_problem.output_scaling_range_sigma,
                                      input_range=reduced_problem.input_range,
                                      output_range=reduced_problem.output_range_sigma,
                                      index=0, verbose=True)
    print(f"Error: {error_numpy_sigma[i]}")

# Online phase
# Define a parameter
online_mu = np.array([-2.3, 0.47, 0.57, 0.67, 3.4])

# Compute FEM solution
(solution_sigma, solution_u) = problem_parametric.solve(online_mu)

# Compute RB solution
rb_solution_sigma = \
    reduced_problem.reconstruct_solution_sigma(online_nn(reduced_problem,
                                                     problem_parametric,
                                                     online_mu, model_sigma,
                                                     len(reduced_problem._basis_functions_sigma),
                                                     input_scaling_range=[-1., 1.],
                                                     output_scaling_range=reduced_problem.output_scaling_range_sigma,
                                                     input_range=reduced_problem.input_range,
                                                     output_range=reduced_problem.output_range_sigma))

'''
rb_solution_u = \
    reduced_problem.reconstruct_solution_u(online_nn(reduced_problem,
                                                     problem_parametric,
                                                     online_mu, model_u,
                                                     len(reduced_problem._basis_functions_u),
                                                     input_scaling_range=[-1., 1.],
                                                     output_scaling_range=reduced_problem.output_scaling_range_u,
                                                     input_range=reduced_problem.input_range,
                                                     output_range=reduced_problem.output_range_u))
'''

solution_sigma_error = dolfinx.fem.Function(problem_parametric._Q)
solution_sigma_error.x.array[:] = abs(solution_sigma.x.array - rb_solution_sigma.x.array)

'''
with dolfinx.io.XDMFFile(mesh.comm, "dlrbnicsx_solution/fem_sigma_online_mu.xdmf",
                            "w") as solution_file:
    solution_file.write_mesh(mesh)
    solution_file.write_function(solution_sigma)

with dolfinx.io.XDMFFile(mesh.comm, "dlrbnicsx_solution/rb_sigma_online_mu.xdmf",
                            "w") as solution_file:
    solution_file.write_mesh(mesh)
    solution_file.write_function(rb_solution_sigma)
'''
