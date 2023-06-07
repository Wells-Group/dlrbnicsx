import abc
import itertools
# import typing
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import math

import ufl
import dolfinx
# import mdfenicsx

from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion

import rbnicsx
import rbnicsx.backends
import rbnicsx.online

from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory import Tanh
from dlrbnicsx.dataset.custom_dataset import CustomDataset
from dlrbnicsx.interface.wrappers import DataLoader
from dlrbnicsx.train_validate_test.train_validate_test import \
    train_nn, validate_nn, online_nn, error_analysis

import matplotlib.pyplot as plt

Re = 400
    
class ProblemOnDeformedDomain(abc.ABC):
    def __init__(self, mesh, subdomains, boundaries, meshDeformationContext):
        self._mesh = mesh
        self.gdim = self._mesh.geometry.dim
        self._subdomains = subdomains
        self._boundaries = boundaries
        self._boundary_markers = [1,2,3,4]
        self._meshDeformationContext = meshDeformationContext
        x_elem = ufl.VectorElement("CG", mesh.ufl_cell(), 2)
        q_elem = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)
        v_elem  = ufl.MixedElement([x_elem, q_elem])
        self._W = dolfinx.fem.FunctionSpace(self._mesh, v_elem)
        self._test = ufl.TestFunction(self._W)
        self._solution= dolfinx.fem.Function(self._W)

        self._test = ufl.split(self._test)
        self._trial = ufl.split(self._solution)

    @property
    def assemble_bcs(self):
        bcs = list()
        V, _ = self._W.sub(0).collapse()
        u_x = 1.0

        passing_velocity = dolfinx.fem.Function(V)
        noslip = dolfinx.fem.Function(V)
        passing_velocity.interpolate(lambda x: np.stack((u_x * np.ones(x.shape[1]), np.zeros(x.shape[1]))))

        dofs_bottom = dolfinx.fem.locate_dofs_topological((self._W.sub(0),V), self.gdim -1, self._boundaries.find(1))
        dofs_right = dolfinx.fem.locate_dofs_topological((self._W.sub(0),V), self.gdim -1, self._boundaries.find(2))
        dofs_left = dolfinx.fem.locate_dofs_topological((self._W.sub(0),V), self.gdim -1, self._boundaries.find(4))
        dofs_top = dolfinx.fem.locate_dofs_topological((self._W.sub(0),V), self.gdim -1, self._boundaries.find(3))

        bcs.append(dolfinx.fem.dirichletbc(passing_velocity, dofs_top, self._W.sub(0)))
        bcs.append(dolfinx.fem.dirichletbc(noslip, dofs_bottom,self._W.sub(0)))
        bcs.append(dolfinx.fem.dirichletbc(noslip, dofs_left,self._W.sub(0)))
        bcs.append(dolfinx.fem.dirichletbc(noslip, dofs_right,self._W.sub(0)))  
    
        return bcs
    
    @property
    def residual_term(self):
        (v, q) = self._test
        (u, p) = self._trial

        mu = self._mu

        nu = max(mu[0], mu[1]) / Re

        return ufl.inner(ufl.grad(u)* u,v) * ufl.dx\
                + nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx\
                - ufl.div(v) * p * ufl.dx\
                - q * ufl.div(u) * ufl.dx

    @property
    def set_problem(self):
        problemNonlinear = dolfinx.fem.petsc.NonlinearProblem(self.residual_term, self._solution, bcs = self.assemble_bcs)
        return problemNonlinear
    
    def solve(self, mu):
        self._mu = mu
        self._bcs_geometric = [lambda x: (mu[0] * x[0] -x[0] , x[1] - x[1]), # Bottom
                                lambda x: (mu[0]* x[0] + np.cos(mu[2]) * mu[1] * x[1] -x[0], np.sin(mu[2]) * mu[1] * x[1] -x[1]), # Right
                                lambda x: (mu[0]* x[0] + np.cos(mu[2]) * mu[1] -x[0], np.sin(mu[2]) * mu[1] +  0.0 * x[1]-x[1]), # Top
                                lambda x: (np.cos(mu[2]) * mu[1] * x[1] -x[0], np.sin(mu[2]) * mu[1] * x[1]-x[1]) # Left
                                ]
        problemNonlinear = self.set_problem
        solution = dolfinx.fem.Function(self._W)
        with self._meshDeformationContext(self._mesh, self._boundaries, self._boundary_markers, self._bcs_geometric, is_deformation = True, reset_reference = True) as mesh_class:
            solver = dolfinx.nls.petsc.NewtonSolver(mesh_class._mesh.comm, problemNonlinear)
            solver.max_it = 100
            solver.rtol = 1e-6
            self._solution.x.set(0.0)
            n, converged = solver.solve(self._solution)
            assert(converged)
            solution.x.array[:] = self._solution.x.array.copy()
            print(f"Computed solution array: {solution.x.array}")
            print(f"Number of iterations: {n}")
            (solution_u, solution_p) = solution.split()
            solution_u, solution_p = (solution.sub(0).collapse(), solution.sub(1).collapse())
            return solution_u, solution_p
            
class PODANNReducedProblem(abc.ABC):
    def __init__(self, problem) -> None:
        V, _ = problem._W.sub(0).collapse()
        Q, _ = problem._W.sub(1).collapse()
        self._basis_functions_u = rbnicsx.backends.FunctionsList(V)
        self._basis_functions_p = rbnicsx.backends.FunctionsList(Q)

        u, p = ufl.TrialFunction(V), ufl.TrialFunction(Q)
        v, q = ufl.TestFunction(V), ufl.TestFunction(Q)

        self._inner_product_u = ufl.inner(u,v) * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v))* ufl.dx
        self._inner_product_action_u = rbnicsx.backends.bilinear_form_action ( self._inner_product_u, part = "real")

        self._inner_product_p = ufl.inner(p,q) * ufl.dx
        self._inner_product_action_p = rbnicsx.backends.bilinear_form_action ( self._inner_product_p, part = "real")

        
        self.input_scaling_range = [-1., 1.]
        self.output_scaling_range = [-1., 1.]
        self.input_range = \
            np.array([[0.8, 0.8], [1.1, 1.2]])  # TODO: keine Ahnung was das für Parameter sein sollen
        self.output_range = [None, None]
        self.loss_fn = "MSE"
        self.learning_rate = 1e-4
        self.optimizer = "Adam"
        self.regularisation = "EarlyStopping"

    def reconstruct_solution_u(self, reduced_solution):
        return self._basis_functions_u[:reduced_solution.size] * reduced_solution
    
    def reconstruct_solution_p(self, reduced_solution):
        return self._basis_functions_p[:reduced_solution.size] * reduced_solution
    
    def compute_norm_u(self, function):
        #TODO: warum sind die zwei functions getrennt?
        return np.sqrt(self._inner_product_action_u(function)(function))
    
    def compute_norm_p(self, function):
        return np.sqrt(self._inner_product_action_p(function)(function))
    
    def project_snapshot_u(self, solution, N):
        return self._project_snapshot_u(solution, N)
    
    def project_snapshot_p(self, solution, N):
        return self._project_snapshot_p(solution, N)
    
    def _project_snapshot_u(self, solution, N):
        projected_snapshot_u = rbnicsx.online.create_vector(N)
        A = rbnicsx.backends.project_matrix(self._inner_product_action_u, self._basis_functions_u[:N])
        F = rbnicsx.backends.project_vector(self._inner_product_action_u(solution), self._basis_functions_u[:N])
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
        A = rbnicsx.backends.project_matrix(self._inner_product_action_p, self._basis_functions_p[:N])
        F = rbnicsx.backends.project_vector(self._inner_product_action_p(solution), self._basis_functions_p[:N])
        ksp = PETSc.KSP()
        ksp.create(projected_snapshot_p.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
        ksp.solve(F, projected_snapshot_p)
        return projected_snapshot_p
    
    def norm_error_u(self, u, v):
        self.compute_norm_u(u-v)/self.compute_norm_u(u)

    def norm_error_p(self, p, q):
        self.compute_norm_p(p-q)/self.compute_norm_p(p)


mesh_comm = MPI.COMM_WORLD
gdim = 2
gmsh_model_rank = 0
mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh", mesh_comm, gmsh_model_rank, gdim = gdim)

#  Mesh deformation parameters

mu = np.array([1.0, 2/np.sqrt(3), math.pi/3])

# FEM solve
problem_parametric = ProblemOnDeformedDomain(mesh, cell_tags, facet_tags, HarmonicMeshMotion)

solution_u, solution_p = problem_parametric.solve(mu)

computed_file = "results/solution_computed.xdmf"

with HarmonicMeshMotion(mesh, facet_tags,
                           problem_parametric._boundary_markers,
                           problem_parametric._bcs_geometric,
                           reset_reference=True,
                           is_deformation=True) as mesh_class:
    solution_u.name = "Velocity"
    solution_p.name = "Pressure"
    with dolfinx.io.XDMFFile(mesh.comm, computed_file,
                             "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(solution_u, 0.0)
        solution_file.write_function(solution_p, 0.0)

"""
POD
"""
def generate_training_set(samples=[4, 4, 4]):
    # Todo: was sind das für Parameter?
    training_set_0 = np.linspace(1.0, 2.0, samples[0])
    training_set_1 = np.linspace(1.0, 2.0, samples[1])
    training_set_2 = np.linspace(np.pi/6, 5*np.pi/6, samples[2])
    training_set = np.array(list(itertools.product(training_set_0,
                                                   training_set_1,
                                                   training_set_2)))
    return training_set


training_set = rbnicsx.io.on_rank_zero(mesh.comm, generate_training_set)

Nmax = 30

print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
print("")

print("set up snapshots matrix")
V,_ = problem_parametric._W.sub(0).collapse()
Q,_ = problem_parametric._W.sub(1).collapse()
snapshots_matrix_u = rbnicsx.backends.FunctionsList(V)
snapshots_matrix_p = rbnicsx.backends.FunctionsList(Q)

print("set up reduced problem")
reduced_problem = PODANNReducedProblem(problem_parametric)

print("")

for (mu_index, mu) in enumerate(training_set):
    print(rbnicsx.io.TextLine(str(mu_index+1), fill="#"))

    print("Parameter number ", (mu_index+1), "of", training_set.shape[0])
    print("high fidelity solve for mu =", mu)
    (snapshot_u, snapshot_p) = problem_parametric.solve(mu)

    print("update snapshots matrix")
    snapshots_matrix_u.append(snapshot_u)
    snapshots_matrix_p.append(snapshot_p)

    print("")

print(rbnicsx.io.TextLine("perform POD", fill="#"))
eigenvalues_u, modes_u, _ = \
    rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_matrix_u,
                                    reduced_problem._inner_product_action_u,
                                    N=Nmax, tol=1.e-6)
eigenvalues_p, modes_p, _ = \
    rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_matrix_p,
                                    reduced_problem._inner_product_action_p,
                                    N=Nmax, tol=1.e-6)

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
plt.title("Eigenvalue decay u", fontsize=24)
plt.tight_layout()
plt.savefig("results/eigenvalue_decay_u.png")
# plt.show()

positive_eigenvalues_p = np.where(eigenvalues_p > 0., eigenvalues_p, np.nan)
singular_values_p = np.sqrt(positive_eigenvalues_p)

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
plt.title("Eigenvalue decay p", fontsize=24)
plt.tight_layout()
plt.savefig("results/eigenvalue_decay_p.png")
# plt.show()

print(f"Velocity reduced basis size: {len(reduced_problem._basis_functions_u)}")
print(f"Pressure reduced basis size: {len(reduced_problem._basis_functions_p)}")

"""
ANN
"""
def generate_ann_input_set(samples=[4, 4, 4]):
    """Generate an equispaced training set using numpy."""
    # TODO woher kommen diese Parameter
    training_set_0 = np.linspace(1.0, 2.0, samples[0])
    training_set_1 = np.linspace(1.0, 2.0, samples[1])
    training_set_2 = np.linspace(np.pi/6, np.pi/6*5, samples[2])
    training_set = np.array(list(itertools.product(training_set_0,
                                                   training_set_1,
                                                   training_set_2)))
    training_set = training_set.astype("f")
    return training_set


def generate_ann_output_set(problem, reduced_problem,
                            input_set, mode=None):
    output_set_u = np.zeros([input_set.shape[0], len(reduced_problem._basis_functions_u)])
    output_set_p = np.zeros([input_set.shape[0], len(reduced_problem._basis_functions_p)])
    rb_size_u = len(reduced_problem._basis_functions_u)
    rb_size_p = len(reduced_problem._basis_functions_p)
    for i in range(input_set.shape[0]):
        if mode is None:
            print(f"Parameter number {i+1} of {input_set.shape[0]}")
            print(f"Parameter: {input_set[i,:]}")
        else:
            print(f"{mode} parameter number {i+1} of {input_set.shape[0]}")
            print(f"Parameter: {input_set[i,:]}")
        (snapshot_u, snapshot_p) = problem.solve(input_set[i, :])
        output_set_u[i, :] = reduced_problem.project_snapshot(snapshot_u,rb_size_u).array #.astype("f")
        output_set_p[i, :] = reduced_problem.project_snapshot(snapshot_p,rb_size_p).array #.astype("f")
    return output_set_u, output_set_p


# Training dataset
ann_input_set = generate_ann_input_set(samples=[6, 6, 7])
np.random.shuffle(ann_input_set)
ann_output_set_u, ann_output_set_p = generate_ann_output_set(problem_parametric, reduced_problem,
                                         ann_input_set, mode="Training")

num_training_samples = int(0.7 * ann_input_set.shape[0])
num_validation_samples = ann_input_set.shape[0] - num_training_samples

input_training_set = ann_input_set[:num_training_samples, :]
output_training_set_u = ann_output_set_u[:num_training_samples, :]
output_training_set_p = ann_output_set_p[:num_training_samples, :]

input_validation_set = ann_input_set[num_training_samples:, :]
output_validation_set_u = ann_output_set_u[num_training_samples:, :]
output_validation_set_p = ann_output_set_p[num_training_samples:, :]

reduced_problem.output_range_u = (np.min(ann_output_set_u), np.max(ann_output_set_u))
reduced_problem.output_range_p = (np.min(ann_output_set_p), np.max(ann_output_set_p))

customDataset = CustomDataset(problem_parametric, reduced_problem,
                              len(reduced_problem._basis_functions_u),
                              input_training_set, output_training_set_u,
                              input_scaling_range = reduced_problem.input_scaling_range_u,
                              output_scaling_range = reduced_problem.output_scaling_range_u,
                              input_range = reduced_problem.input_range_u,
                              output_range = reduced_problem.output_range_u)
train_dataloader_u = DataLoader(customDataset, batch_size=30, shuffle=True)

customDataset = CustomDataset(problem_parametric, reduced_problem,
                                len(reduced_problem._basis_functions_p),
                                input_training_set, output_training_set_p,
                                input_scaling_range = reduced_problem.input_scaling_range_p,
                                output_scaling_range = reduced_problem.output_scaling_range_p,
                                input_range = reduced_problem.input_range_p,
                                output_range = reduced_problem.output_range_p)

train_dataloader_p = DataLoader(customDataset, batch_size=30, shuffle=True)

customDataset = CustomDataset(problem_parametric, reduced_problem,
                              len(reduced_problem._basis_functions_u),
                              input_validation_set, output_validation_set_u,
                              input_scaling_range = reduced_problem.input_scaling_range_u,
                              output_scaling_range = reduced_problem.output_scaling_range_u,
                              input_range = reduced_problem.input_range_u,
                              output_range = reduced_problem.output_range_u)

valid_dataloader_u = DataLoader(customDataset, shuffle=False)

customDataset = CustomDataset(problem_parametric, reduced_problem,
                                len(reduced_problem._basis_functions_p),
                                input_validation_set, output_validation_set_p,
                                input_scaling_range = reduced_problem.input_scaling_range_p,
                                output_scaling_range = reduced_problem.output_scaling_range_p,
                                input_range = reduced_problem.input_range_p,
                                output_range = reduced_problem.output_range_p)

valid_dataloader_p = DataLoader(customDataset, shuffle=False)

# ANN model
model_u = HiddenLayersNet(training_set.shape[1], [30, 30],
                        len(reduced_problem._basis_functions_u), Tanh())

model_p = HiddenLayersNet(training_set.shape[1], [30, 30],
                        len(reduced_problem._basis_functions_p), Tanh())

# Training of ANN
training_loss_u = list()
validation_loss_u = list()

max_epochs_u = 20000
min_validation_loss_u = None
for epochs in range(max_epochs_u):
    print(f"Epoch: {epochs+1}/{max_epochs_u}")
    current_training_loss = train_nn(reduced_problem, train_dataloader_u,
                                     model_u)
    training_loss_u.append(current_training_loss)
    current_validation_loss = validate_nn(reduced_problem, valid_dataloader_u,
                                          model_u)
    validation_loss_u.append(current_validation_loss)
    if epochs > 0 and current_validation_loss > 1.01 * min_validation_loss_u \
       and reduced_problem.regularisation_u == "EarlyStopping":
        # 1% safety margin against min_validation_loss
        # before invoking early stopping criteria
        print(f"Early stopping criteria invoked at epoch: {epochs+1}")
        break
    min_validation_loss_u = min(validation_loss_u)

training_loss_p = list()
validation_loss_p = list()

max_epochs_p = 20000
min_validation_loss_p = None
for epochs in range(max_epochs_p):
    print(f"Epoch: {epochs+1}/{max_epochs_p}")
    current_training_loss = train_nn(reduced_problem, train_dataloader_p,
                                     model_p)
    training_loss_p.append(current_training_loss)
    current_validation_loss = validate_nn(reduced_problem, valid_dataloader_p,
                                          model_p)
    validation_loss_p.append(current_validation_loss)
    if epochs > 0 and current_validation_loss > 1.01 * min_validation_loss_p \
       and reduced_problem.regularisation_p == "EarlyStopping":
        # 1% safety margin against min_validation_loss
        # before invoking early stopping criteria
        print(f"Early stopping criteria invoked at epoch: {epochs+1}")
        break
    min_validation_loss_p = min(validation_loss_p)

# Error analysis dataset
print("\n")
print("Generating error analysis (only input/parameters) dataset")
print("\n")
error_analysis_set_u = generate_ann_input_set(samples=[3, 3, 3])
error_numpy_u = np.zeros(error_analysis_set_u.shape[0])

for i in range(error_analysis_set_u.shape[0]):
    print(f"Error analysis {i+1} of {error_analysis_set_u.shape[0]}")
    print(f"Parameter: : {error_analysis_set_u[i,:]}")
    error_numpy_u[i] = error_analysis(reduced_problem, problem_parametric,
                                    error_analysis_set_u[i, :], model_u,
                                    len(reduced_problem._basis_functions_u),
                                    online_nn, device=None,
                                    norm_error = reduced_problem.norm_error_u,
                                    reconstruct_solution = reduced_problem.reconstruct_solution_u,
                                    input_scaling_range = reduced_problem.input_scaling_range_u,
                                    output_scaling_range = reduced_problem.output_scaling_range_u,
                                    input_range = reduced_problem.input_range_u,
                                    output_range = reduced_problem.output_range_u,
                                    index=0)
    
    print(f"Error: {error_numpy_u[i]}")

print("\n")
print("Generating error analysis (only input/parameters) dataset")
print("\n")
error_analysis_set_p = generate_ann_input_set(samples=[3, 3, 3])
error_numpy_p = np.zeros(error_analysis_set_p.shape[0])

for i in range(error_analysis_set_p.shape[0]):
    print(f"Error analysis {i+1} of {error_analysis_set_p.shape[0]}")
    print(f"Parameter: : {error_analysis_set_p[i,:]}")
    error_numpy_p[i] = error_analysis(reduced_problem, problem_parametric,
                                    error_analysis_set_p[i, :], model_p,
                                    len(reduced_problem._basis_functions_p),
                                    online_nn, device=None,
                                    norm_error = reduced_problem.norm_error_p,
                                    reconstruct_solution = reduced_problem.reconstruct_solution_p,
                                    input_scaling_range = reduced_problem.input_scaling_range_p,
                                    output_scaling_range = reduced_problem.output_scaling_range_p,
                                    input_range = reduced_problem.input_range_p,
                                    output_range = reduced_problem.output_range_p,
                                    index=1)
    
    print(f"Error: {error_numpy_p[i]}")

# Online phase at parameter online_mu
online_mu = np.array([1.0, 1.0, np.pi/2])
(solution_u, solution_p) = problem_parametric.solve(online_mu)
rb_solution_u = \
    reduced_problem.reconstruct_solution_u(
        online_nn(reduced_problem, problem_parametric, online_mu, model_u,
                  len(reduced_problem._basis_functions_u), device=None,
                  input_scaling_range = reduced_problem.input_scaling_range_u,
                  output_scaling_range = reduced_problem.output_scaling_range_u,
                  input_range = reduced_problem.input_range_u,
                  output_range = reduced_problem.output_range_u))

rb_solution_p = \
    reduced_problem.reconstruct_solution_p(
        online_nn(reduced_problem, problem_parametric, online_mu, model_p,
                    len(reduced_problem._basis_functions_p), device=None,
                    input_scaling_range = reduced_problem.input_scaling_range_p,
                    output_scaling_range = reduced_problem.output_scaling_range_p,
                    input_range = reduced_problem.input_range_p,
                    output_range = reduced_problem.output_range_p))



solution_velocity_error = dolfinx.fem.Function(V)
solution_pressure_error = dolfinx.fem.Function(Q)

solution_velocity_error.x.array[:] = abs(solution_u.x.array - rb_solution_u.x.array)

solution_pressure_error.x.array[:] = abs(solution_p.x.array - rb_solution_p.x.array)

solution_p.name = "Pressure"
solution_u.name = "Velocity"
rb_solution_u.name = "RB Velocity"
rb_solution_p.name = "RB Pressure"
solution_velocity_error.name = "Velocity Error"
solution_pressure_error.name = "Pressure Error"

with HarmonicMeshMotion(problem_parametric._mesh, problem_parametric._boundaries,
                        problem_parametric._boundary_markers, problem_parametric._bcs_geometric,
                        reset_reference=True, is_deformation=True):

    with dolfinx.io.XDMFFile(mesh.comm, "results/fem_v_online_mu.xdmf",
                             "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(solution_u)
        solution_file.write_function(solution_p)

    with dolfinx.io.XDMFFile(mesh.comm, "results/rb_online_mu.xdmf",
                             "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(rb_solution_u)
        solution_file.write_function(rb_solution_p)

    with dolfinx.io.XDMFFile(mesh.comm, "results/error_online_mu.xdmf",
                             "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(solution_velocity_error)
        solution_file.write_function(solution_pressure_error)
