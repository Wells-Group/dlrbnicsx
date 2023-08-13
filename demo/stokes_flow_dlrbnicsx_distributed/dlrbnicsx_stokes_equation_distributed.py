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
    init_cpu_process_group, get_optimiser, get_loss_func, share_model
from dlrbnicsx.train_validate_test.train_validate_test_distributed \
    import train_nn, validate_nn, online_nn, error_analysis


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
mu = np.array([0.93, 1.03])

problem_parametric = ProblemOnDeformedDomain(mesh, cell_tags, facet_tags,
                                             HarmonicMeshMotion)
solution_vel_mu, solution_pre_mu = problem_parametric.solve(mu)

itemsize = MPI.DOUBLE.Get_size()
para_dim = 2
num_dofs_u = solution_vel_mu.x.array.shape[0]
num_dofs_p = solution_pre_mu.x.array.shape[0]

pod_samples = [3, 3]
ann_samples = [3, 4]
error_analysis_samples = [4, 3]
num_snapshots = np.product(pod_samples)

if world_comm.rank == 0:
    nbytes_para = itemsize * num_snapshots * para_dim
    nbytes_dofs_u = itemsize * num_snapshots * num_dofs_u
    nbytes_dofs_p = itemsize * num_snapshots * num_dofs_p
else:
    nbytes_para = 0
    nbytes_dofs_u = 0
    nbytes_dofs_p = 0

# POD Starts ###


def generate_training_set(samples=[12, 12]):  # (samples=[6, 6]):
    # Select input samples for POD
    training_set_0 = np.linspace(0.5, 1., samples[0])
    training_set_1 = np.linspace(0.5, 1., samples[1])
    training_set = \
        np.array(list(itertools.product(training_set_0, training_set_1)))
    return training_set


win0 = MPI.Win.Allocate_shared(nbytes_para, itemsize, comm=MPI.COMM_WORLD)
buf0, itemsize = win0.Shared_query(0)
training_set = np.ndarray(buffer=buf0, dtype="d", shape=(num_snapshots, para_dim))

if world_comm.rank == 0:
    training_set[:, :] = generate_training_set(samples=pod_samples)

world_comm.Barrier()

win1 = MPI.Win.Allocate_shared(nbytes_dofs_u, itemsize, comm=MPI.COMM_WORLD)
buf1, itemsize = win1.Shared_query(0)
training_set_solution_u = np.ndarray(buffer=buf1, dtype="d", shape=(num_snapshots, num_dofs_u))

win2 = MPI.Win.Allocate_shared(nbytes_dofs_p, itemsize, comm=MPI.COMM_WORLD)
buf2, itemsize = win2.Shared_query(0)
training_set_solution_p = np.ndarray(buffer=buf2, dtype="d", shape=(num_snapshots, num_dofs_p))

# Solution manifold
indices = np.arange(world_comm.rank, num_snapshots, world_comm.size)

for i in indices:
    print(f"Solving FEM problem {i+1}/{num_snapshots}")
    (sol_u, sol_p) = problem_parametric.solve(training_set[i, :])
    training_set_solution_u[i, :] = sol_u.x.array
    training_set_solution_p[i, :] = sol_p.x.array

world_comm.Barrier()

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

for i in range(num_snapshots):
    snapshot_u = dolfinx.fem.Function(V)
    print(i, training_set_solution_u[i,:])
    snapshot_u.x.array[:] = training_set_solution_u[i, :]

    print(f"Update snapshots matrix: {i+1}/{num_snapshots}")
    snapshots_matrix_u.append(snapshot_u)

    snapshot_p = dolfinx.fem.Function(Q)
    snapshot_p.x.array[:] = training_set_solution_p[i, :]

    print(f"Update snapshots matrix: {i+1}/{num_snapshots}")
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

positive_eigenvalues_p = np.where(eigenvalues_p > 0., eigenvalues_p, np.nan)
singular_values_p = np.sqrt(positive_eigenvalues_p)

if world_comm.rank == 0:
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

print(f"Velocity eigenvalues: {positive_eigenvalues_u}")
print(f"Pressure eigenvalues: {positive_eigenvalues_p}")

# ### POD Ends ###

# Creating dataset


def generate_ann_input_set(samples=ann_samples):
    # Select samples from the parameter space for ANN
    training_set_0 = np.linspace(0.5, 1., samples[0])
    training_set_1 = np.linspace(0.5, 1., samples[1])
    training_set = np.array(list(itertools.product(training_set_0, training_set_1))).astype("f")
    return training_set


def generate_ann_output_set(problem, reduced_problem, input_set,
                            output_set_u, output_set_p,
                            indices, mode=None):
    # Compute output set for ANN based on input set
    rb_size_u = len(reduced_problem._basis_functions_u)
    rb_size_p = len(reduced_problem._basis_functions_p)
    for i in indices:
        if mode is None:
            print(f"Parameter {i+1}/{input_set.shape[0]}")
        else:
            print(f"{mode} parameter number {i+1}/{input_set.shape[0]}")
        (solution_u, solution_p) = problem.solve(input_set[i, :])
        output_set_u[i, :] = \
            reduced_problem.project_snapshot_u(solution_u, rb_size_u).array
        output_set_p[i, :] = \
            reduced_problem.project_snapshot_p(solution_p, rb_size_p).array

num_ann_input_samples = np.product(ann_samples)
num_training_samples = int(0.7 * num_ann_input_samples)
num_validation_samples = num_ann_input_samples - int(0.7 * num_ann_input_samples)
itemsize = MPI.DOUBLE.Get_size()

if world_comm.rank == 0:
    ann_input_set = generate_ann_input_set(samples=ann_samples)
    np.random.shuffle(ann_input_set)
    nbytes_para_ann_training = num_training_samples * itemsize * para_dim
    nbytes_dofs_ann_training_u = num_training_samples * itemsize * \
        len(reduced_problem._basis_functions_u)
    nbytes_dofs_ann_training_p = num_training_samples * itemsize * \
        len(reduced_problem._basis_functions_p)
    nbytes_para_ann_validation = num_validation_samples * itemsize * para_dim
    nbytes_dofs_ann_validation_u = num_validation_samples * itemsize * \
        len(reduced_problem._basis_functions_u)
    nbytes_dofs_ann_validation_p = num_validation_samples * itemsize * \
        len(reduced_problem._basis_functions_p)
else:
    nbytes_para_ann_training = 0
    nbytes_dofs_ann_training_u = 0
    nbytes_dofs_ann_training_p = 0
    nbytes_para_ann_validation = 0
    nbytes_dofs_ann_validation_u = 0
    nbytes_dofs_ann_validation_p = 0

world_comm.barrier()

win3 = MPI.Win.Allocate_shared(nbytes_para_ann_training, itemsize,
                               comm=MPI.COMM_WORLD)
buf3, itemsize = win3.Shared_query(0)
input_training_set = \
    np.ndarray(buffer=buf3, dtype="d",
               shape=(num_training_samples, para_dim))

win4 = MPI.Win.Allocate_shared(nbytes_para_ann_validation, itemsize,
                               comm=MPI.COMM_WORLD)
buf4, itemsize = win4.Shared_query(0)
input_validation_set = \
    np.ndarray(buffer=buf4, dtype="d",
               shape=(num_validation_samples, para_dim))

win5 = MPI.Win.Allocate_shared(nbytes_dofs_ann_training_u, itemsize,
                               comm=MPI.COMM_WORLD)
buf5, itemsize = win5.Shared_query(0)
output_training_set_u = \
    np.ndarray(buffer=buf5, dtype="d",
               shape=(num_training_samples,
                      len(reduced_problem._basis_functions_u)))

win6 = MPI.Win.Allocate_shared(nbytes_dofs_ann_validation_u, itemsize,
                               comm=MPI.COMM_WORLD)
buf6, itemsize = win6.Shared_query(0)
output_validation_set_u = \
    np.ndarray(buffer=buf6, dtype="d",
               shape=(num_validation_samples,
                      len(reduced_problem._basis_functions_u)))

win7 = MPI.Win.Allocate_shared(nbytes_dofs_ann_training_p, itemsize,
                               comm=MPI.COMM_WORLD)
buf7, itemsize = win7.Shared_query(0)
output_training_set_p = \
    np.ndarray(buffer=buf7, dtype="d",
               shape=(num_training_samples,
                      len(reduced_problem._basis_functions_p)))

win8 = MPI.Win.Allocate_shared(nbytes_dofs_ann_validation_p, itemsize,
                               comm=MPI.COMM_WORLD)
buf8, itemsize = win8.Shared_query(0)
output_validation_set_p = \
    np.ndarray(buffer=buf8, dtype="d",
               shape=(num_validation_samples,
                      len(reduced_problem._basis_functions_p)))

if world_comm.rank == 0:
    input_training_set[:, :] = \
        ann_input_set[:num_training_samples, :]
    input_validation_set[:, :] = \
        ann_input_set[num_training_samples:, :]
    output_training_set_u[:, :] = \
        np.zeros([num_training_samples,
                  len(reduced_problem._basis_functions_u)])
    output_validation_set_u[:, :] = \
        np.zeros([num_validation_samples,
                  len(reduced_problem._basis_functions_u)])
    output_training_set_p[:, :] = \
        np.zeros([num_training_samples,
                  len(reduced_problem._basis_functions_p)])
    output_validation_set_p[:, :] = \
        np.zeros([num_validation_samples,
                  len(reduced_problem._basis_functions_p)])

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
                        input_training_set, output_training_set_u,
                        output_training_set_p, training_set_indices,
                        mode="Training")

generate_ann_output_set(problem_parametric, reduced_problem,
                        input_validation_set, output_validation_set_u,
                        output_validation_set_p, validation_set_indices,
                        mode="Validation")

world_comm.Barrier()

reduced_problem.output_range_u[0] = min(np.min(output_training_set_u), np.min(output_validation_set_u))
reduced_problem.output_range_u[1] = max(np.max(output_training_set_u), np.max(output_validation_set_u))
reduced_problem.output_range_p[0] = min(np.min(output_training_set_p), np.min(output_validation_set_p))
reduced_problem.output_range_p[1] = max(np.max(output_training_set_p), np.max(output_validation_set_p))

print("\n")

cpu_group0_procs = world_comm.group.Incl([0, 1])
cpu_group0_comm = world_comm.Create_group(cpu_group0_procs)

cpu_group1_procs = world_comm.group.Incl([2, 3])
cpu_group1_comm = world_comm.Create_group(cpu_group1_procs)

# ANN model
model_u = HiddenLayersNet(input_training_set.shape[1], [30, 30],
                          len(reduced_problem._basis_functions_u), Tanh())
model_p = HiddenLayersNet(input_training_set.shape[1], [15, 15],
                          len(reduced_problem._basis_functions_p), Tanh())

if cpu_group0_comm != MPI.COMM_NULL:
    init_cpu_process_group(cpu_group0_comm)

    training_set_indices_cpu_u = np.arange(cpu_group0_comm.rank,
                                           input_training_set.shape[0],
                                           cpu_group0_comm.size)
    validation_set_indices_cpu_u = np.arange(cpu_group0_comm.rank,
                                             input_validation_set.shape[0],
                                             cpu_group0_comm.size)

    customDataset = CustomPartitionedDataset(reduced_problem, input_training_set,
                                             output_training_set_u, training_set_indices_cpu_u,
                                             input_scaling_range=reduced_problem.input_scaling_range_u,
                                             output_scaling_range=reduced_problem.output_scaling_range_u,
                                             input_range=reduced_problem.input_range_u,
                                             output_range=reduced_problem.output_range_u, verbose=True)
    train_dataloader_u = DataLoader(customDataset, batch_size=3, shuffle=False)# shuffle=True)

    customDataset = CustomPartitionedDataset(reduced_problem, input_validation_set,
                                             output_validation_set_u, validation_set_indices_cpu_u,
                                             input_scaling_range=reduced_problem.input_scaling_range_u,
                                             output_scaling_range=reduced_problem.output_scaling_range_u,
                                             input_range=reduced_problem.input_range_u,
                                             output_range=reduced_problem.output_range_u, verbose=True)
    valid_dataloader_u = DataLoader(customDataset, shuffle=False)

    path = "model_u.pth"
    # save_model(model_u, path)
    load_model(model_u, path)

    model_synchronise(model_u, verbose=True)

    # Training of ANN
    training_loss_u = list()
    validation_loss_u = list()

    max_epochs_u = 1 # 50 # 20000
    min_validation_loss_u = None
    start_epoch = 0
    checkpoint_path_u = "checkpoint_u"
    checkpoint_epoch_u = 10

    learning_rate_u = 5.e-6
    optimiser_u = get_optimiser(model_u, "Adam", learning_rate_u)
    loss_fn_u = get_loss_func("MSE", reduction="sum")

    if os.path.exists(checkpoint_path_u):
        start_epoch, min_validation_loss_u = \
            load_checkpoint(checkpoint_path_u, model_u, optimiser_u)

    import time
    start_time = time.time()
    for epochs in range(start_epoch, max_epochs_u):
        if epochs > 0 and epochs % checkpoint_epoch_u == 0:
            save_checkpoint(checkpoint_path_u, epochs, model_u, optimiser_u,
                            min_validation_loss_u)
        print(f"Epoch: {epochs+1}/{max_epochs_u}")
        current_training_loss = train_nn(reduced_problem, train_dataloader_u,
                                         model_u, loss_fn_u, optimiser_u)
        training_loss_u.append(current_training_loss)
        current_validation_loss = validate_nn(reduced_problem,
                                              valid_dataloader_u,
                                              model_u, loss_fn_u)
        validation_loss_u.append(current_validation_loss)
        if epochs > 0 and current_validation_loss > min_validation_loss_u \
        and reduced_problem.regularisation_u == "EarlyStopping":
            # 1% safety margin against min_validation_loss
            # before invoking early stopping criteria
            print(f"Early stopping criteria invoked at epoch: {epochs+1}")
            break
        min_validation_loss_u = min(validation_loss_u)
    end_time = time.time()
    elapsed_time_u = end_time - start_time

exit()

model_root_process = 0
share_model(model_u, world_comm, model_root_process)

if cpu_group1_comm != MPI.COMM_NULL:
    init_cpu_process_group(cpu_group1_comm)

    training_set_indices_cpu_p = np.arange(cpu_group1_comm.rank,
                                           input_training_set.shape[0],
                                           cpu_group1_comm.size)
    validation_set_indices_cpu_p = np.arange(cpu_group1_comm.rank,
                                             input_validation_set.shape[0],
                                             cpu_group1_comm.size)

    customDataset = CustomPartitionedDataset(reduced_problem, input_training_set,
                                             output_training_set_p, training_set_indices_cpu_p,
                                             input_scaling_range=reduced_problem.input_scaling_range_p,
                                             output_scaling_range=reduced_problem.output_scaling_range_p,
                                             input_range=reduced_problem.input_range_p,
                                             output_range=reduced_problem.output_range_p, verbose=True)
    train_dataloader_p = DataLoader(customDataset, batch_size=3, shuffle=False)# shuffle=True)

    customDataset = CustomPartitionedDataset(reduced_problem, input_validation_set,
                                             output_validation_set_p, validation_set_indices_cpu_p,
                                             input_scaling_range=reduced_problem.input_scaling_range_p,
                                             output_scaling_range=reduced_problem.output_scaling_range_p,
                                             input_range=reduced_problem.input_range_p,
                                             output_range=reduced_problem.output_range_p, verbose=True)
    valid_dataloader_p = DataLoader(customDataset, shuffle=False)

    path = "model_p.pth"
    # save_model(model_p, path)
    load_model(model_p, path)

    model_synchronise(model_p, verbose=True)

    # Training of ANN
    training_loss_p = list()
    validation_loss_p = list()

    max_epochs_p = 1 # 50 # 20000
    min_validation_loss_p = None
    start_epoch = 0
    checkpoint_path_p = "checkpoint_p"
    checkpoint_epoch_p = 10

    learning_rate_p = 5.e-6
    optimiser_p = get_optimiser(model_p, "Adam", learning_rate_p)
    loss_fn_p = get_loss_func("MSE", reduction="sum")

    if os.path.exists(checkpoint_path_p):
        start_epoch, min_validation_loss_p = \
            load_checkpoint(checkpoint_path_p, model_p, optimiser_p)

    import time
    start_time = time.time()
    for epochs in range(start_epoch, max_epochs_p):
        if epochs > 0 and epochs % checkpoint_epoch_p == 0:
            save_checkpoint(checkpoint_path_p, epochs, model_p, optimiser_p,
                            min_validation_loss_p)
        print(f"Epoch: {epochs+1}/{max_epochs_p}")
        current_training_loss = train_nn(reduced_problem, train_dataloader_p,
                                         model_p, loss_fn_p, optimiser_p)
        training_loss_p.append(current_training_loss)
        current_validation_loss = validate_nn(reduced_problem,
                                              valid_dataloader_p,
                                              model_p, loss_fn_p)
        validation_loss_p.append(current_validation_loss)
        if epochs > 0 and current_validation_loss > min_validation_loss_p \
        and reduced_problem.regularisation_p == "EarlyStopping":
            # 1% safety margin against min_validation_loss
            # before invoking early stopping criteria
            print(f"Early stopping criteria invoked at epoch: {epochs+1}")
            break
        min_validation_loss_p = min(validation_loss_p)
    end_time = time.time()
    elapsed_time_p = end_time - start_time



model_root_process = 0
share_model(model_p, world_comm, model_root_process)

if cpu_group0_comm != MPI.COMM_NULL:
    os.system(f"rm {checkpoint_path_u}")

if cpu_group1_comm != MPI.COMM_NULL:
    os.system(f"rm {checkpoint_path_p}")
exit()
# Error analysis dataset
print("\n")
print("Generating error analysis (only input/parameters) dataset")
print("\n")
if rank == 0:
    error_analysis_set_u = generate_ann_input_set(samples=error_analysis_samples)
else:
    error_analysis_set_u = np.zeros_like(generate_ann_input_set(samples=error_analysis_samples))

world_comm.Bcast(error_analysis_set_u, root=0)

world_comm.Barrier()
error_numpy_u = np.zeros(error_analysis_set_u.shape[0])
error_numpy_recv_u = np.zeros(error_analysis_set_u.shape[0])
indices = np.arange(rank, error_analysis_set_u.shape[0], size)

for i in indices:
    print(f"Error analysis parameter number {i+1} of ")
    print(f"{error_analysis_set_u.shape[0]}: {error_analysis_set_u[i, :]}")
    error_numpy_u[i] = error_analysis(reduced_problem, problem_parametric,
                                      error_analysis_set_u[i, :], model_u,
                                      len(reduced_problem._basis_functions_u),
                                      online_nn,
                                      norm_error=reduced_problem.norm_error_u,
                                      reconstruct_solution=reduced_problem.reconstruct_solution_u,
                                      input_scaling_range=reduced_problem.input_scaling_range_u,
                                      output_scaling_range=reduced_problem.output_scaling_range_u,
                                      input_range=reduced_problem.input_range_u,
                                      output_range=reduced_problem.output_range_u,
                                      index=0)
    print(f"Error: {error_numpy_u[i]}")

world_comm.Barrier()
world_comm.Allreduce(error_numpy_u, error_numpy_recv_u, op=MPI.SUM)

# Error analysis dataset
print("\n")
print("Generating error analysis (only input/parameters) dataset")
print("\n")
if rank == 0:
    error_analysis_set_p = generate_ann_input_set(samples=error_analysis_samples)
else:
    error_analysis_set_p = np.zeros_like(generate_ann_input_set(samples=error_analysis_samples))

world_comm.Bcast(error_analysis_set_p, root=0)

world_comm.Barrier()
error_numpy_p = np.zeros(error_analysis_set_p.shape[0])
error_numpy_recv_p = np.zeros(error_analysis_set_p.shape[0])
indices = np.arange(rank, error_analysis_set_p.shape[0], size)

for i in indices:
    print(f"Error analysis parameter number {i+1} of ")
    print(f"{error_analysis_set_p.shape[0]}: {error_analysis_set_p[i,:]}")
    error_numpy_p[i] = error_analysis(reduced_problem, problem_parametric,
                                      error_analysis_set_p[i, :], model_p,
                                      len(reduced_problem._basis_functions_p),
                                      online_nn,
                                      norm_error=reduced_problem.norm_error_p,
                                      reconstruct_solution=reduced_problem.reconstruct_solution_p,
                                      input_scaling_range=reduced_problem.input_scaling_range_p,
                                      output_scaling_range=reduced_problem.output_scaling_range_p,
                                      input_range=reduced_problem.input_range_p,
                                      output_range=reduced_problem.output_range_p,
                                      index=1)
    print(f"Error: {error_numpy_p[i]}")

world_comm.Barrier()
world_comm.Allreduce(error_numpy_p, error_numpy_recv_p, op=MPI.SUM)

# Online phase

if rank == 0:
    # Define a parameter
    online_mu = np.array([0.8, 0.9])

    # Compute FEM solution
    (solution_u, solution_p) = problem_parametric.solve(online_mu)

    solution_p1 = dolfinx.fem.Function(Q)
    solution_p2 = dolfinx.fem.Function(Q)
    solution_p1.x.array[:] = -solution_p.x.array

    # Compute RB solution
    rb_solution_u = \
        reduced_problem.reconstruct_solution_u(online_nn(reduced_problem,
                                                         problem_parametric,
                                                         online_mu, model_u,
                                                         len(reduced_problem._basis_functions_u),
                                                         input_scaling_range=reduced_problem.input_scaling_range_u,
                                                         output_scaling_range=reduced_problem.output_scaling_range_u,
                                                         input_range=reduced_problem.input_range_u,
                                                         output_range=reduced_problem.output_range_u))
    rb_solution_p = \
        reduced_problem.reconstruct_solution_p(online_nn(reduced_problem,
                                                         problem_parametric,
                                                         online_mu, model_p,
                                                         len(reduced_problem._basis_functions_p),
                                                         input_scaling_range=reduced_problem.input_scaling_range_p,
                                                         output_scaling_range=reduced_problem.output_scaling_range_p,
                                                         input_range=reduced_problem.input_range_p,
                                                         output_range=reduced_problem.output_range_p))

    solution_p2.x.array[:] = -rb_solution_p.x.array

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
            # solution_file.write_function(solution_p)
            solution_file.write_function(solution_p1)

        with dolfinx.io.XDMFFile(mesh.comm, "dlrbnicsx_solution/rb_velocity_online_mu.xdmf",
                                 "w") as solution_file:
            solution_file.write_mesh(mesh)
            solution_file.write_function(rb_solution_u)

        with dolfinx.io.XDMFFile(mesh.comm, "dlrbnicsx_solution/rb_pressure_online_mu.xdmf",
                                 "w") as solution_file:
            solution_file.write_mesh(mesh)
            # solution_file.write_function(rb_solution_p)
            solution_file.write_function(solution_p2)

        with dolfinx.io.XDMFFile(mesh.comm, "dlrbnicsx_solution/error_velocity_online_mu.xdmf",
                                 "w") as solution_file:
            solution_file.write_mesh(mesh)
            solution_file.write_function(solution_velocity_error)

        with dolfinx.io.XDMFFile(mesh.comm, "dlrbnicsx_solution/error_pressure_online_mu.xdmf",
                                 "w") as solution_file:
            solution_file.write_mesh(mesh)
            solution_file.write_function(solution_pressure_error)
