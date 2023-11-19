import abc

import ufl
import dolfinx
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
import itertools
import matplotlib.pyplot as plt

import rbnicsx
import rbnicsx.backends
import rbnicsx.online

from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion


class CustomMeshDeformation(HarmonicMeshMotion):
    def __init__(self, mesh, boundaries, boundary_markers,
                 bc_function_list, mu, reset_reference=True,
                 is_deformation=True):
        super().__init__(mesh, boundaries, boundary_markers,
                         bc_function_list, reset_reference,
                         is_deformation)
        self.mu = mu

    def __enter__(self):
        gdim = self._mesh.geometry.dim
        mu = self.mu
        self.shape_parametrization = self.solve()
        self._mesh.geometry.x[:, 0] += \
            (mu[2] - 1) * self._mesh.geometry.x[:, 0]
        self._mesh.geometry.x[:, :gdim] += \
            self.shape_parametrization.x.array.\
            reshape(self._reference_coordinates.shape[0], gdim)
        self._mesh.geometry.x[:, 0] -= \
           self._mesh.comm.allreduce(min(self._mesh.geometry.x[:, 0]), op=MPI.MIN)


class ProblemOnDeformedDomain(abc.ABC):
    def __init__(self, mesh, subdomains, boundaries, meshDeformationContext):
        self._mesh = mesh
        self._subdomains = subdomains
        self._boundaries = boundaries
        self._boundary_markers = [1, 2, 3, 4]
        self.gdim = mesh.geometry.dim
        self._meshDeformationContext = meshDeformationContext
        self._V = dolfinx.fem.FunctionSpace(self._mesh, ("CG", 2))
        self._dirichletFunc = dolfinx.fem.Function(self._V)
        self.trial, self.test = ufl.TrialFunction(self._V), ufl.TestFunction(self._V)
        u, v = self.trial, self.test
        self._inner_product = ufl.inner(u, v) * ufl.dx + \
            ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        self._inner_product_action = \
            rbnicsx.backends.bilinear_form_action(self._inner_product,
                                                  part="real")
        self._solution = dolfinx.fem.Function(self._V)

    @property
    def assemble_bcs(self):
        bcs = list()
        for i in self._boundary_markers:
            dofs = \
                dolfinx.fem.locate_dofs_topological(
                    self._V, self.gdim - 1, self._boundaries.find(i))
            bcs.append(dolfinx.fem.dirichletbc(self._dirichletFunc, dofs))
        return bcs

    @property
    def source_term(self):
        return -ufl.div(ufl.exp(self._dirichletFunc) *
                        ufl.grad(self._dirichletFunc))

    @property
    def residual_term(self):
        return ufl.inner(ufl.exp(self._solution) * ufl.grad(self._solution),
                         ufl.grad(self.test)) * ufl.dx - \
            ufl.inner(self.source_term, self.test) * ufl.dx

    @property
    def set_problem(self):
        problemNonlinear = \
            NonlinearProblem(self.residual_term, self._solution,
                             bcs=self.assemble_bcs)
        return problemNonlinear

    def solve(self, mu):
        self._bcs_geometric = \
            [lambda x: (0. * x[0], mu[0] * np.sin(x[0] * np.pi)),
             lambda x: (0. * x[0], 0. * x[1]),
             lambda x: (0 * x[1], mu[1] * np.sin(x[0] * np.pi)),
             lambda x: (0. * x[0], 0. * x[1])]
        problemNonlinear = self.set_problem
        solution = dolfinx.fem.Function(self._V)
        with self._meshDeformationContext(self._mesh, self._boundaries,
                                          self._boundary_markers,
                                          self._bcs_geometric, mu):
            solver = NewtonSolver(self._mesh.comm, problemNonlinear)
            solver.convergence_criterion = "incremental"
            solver.rtol = 1.e-6
            solver.report = True
            ksp = solver.krylov_solver
            ksp.setFromOptions()
            self._dirichletFunc.interpolate(lambda x:
                                            x[1] * np.sin(x[0] * np.pi)
                                            * np.cos(x[1] * np.pi))
            n, converged = solver.solve(self._solution)
            assert (converged)
            solution.x.array[:] = self._solution.x.array.copy()
            # print(f"Computed solution array: {solution.x.array}")
            print(f"Number of iterations: {n:d}")
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
        self.problem = problem
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
        #self.loss_fn = "MSE"
        #self.learning_rate = 1e-5
        #self.optimizer = "Adam"
        self.regularisation = "EarlyStopping"

    '''
    def _inner_product_action(self, fun_j):
        def _(fun_i):
            return fun_i.vector.dot(fun_j.vector)
        return _
    '''

    def reconstruct_solution(self, reduced_solution):
        """Reconstructed reduced solution on the high fidelity space."""
        return self._basis_functions[:reduced_solution.size] * \
            reduced_solution

    def compute_norm(self, function):
        """Compute the norm of a function inner product
        on the reference domain."""
        return np.sqrt(self._inner_product_action(function)(function))
        # return np.sqrt(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(function, function) * ufl.dx +\
        #     ufl.inner(ufl.grad(function), ufl.grad(function)) * ufl.dx)))

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
        absolute_error = dolfinx.fem.Function(self.problem._V)
        absolute_error.x.array[:] = u.x.array - v.x.array
        return self.compute_norm(absolute_error)/self.compute_norm(u) # self.compute_norm(u-v)/self.compute_norm(u)

# Read unit square mesh with Triangular elements
world_comm = MPI.COMM_WORLD
world_rank = world_comm.rank
world_size = world_comm.size

if world_comm.size == 8:
    group0_procs = world_comm.group.Incl([0, 1, 2, 3])
    gpu_group0_comm = world_comm.Create_group(group0_procs)

    group1_procs = world_comm.group.Incl([4, 5, 6, 7])
    gpu_group1_comm = world_comm.Create_group(group1_procs)

    gpu_comm_list = [gpu_group0_comm, gpu_group1_comm]

    for i in range(len(gpu_comm_list)):
        if gpu_comm_list[i] != MPI.COMM_NULL:
            print(f"Process {world_rank}, gpu comm list: {i}")

    if gpu_group0_comm != MPI.COMM_NULL:

        fem0_procs = gpu_group0_comm.group.Incl([0, 1])
        fem0_procs_comm = gpu_group0_comm.Create_group(fem0_procs)

        fem1_procs = gpu_group0_comm.group.Incl([2, 3])
        fem1_procs_comm = gpu_group0_comm.Create_group(fem1_procs)

        cuda_num = 0

    if gpu_group1_comm != MPI.COMM_NULL:

        fem2_procs = gpu_group1_comm.group.Incl([0, 1])
        fem2_procs_comm = gpu_group1_comm.Create_group(fem2_procs)

        fem3_procs = gpu_group1_comm.group.Incl([2, 3])
        fem3_procs_comm = gpu_group1_comm.Create_group(fem3_procs)

        cuda_num = 1
elif world_comm.size == 4:
    group0_procs = world_comm.group.Incl([0, 1])
    gpu_group0_comm = world_comm.Create_group(group0_procs)

    group1_procs = world_comm.group.Incl([2, 3])
    gpu_group1_comm = world_comm.Create_group(group1_procs)

    gpu_comm_list = [gpu_group0_comm, gpu_group1_comm]

    for i in range(len(gpu_comm_list)):
        if gpu_comm_list[i] != MPI.COMM_NULL:
            print(f"Process {world_rank}, gpu comm list: {i}")

    if gpu_group0_comm != MPI.COMM_NULL:

        fem0_procs = gpu_group0_comm.group.Incl([0])
        fem0_procs_comm = gpu_group0_comm.Create_group(fem0_procs)

        fem1_procs = gpu_group0_comm.group.Incl([1])
        fem1_procs_comm = gpu_group0_comm.Create_group(fem1_procs)

        cuda_num = 0

    if gpu_group1_comm != MPI.COMM_NULL:

        fem2_procs = gpu_group1_comm.group.Incl([0])
        fem2_procs_comm = gpu_group1_comm.Create_group(fem2_procs)

        fem3_procs = gpu_group1_comm.group.Incl([1])
        fem3_procs_comm = gpu_group1_comm.Create_group(fem3_procs)

        cuda_num = 1
elif world_comm.size == 1:
    gpu_group0_comm = MPI.COMM_NULL
    gpu_group1_comm = MPI.COMM_NULL
    mu = np.array([0.2, -0.2, 1.])
    mesh_comm = MPI.COMM_WORLD
    cuda_num = 0
    gpu_comm_list = [gpu_group0_comm, gpu_group1_comm]
    fem_comm_list = [MPI.COMM_WORLD]

else:
    raise NotImplementedError("Please use 1, 4 or 8 processes")

if gpu_group0_comm != MPI.COMM_NULL:
    if fem0_procs_comm != MPI.COMM_NULL:
        print(
            f"Group 0: World rank {world_rank}, FEM 0 rank {fem0_procs_comm.rank}, Cuda number {cuda_num}")
        mu = np.array([0.2, -0.2, 1.])
        mesh_comm = fem0_procs_comm

    if fem1_procs_comm != MPI.COMM_NULL:
        print(
            f"Group 0: World rank {world_rank}, FEM 1 rank {fem1_procs_comm.rank}, Cuda number {cuda_num}")
        mu = np.array([0.2, -0.2, 1.])  # np.array([0.3, -0.25, 1.5])
        mesh_comm = fem1_procs_comm

    fem_comm_list = [fem0_procs_comm, fem1_procs_comm]

if gpu_group1_comm != MPI.COMM_NULL:
    if fem2_procs_comm != MPI.COMM_NULL:
        print(
            f"Group 1: World rank {world_rank}, FEM 2 rank {fem2_procs_comm.rank}, Cuda number {cuda_num}")
        mu = np.array([0.2, -0.2, 1.])  # np.array([0.28, -0.3, 2.3])
        mesh_comm = fem2_procs_comm
    if fem3_procs_comm != MPI.COMM_NULL:
        print(
            f"Group 1: World rank {world_rank}, FEM 3 rank {fem3_procs_comm.rank}, Cuda number {cuda_num}")
        mu = np.array([0.2, -0.2, 1.])  # np.array([0.22, -0.4, 3.])
        mesh_comm = fem3_procs_comm

    fem_comm_list = [fem2_procs_comm, fem3_procs_comm]

gdim = 2
gmsh_model_rank = 0
mesh, cell_tags, facet_tags = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh",
                                    mesh_comm, gmsh_model_rank, gdim=gdim)

problem_parametric = ProblemOnDeformedDomain(mesh, cell_tags, facet_tags,
                                  CustomMeshDeformation)
solution = problem_parametric.solve(mu)

itemsize = MPI.DOUBLE.Get_size()
num_snapshots = 64
para_dim = 3


def generate_training_set(samples=[4, 4, 4]):
    training_set_0 = np.linspace(0.2, 0.3, samples[0])
    training_set_1 = np.linspace(-0.2, -0.3, samples[1])
    training_set_2 = np.linspace(1., 4., samples[2])
    training_set = np.array(list(itertools.product(training_set_0,
                                                   training_set_1,
                                                   training_set_2)))
    return training_set


if world_comm.rank == 0:
    nbytes = num_snapshots * para_dim * itemsize
else:
    nbytes = 0

win0 = MPI.Win.Allocate_shared(nbytes, itemsize, comm=MPI.COMM_WORLD)
buf0, itemsize = win0.Shared_query(0)
para_matrix = np.ndarray(buffer=buf0, dtype="d", shape=(num_snapshots, para_dim))

if world_comm.rank == 0:
    para_matrix[:, :] = generate_training_set()

world_comm.barrier()

# print(f"Rank {world_comm.rank}, Para matrix {para_matrix}")

for i in range(len(gpu_comm_list)):
    if gpu_comm_list[i] != MPI.COMM_NULL:
        gpu_indices = np.arange(i, para_matrix.shape[0], len(gpu_comm_list))
        for j in range(len(fem_comm_list)):
            if fem_comm_list[j] != MPI.COMM_NULL:
                cpu_indices = gpu_indices[np.arange(j, gpu_indices.shape[0], len(fem_comm_list))]

if world_comm.size == 1:
    cpu_indices = np.arange(0, para_matrix.shape[0])

func_empty = dolfinx.fem.Function(problem_parametric._V)
rstart, rend = func_empty.vector.getOwnershipRange()
num_dofs = mesh.comm.allreduce(rend, op=MPI.MAX) - mesh.comm.allreduce(rstart, op=MPI.MIN)

if world_comm.rank == 0:
    nbytes = num_snapshots * num_dofs * itemsize
else:
    nbytes = 0

world_comm.barrier()

win0 = MPI.Win.Allocate_shared(nbytes, itemsize, comm=MPI.COMM_WORLD)
buf0, itemsize = win0.Shared_query(0)
snapshot_arrays = np.ndarray(buffer=buf0, dtype="d", shape=(num_snapshots, num_dofs))
snapshots_matrix = rbnicsx.backends.FunctionsList(problem_parametric._V)
Nmax = 10

for j in range(len(fem_comm_list)):
    if fem_comm_list[j] != MPI.COMM_NULL:
        for mu_index in cpu_indices:
            solution = problem_parametric.solve(para_matrix[mu_index, :])
            rstart, rend = solution.vector.getOwnershipRange()
            snapshot_arrays[mu_index, rstart:rend] = solution.vector[rstart:rend]

world_comm.barrier()

for j in range(len(fem_comm_list)):
    if fem_comm_list[j] != MPI.COMM_NULL:
        for i in range(snapshot_arrays.shape[0]):
            solution_empty = dolfinx.fem.Function(problem_parametric._V)
            rstart, rend = solution_empty.vector.getOwnershipRange()
            solution_empty.vector[rstart:rend] = snapshot_arrays[i, rstart:rend]
            solution_empty.x.scatter_forward()
            solution_empty.vector.assemble()
            snapshots_matrix.append(solution_empty)

print("Set up reduced problem")
reduced_problem = PODANNReducedProblem(problem_parametric)

print("")

print(rbnicsx.io.TextLine("Perform POD", fill="#"))

eigenvalues, modes, _ = \
    rbnicsx.backends.proper_orthogonal_decomposition(
        snapshots_matrix, reduced_problem._inner_product_action, N=Nmax, tol=1e-6)

# print(f"My Rank {world_comm.rank}, Eigenvalues: {eigenvalues[:Nmax]}")

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

print(f"Eigenvalues: {positive_eigenvalues}")
