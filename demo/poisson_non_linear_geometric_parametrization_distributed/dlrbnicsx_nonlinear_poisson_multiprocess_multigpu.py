import abc
import itertools
# import typing
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

import ufl
import dolfinx
# import mdfenicsx

from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion

import rbnicsx
import rbnicsx.backends
import rbnicsx.online

# import dlrbnicsx

from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory import Tanh
from dlrbnicsx.dataset.custom_partitioned_dataset \
    import CustomPartitionedDataset
from dlrbnicsx.dataset.snapshot_indices_split \
    import parameter_dataset_split, snapshot_matrix_write
from dlrbnicsx.interface.wrappers import DataLoader
from dlrbnicsx.train_validate_test.train_validate_test_distributed \
    import train_nn, validate_nn, online_nn, error_analysis

import matplotlib.pyplot as plt
import torch.distributed as dist  # TODO

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
            #print(f"Parameter {mu}, \n Computed solution norm: {np.sqrt(self._inner_product_action(solution)(solution))}")# Computed solution array: {solution.x.array}")
            print(f"Number of interations: {n:d}")
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
        """Reconstructed reduced solution on the high fidelity space."""
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


def generate_training_set(samples=[2, 2, 4]):
    training_set_0 = np.linspace(0.2, 0.3, samples[0])
    training_set_1 = np.linspace(-0.2, -0.4, samples[1])
    training_set_2 = np.linspace(1., 4., samples[2])
    training_set = np.array(list(itertools.product(training_set_0,
                                                   training_set_1,
                                                   training_set_2)))
    return training_set


global_comm = MPI.COMM_WORLD

if global_comm.size == 8:
    group0_procs = global_comm.group.Incl([0, 1])
    group0_comm = global_comm.Create_group(group0_procs)

    group1_procs = global_comm.group.Incl([2, 3])
    group1_comm = global_comm.Create_group(group1_procs)

    group2_procs = global_comm.group.Incl([4, 5])
    group2_comm = global_comm.Create_group(group2_procs)

    group3_procs = global_comm.group.Incl([6, 7])
    group3_comm = global_comm.Create_group(group3_procs)

    comm_list = [group0_comm, group1_comm, group2_comm, group3_comm]
elif global_comm.size == 1:
    comm_list = [global_comm.Create_group(global_comm.group.Incl([0]))]
elif global_comm.size == 4:
    group0_procs = global_comm.group.Incl([0, 1])
    group0_comm = global_comm.Create_group(group0_procs)

    group1_procs = global_comm.group.Incl([2, 3])
    group1_comm = global_comm.Create_group(group1_procs)
    
    comm_list = [group0_comm, group1_comm]
elif global_comm.size == 6:
    group0_procs = global_comm.group.Incl([0, 1])
    group0_comm = global_comm.Create_group(group0_procs)

    group1_procs = global_comm.group.Incl([2, 3])
    group1_comm = global_comm.Create_group(group1_procs)

    group2_procs = global_comm.group.Incl([4, 5])
    group2_comm = global_comm.Create_group(group2_procs)

    comm_list = [group0_comm, group1_comm, group2_comm]

global_comm.Barrier()

# Read mesh
for i in range(len(comm_list)):
    if comm_list[i] != MPI.COMM_NULL:
        mesh_comm = comm_list[i]
        gdim = 2
        gmsh_model_rank = 0
        mesh, cell_tags, facet_tags = \
            dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh",
                                            mesh_comm, gmsh_model_rank, gdim=gdim)

num_samples = 16
dim_sample = 3

local_indices, para_samples = parameter_dataset_split(
    generate_training_set, comm_list, num_samples, dim_sample, global_comm)

global_comm.Barrier()  # NOTE Barrier only for good output print


problem_parametric = ProblemOnDeformedDomain(
    mesh, cell_tags, facet_tags, CustomMeshDeformation)

for i in range(len(comm_list)):
    if comm_list[i] != MPI.COMM_NULL:
        solution_mu = problem_parametric.solve(para_samples[0,:])
        _, rend = solution_mu.vector.getOwnershipRange()
        num_dofs = comm_list[i].allreduce(rend, op=MPI.MAX)
        print(f"Rank {global_comm.rank}, Num dofs {num_dofs}")

fem_snapshots = \
    snapshot_matrix_write(local_indices, para_samples, problem_parametric,
                          comm_list, num_samples, num_dofs, global_comm)

global_comm.Barrier()

Nmax = 30

for i in range(fem_snapshots.shape[0]):
    print(f"Rank {global_comm.rank}, Index {i}, 2-norm: {np.linalg.norm(fem_snapshots[i,:])}")
    global_comm.Barrier() # NOTE Barrier only for output formatting
