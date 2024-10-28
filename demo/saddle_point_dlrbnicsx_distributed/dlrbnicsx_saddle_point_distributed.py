import time
import abc

import dolfinx
import ufl
from basix.ufl import element, mixed_element
import basix

import rbnicsx
import rbnicsx.backends
import rbnicsx.online

from mpi4py import MPI
from petsc4py import PETSc
from smt.sampling_methods import LHS

import numpy as np
import itertools
import abc
import matplotlib.pyplot as plt
import os

from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory \
    import Tanh, Sigmoid
from dlrbnicsx.dataset.custom_partitioned_dataset \
    import CustomPartitionedDataset
from dlrbnicsx.interface.wrappers import DataLoader, save_model, \
    load_model, save_checkpoint, load_checkpoint, model_synchronise, \
    init_cpu_process_group, get_optimiser, get_loss_func, share_model
from dlrbnicsx.train_validate_test.train_validate_test_distributed \
    import train_nn, validate_nn, online_nn, error_analysis

class ParametricProblem(abc.ABC):
    # Define FEM problem
    def __init__(self, mesh):
        # Mesh, Subdomains, Boundaries
        self._mesh = mesh
        self._boundaries = self.mark_boundaries
        pol_degree = 1

        # Define function space, Trial and Test Function
        Q_el = basix.ufl.element("RT", mesh.basix_cell(),
                                      pol_degree)
        P_el = basix.ufl.element("DG", mesh.basix_cell(),
                                      pol_degree - 1)
        V_el = basix.ufl.mixed_element([Q_el, P_el])
        self._V = dolfinx.fem.FunctionSpace(self._mesh, V_el)
        self._Q, self._VQ_map = self._V.sub(0).collapse()
        self._W, self._VW_map = self._V.sub(1).collapse()

        (sigma, u) = ufl.TrialFunctions(self._V)
        (tau, v) = ufl.TestFunctions(self._V)
        self._trial = (sigma, u)
        self._test = (tau, v)

        sigma_collapsed, u_collapsed = \
            ufl.TrialFunction(self._Q), ufl.TrialFunction(self._W)
        tau_collapsed, v_collapsed = \
            ufl.TestFunction(self._Q), ufl.TestFunction(self._W)

        self._inner_product_sigma = ufl.inner(sigma_collapsed,
                                              tau_collapsed) * ufl.dx + \
                                    ufl.inner(ufl.div(sigma_collapsed),
                                              ufl.div(tau_collapsed)) * ufl.dx
        self._inner_product_action_sigma = \
            rbnicsx.backends.bilinear_form_action(self._inner_product_sigma,
                                                  part="real")

        self._inner_product_u = ufl.inner(u_collapsed, v_collapsed) * ufl.dx
        self._inner_product_action_u = \
            rbnicsx.backends.bilinear_form_action(self._inner_product_u,
                                                  part="real")

        self.mu_0 = dolfinx.fem.Constant(mesh, PETSc.ScalarType(-2.))
        self.mu_1 = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.5))
        self.mu_2 = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.5))
        self.mu_3 = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.5))
        self.mu_4 = dolfinx.fem.Constant(mesh, PETSc.ScalarType(3.))

    @property
    def mark_boundaries(self):
        def z_0(x):
            return np.isclose(x[2], 0)

        def y_0(x):
            return np.isclose(x[1], 0)

        def x_0(x):
            return np.isclose(x[0], 0)

        fdim = self._mesh.topology.dim - 1
        x_0_facets = dolfinx.mesh.locate_entities_boundary(self._mesh,
                                                           fdim, x_0)
        y_0_facets = dolfinx.mesh.locate_entities_boundary(self._mesh,
                                                           fdim, y_0)
        z_0_facets = dolfinx.mesh.locate_entities_boundary(self._mesh,
                                                           fdim, z_0)

        marked_facets = np.hstack([z_0_facets, y_0_facets, x_0_facets])
        marked_values = np.hstack([np.full_like(z_0_facets, 1),
                                   np.full_like(y_0_facets, 18),
                                   np.full_like(x_0_facets, 30)])
        sorted_facets = np.argsort(marked_facets)
        boundaries = dolfinx.mesh.meshtags(self._mesh, fdim,
                                           marked_facets[sorted_facets],
                                           marked_values[sorted_facets])

        return boundaries

    @property
    def source_term(self):
        x = ufl.SpatialCoordinate(self._mesh)
        mu_0, mu_1, mu_2, mu_3, mu_4 = \
            self.mu_0, self.mu_1, self.mu_2, self.mu_3, self.mu_4
        mu = [mu_0, mu_1, mu_2, mu_3, mu_4]
        f = 10. * ufl.exp(-mu[0] * ((x[0] - mu[1]) * (x[0] - mu[1]) +
                                    (x[1] - mu[2]) * (x[1] - mu[2]) +
                                    (x[2] - mu[3]) * (x[2] - mu[3])))
        return f

    @property
    def bilinear_form(self):
        (sigma, u) = self._trial
        (tau, v) = self._test
        a = ufl.inner(sigma, tau) * ufl.dx + \
            ufl.inner(u, ufl.div(tau)) * ufl.dx + \
            ufl.inner(ufl.div(sigma), v) * ufl.dx
        return a

    @property
    def linear_form(self):
        f = self.source_term
        (tau, v) = self._test
        L = - ufl.inner(f, v) * ufl.dx
        return L

    @property
    def preconditioner_form(self):
        (sigma, u) = self._trial
        (tau, v) = self._test
        aP = ufl.inner(sigma, tau) * ufl.dx + \
             ufl.inner(ufl.div(sigma), ufl.div(tau)) * ufl.dx + \
             ufl.inner(u, v) * ufl.dx
        return aP

    @property
    def assemble_bcs(self):

        V0 = self._V.sub(0)
        Q, VQ_map = V0.collapse()
        V1 = self._V.sub(1)
        W, VW_map = V1.collapse()

        gdim = self._mesh.geometry.dim

        dofs_x0 = dolfinx.fem.locate_dofs_topological((V0, Q), gdim-1,
                                                      self._boundaries.find(30))

        mu = [self.mu_0.value, self.mu_1.value, self.mu_2.value,
              self.mu_3.value, self.mu_4.value]

        def f1(x):
            values = np.zeros((3, x.shape[1]))
            values[0, :] = np.sin(mu[4] * x[0])
            return values


        f_h1 = dolfinx.fem.Function(Q)
        f_h1.interpolate(f1)
        bc_x0 = dolfinx.fem.dirichletbc(f_h1, dofs_x0, V0)

        dofs_y0 = dolfinx.fem.locate_dofs_topological((V0, Q), gdim-1,
                                                      self._boundaries.find(18))

        def f2(x):
            values = np.zeros((3, x.shape[1]))
            values[1, :] = np.sin(mu[4] * x[1])
            return values

        f_h2 = dolfinx.fem.Function(Q)
        f_h2.interpolate(f2)
        bc_y0 = dolfinx.fem.dirichletbc(f_h2, dofs_y0, V0)

        dofs_z0 = dolfinx.fem.locate_dofs_topological((V0, Q), gdim-1,
                                                      self._boundaries.find(1))

        def f3(x):
            values = np.zeros((3, x.shape[1]))
            values[2, :] = np.sin(mu[4] * x[2])
            return values


        f_h3 = dolfinx.fem.Function(Q)
        f_h3.interpolate(f3)
        bc_z0 = dolfinx.fem.dirichletbc(f_h3, dofs_z0, V0)

        # NOTE
        bcs = [bc_x0, bc_y0, bc_z0]

        return bcs
    '''
    def solve(self, mu):
        self.mu_0.value = mu[0]
        self.mu_1.value = mu[1]
        self.mu_2.value = mu[2]
        self.mu_3.value = mu[3]
        self.mu_4.value = mu[4]

        a = self.bilinear_form
        L = self.linear_form
        aP = self.preconditioner_form
        bcs = self.assemble_bcs

        # TODO Twice collapse solve and assemble_bcs, do only one collapse and save in self
        V0 = self._V.sub(0)
        Q, VQ_map = V0.collapse()
        V1 = self._V.sub(1)
        W, VW_map = V1.collapse()

        a_cpp = dolfinx.fem.form(a)
        l_cpp = dolfinx.fem.form(L)
        aP_cpp = dolfinx.fem.form(aP)

        A = dolfinx.fem.petsc.assemble_matrix(a_cpp, bcs=bcs)
        A.assemble()
        P = dolfinx.fem.petsc.assemble_matrix(aP_cpp, bcs=bcs)
        P.assemble()
        L = dolfinx.fem.petsc.assemble_vector(l_cpp)
        dolfinx.fem.petsc.apply_lifting(L, [a_cpp], [bcs])
        L.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(L, bcs)

        # Solver setup
        ksp = PETSc.KSP()
        ksp.create(self._mesh.comm)
        ksp.setOperators(A, P)
        ksp.setType("gmres")
        pc = ksp.getPC()
        pc.setType("fieldsplit")
        # NOTE see https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.PC.CompositeType.html
        pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

        ksp.getPC().setFactorSolverType("superlu")

        # NOTE Since setFieldSplitIS for ISq is called zero-th and for ISw is called first --> subksps[0] corressponds to ISq and subksps[1] corressponds to ISw
        ISq = PETSc.IS().createGeneral(VQ_map, self._mesh.comm)
        ISw = PETSc.IS().createGeneral(VW_map, self._mesh.comm)
        pc.setFieldSplitIS(("sigma",ISq))
        pc.setFieldSplitIS(("u",ISw))
        pc.setUp()

        subksps = pc.getFieldSplitSubKSP()
        subksps[0].setType("preonly")
        subksps[0].getPC().setType("lu")
        # subksps[0].rtol = 1.e-12
        subksps[1].setType("preonly")
        subksps[1].getPC().setType("ilu")
        # subksps[1].rtol = 1.e-12
        ksp.rtol = 1.e-8 # NOTE or ksp.setTolerances(1e-8) # rtol is first argument of setTolerances

        # ksp.setConvergenceHistory()
        ksp.setFromOptions()
        w_h = dolfinx.fem.Function(self._V)
        solve_start_time = time.process_time()
        ksp.solve(L, w_h.vector)
        solve_end_time = time.process_time()
        print(f"Number of iterations: {ksp.getIterationNumber()}")
        print(f"Convergence reason: {ksp.getConvergedReason()}")
        # print(f"Convergence history: {ksp.getConvergenceHistory()}")
        ksp.destroy()
        A.destroy()
        L.destroy()
        w_h.x.scatter_forward()
        sigma_h, u_h = w_h.split()
        sigma_h = sigma_h.collapse()
        u_h = u_h.collapse()
        print(f"Solve time: {solve_end_time - solve_start_time}")
        return sigma_h, u_h
    '''

    def solve(self, mu):
        self.mu_0.value = mu[0]
        self.mu_1.value = mu[1]
        self.mu_2.value = mu[2]
        self.mu_3.value = mu[3]
        self.mu_4.value = mu[4]

        a = self.bilinear_form
        L = self.linear_form
        bcs = self.assemble_bcs

        V0 = self._V.sub(0)
        Q, VQ_map = V0.collapse()
        V1 = self._V.sub(1)
        W, VW_map = V1.collapse()

        a_cpp = dolfinx.fem.form(a)
        l_cpp = dolfinx.fem.form(L)

        A = dolfinx.fem.petsc.assemble_matrix(a_cpp, bcs=bcs)
        A.assemble()
        L = dolfinx.fem.petsc.assemble_vector(l_cpp)
        dolfinx.fem.petsc.apply_lifting(L, [a_cpp], [bcs])
        L.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(L, bcs)

        ksp = PETSc.KSP()
        ksp.create(mesh.comm)
        ksp.setOperators(A)
        # Set GMRES solver
        ksp.setType("gmres")
        ksp.setGMRESRestart(100)
        # Convergence criteria based on residual tolerance
        ksp.rtol = 1.e-8
        # Solve and see convergence details
        ksp.setFromOptions()
        w_h = dolfinx.fem.Function(self._V)
        solve_start_time = time.process_time()
        ksp.solve(L, w_h.vector)
        solve_end_time = time.process_time()
        print(f"Number of iterations: {ksp.getIterationNumber()}")
        print(f"Convergence reason: {ksp.getConvergedReason()}")
        ksp.destroy()
        A.destroy()
        L.destroy()
        w_h.x.scatter_forward()
        # Split the FEM solutions sigma and u
        sigma_h, u_h = w_h.split()
        sigma_h = sigma_h.collapse()
        u_h = u_h.collapse()
        print(f"Solve time: {solve_end_time - solve_start_time}")
        return sigma_h, u_h

class PODANNReducedProblem(abc.ABC):
    # Define Reduced problem class
    def __init__(self, problem) -> None:
        Q, _ = problem._V.sub(0).collapse()
        W, _ = problem._V.sub(1).collapse()
        self._basis_functions_sigma = rbnicsx.backends.FunctionsList(Q)
        self._basis_functions_u = rbnicsx.backends.FunctionsList(W)
        sigma_collapsed, u_collapsed = ufl.TrialFunction(Q), ufl.TrialFunction(W)
        tau_collapsed, v_collapsed = ufl.TestFunction(Q), ufl.TestFunction(W)
        self._inner_product_sigma = ufl.inner(sigma_collapsed, tau_collapsed) * ufl.dx + \
            ufl.inner(ufl.div(sigma_collapsed), ufl.div(tau_collapsed)) * ufl.dx
        self._inner_product_action_sigma = \
            rbnicsx.backends.bilinear_form_action(self._inner_product_sigma,
                                                  part="real")
        self._inner_product_u = ufl.inner(u_collapsed, v_collapsed) * ufl.dx
        self._inner_product_action_u = \
            rbnicsx.backends.bilinear_form_action(self._inner_product_u,
                                                  part="real")
        self.input_scaling_range = [-1., 1.]
        self.input_range = \
            np.array([[-2.5, 0., 0.2, 0.2, 2.5],
                      [-1.5, 1., 0.8, 0.8, 3.5]])

        self.output_scaling_range_sigma = [-1., 1.]
        self.output_range_sigma = [None, None]
        self.regularisation_sigma = "EarlyStopping"

        self.output_scaling_range_u = [-1., 1.]
        self.output_range_u = [None, None]
        self.regularisation_u = "EarlyStopping"

    def reconstruct_solution_sigma(self, reduced_solution):
        """Reconstructed reduced VELOCITY solution on the high fidelity space."""
        return self._basis_functions_sigma[:reduced_solution.size] * \
            reduced_solution

    def reconstruct_solution_u(self, reduced_solution):
        """Reconstructed reduced PRESSURE solution on the high fidelity space."""
        return self._basis_functions_u[:reduced_solution.size] * \
            reduced_solution

    def compute_norm_sigma(self, function):
        """Compute the norm of a VELOCITY function inner product
        on the reference domain."""
        return np.sqrt(self._inner_product_action_sigma(function)(function))

    def compute_norm_u(self, function):
        """Compute the norm of a PRESSURE function inner product
        on the reference domain."""
        return np.sqrt(self._inner_product_action_u(function)(function))

    def project_snapshot_sigma(self, solution, N):
        # Project VELOCITY FEM solution on RB space
        return self._project_snapshot_sigma(solution, N)

    def project_snapshot_u(self, solution, N):
        # Project PRESSURE FEM solution on RB space
        return self._project_snapshot_u(solution, N)

    def _project_snapshot_sigma(self, solution, N):
        projected_snapshot_sigma = rbnicsx.online.create_vector(N)
        A = rbnicsx.backends.\
            project_matrix(self._inner_product_action_sigma,
                           self._basis_functions_sigma[:N])
        F = rbnicsx.backends.\
            project_vector(self._inner_product_action_sigma(solution),
                           self._basis_functions_sigma[:N])
        ksp = PETSc.KSP()
        ksp.create(projected_snapshot_sigma.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
        ksp.solve(F, projected_snapshot_sigma)
        return projected_snapshot_sigma

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

    def norm_error_sigma(self, sigma, tau):
        # Relative error norm for SIGMA
        return self.compute_norm_sigma(sigma-tau)/self.compute_norm_sigma(sigma)

    def norm_error_u(self, u, v):
        # Relative error norm for U
        return self.compute_norm_u(u-v)/self.compute_norm_u(u)


# MPI Communicators
world_comm = MPI.COMM_WORLD
if world_comm.size == 8:
    fem0_procs = world_comm.group.Incl([0, 1])
    fem0_procs_comm = world_comm.Create_group(fem0_procs)

    fem1_procs = world_comm.group.Incl([2, 3])
    fem1_procs_comm = world_comm.Create_group(fem1_procs)

    fem2_procs = world_comm.group.Incl([4, 5])
    fem2_procs_comm = world_comm.Create_group(fem2_procs)

    fem3_procs = world_comm.group.Incl([6, 7])
    fem3_procs_comm = world_comm.Create_group(fem3_procs)

    fem_comm_list = [fem0_procs_comm, fem1_procs_comm,
                     fem2_procs_comm, fem3_procs_comm]

elif world_comm.size == 4:
    '''
    fem0_procs = world_comm.group.Incl([0])
    fem0_procs_comm = world_comm.Create_group(fem0_procs)

    fem1_procs = world_comm.group.Incl([1])
    fem1_procs_comm = world_comm.Create_group(fem1_procs)

    fem2_procs = world_comm.group.Incl([2])
    fem2_procs_comm = world_comm.Create_group(fem2_procs)

    fem3_procs = world_comm.group.Incl([3])
    fem3_procs_comm = world_comm.Create_group(fem3_procs)

    fem_comm_list = [fem0_procs_comm, fem1_procs_comm,
                     fem2_procs_comm, fem3_procs_comm]
    '''

    fem0_procs = world_comm.group.Incl([0, 1])
    fem0_procs_comm = world_comm.Create_group(fem0_procs)

    fem1_procs = world_comm.group.Incl([2, 3])
    fem1_procs_comm = world_comm.Create_group(fem1_procs)

    fem_comm_list = [fem0_procs_comm, fem1_procs_comm]

elif world_comm.size == 1:
    fem0_procs = world_comm.group.Incl([0])
    fem0_procs_comm = world_comm.Create_group(fem0_procs)
    # fem0_procs_comm = MPI.COMM_WORLD or MPI.COMM_SELF
    fem_comm_list = [fem0_procs_comm]
else:
    raise NotImplementedError("Please use 1, 4 or 8 processes")

# Mesh, subdomains, Boundaries
for comm_i in fem_comm_list:
    if comm_i != MPI.COMM_NULL:
        mesh_comm = comm_i

nx, ny, nz = 20, 20, 20
mesh = dolfinx.mesh.create_box(mesh_comm,
                               [[0.0, 0.0, 0.0], [1., 1, 1]],
                               [nx, ny, nz],
                               dolfinx.mesh.CellType.tetrahedron)
problem_parametric = ParametricProblem(mesh)


# mu_range = ((-2.5, -1.5), (0., 1.), (0.2, 0.8), (0.2, 0.8), (2.5, 3.5))
# mu = np.array([-2., 0.5, 0.5, 0.5, 3.])
mu = np.array([-1., 1.5, 0.7, 0.3, 3.4])

para_dim = 5
ann_input_samples_num = 1100
error_analysis_samples_num = 500
num_snapshots = 1000
itemsize = MPI.DOUBLE.Get_size()

sigma_h, u_h = problem_parametric.solve(mu)

sigma_norm = mesh.comm.allreduce(dolfinx.fem.assemble_scalar
                                 (dolfinx.fem.form(ufl.inner(sigma_h, sigma_h) *
                                                   ufl.dx +
                                                   ufl.inner(ufl.div(sigma_h),
                                                             ufl.div(sigma_h)) *
                                                             ufl.dx)), op=MPI.SUM)
u_norm = mesh.comm.allreduce(dolfinx.fem.assemble_scalar
                             (dolfinx.fem.form(ufl.inner(u_h, u_h) *
                                               ufl.dx)), op=MPI.SUM)

print(f"sigma norm: {sigma_norm}, u norm: {u_norm}")

rstart_sigma, rend_sigma = sigma_h.vector.getOwnershipRange()
num_dofs_sigma = mesh_comm.allreduce(rend_sigma, op=MPI.MAX) - mesh_comm.allreduce(rstart_sigma, op=MPI.MIN)
rstart_u, rend_u = u_h.vector.getOwnershipRange()
num_dofs_u = mesh_comm.allreduce(rend_u, op=MPI.MAX) - mesh_comm.allreduce(rstart_u, op=MPI.MIN)

computed_file_sigma = "dlrbnicsx_solution_saddle_point/solution_computed_sigma.xdmf"
computed_file_u = "dlrbnicsx_solution_saddle_point/solution_computed_u.xdmf"
'''
if world_comm.rank == 0:
    with dolfinx.io.XDMFFile(mesh.comm, computed_file_sigma,
                            "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(sigma_h)

    with dolfinx.io.XDMFFile(mesh.comm, computed_file_u,
                                "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(u_h)
'''

# POD Starts ###
Nmax_sigma = 30
Nmax_u = 30
tol_sigma = 1.e-4
tol_u = 1.e-4

def generate_training_set(num_samples, para_dim):
    # training_set = np.random.uniform(size=(num_samples, para_dim))
    training_set = np.zeros((num_samples, para_dim))
    training_set[:, 0] = np.linspace(0., 1., num=num_samples)
    training_set[:, 1] = np.linspace(0., 1., num=num_samples)
    training_set[:, 2] = np.linspace(0., 1., num=num_samples)
    training_set[:, 3] = np.linspace(0., 1., num=num_samples)
    training_set[:, 4] = np.linspace(0., 1., num=num_samples)
    training_set[:, 0] = (-1.5 + 2.5) * training_set[:, 0] - 2.5
    training_set[:, 1] = (1. - 0.) * training_set[:, 1] + 0.
    training_set[:, 2] = (0.8 - 0.2) * training_set[:, 2] + 0.2
    training_set[:, 3] = (0.8 - 0.2) * training_set[:, 3] + 0.2
    training_set[:, 4] = (3.5 - 2.5) * training_set[:, 3] + 2.5
    return training_set

if world_comm.rank == 0:
    nbytes_para = num_snapshots * para_dim * itemsize
else:
    nbytes_para = 0

win0 = MPI.Win.Allocate_shared(nbytes_para, itemsize, comm=MPI.COMM_WORLD)
buf0, itemsize = win0.Shared_query(0)
para_matrix = np.ndarray(buffer=buf0, dtype="d",
                               shape=(num_snapshots, para_dim))

if world_comm.rank == 0:
    para_matrix[:, :] = generate_training_set(num_snapshots, para_dim)

world_comm.barrier()

for i in range(len(fem_comm_list)):
    if fem_comm_list[i] != MPI.COMM_NULL:
        cpu_indices_sigma = np.arange(i, para_matrix.shape[0], len(fem_comm_list))

print(f"CPU Indices: {cpu_indices_sigma}")

if world_comm.rank == 0:
    nbytes_dofs_sigma = num_snapshots * num_dofs_sigma * itemsize
else:
    nbytes_dofs_sigma = 0

world_comm.barrier()

win1 = MPI.Win.Allocate_shared(nbytes_dofs_sigma, itemsize, comm=MPI.COMM_WORLD)
buf1, itemsize = win1.Shared_query(0)
snapshot_arrays_sigma = np.ndarray(buffer=buf1, dtype="d",
                             shape=(num_snapshots, num_dofs_sigma))
snapshots_matrix_sigma = rbnicsx.backends.FunctionsList(problem_parametric._Q)

if world_comm.rank == 0:
    nbytes_dofs_u = num_snapshots * num_dofs_u * itemsize
else:
    nbytes_dofs_u = 0

world_comm.barrier()

win2 = MPI.Win.Allocate_shared(nbytes_dofs_u, itemsize, comm=MPI.COMM_WORLD)
buf2, itemsize = win2.Shared_query(0)
snapshot_arrays_u = np.ndarray(buffer=buf2, dtype="d",
                             shape=(num_snapshots, num_dofs_u))
snapshots_matrix_u = rbnicsx.backends.FunctionsList(problem_parametric._W)

# NOTE  Redundant check of if fem_comm_list[i] != MPI.COMM_NULL is removed

for mu_index in cpu_indices_sigma:
    print(f"Parameter number {mu_index+1} of {para_matrix.shape[0]}: {para_matrix[mu_index,:]}")
    solution_sigma, solution_u = problem_parametric.solve(para_matrix[mu_index, :])
    snapshot_arrays_sigma[mu_index, rstart_sigma:rend_sigma] = solution_sigma.vector[rstart_sigma:rend_sigma]
    snapshot_arrays_u[mu_index, rstart_u:rend_u] = solution_u.vector[rstart_u:rend_u]

world_comm.barrier()

for i in range(snapshot_arrays_sigma.shape[0]):
    solution_empty = dolfinx.fem.Function(problem_parametric._Q)
    solution_empty.vector[rstart_sigma:rend_sigma] = snapshot_arrays_sigma[i, rstart_sigma:rend_sigma]
    solution_empty.x.scatter_forward()
    # TODO Recall why .assemble()?
    solution_empty.vector.assemble()
    snapshots_matrix_sigma.append(solution_empty)
    del(solution_empty)

for i in range(snapshot_arrays_u.shape[0]):
    solution_empty = dolfinx.fem.Function(problem_parametric._W)
    solution_empty.vector[rstart_u:rend_u] = snapshot_arrays_u[i, rstart_u:rend_u]
    solution_empty.x.scatter_forward()
    # TODO Recall why .assemble()?
    solution_empty.vector.assemble()
    snapshots_matrix_u.append(solution_empty)
    del(solution_empty)

print("Set up reduced problem")
reduced_problem = PODANNReducedProblem(problem_parametric)

print("")

print(rbnicsx.io.TextLine("Perform POD (Sigma)", fill="#"))
eigenvalues_sigma, modes_sigma, _ = \
    rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_matrix_sigma,
                                    problem_parametric._inner_product_action_sigma,
                                    N=Nmax_sigma, tol=tol_sigma)

reduced_problem._basis_functions_sigma.extend(modes_sigma)
reduced_size_sigma = len(reduced_problem._basis_functions_sigma)
print(f"Sigma RB size: {reduced_size_sigma}, Sigma eigenvalues: {eigenvalues_sigma}")

print(rbnicsx.io.TextBox("POD-Galerkin offline phase ends", fill="="))

positive_eigenvalues_sigma = np.where(eigenvalues_sigma > 0., eigenvalues_sigma, np.nan)
singular_values_sigma = np.sqrt(positive_eigenvalues_sigma)

if world_comm.rank == 0:
    plt.figure(figsize=[8, 10])
    xint = list()
    yval = list()

    for x, y in enumerate(eigenvalues_sigma[:len(reduced_problem._basis_functions_sigma)]):
        yval.append(y)
        xint.append(x+1)

    plt.plot(xint, yval, "*-", color="orange")
    plt.xlabel("Eigenvalue number", fontsize=18)
    plt.ylabel("Eigenvalue", fontsize=18)
    plt.xticks(xint)
    plt.yscale("log")
    plt.title("Eigenvalue decay", fontsize=24)
    plt.tight_layout()
    plt.savefig("eigenvalue_decay_sigma")

del(snapshot_arrays_sigma)

print(rbnicsx.io.TextLine("Perform POD (U)", fill="#"))
eigenvalues_u, modes_u, _ = \
    rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_matrix_u,
                                    problem_parametric._inner_product_action_u,
                                    N=Nmax_u, tol=tol_u)

reduced_problem._basis_functions_u.extend(modes_u)
reduced_size_u = len(reduced_problem._basis_functions_u)
print(f"U RB size: {reduced_size_u}, U eigenvalues: {eigenvalues_u}")

print(rbnicsx.io.TextBox("POD-Galerkin offline phase ends", fill="="))

positive_eigenvalues_u = np.where(eigenvalues_u > 0., eigenvalues_u, np.nan)
singular_values_u = np.sqrt(positive_eigenvalues_u)

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

del(snapshot_arrays_u)

# ### POD Ends ###

sigma_h_projected = reduced_problem.project_snapshot_sigma(sigma_h, reduced_size_sigma)
sigma_h_reconstructed = reduced_problem.reconstruct_solution_sigma(sigma_h_projected)
print(sigma_h_reconstructed.x.array.shape, sigma_h.x.array.shape)
sigma_norm = reduced_problem.compute_norm_sigma(sigma_h_reconstructed)
sigma_error = reduced_problem.norm_error_sigma(sigma_h, sigma_h_reconstructed)
print(f"Norm reconstructed: {sigma_norm}, projection error: {sigma_error}")

# ### Projection error samples ###
# Creating dataset
def generate_projection_error_set(num_projection_samples=10):
    '''
    xlimits = np.array([[-1., 1.], [0.4, 0.6],
                        [0.4, 0.6], [0.4, 0.6],
                        [2.5, 3.5]])
    sampling = LHS(xlimits=xlimits)
    training_set = sampling(num_projection_samples)
    '''
    training_set = np.zeros((num_projection_samples, para_dim))
    training_set[:, 0] = np.linspace(0., 1., num=num_projection_samples)
    training_set[:, 1] = np.linspace(0., 1., num=num_projection_samples)
    training_set[:, 2] = np.linspace(0., 1., num=num_projection_samples)
    training_set[:, 3] = np.linspace(0., 1., num=num_projection_samples)
    training_set[:, 4] = np.linspace(0., 1., num=num_projection_samples)
    training_set[:, 0] = (-1.5 + 2.5) * training_set[:, 0] - 2.5
    training_set[:, 1] = (1. - 0.) * training_set[:, 1] + 0.
    training_set[:, 2] = (0.8 - 0.2) * training_set[:, 2] + 0.2
    training_set[:, 3] = (0.8 - 0.2) * training_set[:, 3] + 0.2
    training_set[:, 4] = (3.5 - 2.5) * training_set[:, 3] + 2.5
    return training_set

if world_comm.rank == 0:
    nbytes_para_projection_error = error_analysis_samples_num * itemsize * para_dim
    nbytes_projection_error_array_sigma = error_analysis_samples_num * itemsize
    nbytes_projection_error_array_u = error_analysis_samples_num * itemsize
else:
    nbytes_para_projection_error = 0
    nbytes_projection_error_array_sigma = 0
    nbytes_projection_error_array_u = 0

world_comm.barrier()

win3 = MPI.Win.Allocate_shared(nbytes_para_projection_error, itemsize,
                               comm=MPI.COMM_WORLD)
buf3, itemsize = win3.Shared_query(0)
projection_error_samples = \
    np.ndarray(buffer=buf3, dtype="d",
               shape=(error_analysis_samples_num, para_dim))

win4 = MPI.Win.Allocate_shared(nbytes_projection_error_array_sigma, itemsize,
                               comm=MPI.COMM_WORLD)
buf4, itemsize = win4.Shared_query(0)
projection_error_array_sigma = \
    np.ndarray(buffer=buf4, dtype="d",
               shape=(error_analysis_samples_num))

win5 = MPI.Win.Allocate_shared(nbytes_projection_error_array_u, itemsize,
                               comm=MPI.COMM_WORLD)
buf5, itemsize = win5.Shared_query(0)
projection_error_array_u = \
    np.ndarray(buffer=buf5, dtype="d",
               shape=(error_analysis_samples_num))

if world_comm.rank == 0:
    projection_error_samples[:, :] = \
        generate_projection_error_set(num_projection_samples=error_analysis_samples_num)

world_comm.Barrier()

for j in range(len(fem_comm_list)):
    if fem_comm_list[j] != MPI.COMM_NULL:
        projection_error_indices = \
            np.arange(j, projection_error_samples.shape[0],
                      len(fem_comm_list))

print(f"Rank: {world_comm.rank}, Indices (projection error): {projection_error_indices}")

for k in projection_error_indices:
    print(f"Index: {k}")
    fem_sol_sigma, fem_sol_u = problem_parametric.solve(projection_error_samples[k, :])
    reconstructed_sol_sigma = \
        reduced_problem.reconstruct_solution_sigma(
            reduced_problem.project_snapshot_sigma(fem_sol_sigma,
                                                   reduced_size_sigma))
    reconstructed_sol_u = \
        reduced_problem.reconstruct_solution_u(
            reduced_problem.project_snapshot_u(fem_sol_u,
                                                reduced_size_u))
    projection_error_array_sigma[k] = \
        reduced_problem.norm_error_sigma(fem_sol_sigma, reconstructed_sol_sigma)
    projection_error_array_u[k] = \
        reduced_problem.norm_error_u(fem_sol_u, reconstructed_sol_u)

print(f"Rank: {world_comm.rank}, \nProjection errors (sigma): {projection_error_array_sigma}, \nProjection errors (u): {projection_error_array_u} ")

'''
if fem_comm_list[0] != MPI.COMM_NULL:
    fem_error_file \
        = "projection_error/fem_solution_sigma.xdmf"
    with dolfinx.io.XDMFFile(mesh.comm, fem_error_file,
                            "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(fem_sol_sigma)

    rb_error_file \
        = "projection_error/rb_solution_sigma.xdmf"
    with dolfinx.io.XDMFFile(mesh.comm, rb_error_file,
                            "w") as solution_file:
        # NOTE scatter_forward not considered for online solution
        solution_file.write_mesh(mesh)
        solution_file.write_function(reconstructed_sol_sigma)

    error_function_sigma = dolfinx.fem.Function(problem_parametric._Q)
    error_function_sigma.vector[rstart_sigma:rend_sigma] = \
        abs(fem_sol_sigma.vector[rstart_sigma:rend_sigma] -
            reconstructed_sol_sigma.vector[rstart_sigma:rend_sigma])

    fem_rb_error_file \
        = "projection_error/projection_error_sigma.xdmf"
    with dolfinx.io.XDMFFile(mesh.comm, fem_rb_error_file,
                            "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(error_function_sigma)

    fem_error_file \
        = "projection_error/fem_solution_u.xdmf"
    with dolfinx.io.XDMFFile(mesh.comm, fem_error_file,
                            "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(fem_sol_u)

    rb_error_file \
        = "projection_error/rb_solution_u.xdmf"
    with dolfinx.io.XDMFFile(mesh.comm, rb_error_file,
                            "w") as solution_file:
        # NOTE scatter_forward not considered for online solution
        solution_file.write_mesh(mesh)
        solution_file.write_function(reconstructed_sol_u)

    error_function_u = dolfinx.fem.Function(problem_parametric._W)
    error_function_u.vector[rstart_u:rend_u] = \
        abs(fem_sol_u.vector[rstart_u:rend_u] -
            reconstructed_sol_u.vector[rstart_u:rend_u])
    fem_rb_error_file \
        = "projection_error/projection_error_u.xdmf"
    with dolfinx.io.XDMFFile(mesh.comm, fem_rb_error_file,
                            "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(error_function_u)
'''
# ### Projection error ends ###

# Creating dataset
def generate_ann_input_set(num_ann_samples=10):
    '''
    # ((-2.5, -1.5), (0., 1.), (0.2, 0.8), (0.2, 0.8), (2.5, 3.5))
    xlimits = np.array([[-2.5, -1.5], [0., 1.],
                        [0.2, 0.8], [0.2, 0.8],
                        [2.5, 3.5]])
    sampling = LHS(xlimits=xlimits)
    training_set = sampling(num_ann_samples)
    '''
    training_set = np.zeros((num_ann_samples, para_dim))
    training_set[:, 0] = np.linspace(0., 1., num=num_ann_samples)
    training_set[:, 1] = np.linspace(0., 1., num=num_ann_samples)
    training_set[:, 2] = np.linspace(0., 1., num=num_ann_samples)
    training_set[:, 3] = np.linspace(0., 1., num=num_ann_samples)
    training_set[:, 4] = np.linspace(0., 1., num=num_ann_samples)
    training_set[:, 0] = (-1.5 + 2.5) * training_set[:, 0] - 2.5
    training_set[:, 1] = (1. - 0.) * training_set[:, 1] + 0.
    training_set[:, 2] = (0.8 - 0.2) * training_set[:, 2] + 0.2
    training_set[:, 3] = (0.8 - 0.2) * training_set[:, 3] + 0.2
    training_set[:, 4] = (3.5 - 2.5) * training_set[:, 3] + 2.5
    return training_set

def generate_ann_output_set(problem, reduced_problem, input_set,
                            output_set_sigma, output_set_u,
                            indices, mode=None):
    # Solve the FE problem at given input_sets and
    # project on the RB space
    rb_size_sigma = len(reduced_problem._basis_functions_sigma)
    rb_size_u = len(reduced_problem._basis_functions_u)
    for i in indices:
        if mode is None:
            print(f"Parameter {i+1}/{input_set.shape[0]}")
        else:
            print(f"{mode} parameter number {i+1}/{input_set.shape[0]}")
        solution_sigma, solution_u = problem.solve(input_set[i, :])
        output_set_sigma[i, :] = \
            reduced_problem.project_snapshot_sigma(solution_sigma,
                                                   rb_size_sigma).array
        output_set_u[i, :] = \
            reduced_problem.project_snapshot_u(solution_u,
                                               rb_size_u).array

num_training_samples = int(0.7 * ann_input_samples_num)
num_validation_samples = \
    ann_input_samples_num - int(0.7 * ann_input_samples_num)
itemsize = MPI.DOUBLE.Get_size()

if world_comm.rank == 0:
    ann_input_set = generate_ann_input_set(num_ann_samples=ann_input_samples_num)
    # np.random.shuffle(ann_input_set)
    nbytes_para_ann_training = num_training_samples * itemsize * para_dim
    nbytes_para_ann_validation = num_validation_samples * itemsize * para_dim
    nbytes_dofs_ann_training_sigma = num_training_samples * itemsize * \
        len(reduced_problem._basis_functions_sigma)
    nbytes_dofs_ann_validation_sigma = num_validation_samples * itemsize * \
        len(reduced_problem._basis_functions_sigma)
    nbytes_dofs_ann_training_u = num_training_samples * itemsize * \
        len(reduced_problem._basis_functions_u)
    nbytes_dofs_ann_validation_u = num_validation_samples * itemsize * \
        len(reduced_problem._basis_functions_u)
else:
    nbytes_para_ann_training = 0
    nbytes_para_ann_validation = 0
    nbytes_dofs_ann_training_sigma = 0
    nbytes_dofs_ann_validation_sigma = 0
    nbytes_dofs_ann_training_u = 0
    nbytes_dofs_ann_validation_u = 0

world_comm.barrier()

win6 = MPI.Win.Allocate_shared(nbytes_para_ann_training, itemsize,
                               comm=MPI.COMM_WORLD)
buf6, itemsize = win6.Shared_query(0)
input_training_set = \
    np.ndarray(buffer=buf6, dtype="d",
               shape=(num_training_samples, para_dim))

win7 = MPI.Win.Allocate_shared(nbytes_para_ann_validation, itemsize,
                               comm=MPI.COMM_WORLD)
buf7, itemsize = win7.Shared_query(0)
input_validation_set = \
    np.ndarray(buffer=buf7, dtype="d",
               shape=(num_validation_samples, para_dim))

win8 = MPI.Win.Allocate_shared(nbytes_dofs_ann_training_sigma, itemsize,
                               comm=MPI.COMM_WORLD)
buf8, itemsize = win8.Shared_query(0)
output_training_set_sigma = \
    np.ndarray(buffer=buf8, dtype="d",
               shape=(num_training_samples,
                      len(reduced_problem._basis_functions_sigma)))

win9 = MPI.Win.Allocate_shared(nbytes_dofs_ann_validation_sigma, itemsize,
                               comm=MPI.COMM_WORLD)
buf9, itemsize = win9.Shared_query(0)
output_validation_set_sigma = \
    np.ndarray(buffer=buf9, dtype="d",
               shape=(num_validation_samples,
                      len(reduced_problem._basis_functions_sigma)))

win10 = MPI.Win.Allocate_shared(nbytes_dofs_ann_training_u, itemsize,
                               comm=MPI.COMM_WORLD)
buf10, itemsize = win10.Shared_query(0)
output_training_set_u = \
    np.ndarray(buffer=buf10, dtype="d",
               shape=(num_training_samples,
                      len(reduced_problem._basis_functions_u)))

win11 = MPI.Win.Allocate_shared(nbytes_dofs_ann_validation_u, itemsize,
                               comm=MPI.COMM_WORLD)
buf11, itemsize = win11.Shared_query(0)
output_validation_set_u = \
    np.ndarray(buffer=buf11, dtype="d",
               shape=(num_validation_samples,
                      len(reduced_problem._basis_functions_u)))

if world_comm.rank == 0:
    input_training_set[:, :] = \
        ann_input_set[:num_training_samples, :]
    input_validation_set[:, :] = \
        ann_input_set[num_training_samples:, :]
    output_training_set_sigma[:, :] = \
        np.zeros([num_training_samples,
                  len(reduced_problem._basis_functions_sigma)])
    output_validation_set_sigma[:, :] = \
        np.zeros([num_validation_samples,
                  len(reduced_problem._basis_functions_sigma)])
    output_training_set_u[:, :] = \
        np.zeros([num_training_samples,
                  len(reduced_problem._basis_functions_u)])
    output_validation_set_u[:, :] = \
        np.zeros([num_validation_samples,
                  len(reduced_problem._basis_functions_u)])

world_comm.Barrier()

for j in range(len(fem_comm_list)):
    if fem_comm_list[j] != MPI.COMM_NULL:
        training_set_indices = \
            np.arange(j, input_training_set.shape[0], len(fem_comm_list))
        validation_set_indices = \
            np.arange(j, input_validation_set.shape[0], len(fem_comm_list))

# world_comm.Barrier()

# Training dataset
generate_ann_output_set(problem_parametric, reduced_problem,
                        input_training_set, output_training_set_sigma,
                        output_training_set_u, training_set_indices, mode="Training")

generate_ann_output_set(problem_parametric, reduced_problem,
                        input_validation_set, output_validation_set_sigma,
                        output_validation_set_u, validation_set_indices, mode="Validation")

world_comm.Barrier()

reduced_problem.output_range_sigma[0] = min(np.min(output_training_set_sigma), np.min(output_validation_set_sigma))
reduced_problem.output_range_sigma[1] = max(np.max(output_training_set_sigma), np.max(output_validation_set_sigma))
reduced_problem.output_range_u[0] = min(np.min(output_training_set_u), np.min(output_validation_set_u))
reduced_problem.output_range_u[1] = max(np.max(output_training_set_u), np.max(output_validation_set_u))

print("\n")

if world_comm.size == 8:
    cpu_group0_procs_sigma = world_comm.group.Incl([0, 1])
    cpu_group0_comm_sigma = world_comm.Create_group(cpu_group0_procs_sigma)

    cpu_group1_procs_sigma = world_comm.group.Incl([2, 3])
    cpu_group1_comm_sigma = world_comm.Create_group(cpu_group1_procs_sigma)

    cpu_group2_procs_sigma = world_comm.group.Incl([4, 5])
    cpu_group2_comm_sigma = world_comm.Create_group(cpu_group2_procs_sigma)

    cpu_group3_procs_sigma = world_comm.group.Incl([6, 7])
    cpu_group3_comm_sigma = world_comm.Create_group(cpu_group3_procs_sigma)

    ann_comm_list_sigma = \
        [cpu_group0_comm_sigma, cpu_group1_comm_sigma,
         cpu_group2_comm_sigma, cpu_group3_comm_sigma]

elif world_comm.size == 4:
    '''
    cpu_group0_procs_sigma = world_comm.group.Incl([0])
    cpu_group0_comm_sigma = \
        world_comm.Create_group(cpu_group0_procs_sigma)

    cpu_group1_procs_sigma = world_comm.group.Incl([1])
    cpu_group1_comm_sigma = \
        world_comm.Create_group(cpu_group1_procs_sigma)

    cpu_group2_procs_sigma = world_comm.group.Incl([2])
    cpu_group2_comm_sigma = \
        world_comm.Create_group(cpu_group2_procs_sigma)

    cpu_group3_procs_sigma = world_comm.group.Incl([3])
    cpu_group3_comm_sigma = world_comm.Create_group(cpu_group3_procs_sigma)

    ann_comm_list_sigma = \
        [cpu_group0_comm_sigma, cpu_group1_comm_sigma,
         cpu_group2_comm_sigma, cpu_group3_comm_sigma]
    '''
    cpu_group0_procs_sigma = world_comm.group.Incl([0, 1])
    cpu_group0_comm_sigma = \
        world_comm.Create_group(cpu_group0_procs_sigma)

    cpu_group1_procs_sigma = world_comm.group.Incl([2, 3])
    cpu_group1_comm_sigma = \
        world_comm.Create_group(cpu_group1_procs_sigma)

    ann_comm_list_sigma = \
        [cpu_group0_comm_sigma, cpu_group1_comm_sigma]

elif world_comm.size == 1:
    cpu_group0_procs_sigma = world_comm.group.Incl([0])
    cpu_group0_comm_sigma = world_comm.Create_group(cpu_group0_procs_sigma)

    ann_comm_list_sigma = [cpu_group0_comm_sigma]

else:
    raise NotImplementedError("Please use 1,4 or 8 processes")

print(f"Rank: {world_comm.rank}, Training set indices: {training_set_indices}, Validation set indices: {validation_set_indices}")

# ANN model
model_sigma0 = HiddenLayersNet(para_dim, [45, 45, 45],
                               len(reduced_problem._basis_functions_sigma),
                               Tanh())

model_sigma1 = HiddenLayersNet(para_dim, [55, 55, 55],
                               len(reduced_problem._basis_functions_sigma),
                               Tanh())

model_sigma2 = HiddenLayersNet(para_dim, [65, 65, 65],
                               len(reduced_problem._basis_functions_sigma),
                               Tanh())

model_sigma3 = HiddenLayersNet(para_dim, [75, 75, 75],
                               len(reduced_problem._basis_functions_sigma),
                               Tanh())

if world_comm.size == 8:
    ann_model_list_sigma = [model_sigma0, model_sigma1,
                            model_sigma2, model_sigma3]
    path_list_sigma = ["model_sigma0.pth", "model_sigma1.pth",
                       "model_sigma2.pth", "model_sigma3.pth"]
    checkpoint_path_list_sigma = \
        ["checkpoint_sigma0", "checkpoint_sigma1",
         "checkpoint_sigma2", "checkpoint_sigma3"]
    model_root_process_list_sigma = [0, 3, 4, 7]
    trained_model_path_list_sigma = \
        ["trained_model_sigma0.pth", "trained_model_sigma1.pth",
         "trained_model_sigma2.pth", "trained_model_sigma3.pth"]
elif world_comm.size == 4:
    ann_model_list_sigma = [model_sigma0, model_sigma1,
                            model_sigma2, model_sigma3]
    path_list_sigma = ["model_sigma0.pth", "model_sigma1.pth",
                       "model_sigma2.pth", "model_sigma3.pth"]
    checkpoint_path_list_sigma = \
        ["checkpoint_sigma0", "checkpoint_sigma1",
         "checkpoint_sigma2", "checkpoint_sigma3"]
    model_root_process_list_sigma = [0, 2] # [0, 1, 2, 3]
    trained_model_path_list_sigma = \
        ["trained_model_sigma0.pth", "trained_model_sigma1.pth",
         "trained_model_sigma2.pth", "trained_model_sigma3.pth"]
elif world_comm.size == 1:
    ann_model_list_sigma = [model_sigma0]
    path_list_sigma = ["model_sigma0.pth"]
    checkpoint_path_list_sigma = ["checkpoint_sigma0"]
    model_root_process_list_sigma = [0]
    trained_model_path_list_sigma = ["trained_model_sigma0.pth"]

for j in range(len(ann_comm_list_sigma)):
    if ann_comm_list_sigma[j] != MPI.COMM_NULL:
        init_cpu_process_group(ann_comm_list_sigma[j])

        training_set_indices_cpu_sigma = \
            np.arange(ann_comm_list_sigma[j].rank,
                      input_training_set.shape[0],
                      ann_comm_list_sigma[j].size)
        validation_set_indices_cpu_sigma = \
            np.arange(ann_comm_list_sigma[j].rank,
                      input_validation_set.shape[0],
                      ann_comm_list_sigma[j].size)

        customDataset_sigma = \
            CustomPartitionedDataset(reduced_problem,
                                     input_training_set,
                                     output_training_set_sigma,
                                     training_set_indices_cpu_sigma,
                                     input_scaling_range=reduced_problem.input_scaling_range,
                                     output_scaling_range=reduced_problem.output_scaling_range_sigma,
                                     input_range=reduced_problem.input_range,
                                     output_range=reduced_problem.output_range_sigma
                                     )
        train_dataloader_sigma = \
            DataLoader(customDataset_sigma, batch_size=input_training_set.shape[0], shuffle=False)# shuffle=True)

        customDataset_sigma = \
            CustomPartitionedDataset(reduced_problem,
                                     input_validation_set,
                                     output_validation_set_sigma,
                                     validation_set_indices_cpu_sigma,
                                     input_scaling_range=reduced_problem.input_scaling_range,
                                     output_scaling_range=reduced_problem.output_scaling_range_sigma,
                                     input_range=reduced_problem.input_range,
                                     output_range=reduced_problem.output_range_sigma
                                    )
        valid_dataloader_sigma = \
            DataLoader(customDataset_sigma, batch_size=input_validation_set.shape[0], shuffle=False)

        # save_model(ann_model_list_sigma[j], path_list_sigma[j])
        load_model(ann_model_list_sigma[j], path_list_sigma[j])

        model_synchronise(ann_model_list_sigma[j], verbose=False)

        # Training of ANN
        training_loss = list()
        validation_loss = list()

        max_epochs_sigma = 50000
        min_validation_loss_sigma = None
        start_epoch_sigma = 0
        checkpoint_epoch_sigma = 10

        learning_rate_sigma = 5.e-6 # 1.e-4
        optimiser_sigma = get_optimiser(ann_model_list_sigma[j], "Adam", learning_rate_sigma)
        loss_fn_sigma = get_loss_func("MSE", reduction="sum")

        if os.path.exists(checkpoint_path_list_sigma[j]):
            start_epoch_sigma, min_validation_loss_sigma = \
                load_checkpoint(checkpoint_path_list_sigma[j], ann_model_list_sigma[j],
                                optimiser_sigma)

        import time
        start_time = time.process_time()
        for epochs in range(start_epoch_sigma, max_epochs_sigma):
            if epochs > 0 and epochs % checkpoint_epoch_sigma == 0:
                save_checkpoint(checkpoint_path_list_sigma[j], epochs,
                                ann_model_list_sigma[j], optimiser_sigma,
                                min_validation_loss_sigma)
            print(f"Epoch: {epochs+1}/{max_epochs_sigma}")
            current_training_loss = train_nn(reduced_problem,
                                             train_dataloader_sigma,
                                             ann_model_list_sigma[j],
                                             loss_fn_sigma, optimiser_sigma)
            training_loss.append(current_training_loss)
            current_validation_loss = validate_nn(reduced_problem,
                                                  valid_dataloader_sigma,
                                                  ann_model_list_sigma[j],
                                                  loss_fn_sigma)
            validation_loss.append(current_validation_loss)
            if epochs > 0 and current_validation_loss > 1.01 * min_validation_loss_sigma:
                # 1% safety margin against min_validation_loss
                # before invoking early stopping criteria
                print(f"Early stopping criteria invoked at epoch: {epochs+1}")
                break
            min_validation_loss_sigma = min(validation_loss)
        end_time = time.process_time()
        elapsed_time = end_time - start_time

        os.system(f"rm {checkpoint_path_list_sigma[j]}")

print(f"ANN training time (sigma): {elapsed_time}")
world_comm.Barrier()

for j in range(len(model_root_process_list_sigma)):
    share_model(ann_model_list_sigma[j], world_comm,
                model_root_process_list_sigma[j])
    # save_model(ann_model_list_sigma[j], trained_model_path_list_sigma[j])

# Error analysis dataset
print("\n")
print("Generating error analysis (only input/parameters) dataset")
print("\n")

itemsize = MPI.DOUBLE.Get_size()

if world_comm.rank == 0:
    nbytes_para_sigma = error_analysis_samples_num * itemsize * para_dim
    nbytes_error_sigma = error_analysis_samples_num * itemsize
else:
    nbytes_para_sigma = 0
    nbytes_error_sigma = 0

win12 = MPI.Win.Allocate_shared(nbytes_para_sigma, itemsize,
                               comm=world_comm)
buf12, itemsize = win12.Shared_query(0)
error_analysis_set = \
    np.ndarray(buffer=buf12, dtype="d",
               shape=(error_analysis_samples_num,
                      para_dim))

win13 = MPI.Win.Allocate_shared(nbytes_error_sigma, itemsize,
                               comm=world_comm)
buf13, itemsize = win13.Shared_query(0)
error_numpy_sigma0 = np.ndarray(buffer=buf13, dtype="d",
                         shape=(error_analysis_samples_num))

win14 = MPI.Win.Allocate_shared(nbytes_error_sigma, itemsize,
                               comm=world_comm)
buf14, itemsize = win14.Shared_query(0)
error_numpy_sigma1 = np.ndarray(buffer=buf14, dtype="d",
                         shape=(error_analysis_samples_num))

win15 = MPI.Win.Allocate_shared(nbytes_error_sigma, itemsize,
                               comm=world_comm)
buf15, itemsize = win15.Shared_query(0)
error_numpy_sigma2 = np.ndarray(buffer=buf15, dtype="d",
                         shape=(error_analysis_samples_num))

win16 = MPI.Win.Allocate_shared(nbytes_error_sigma, itemsize,
                               comm=world_comm)
buf16, itemsize = win16.Shared_query(0)
error_numpy_sigma3 = np.ndarray(buffer=buf16, dtype="d",
                         shape=(error_analysis_samples_num))

if world_comm.rank == 0:
    error_analysis_set[:, :] = \
        generate_ann_input_set(num_ann_samples=error_analysis_samples_num)
    print(f"Error analysis set generated")

world_comm.Barrier()

if world_comm.size != 1:
    error_array_list_sigma = [error_numpy_sigma0, error_numpy_sigma1,
                              error_numpy_sigma2, error_numpy_sigma3]
else:
    error_array_list_sigma = [error_numpy_sigma0]

for j in range(len(fem_comm_list)):
    if fem_comm_list[j] != MPI.COMM_NULL:
        error_analysis_indices = \
            np.arange(j, error_analysis_set.shape[0], len(fem_comm_list))
        print(f"Error analysis indices: {error_analysis_indices}")
        for i in error_analysis_indices:
            for array_num in range(len(error_array_list_sigma)):
                error_array_list_sigma[array_num][i] = \
                    error_analysis(reduced_problem, problem_parametric,
                                   error_analysis_set[i, :],
                                   ann_model_list_sigma[array_num],
                                   len(reduced_problem._basis_functions_sigma),
                                   online_nn,
                                   norm_error=reduced_problem.norm_error_sigma,
                                   reconstruct_solution=reduced_problem.reconstruct_solution_sigma,
                                   input_scaling_range=reduced_problem.input_scaling_range,
                                   output_scaling_range=reduced_problem.output_scaling_range_sigma,
                                   input_range=reduced_problem.input_range,
                                   output_range=reduced_problem.output_range_sigma,
                                   index=0
                                )
                print(f"Error analysis (sigma) {i+1} of {error_analysis_set.shape[0]}, Model {array_num}, Error: {error_array_list_sigma[array_num][i]}")

if world_comm.size == 8:
    cpu_group0_procs_u = world_comm.group.Incl([0, 1])
    cpu_group0_comm_u = world_comm.Create_group(cpu_group0_procs_u)

    cpu_group1_procs_u = world_comm.group.Incl([2, 3])
    cpu_group1_comm_u = world_comm.Create_group(cpu_group1_procs_u)

    cpu_group2_procs_u = world_comm.group.Incl([4, 5])
    cpu_group2_comm_u = world_comm.Create_group(cpu_group2_procs_u)

    cpu_group3_procs_u = world_comm.group.Incl([6, 7])
    cpu_group3_comm_u = world_comm.Create_group(cpu_group3_procs_u)

    ann_comm_list_u = \
        [cpu_group0_comm_u, cpu_group1_comm_u,
         cpu_group2_comm_u, cpu_group3_comm_u]

elif world_comm.size == 4:
    '''
    cpu_group0_procs_u = world_comm.group.Incl([0])
    cpu_group0_comm_u = \
        world_comm.Create_group(cpu_group0_procs_u)

    cpu_group1_procs_u = world_comm.group.Incl([1])
    cpu_group1_comm_u = \
        world_comm.Create_group(cpu_group1_procs_u)

    cpu_group2_procs_u = world_comm.group.Incl([2])
    cpu_group2_comm_u = \
        world_comm.Create_group(cpu_group2_procs_u)

    cpu_group3_procs_u = world_comm.group.Incl([3])
    cpu_group3_comm_u = world_comm.Create_group(cpu_group3_procs_u)

    ann_comm_list_u = \
        [cpu_group0_comm_u, cpu_group1_comm_u,
         cpu_group2_comm_u, cpu_group3_comm_u]
    '''
    cpu_group0_procs_u = world_comm.group.Incl([0, 1])
    cpu_group0_comm_u = \
        world_comm.Create_group(cpu_group0_procs_u)

    cpu_group1_procs_u = world_comm.group.Incl([2, 3])
    cpu_group1_comm_u = \
        world_comm.Create_group(cpu_group1_procs_u)

    ann_comm_list_u = \
        [cpu_group0_comm_u, cpu_group1_comm_u]

elif world_comm.size == 1:
    cpu_group0_procs_u = world_comm.group.Incl([0])
    cpu_group0_comm_u = world_comm.Create_group(cpu_group0_procs_u)

    ann_comm_list_u = [cpu_group0_comm_u]

else:
    raise NotImplementedError("Please use 1,4 or 8 processes")

print(f"Rank: {world_comm.rank}, Training set indices: {training_set_indices}, Validation set indices: {validation_set_indices}")

# ANN model
model_u0 = HiddenLayersNet(para_dim, [45, 45, 45],
                           len(reduced_problem._basis_functions_u),
                           Tanh())

model_u1 = HiddenLayersNet(para_dim, [55, 55, 55],
                           len(reduced_problem._basis_functions_u),
                           Tanh())

model_u2 = HiddenLayersNet(para_dim, [65, 65, 65],
                           len(reduced_problem._basis_functions_u),
                           Tanh())

model_u3 = HiddenLayersNet(para_dim, [75, 75, 75],
                           len(reduced_problem._basis_functions_u),
                           Tanh())

if world_comm.size == 8:
    ann_model_list_u = [model_u0, model_u1,
                        model_u2, model_u3]
    path_list_u = ["model_u0.pth", "model_u1.pth",
                   "model_u2.pth", "model_u3.pth"]
    checkpoint_path_list_u = \
        ["checkpoint_u0", "checkpoint_u1",
         "checkpoint_u2", "checkpoint_u3"]
    model_root_process_list_u = [0, 2, 4, 6]
    trained_model_path_list_u = \
        ["trained_model_u0.pth", "trained_model_u1.pth",
         "trained_model_u2.pth", "trained_model_u3.pth"]
elif world_comm.size == 4:
    ann_model_list_u = [model_u0, model_u1,
                        model_u2, model_u3]
    path_list_u = ["model_u0.pth", "model_u1.pth",
                   "model_u2.pth", "model_u3.pth"]
    checkpoint_path_list_u = \
        ["checkpoint_u0", "checkpoint_u1",
         "checkpoint_u2", "checkpoint_u3"]
    model_root_process_list_u = [0, 2] # [0, 1, 2, 3]
    trained_model_path_list_u = \
        ["trained_model_u0.pth", "trained_model_u1.pth",
         "trained_model_u2.pth", "trained_model_u3.pth"]
elif world_comm.size == 1:
    ann_model_list_u = [model_u0]
    path_list_u = ["model_u0.pth"]
    checkpoint_path_list_u = ["checkpoint_u0"]
    model_root_process_list_u = [0]
    trained_model_path_list_u = ["trained_model_u0.pth"]

for j in range(len(ann_comm_list_u)):
    if ann_comm_list_u[j] != MPI.COMM_NULL:
        # TODO see whether init_cpu_process_group needs to be called twice
        # init_cpu_process_group(ann_comm_list_u[j])

        training_set_indices_cpu_u = \
            np.arange(ann_comm_list_u[j].rank,
                      input_training_set.shape[0],
                      ann_comm_list_u[j].size)
        validation_set_indices_cpu_u = \
            np.arange(ann_comm_list_u[j].rank,
                      input_validation_set.shape[0],
                      ann_comm_list_u[j].size)

        customDataset_u = \
            CustomPartitionedDataset(reduced_problem,
                                     input_training_set,
                                     output_training_set_u,
                                     training_set_indices_cpu_u,
                                     input_scaling_range=reduced_problem.input_scaling_range,
                                     output_scaling_range=reduced_problem.output_scaling_range_u,
                                     input_range=reduced_problem.input_range,
                                     output_range=reduced_problem.output_range_u
                                     )
        train_dataloader_u = \
            DataLoader(customDataset_u, batch_size=input_training_set.shape[0], shuffle=False)# shuffle=True)

        customDataset_u = \
            CustomPartitionedDataset(reduced_problem,
                                     input_validation_set,
                                     output_validation_set_u,
                                     validation_set_indices_cpu_u,
                                     input_scaling_range=reduced_problem.input_scaling_range,
                                     output_scaling_range=reduced_problem.output_scaling_range_u,
                                     input_range=reduced_problem.input_range,
                                     output_range=reduced_problem.output_range_u
                                    )
        valid_dataloader_u = \
            DataLoader(customDataset_u, batch_size=input_validation_set.shape[0], shuffle=False)

        # save_model(ann_model_list_u[j], path_list_u[j])
        load_model(ann_model_list_u[j], path_list_u[j])

        model_synchronise(ann_model_list_u[j], verbose=False)

        # Training of ANN
        training_loss = list()
        validation_loss = list()

        max_epochs_u = 50 # 50000
        min_validation_loss_u = None
        start_epoch_u = 0
        checkpoint_epoch_u = 10

        learning_rate_u = 5.e-6 # 1.e-4
        optimiser_u = get_optimiser(ann_model_list_u[j], "Adam", learning_rate_u)
        loss_fn_u = get_loss_func("MSE", reduction="sum")

        if os.path.exists(checkpoint_path_list_u[j]):
            start_epoch_u, min_validation_loss_u = \
                load_checkpoint(checkpoint_path_list_u[j], ann_model_list_u[j],
                                optimiser_u)

        import time
        start_time = time.process_time()
        for epochs in range(start_epoch_u, max_epochs_u):
            if epochs > 0 and epochs % checkpoint_epoch_u == 0:
                save_checkpoint(checkpoint_path_list_u[j], epochs,
                                ann_model_list_u[j], optimiser_u,
                                min_validation_loss_u)
            print(f"Epoch: {epochs+1}/{max_epochs_u}")
            current_training_loss = train_nn(reduced_problem,
                                             train_dataloader_u,
                                             ann_model_list_u[j],
                                             loss_fn_u, optimiser_u)
            training_loss.append(current_training_loss)
            current_validation_loss = validate_nn(reduced_problem,
                                                  valid_dataloader_u,
                                                  ann_model_list_u[j],
                                                  loss_fn_u)
            validation_loss.append(current_validation_loss)
            if epochs > 0 and current_validation_loss > 1.01 * min_validation_loss_u:
                # 1% safety margin against min_validation_loss
                # before invoking early stopping criteria
                print(f"Early stopping criteria invoked at epoch: {epochs+1}")
                break
            min_validation_loss_u = min(validation_loss)
        end_time = time.process_time()
        elapsed_time = end_time - start_time

        os.system(f"rm {checkpoint_path_list_u[j]}")

print(f"ANN training time (U): {elapsed_time}")
world_comm.Barrier()

for j in range(len(model_root_process_list_u)):
    share_model(ann_model_list_u[j], world_comm,
                model_root_process_list_u[j])
    # save_model(ann_model_list_u[j], trained_model_path_list_u[j])

# Error analysis dataset
print("\n")
print("Generating error analysis (only input/parameters) dataset")
print("\n")

itemsize = MPI.DOUBLE.Get_size()

if world_comm.rank == 0:
    nbytes_para_u = error_analysis_samples_num * itemsize * para_dim
    nbytes_error_u = error_analysis_samples_num * itemsize
else:
    nbytes_para_u = 0
    nbytes_error_u = 0

win17 = MPI.Win.Allocate_shared(nbytes_para_u, itemsize,
                               comm=world_comm)
buf17, itemsize = win17.Shared_query(0)
error_analysis_set = \
    np.ndarray(buffer=buf17, dtype="d",
               shape=(error_analysis_samples_num,
                      para_dim))

win18 = MPI.Win.Allocate_shared(nbytes_error_u, itemsize,
                               comm=world_comm)
buf18, itemsize = win18.Shared_query(0)
error_numpy_u0 = np.ndarray(buffer=buf18, dtype="d",
                         shape=(error_analysis_samples_num))

win19 = MPI.Win.Allocate_shared(nbytes_error_u, itemsize,
                               comm=world_comm)
buf19, itemsize = win19.Shared_query(0)
error_numpy_u1 = np.ndarray(buffer=buf19, dtype="d",
                         shape=(error_analysis_samples_num))

win20 = MPI.Win.Allocate_shared(nbytes_error_u, itemsize,
                               comm=world_comm)
buf20, itemsize = win20.Shared_query(0)
error_numpy_u2 = np.ndarray(buffer=buf20, dtype="d",
                         shape=(error_analysis_samples_num))

win21 = MPI.Win.Allocate_shared(nbytes_error_u, itemsize,
                               comm=world_comm)
buf21, itemsize = win21.Shared_query(0)
error_numpy_u3 = np.ndarray(buffer=buf21, dtype="d",
                         shape=(error_analysis_samples_num))

if world_comm.rank == 0:
    error_analysis_set[:, :] = \
        generate_ann_input_set(num_ann_samples=error_analysis_samples_num)
    print(f"Error analysis set generated")

world_comm.Barrier()

if world_comm.size != 1:
    error_array_list_u = [error_numpy_u0, error_numpy_u1,
                          error_numpy_u2, error_numpy_u3]
else:
    error_array_list_u = [error_numpy_u0]

for j in range(len(fem_comm_list)):
    if fem_comm_list[j] != MPI.COMM_NULL:
        error_analysis_indices = \
            np.arange(j, error_analysis_set.shape[0], len(fem_comm_list))
        print(f"Error analysis indices: {error_analysis_indices}")
        for i in error_analysis_indices:
            for array_num in range(len(error_array_list_u)):
                error_array_list_u[array_num][i] = \
                    error_analysis(reduced_problem, problem_parametric,
                                   error_analysis_set[i, :],
                                   ann_model_list_u[array_num],
                                   len(reduced_problem._basis_functions_u),
                                   online_nn,
                                   norm_error=reduced_problem.norm_error_u,
                                   reconstruct_solution=reduced_problem.reconstruct_solution_u,
                                   input_scaling_range=reduced_problem.input_scaling_range,
                                   output_scaling_range=reduced_problem.output_scaling_range_u,
                                   input_range=reduced_problem.input_range,
                                   output_range=reduced_problem.output_range_u,
                                   index=1
                                )
                print(f"Error analysis (U) {i+1} of {error_analysis_set.shape[0]}, Model {array_num}, Error: {error_array_list_u[array_num][i]}")

if fem_comm_list[0] != MPI.COMM_NULL:
    # Online phase at parameter online_mu
    online_mu = np.array([1., 0.4, 0.55, 0.27, 3.])
    fem_start_time_0 = time.process_time()
    fem_solution_sigma, fem_solution_u = problem_parametric.solve(online_mu)
    fem_end_time_0 = time.process_time()
    # First compute the RB solution using online_nn.
    # Next this solution is reconstructed on FE space
    rb_start_time_0 = time.process_time()
    rb_solution_sigma = \
        reduced_problem.reconstruct_solution_sigma(
            online_nn(reduced_problem, problem_parametric, online_mu, model_sigma0,
                      len(reduced_problem._basis_functions_sigma),
                      input_scaling_range=reduced_problem.input_scaling_range,
                      output_scaling_range=reduced_problem.output_scaling_range_sigma,
                      input_range=reduced_problem.input_range,
                      output_range=reduced_problem.output_range_sigma
                      ))
            # TODO Replace model_sigma0 with best model model_sigmaX
    rb_end_time_0 = time.process_time()

    rb_start_time_1 = time.process_time()
    rb_solution_u = \
        reduced_problem.reconstruct_solution_u(
            online_nn(reduced_problem, problem_parametric, online_mu, model_u0,
                      len(reduced_problem._basis_functions_u),
                      input_scaling_range=reduced_problem.input_scaling_range,
                      output_scaling_range=reduced_problem.output_scaling_range_u,
                      input_range=reduced_problem.input_range,
                      output_range=reduced_problem.output_range_u
                      ))
            # TODO Replace model_u0 with best model model_uX
    rb_end_time_1 = time.process_time()

    # Post processing
    # TODO make plotting work on CSD3

    '''
    fem_online_file \
        = "dlrbnicsx_solution_mixed_poisson_0/fem_online_mu_computed_sigma.xdmf"
    with dolfinx.io.XDMFFile(mesh.comm, fem_online_file,
                            "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(fem_solution_sigma)

    rb_online_file \
        = "dlrbnicsx_solution_mixed_poisson_0/rb_online_mu_computed_sigma.xdmf"
    with dolfinx.io.XDMFFile(mesh.comm, rb_online_file,
                            "w") as solution_file:
        # NOTE scatter_forward not considered for online solution
        solution_file.write_mesh(mesh)
        solution_file.write_function(rb_solution_sigma)

    error_function_sigma = dolfinx.fem.Function(problem_parametric._Q)
    error_function_sigma.vector[rstart_sigma:rend_sigma] = \
        abs(fem_solution_sigma.vector[rstart_sigma:rend_sigma] -
            rb_solution_sigma.vector[rstart_sigma:rend_sigma])
    fem_rb_error_file \
        = "dlrbnicsx_solution_mixed_poisson_0/fem_rb_error_computed_sigma.xdmf"
    with dolfinx.io.XDMFFile(mesh.comm, fem_rb_error_file,
                            "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(error_function_sigma)
    '''

    print(f"FEM time 0: {fem_end_time_0 - fem_start_time_0}")
    print(f"RB time (sigma) 0: {rb_end_time_0 - rb_start_time_0}")
    print(f"RB time (u) 0: {rb_end_time_1 - rb_start_time_1}")
    print(f"Speedup 0: {(fem_end_time_0 - fem_start_time_0)/((rb_end_time_0 - rb_start_time_0) + (rb_end_time_1 - rb_start_time_1))}")
