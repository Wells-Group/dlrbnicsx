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
        ksp.destroy()
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
        self.regularisation = "EarlyStopping"

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

# MPI coomunicator variables
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

elif world_comm.size == 1:
    fem0_procs = world_comm.group.Incl([0])
    fem0_procs_comm = world_comm.Create_group(fem0_procs)
    # fem0_procs_comm = MPI.COMM_WORLD or MPI.COMM_SELF
    fem_comm_list = [fem0_procs_comm]

else:
    raise NotImplementedError("Please use 1, 4 or 8 processes")

# Import mesh in dolfinx
gdim = 3

for comm_i in fem_comm_list:
    if comm_i != MPI.COMM_NULL:
        mesh_comm = comm_i

gmsh_model_rank = 0
mesh, subdomains, boundaries = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/3d_mesh.msh", mesh_comm,
                                    gmsh_model_rank, gdim=gdim)
# Boundary markers: x=1 is 22, x=0 is 30, y=1 is 26, y=0 is 18, z=1 is 31, z=0 is 1

# Parameters
mu = np.array([-2., 0.5, 0.5, 0.5, 3.])

# FEM solve
problem_parametric = ParametricProblem(mesh, subdomains,
                                       boundaries)
sigma_sol, u_sol = problem_parametric.solve(mu)
print(sigma_sol.x.array, np.sqrt((mesh.comm.allreduce(np.linalg.norm(sigma_sol.x.array)**2, op=MPI.SUM))))
print(u_sol.x.array, np.sqrt((mesh.comm.allreduce(np.linalg.norm(u_sol.x.array)**2, op=MPI.SUM))))

itemsize = MPI.DOUBLE.Get_size()
para_dim_sigma = 5
rstart_sigma, rend_sigma = sigma_sol.vector.getOwnershipRange()
num_dofs_sigma = mesh_comm.allreduce(rend_sigma, op=MPI.MAX) - mesh_comm.allreduce(rstart_sigma, op=MPI.MIN)
rstart_u, rend_u = u_sol.vector.getOwnershipRange()
num_dofs_u = mesh_comm.allreduce(rend_u, op=MPI.MAX) - mesh_comm.allreduce(rstart_u, op=MPI.MIN)

num_pod_samples_sigma = [3, 2, 4, 3, 2] # [4, 3, 4, 3, 2]
num_projection_error_samples_sigma = 100 # 200
num_ann_samples_sigma = 300
num_error_analysis_samples_sigma = 100
num_snapshots_sigma = np.product(num_pod_samples_sigma)
nbytes_para_sigma = itemsize * num_snapshots_sigma * para_dim_sigma
nbytes_dofs_sigma = itemsize * num_snapshots_sigma * num_dofs_sigma

# TODO more benchmarking with dolfinx implementation for correctness

def generate_training_set(samples=num_pod_samples_sigma):
    # Select input samples for POD
    training_set_0 = np.linspace(-1., 1., samples[0])
    training_set_1 = np.linspace(0.4, 0.6, samples[1])
    training_set_2 = np.linspace(0.4, 0.6, samples[2])
    training_set_3 = np.linspace(0.4, 0.6, samples[3])
    training_set_4 = np.linspace(2.5, 3.5, samples[4])
    training_set = np.array(list(itertools.product(training_set_0,
                                                   training_set_1,
                                                   training_set_2,
                                                   training_set_3,
                                                   training_set_4)))
    return training_set

if world_comm.rank == 0:
    nbytes_para_sigma = num_snapshots_sigma * para_dim_sigma * itemsize
else:
    nbytes_para_sigma = 0

win0 = MPI.Win.Allocate_shared(nbytes_para_sigma, itemsize, comm=MPI.COMM_WORLD)
buf0, itemsize = win0.Shared_query(0)
para_matrix_sigma = np.ndarray(buffer=buf0, dtype="d",
                               shape=(num_snapshots_sigma, para_dim_sigma))

if world_comm.rank == 0:
    para_matrix_sigma[:, :] = generate_training_set()

world_comm.barrier()

for i in range(len(fem_comm_list)):
    if fem_comm_list[i] != MPI.COMM_NULL:
        cpu_indices_sigma = np.arange(i, para_matrix_sigma.shape[0], len(fem_comm_list))

print(f"cpu_indices_sigma: {cpu_indices_sigma}")

if world_comm.rank == 0:
    nbytes_dofs_sigma = num_snapshots_sigma * num_dofs_sigma * itemsize
else:
    nbytes_dofs_sigma = 0

world_comm.barrier()

win1 = MPI.Win.Allocate_shared(nbytes_dofs_sigma, itemsize, comm=MPI.COMM_WORLD)
buf1, itemsize = win1.Shared_query(0)
snapshot_arrays_sigma = np.ndarray(buffer=buf1, dtype="d",
                             shape=(num_snapshots_sigma, num_dofs_sigma))
snapshots_matrix_sigma = rbnicsx.backends.FunctionsList(problem_parametric._Q)
Nmax_sigma = 50

# NOTE  Redundant check of if fem_comm_list[i] != MPI.COMM_NULL is removed

for mu_index in cpu_indices_sigma:
    print(f"Parameter number {mu_index+1} of {para_matrix_sigma.shape[0]}: {para_matrix_sigma[mu_index,:]}")
    solution_sigma, solution_u = problem_parametric.solve(para_matrix_sigma[mu_index, :])
    snapshot_arrays_sigma[mu_index, rstart_sigma:rend_sigma] = solution_sigma.vector[rstart_sigma:rend_sigma]

world_comm.barrier()

for i in range(snapshot_arrays_sigma.shape[0]):
    solution_empty = dolfinx.fem.Function(problem_parametric._Q)
    solution_empty.vector[rstart_sigma:rend_sigma] = snapshot_arrays_sigma[i, rstart_sigma:rend_sigma]
    solution_empty.x.scatter_forward()
    # TODO Recall why .assemble()?
    solution_empty.vector.assemble()
    snapshots_matrix_sigma.append(solution_empty)

print("Set up reduced problem")
reduced_problem = PODANNReducedProblem(problem_parametric)

print("")

print(rbnicsx.io.TextLine("Perform POD", fill="#"))
eigenvalues_sigma, modes_sigma, _ = \
    rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_matrix_sigma,
                                    problem_parametric._inner_product_sigma_action,
                                    N=Nmax_sigma, tol=1e-6)

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

# ### POD Ends ###

del(snapshot_arrays_sigma)

sigma_sol_projected = reduced_problem.project_snapshot_sigma(sigma_sol, reduced_size_sigma)
sigma_sol_reconstructed = reduced_problem.reconstruct_solution_sigma(sigma_sol_projected)
print(sigma_sol_reconstructed.x.array.shape, sigma_sol.x.array.shape)
sigma_norm = reduced_problem.compute_norm_sigma(sigma_sol_reconstructed)
sigma_error = reduced_problem.norm_error_sigma(sigma_sol, sigma_sol_reconstructed)
print(f"Norm reconstructed: {sigma_norm}, Error: {sigma_error}")

# ### Projection error samples ###
# Creating dataset
def generate_projection_error_set(num_projection_samples=10):
    xlimits = np.array([[-1., 1.], [0.4, 0.6],
                        [0.4, 0.6], [0.4, 0.6],
                        [2.5, 3.5]])
    sampling = LHS(xlimits=xlimits)
    training_set = sampling(num_projection_samples)
    return training_set

if world_comm.rank == 0:
    nbytes_para_projection_error_sigma = num_projection_error_samples_sigma * itemsize * para_dim_sigma
    nbytes_projection_error_array_sigma = num_projection_error_samples_sigma * itemsize
else:
    nbytes_para_projection_error_sigma = 0
    nbytes_projection_error_array_sigma = 0

world_comm.barrier()

win02 = MPI.Win.Allocate_shared(nbytes_para_projection_error_sigma, itemsize,
                                comm=MPI.COMM_WORLD)
buf02, itemsize = win02.Shared_query(0)
projection_error_samples_sigma = \
    np.ndarray(buffer=buf02, dtype="d",
               shape=(num_projection_error_samples_sigma, para_dim_sigma))

win03 = MPI.Win.Allocate_shared(nbytes_projection_error_array_sigma, itemsize,
                               comm=MPI.COMM_WORLD)
buf03, itemsize = win03.Shared_query(0)
projection_error_array_sigma = \
    np.ndarray(buffer=buf03, dtype="d",
               shape=(num_projection_error_samples_sigma))

if world_comm.rank == 0:
    projection_error_samples_sigma[:, :] = \
        generate_projection_error_set(num_projection_samples=num_projection_error_samples_sigma)

world_comm.Barrier()

for j in range(len(fem_comm_list)):
    if fem_comm_list[j] != MPI.COMM_NULL:
        projection_error_indices_sigma = \
            np.arange(j, projection_error_samples_sigma.shape[0],
                      len(fem_comm_list))

print(f"Rank: {world_comm.rank}, Indices (projection error): {projection_error_indices_sigma}")

for k in projection_error_indices_sigma:
    fem_sol_sigma, _ = problem_parametric.solve(projection_error_samples_sigma[k, :])
    reconstructed_sol = \
        reduced_problem.reconstruct_solution_sigma(
            reduced_problem.project_snapshot_sigma(fem_sol_sigma,
                                                   reduced_size_sigma))
    projection_error_array_sigma[k] = \
        reduced_problem.norm_error_sigma(fem_sol_sigma, reconstructed_sol)

print(f"Rank: {world_comm.rank}, Projection error: {projection_error_array_sigma[projection_error_indices_sigma]}")

# ### Projection error ends ###
exit()


# Creating dataset
def generate_ann_input_set(num_ann_samples=10):
    xlimits = np.array([[-1., 1.], [0.4, 0.6],
                        [0.4, 0.6], [0.4, 0.6],
                        [2.5, 3.5]])
    sampling = LHS(xlimits=xlimits)
    training_set = sampling(num_ann_samples)
    return training_set

def generate_ann_output_set_sigma(problem, reduced_problem, input_set,
                                  output_set_sigma, indices, mode=None):
    # Solve the FE problem at given input_sets and
    # project on the RB space
    rb_size_sigma = len(reduced_problem._basis_functions_sigma)
    for i in indices:
        if mode is None:
            print(f"Parameter {i+1}/{input_set.shape[0]}")
        else:
            print(f"{mode} parameter number {i+1}/{input_set.shape[0]}")
        solution_sigma, _ = problem.solve(input_set[i, :])
        output_set_sigma[i, :] = \
            reduced_problem.project_snapshot_sigma(solution_sigma,
                                                   rb_size_sigma).array

num_ann_input_samples_sigma = np.product(num_ann_samples_sigma)
num_training_samples_sigma = int(0.7 * num_ann_input_samples_sigma)
num_validation_samples_sigma = \
    num_ann_input_samples_sigma - int(0.7 * num_ann_input_samples_sigma)
itemsize = MPI.DOUBLE.Get_size()

if world_comm.rank == 0:
    ann_input_set_sigma = generate_ann_input_set(num_ann_samples=num_ann_samples_sigma)
    np.random.shuffle(ann_input_set_sigma)
    nbytes_para_ann_training_sigma = num_training_samples_sigma * itemsize * para_dim_sigma
    nbytes_dofs_ann_training_sigma = num_training_samples_sigma * itemsize * \
        len(reduced_problem._basis_functions_sigma)
    nbytes_para_ann_validation_sigma = num_validation_samples_sigma * itemsize * para_dim_sigma
    nbytes_dofs_ann_validation_sigma = num_validation_samples_sigma * itemsize * \
        len(reduced_problem._basis_functions_sigma)
else:
    nbytes_para_ann_training_sigma = 0
    nbytes_dofs_ann_training_sigma = 0
    nbytes_para_ann_validation_sigma = 0
    nbytes_dofs_ann_validation_sigma = 0

world_comm.barrier()

win2 = MPI.Win.Allocate_shared(nbytes_para_ann_training_sigma, itemsize,
                               comm=MPI.COMM_WORLD)
buf2, itemsize = win2.Shared_query(0)
input_training_set_sigma = \
    np.ndarray(buffer=buf2, dtype="d",
               shape=(num_training_samples_sigma, para_dim_sigma))

win3 = MPI.Win.Allocate_shared(nbytes_para_ann_validation_sigma, itemsize,
                               comm=MPI.COMM_WORLD)
buf3, itemsize = win3.Shared_query(0)
input_validation_set_sigma = \
    np.ndarray(buffer=buf3, dtype="d",
               shape=(num_validation_samples_sigma, para_dim_sigma))

win4 = MPI.Win.Allocate_shared(nbytes_dofs_ann_training_sigma, itemsize,
                               comm=MPI.COMM_WORLD)
buf4, itemsize = win4.Shared_query(0)
output_training_set_sigma = \
    np.ndarray(buffer=buf4, dtype="d",
               shape=(num_training_samples_sigma,
                      len(reduced_problem._basis_functions_sigma)))

win5 = MPI.Win.Allocate_shared(nbytes_dofs_ann_validation_sigma, itemsize,
                               comm=MPI.COMM_WORLD)
buf5, itemsize = win5.Shared_query(0)
output_validation_set_sigma = \
    np.ndarray(buffer=buf5, dtype="d",
               shape=(num_validation_samples_sigma,
                      len(reduced_problem._basis_functions_sigma)))

if world_comm.rank == 0:
    input_training_set_sigma[:, :] = \
        ann_input_set_sigma[:num_training_samples_sigma, :]
    input_validation_set_sigma[:, :] = \
        ann_input_set_sigma[num_training_samples_sigma:, :]
    output_training_set_sigma[:, :] = \
        np.zeros([num_training_samples_sigma,
                  len(reduced_problem._basis_functions_sigma)])
    output_validation_set_sigma[:, :] = \
        np.zeros([num_validation_samples_sigma,
                  len(reduced_problem._basis_functions_sigma)])

world_comm.Barrier()

for j in range(len(fem_comm_list)):
    if fem_comm_list[j] != MPI.COMM_NULL:
        training_set_indices_sigma = \
            np.arange(j, input_training_set_sigma.shape[0], len(fem_comm_list))

        validation_set_indices_sigma = \
            np.arange(j, input_validation_set_sigma.shape[0], len(fem_comm_list))

# world_comm.Barrier()

# Training dataset
generate_ann_output_set_sigma(problem_parametric, reduced_problem,
                              input_training_set_sigma, output_training_set_sigma,
                              training_set_indices_sigma, mode="Training")

generate_ann_output_set_sigma(problem_parametric, reduced_problem,
                        input_validation_set_sigma, output_validation_set_sigma,
                        validation_set_indices_sigma, mode="Validation")

world_comm.Barrier()

reduced_problem.output_range_sigma[0] = min(np.min(output_training_set_sigma), np.min(output_validation_set_sigma))
reduced_problem.output_range_sigma[1] = max(np.max(output_training_set_sigma), np.max(output_validation_set_sigma))

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

elif world_comm.size == 1:
    cpu_group0_procs_sigma = world_comm.group.Incl([0])
    cpu_group0_comm_sigma = world_comm.Create_group(cpu_group0_procs_sigma)
    
    ann_comm_list_sigma = [cpu_group0_comm_sigma]

else:
    raise NotImplementedError("Please use 1,4 or 8 processes")

# ANN model
model_sigma0 = HiddenLayersNet(input_training_set_sigma.shape[1], [50, 50, 50],
                         len(reduced_problem._basis_functions_sigma), Tanh())

model_sigma1 = HiddenLayersNet(input_training_set_sigma.shape[1], [55, 55, 55],
                         len(reduced_problem._basis_functions_sigma), Tanh())

model_sigma2 = HiddenLayersNet(input_training_set_sigma.shape[1], [60, 60, 60],
                         len(reduced_problem._basis_functions_sigma), Tanh())

model_sigma3 = HiddenLayersNet(input_training_set_sigma.shape[1], [65, 65, 65],
                         len(reduced_problem._basis_functions_sigma), Tanh())

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
    model_root_process_list_sigma = [0, 1, 2, 3]
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
                      input_training_set_sigma.shape[0],
                      ann_comm_list_sigma[j].size)
        validation_set_indices_cpu_sigma = \
            np.arange(ann_comm_list_sigma[j].rank,
                      input_validation_set_sigma.shape[0],
                      ann_comm_list_sigma[j].size)
        
        customDataset_sigma = \
            CustomPartitionedDataset(reduced_problem,
                                     input_training_set_sigma,
                                     output_training_set_sigma,
                                     training_set_indices_cpu_sigma,
                                     input_scaling_range=reduced_problem.input_scaling_range,
                                     output_scaling_range=reduced_problem.output_scaling_range_sigma,
                                     input_range=reduced_problem.input_range,
                                     output_range=reduced_problem.output_range_sigma
                                     )
        train_dataloader_sigma = \
            DataLoader(customDataset_sigma, batch_size=input_training_set_sigma.shape[0], shuffle=False)# shuffle=True)

        customDataset_sigma = \
            CustomPartitionedDataset(reduced_problem,
                                     input_validation_set_sigma,
                                     output_validation_set_sigma,
                                     validation_set_indices_cpu_sigma,
                                     input_scaling_range=reduced_problem.input_scaling_range,
                                     output_scaling_range=reduced_problem.output_scaling_range_sigma,
                                     input_range=reduced_problem.input_range,
                                     output_range=reduced_problem.output_range_sigma
                                    )
        valid_dataloader_sigma = \
            DataLoader(customDataset_sigma, batch_size=input_validation_set_sigma.shape[0], shuffle=False)
        
        save_model(ann_model_list_sigma[j], path_list_sigma[j])
        # load_model(ann_model_list_sigma[j], path_list_sigma[j])
        
        model_synchronise(ann_model_list_sigma[j], verbose=False)
        
        # Training of ANN
        training_loss = list()
        validation_loss = list()
        
        max_epochs_sigma = 50000
        min_validation_loss_sigma = None
        start_epoch_sigma = 0
        checkpoint_epoch_sigma = 10
        
        learning_rate_sigma = 1.e-4
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
            if epochs > 0 and current_validation_loss > 1.01 * min_validation_loss_sigma \
            and reduced_problem.regularisation == "EarlyStopping":
                # 1% safety margin against min_validation_loss
                # before invoking early stopping criteria
                print(f"Early stopping criteria invoked at epoch: {epochs+1}")
                break
            min_validation_loss_sigma = min(validation_loss)
        end_time = time.process_time()
        elapsed_time = end_time - start_time

        os.system(f"rm {checkpoint_path_list_sigma[j]}")

world_comm.Barrier()

for j in range(len(model_root_process_list_sigma)):
    share_model(ann_model_list_sigma[j], world_comm,
                model_root_process_list_sigma[j])
    save_model(ann_model_list_sigma[j], trained_model_path_list_sigma[j])

# Error analysis dataset
print("\n")
print("Generating error analysis (only input/parameters) dataset")
print("\n")

error_analysis_num_para_sigma = np.product(num_error_analysis_samples_sigma)
itemsize = MPI.DOUBLE.Get_size()

if world_comm.rank == 0:
    nbytes_para_sigma = error_analysis_num_para_sigma * itemsize * para_dim_sigma
    nbytes_error_sigma = error_analysis_num_para_sigma * itemsize
else:
    nbytes_para_sigma = 0
    nbytes_error_sigma = 0

win6 = MPI.Win.Allocate_shared(nbytes_para_sigma, itemsize,
                               comm=world_comm)
buf6, itemsize = win6.Shared_query(0)
error_analysis_set_sigma = \
    np.ndarray(buffer=buf6, dtype="d",
               shape=(error_analysis_num_para_sigma,
                      para_dim_sigma))

win7 = MPI.Win.Allocate_shared(nbytes_error_sigma, itemsize,
                               comm=world_comm)
buf7, itemsize = win7.Shared_query(0)
error_numpy_sigma0 = np.ndarray(buffer=buf7, dtype="d",
                         shape=(error_analysis_num_para_sigma))

win8 = MPI.Win.Allocate_shared(nbytes_error_sigma, itemsize,
                               comm=world_comm)
buf8, itemsize = win8.Shared_query(0)
error_numpy_sigma1 = np.ndarray(buffer=buf8, dtype="d",
                         shape=(error_analysis_num_para_sigma))

win9 = MPI.Win.Allocate_shared(nbytes_error_sigma, itemsize,
                               comm=world_comm)
buf9, itemsize = win9.Shared_query(0)
error_numpy_sigma2 = np.ndarray(buffer=buf9, dtype="d",
                         shape=(error_analysis_num_para_sigma))

win10 = MPI.Win.Allocate_shared(nbytes_error_sigma, itemsize,
                               comm=world_comm)
buf10, itemsize = win10.Shared_query(0)
error_numpy_sigma3 = np.ndarray(buffer=buf10, dtype="d",
                         shape=(error_analysis_num_para_sigma))

if world_comm.rank == 0:
    error_analysis_set_sigma[:, :] = \
        generate_ann_input_set(num_ann_samples=num_error_analysis_samples_sigma)
    print(f"Error analysis set generated")

world_comm.Barrier()

if world_comm.size != 1:
    error_array_list_sigma = [error_numpy_sigma0, error_numpy_sigma1,
                        error_numpy_sigma2, error_numpy_sigma3]
else:
    error_array_list_sigma = [error_numpy_sigma0]

for j in range(len(fem_comm_list)):
    if fem_comm_list[j] != MPI.COMM_NULL:
        error_analysis_indices_sigma = \
            np.arange(j, error_analysis_set_sigma.shape[0], len(fem_comm_list))
        print(f"Error analysis indices (sigma): {error_analysis_indices_sigma}")
        for i in error_analysis_indices_sigma:
            for array_num in range(len(error_array_list_sigma)):
                error_array_list_sigma[array_num][i] = \
                    error_analysis(reduced_problem, problem_parametric,
                                error_analysis_set_sigma[i, :],
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
                print(f"Error analysis (sigma) {i+1} of {error_analysis_set_sigma.shape[0]}, Model {array_num}, Error: {error_array_list_sigma[array_num][i]}")

if fem_comm_list[0] != MPI.COMM_NULL:
    # Online phase at parameter online_mu
    online_mu = np.array([1., 0.4, 0.55, 0.27, 3.])
    fem_start_time_0 = time.process_time()
    fem_solution_sigma, _ = problem_parametric.solve(online_mu)
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
    print(f"RB time 0: {rb_end_time_0 - rb_start_time_0}")
    print(f"Speedup 0: {(fem_end_time_0 - fem_start_time_0)/(rb_end_time_0 - rb_start_time_0)}")
