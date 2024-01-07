import dolfinx
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, \
    create_vector, apply_lifting, set_bc, NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
import basix
import ufl

import rbnicsx
import rbnicsx.backends
import rbnicsx.online

from mpi4py import MPI
from petsc4py import PETSc
import sympy

import numpy as np
import itertools
import abc
import matplotlib.pyplot as plt
import os

class ProblemParametric(abc.ABC):
    def __init__(self, mesh, subdomains, boundaries):
        self._mesh = mesh
        self._subdomains = subdomains
        self._boundaries = boundaries
        self.dx = ufl.Measure("dx", domain=self._mesh, subdomain_data=self._subdomains)
        self.ds = ufl.Measure("ds", domain=self._mesh, subdomain_data=self._boundaries)
        self.pol_degree = 2
        self.gdim = self._mesh.geometry.dim
        Theta_el = basix.ufl.element("Lagrange", self._mesh.basix_cell(), self.pol_degree)
        Mu_el = basix.ufl.element("Lagrange", self._mesh.basix_cell(), self.pol_degree)
        U_el = basix.ufl.element("Lagrange", self._mesh.basix_cell(), self.pol_degree, shape=(self._mesh.geometry.dim,))
        V_el = basix.ufl.mixed_element([Theta_el, Mu_el, U_el])
        self._V = dolfinx.fem.FunctionSpace(self._mesh, V_el)
        self._U, _ = self._V.sub(2).sub(1).collapse()
        self._V_x = dolfinx.fem.VectorFunctionSpace(self._mesh, ("Lagrange", self.pol_degree))
        self._x = dolfinx.fem.Function(self._V_x)
        self._theta_trial, self._mu_trial, self._u_trial = ufl.TrialFunctions(self._V)
        self._theta_test, self._mu_test, self._u_test = ufl.TestFunctions(self._V)
        # TODO check whether self._x[0] is correct choice below inner products. Here, self._x is NOT ufl.spatialCoordinate

        Theta, _ = self._V.sub(0).collapse()
        Mu, _ = self._V.sub(1).collapse()
        U, _ = self._V.sub(2).collapse()
        self._theta_sub_trial, self._theta_sub_test = ufl.TrialFunction(Theta), ufl.TestFunction(Theta)
        self._mu_sub_trial, self._mu_sub_test = ufl.TrialFunction(Mu), ufl.TestFunction(Mu)
        self._u_sub_trial, self._u_sub_test = ufl.TrialFunction(U), ufl.TestFunction(U)

        '''
        # H^1_r for theta
        self._inner_product_theta = ufl.inner(self._theta_sub_trial, self._theta_sub_test) * self._x[0] * ufl.dx + \
            ufl.inner(ufl.grad(self._theta_sub_trial), ufl.grad(self._theta_sub_test)) * self._x[0] * ufl.dx
        self._inner_product_theta_action = \
            rbnicsx.backends.bilinear_form_action(self._inner_product_theta,
                                                  part="real")
        # L^2_r for mu
        self._inner_product_mu = ufl.inner(self._mu_sub_trial, self._mu_sub_test) * self._x[0] * ufl.dx
        self._inner_product_mu_action = \
            rbnicsx.backends.bilinear_form_action(self._inner_product_mu,
                                                  part="real")
        # H^1_r for u
        self._inner_product_u = ufl.inner(self._u_sub_trial, self._u_sub_test) * self._x[0] * ufl.dx + \
            ufl.inner(ufl.grad(self._u_sub_trial), ufl.grad(self._u_sub_test)) * self._x[0] * ufl.dx
        self._inner_product_u_action = \
            rbnicsx.backends.bilinear_form_action(self._inner_product_u,
                                                  part="real")
        '''

        self._sol_previous = dolfinx.fem.Function(self._V)
        self._theta_previous, self._mu_previous, self._u_previous = ufl.split(self._sol_previous)
        self._sol_current = dolfinx.fem.Function(self._V)
        self._theta_current, self._mu_current, self._u_current = ufl.split(self._sol_current)
        self._theta_current = ufl.variable(self._theta_current)
        v_oc = - self._theta_current + 4.5
        self.dv_oc_dtheta = ufl.diff(v_oc, self._theta_current)

        self.mu_0 = dolfinx.fem.Constant(self._mesh, PETSc.ScalarType(0.)) # Only for initialisation of self.mu_0
        self.mu_1 = dolfinx.fem.Constant(self._mesh, PETSc.ScalarType(0.)) # Only for initialisation of self.mu_1
        self.stiffness_tensor = ufl.as_tensor([
            [259.e9, 75.e9, 107.e9, 0.], [75.e9, 194.e9, 75.e9, 0.],
            [107.e9, 75.e9, 259.e9, 0.], [0., 0., 0., 59.e9]])
        self.num_steps = dolfinx.fem.Constant(self._mesh, PETSc.ScalarType(5)) # dolfinx.fem.Constant(self._mesh, PETSc.ScalarType(20))
        self.dt = (3600 / self.mu_0) / self.num_steps
        self.i_s = 4780 * self.mu_0 * self.mu_1 * 210 / 4

    def _inner_product_theta_action(self, fun_j):
        def _(fun_i):
            return fun_i.vector.dot(fun_j.vector)
        return _

    def _inner_product_mu_action(self, fun_j):
        def _(fun_i):
            return fun_i.vector.dot(fun_j.vector)
        return _

    def _inner_product_u_action(self, fun_j):
        def _(fun_i):
            return fun_i.vector.dot(fun_j.vector)
        return _

    def center_marker(self, x):
        return np.logical_and(np.isclose(x[0], np.zeros_like(x[0]), atol=1.e-10),
                              np.isclose(x[1], np.zeros_like(x[1]), atol=1.e-10))

    @property
    def assemble_bcs(self):
        dofs_sym_x_1 = dolfinx.fem.locate_dofs_topological(self._V.sub(2).sub(0), self.gdim-1, self._boundaries.find(5))
        bc_sym_x_1 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_sym_x_1, self._V.sub(2).sub(0))

        dofs_sym_x_2 = dolfinx.fem.locate_dofs_topological(self._V.sub(2).sub(0), self.gdim-1, self._boundaries.find(6))
        bc_sym_x_2 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_sym_x_2, self._V.sub(2).sub(0))

        center_disp_dofs = dolfinx.fem.locate_dofs_geometrical((self._V.sub(2).sub(1), self._U), self.center_marker)
        bc_center = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), center_disp_dofs[0], self._V.sub(2).sub(1))
        bcs = [bc_sym_x_1, bc_sym_x_2, bc_center]
        return bcs

    def diffusivity_Li(self, theta_current):
        conditions = [ufl.le(theta_current, 0.23), ufl.And(ufl.ge(theta_current, 0.23), ufl.le(theta_current, 0.53)), ufl.And(ufl.ge(theta_current, 0.53), ufl.le(theta_current, 0.61)), ufl.And(ufl.ge(theta_current, 0.61), ufl.le(theta_current, 0.9)), ufl.ge(theta_current, 0.9)]
        interps = [PETSc.ScalarType(0.625e-15), 6.84665027273618e-14 * theta_current**3 - 9.34962575536366e-14 * theta_current**2 + 4.25736371145718e-14 * theta_current - 5.05401645044794e-15, -2.56806952920992e-13 * theta_current**3 + 4.23688536927247e-13 * theta_current**2 - 2.31534303960296e-13 * theta_current + 4.33717198061121e-14, 1.17694155221544e-13 * theta_current**3 - 2.61648490973595e-13 * theta_current**2 + 1.86521283059217e-13 * theta_current - 4.1632916221189e-14, PETSc.ScalarType(0.1e-15)]
        assert len(conditions) == len(interps)
        d_func = ufl.conditional(conditions[0], interps[0], interps[0])
        for i in range(1, len(conditions)):
            d_func = ufl.conditional(conditions[i], interps[i], d_func)
        return ufl.as_tensor([[d_func, 0], [0, 0]])

    def epsilon(self, u):
        return ufl.as_vector([u[0].dx(0), u[1].dx(1), u[0]/self._x[0], (u[0].dx(1) + u[1].dx(0))])

    def epsilon_theta(self, theta_current):
        epsilon_a = 0.2572 * theta_current**5 - 0.7367 * theta_current**4 + 0.7185 * theta_current**3 - 0.2602 * theta_current**2 + 0.0446 * theta_current - 0.0025
        epsilon_b = epsilon_a
        epsilon_c = 0.2362 * theta_current**5 - 1.1269 * theta_current**4 + 2.0545 * theta_current**3 - 1.7512 * theta_current**2 + 0.6531 * theta_current - 0.0489
        return ufl.as_vector([epsilon_a, epsilon_c, epsilon_b, 0.])

    @property
    def residual_term(self):
        n_L = 49200 # Molar density of Lattice sites
        f_far = 96485.3321 # Faraday constant
        r_gas = 8.314 # Gas constant R
        ref_T = 298. # Reference temperature
        a0 = n_L * ufl.inner(self._theta_current - self._theta_previous, self._theta_test) * self._x[0] * self.dx - self.dt * (n_L / (r_gas * ref_T)) * self._theta_current * ufl.inner(self.diffusivity_Li(self._theta_current) * (f_far * self.dv_oc_dtheta * ufl.grad(self._theta_current) - ufl.grad(self._mu_current)), ufl.grad(self._theta_test)) * self._x[0] * self.dx + self.dt * (self.i_s / f_far) * self._theta_test * self._x[0] * (self.ds(2) + self.ds(3))
        a1 = ufl.inner(self._mu_current, self._mu_test) * self._x[0] * self.dx + (1 / n_L) * ufl.inner(ufl.inner(self.stiffness_tensor * (self.epsilon(self._u_current) - self.epsilon_theta(self._theta_current)), ufl.diff(self.epsilon_theta(self._theta_current), self._theta_current)), self._mu_test) * self._x[0] * self.dx
        a2 = ufl.inner(self.stiffness_tensor * self.epsilon(self._u_current), self.epsilon(self._u_test)) * self._x[0] * self.dx - ufl.inner(self.stiffness_tensor * self.epsilon_theta(self._theta_current), self.epsilon(self._u_test)) * self._x[0] * self.dx
        return a0 + a1 + a2

    @property
    def set_problem(self):
        problem = \
            NonlinearProblem(self.residual_term, self._sol_current,
                             bcs=self.assemble_bcs)
        return problem

    def solve(self, mu):
        # TODO stress computation
        self._sol_current.x.array[:] = 0.
        self._sol_current.x.scatter_forward()
        self._sol_previous.x.array[:] = 0.
        self._sol_previous.x.scatter_forward()
        self._sol_current.sub(0).interpolate(lambda x: np.array(0.95 * np.ones(x[0].shape,)))
        self._sol_current.x.scatter_forward()
        self._sol_previous.sub(0).x.array[:] = self._sol_current.sub(0).x.array.copy()

        self._computed_file = f"battery_problem_dlrbnicsx/solution_computed_{mu[0]}_{mu[1]}.xdmf"
        self._solution_file = dolfinx.io.XDMFFile(self._mesh.comm, self._computed_file, "w")
        self._solution_file.write_mesh(self._mesh)
        self._solution_file.write_function(self._sol_current.sub(0), 0)
        self._solution_file.write_function(self._sol_current.sub(1), 0)
        self._solution_file.write_function(self._sol_current.sub(2), 0)

        self._x.interpolate(lambda x: (x[0], x[1]))
        time_list = list()
        time_list.append(0)

        solution_list = list()
        solution_temp = dolfinx.fem.Function(self._V)
        solution_temp.x.array[:] = self._sol_current.x.array.copy()
        solution_temp.x.scatter_forward()
        solution_list.append(solution_temp)

        self.mu_0.value = mu[0]
        self.mu_1.value = mu[1]

        problem = self.set_problem
        solver = NewtonSolver(self._mesh.comm, problem)
        solver.convergence_criterion = "incremental"
        solver.rtol = 1.e-6
        # solver.atol = 1.e-10
        solver.max_it = 20
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        '''
        self._theta_current = self._sol_current.sub(0)
        self._mu_current = self._sol_current.sub(1)
        self._disp_current = self._sol_current.sub(2)
        '''
        current_time = 0
        for i in range(int(self.num_steps.value)):
            current_time += (3600 / self.mu_0.value) / self.num_steps.value # self.dt
            solution_temp = dolfinx.fem.Function(self._V)
            print(f"Time: {current_time}, Step: {i}")
            n, converged = solver.solve(self._sol_current)
            # TODO How to evaluate time when self.mu_0 is involved?
            print(f"Time: {current_time}, Iteration: {n}, Converged: {converged}, Step: {i}")
            self._sol_current.x.scatter_forward()
            self._sol_previous.x.array[:] = self._sol_current.x.array.copy()
            self._solution_file.write_function(self._sol_current.sub(0), current_time)
            self._solution_file.write_function(self._sol_current.sub(1), current_time)
            self._solution_file.write_function(self._sol_current.sub(2), current_time)
            # self._solution_file.write_function(self._theta_current, current_time)
            # self._solution_file.write_function(self._mu_current, current_time)
            # self._solution_file.write_function(self._disp_current, current_time)
            solution_temp.x.array[:] = self._sol_current.x.array.copy()
            solution_temp.x.scatter_forward()
            solution_list.append(solution_temp)
            time_list.append(current_time)
        return (time_list, solution_list)


# Read mesh
world_comm = MPI.COMM_WORLD

if world_comm.size == 16:
    procs_per_communicator = 4
    fem_comm_list = list()
    for i in range(0, world_comm.size, procs_per_communicator):
        fem_procs = world_comm.group.Incl(list(np.arange(i, i + procs_per_communicator).astype("int")))
        fem_comm = world_comm.Create_group(fem_procs)
        fem_comm_list.append(fem_comm)

elif world_comm.size == 8:
    procs_per_communicator = 2
    fem_comm_list = list()
    for i in range(0, world_comm.size, procs_per_communicator):
        fem_procs = world_comm.group.Incl(list(np.arange(i, i + procs_per_communicator).astype("int")))
        fem_comm = world_comm.Create_group(fem_procs)
        fem_comm_list.append(fem_comm)

elif world_comm.size == 1:
    procs_per_communicator = 1
    fem_comm_list = list()
    for i in range(0, world_comm.size, procs_per_communicator):
        fem_procs = world_comm.group.Incl(list(np.arange(i, i + procs_per_communicator).astype("int")))
        fem_comm = world_comm.Create_group(fem_procs)
        fem_comm_list.append(fem_comm)

elif world_comm.size == 128:
    procs_per_communicator = 16
    fem_comm_list = list()
    for i in range(0, world_comm.size, procs_per_communicator):
        fem_procs = world_comm.group.Incl(list(np.arange(i, i + procs_per_communicator).astype("int")))
        fem_comm = world_comm.Create_group(fem_procs)
        fem_comm_list.append(fem_comm)

for comm_i in fem_comm_list:
    if comm_i != MPI.COMM_NULL:
        print(f"World rank: {world_comm.rank}, Comm rank: {comm_i.rank}")
        mesh_comm = comm_i

gdim = 2
# TODO clarify what does gmsh_model_rank correspond to? Is it rank on world_comm or mesh_comm?
gmsh_model_rank = 0
mesh, cell_tags, facet_tags = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh",
                                    mesh_comm, gmsh_model_rank, gdim=gdim)

problem_parametric = ProblemParametric(mesh, cell_tags, facet_tags)
sol_current = dolfinx.fem.Function(problem_parametric._V)

mu = np.array([5., 2.e-6])
problem_parametric.num_steps.value = 5
(time_list, sol_list) = problem_parametric.solve(mu)

mu = np.array([4., 2.e-6])
problem_parametric.num_steps.value = 5
(time_list, sol_list) = problem_parametric.solve(mu)

pod_samples = [len(fem_comm_list), 1] # [3, 1]
para_dim = 2
num_snapshots = np.product(pod_samples)
itemsize = MPI.DOUBLE.Get_size()

# POD Starts ###
def generate_training_set(samples=pod_samples):
    training_set_0 = np.linspace(1., 5., samples[0])
    training_set_1 = np.linspace(2.e-6, 2.e-6, samples[1])
    training_set = np.array(list(itertools.product(training_set_0,
                                                   training_set_1)))
    return training_set

if world_comm.rank == 0:
    nbytes = num_snapshots * para_dim * itemsize
else:
    nbytes = 0

win0 = MPI.Win.Allocate_shared(nbytes, itemsize, comm=MPI.COMM_WORLD)
buf0, itemsize = win0.Shared_query(0)
training_set = np.ndarray(buffer=buf0, dtype="d", shape=(num_snapshots, para_dim))

if world_comm.rank == 0:
    training_set[:, :] = generate_training_set()

world_comm.barrier()

for i in range(len(fem_comm_list)):
    if fem_comm_list[i] != MPI.COMM_NULL:
        cpu_indices = np.arange(i, num_snapshots, len(fem_comm_list))

print(f"World rank: {world_comm.rank}, CPU indices: {cpu_indices}")
exit()

Nmax = 30

print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
print("")

print("set up snapshots matrix")
Theta, _ = problem_parametric._V.sub(0).collapse()
Mu, _ = problem_parametric._V.sub(1).collapse()
U, _ = problem_parametric._V.sub(2).collapse()
snapshots_matrix_theta = rbnicsx.backends.FunctionsList(Theta)
snapshots_matrix_mu = rbnicsx.backends.FunctionsList(Mu)
snapshots_matrix_u = rbnicsx.backends.FunctionsList(U)

print("set up reduced problem") # TODO

print("")

for (mu_index, mu) in enumerate(training_set):
    print(rbnicsx.io.TextLine(str(mu_index+1), fill="#"))
    theta_func = dolfinx.fem.Function(Theta)
    mu_func = dolfinx.fem.Function(Mu)
    u_func = dolfinx.fem

    print("Parameter number ", (mu_index+1), "of", training_set.shape[0])
    print("high fidelity solve for mu =", mu)
    time_steps, snapshot_list = problem_parametric.solve(mu)
    for i in range(len(snapshot_list)):
        theta_func = snapshot_list[i].sub(0).collapse()
        mu_func = snapshot_list[i].sub(1).collapse()
        u_func = snapshot_list[i].sub(2).collapse()

        print("update snapshots matrix")
        snapshots_matrix_theta.append(theta_func)
        snapshots_matrix_mu.append(mu_func)
        snapshots_matrix_u.append(u_func)
    for k in range(len(snapshots_matrix_theta)):
        print(f"Norms: {np.linalg.norm(snapshots_matrix_theta[k].vector)}, {np.linalg.norm(snapshots_matrix_mu[k].vector)}, {np.linalg.norm(snapshots_matrix_u[k].vector)}")
        print("")

print(rbnicsx.io.TextLine("perform POD (Theta)", fill="#"))
eigenvalues_theta, modes_theta, _ = \
    rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_matrix_theta,
                                    problem_parametric._inner_product_theta_action,
                                    N=Nmax, tol=1.e-6)
# reduced_problem._basis_functions.extend(modes)
# reduced_size = len(reduced_problem._basis_functions)
print("")

print(rbnicsx.io.TextBox("POD-Galerkin (Theta) offline phase ends", fill="="))

positive_eigenvalues_theta = np.where(eigenvalues_theta > 0., eigenvalues_theta, np.nan)
singular_values_theta = np.sqrt(positive_eigenvalues_theta)

print(positive_eigenvalues_theta)

print(rbnicsx.io.TextLine("perform POD (Mu)", fill="#"))
eigenvalues_mu, modes_mu, _ = \
    rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_matrix_mu,
                                    problem_parametric._inner_product_mu_action,
                                    N=Nmax, tol=1.e-6)
# reduced_problem._basis_functions.extend(modes)
# reduced_size = len(reduced_problem._basis_functions)
print("")

print(rbnicsx.io.TextBox("POD-Galerkin (Mu) offline phase ends", fill="="))

positive_eigenvalues_mu = np.where(eigenvalues_mu > 0., eigenvalues_mu, np.nan)
singular_values_mu = np.sqrt(positive_eigenvalues_mu)

print(positive_eigenvalues_mu)

print(rbnicsx.io.TextLine("perform POD (U)", fill="#"))
eigenvalues_u, modes_u, _ = \
    rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_matrix_u,
                                    problem_parametric._inner_product_u_action,
                                    N=Nmax, tol=1.e-6)
# reduced_problem._basis_functions.extend(modes)
# reduced_size = len(reduced_problem._basis_functions)
print("")

print(rbnicsx.io.TextBox("POD-Galerkin (U) offline phase ends", fill="="))

positive_eigenvalues_u = np.where(eigenvalues_u > 0., eigenvalues_u, np.nan)
singular_values_u = np.sqrt(positive_eigenvalues_u)

print(positive_eigenvalues_u)

# POD Ends ###
