import dolfinx
import basix
import ufl

from dolfinx.fem.petsc import assemble_matrix, assemble_vector, \
    create_vector, apply_lifting, set_bc, NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver

import rbnicsx
import rbnicsx.backends
import rbnicsx.online

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import sympy

import matplotlib.pyplot as plt

import abc

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
        # H^1_r for theta
        self._inner_product_theta = ufl.inner(self._theta_trial, self._theta_test) * self._x[0] * ufl.dx + \
            ufl.inner(ufl.grad(self._theta_trial), ufl.grad(self._theta_test)) * self._x[0] * ufl.dx
        self._inner_product_theta_action = \
            rbnicsx.backends.bilinear_form_action(self._inner_product_theta,
                                                  part="real")
        # L^2_r for mu
        self._inner_product_mu = ufl.inner(self._mu_trial, self._mu_test) * self._x[0] * ufl.dx
        self._inner_product_mu_action = \
            rbnicsx.backends.bilinear_form_action(self._inner_product_mu,
                                                  part="real")
        # H^1_r for u
        self._inner_product_u = ufl.inner(self._u_trial, self._u_test) * self._x[0] * ufl.dx + \
            ufl.inner(ufl.grad(self._u_trial), ufl.grad(self._u_test)) * self._x[0] * ufl.dx
        self._inner_product_u_action = \
            rbnicsx.backends.bilinear_form_action(self._inner_product_u,
                                                  part="real")

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
        self.num_steps = dolfinx.fem.Constant(self._mesh, PETSc.ScalarType(20))
        self.dt = (3600 / self.mu_0) / self.num_steps
        self.i_s = 4780 * self.mu_0 * self.mu_1 * 210 / 4

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
mesh_comm = MPI.COMM_WORLD
gdim = 2
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
problem_parametric.num_steps.value = 20
(time_list, sol_list) = problem_parametric.solve(mu)
