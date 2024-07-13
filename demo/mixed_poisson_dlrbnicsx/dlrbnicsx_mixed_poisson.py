import numpy as np
import abc
import matplotlib.pyplot as plt
import itertools
import os

from mpi4py import MPI
from petsc4py import PETSc

import ufl
import basix
import dolfinx

import rbnicsx
import rbnicsx.online
import rbnicsx.backends

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

# Parameters
mu = np.array([-2., 0.5, 0.5, 0.5, 3.])

# FEM solve
problem_parametric = ParametricProblem(mesh, subdomains,
                                       boundaries)
sigma_sol, u_sol = problem_parametric.solve(mu)
print(sigma_sol.x.array, np.linalg.norm(sigma_sol.x.array))
print(u_sol.x.array, np.linalg.norm(u_sol.x.array))

with dolfinx.io.XDMFFile(mesh.comm, "parametric_mixed_poisson/sigma.xdmf", "w") as sol_file:
    sol_file.write_mesh(mesh)
    sol_file.write_function(sigma_sol)

with dolfinx.io.XDMFFile(mesh.comm, "parametric_mixed_poisson/u.xdmf", "w") as sol_file:
    sol_file.write_mesh(mesh)
    sol_file.write_function(u_sol)

# TODO more benchmarking with dolfinx implementation for correctness

# POD Starts ###


def generate_training_set(samples=[2, 1, 2, 1, 2]):
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
                                    N=Nmax, tol=1e-10)

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
