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

import numpy as np
import itertools
import abc
import matplotlib.pyplot as plt
import os

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
        # Relative error norm for VELOCITY
        return self.compute_norm_sigma(sigma-tau)/self.compute_norm_sigma(sigma)

    def norm_error_u(self, u, v):
        # Relative error norm for PRESSURE
        return self.compute_norm_u(u-v)/self.compute_norm_u(u)


# Mesh, subdomains, Boundaries
nx, ny, nz = 5, 5, 5
mesh = dolfinx.mesh.create_box(MPI.COMM_WORLD,
                               [[0.0, 0.0, 0.0], [1., 1, 1]],
                               [nx, ny, nz],
                               dolfinx.mesh.CellType.tetrahedron)
problem_parametric = ParametricProblem(mesh)


# mu_range = ((-2.5, -1.5), (0., 1.), (0.2, 0.8), (0.2, 0.8), (2.5, 3.5))
# mu = np.array([-2., 0.5, 0.5, 0.5, 3.])
mu = np.array([-1., 1.5, 0.7, 0.3, 3.4])

para_dim = 5
ann_input_samples_num = 640
error_analysis_samples_num = 7
num_snapshots = 13

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

computed_file_sigma = "dlrbnicsx_solution_saddle_point/solution_computed_sigma.xdmf"
computed_file_u = "dlrbnicsx_solution_saddle_point/solution_computed_u.xdmf"

with dolfinx.io.XDMFFile(mesh.comm, computed_file_sigma,
                         "w") as solution_file:
    solution_file.write_mesh(mesh)
    solution_file.write_function(sigma_h)

with dolfinx.io.XDMFFile(mesh.comm, computed_file_u,
                            "w") as solution_file:
    solution_file.write_mesh(mesh)
    solution_file.write_function(u_h)

# POD Starts ###
def generate_training_set(num_samples, para_dim):
    training_set = np.random.uniform(size=(num_samples, para_dim))
    training_set[:, 0] = (-1.5 + 2.5) * training_set[:, 0] - 2.5
    training_set[:, 1] = (1. - 0.) * training_set[:, 1] + 0.
    training_set[:, 2] = (0.8 - 0.2) * training_set[:, 2] + 0.2
    training_set[:, 3] = (0.8 - 0.2) * training_set[:, 3] + 0.2
    training_set[:, 4] = (3.5 - 2.5) * training_set[:, 3] + 2.5
    return training_set

training_set = generate_training_set(num_snapshots, para_dim)

# Maximum RB size
Nmax_sigma = 30
Nmax_u = 30
tol_sigma = 1.e-4
tol_u = 1.e-4

print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
print("")

print("Set up snapshots matrix")
Q, _ = problem_parametric._V.sub(0).collapse()
W, _ = problem_parametric._V.sub(1).collapse()
snapshots_matrix_sigma = rbnicsx.backends.FunctionsList(Q)
snapshots_matrix_u = rbnicsx.backends.FunctionsList(W)

print("Set up reduced problem")
reduced_problem = PODANNReducedProblem(problem_parametric)

print("")

for (mu_index, mu) in enumerate(training_set):
    print(rbnicsx.io.TextLine(str(mu_index+1), fill="#"))

    print("Parameter number ", (mu_index+1), "of", training_set.shape[0])
    print("High fidelity solve for mu =", mu)
    (snapshot_sigma, snapshot_u) = problem_parametric.solve(mu)

    print("Update snapshots matrix")
    snapshots_matrix_sigma.append(snapshot_sigma)
    snapshots_matrix_u.append(snapshot_u)

    print("")

print(rbnicsx.io.TextLine("Perform POD", fill="#"))
eigenvalues_sigma, modes_sigma, _ = rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_matrix_sigma,
                                    problem_parametric._inner_product_action_sigma,
                                    N=Nmax_sigma, tol=tol_sigma)
eigenvalues_u, modes_u, _ = rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_matrix_u,
                                    problem_parametric._inner_product_action_u,
                                    N=Nmax_u, tol=tol_u)

reduced_problem._basis_functions_sigma.extend(modes_sigma)
reduced_problem._basis_functions_u.extend(modes_u)
print("")

print(rbnicsx.io.TextBox("POD-Galerkin offline phase ends", fill="="))

positive_eigenvalues_sigma = np.where(eigenvalues_sigma > 0., eigenvalues_sigma, np.nan)
singular_values_sigma = np.sqrt(positive_eigenvalues_sigma)

positive_eigenvalues_u = np.where(eigenvalues_u > 0., eigenvalues_u, np.nan)
singular_values_u = np.sqrt(positive_eigenvalues_u)

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

print(f"Sigma eigenvalues: {positive_eigenvalues_sigma}")
print(f"U eigenvalues: {positive_eigenvalues_u}")

# Measure projection error
projection_error_analysis_samples = \
    generate_training_set(error_analysis_samples_num, para_dim)
projection_error_array_sigma = np.zeros(error_analysis_samples_num)
projection_error_array_u = np.zeros(error_analysis_samples_num)

for i in range(projection_error_analysis_samples.shape[0]):
    sigma_h, u_h = problem_parametric.solve(projection_error_analysis_samples[i, :])
    sigma_h_reconstructed = reduced_problem.reconstruct_solution_sigma(
        reduced_problem.project_snapshot_sigma(sigma_h,
                                               len(reduced_problem._basis_functions_sigma)))
    u_h_reconstructed = reduced_problem.reconstruct_solution_u(
        reduced_problem.project_snapshot_u(u_h, len(reduced_problem._basis_functions_u)))
    projection_error_array_sigma[i] = \
        reduced_problem.norm_error_sigma(sigma_h, sigma_h_reconstructed)
    projection_error_array_u[i] = \
        reduced_problem.norm_error_u(u_h, u_h_reconstructed)

print(f"Projection error (sigma): {projection_error_array_sigma}")
print(f"Projection error (u): {projection_error_array_u}")
