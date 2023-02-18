import abc
import itertools
import typing

import dolfinx.fem
import dolfinx.io
import gmsh
import matplotlib.pyplot as plt
import mpi4py.MPI
import multiphenicsx.io
import numpy as np
import numpy.typing
import petsc4py.PETSc
import plotly.graph_objects as go
import rbnicsx.backends
import rbnicsx.online
import rbnicsx.test
import torch
import ufl

from dlrbnicsx.activation_function.activation_function_factory import Tanh
from dlrbnicsx.dataset.custom_dataset import CustomDataset
from dlrbnicsx.interface.wrappers import DataLoader
from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.train_validate_test.train_validate_test import (error_analysis,
                                                               online_nn,
                                                               train_nn,
                                                               validate_nn)

# Import mesh in dolfinx
gdim = 2
mesh, subdomains, boundaries = dolfinx.io.gmshio.read_from_msh(
    "mesh_data/domain_geometry.msh", mpi4py.MPI.COMM_WORLD, 0, gdim=gdim)

# 1. Geometric parametrization


class HarmonicExtension(rbnicsx.backends.MeshMotion):
    """Extend the shape parametrization from the boundary to the interior with an harmonic extension."""
    # TODO Ask Francesco: 1. Reset reference?? 2. Is mesh from global variable

    def __init__(self, mu: np.typing.NDArray[np.float64]) -> None:
        # Define function space
        M = dolfinx.fem.VectorFunctionSpace(mesh, ("Lagrange", mesh.geometry.cmap.degree))
        # Define trial and test functions
        m = ufl.TrialFunction(M)
        n = ufl.TestFunction(M)
        # Define bilinear form of the harmonic extension problem
        a_he = dolfinx.fem.form(ufl.inner(ufl.grad(m), ufl.grad(n)) * ufl.dx)
        a_he_cpp = dolfinx.fem.form(a_he)
        # Define linear form of the harmonic extension problem
        zero_vector = dolfinx.fem.Constant(mesh, np.zeros(mesh.topology.dim, petsc4py.PETSc.ScalarType))
        f_he = dolfinx.fem.form(ufl.inner(zero_vector, n) * ufl.dx)
        f_he_cpp = dolfinx.fem.form(f_he)
        # Define boundary conditions for the harmonic extension problem
        dofs_bc_1 = dolfinx.fem.locate_dofs_topological(M, 1, boundaries.find(1))
        dofs_bc_2 = dolfinx.fem.locate_dofs_topological(M, 1, boundaries.find(2))
        dofs_bc_3 = dolfinx.fem.locate_dofs_topological(M, 1, boundaries.find(3))
        dofs_bc_4 = dolfinx.fem.locate_dofs_topological(M, 1, boundaries.find(4))
        dofs_bc_5 = dolfinx.fem.locate_dofs_topological(M, 1, boundaries.find(5))
        dofs_bc_6 = dolfinx.fem.locate_dofs_topological(M, 1, boundaries.find(6))
        mD_1 = dolfinx.fem.Function(M)
        mD_1.interpolate(lambda x: (x[0], x[1]))
        mD_2 = dolfinx.fem.Function(M)
        mD_2.interpolate(lambda x: (x[0], x[1]))
        mD_3 = dolfinx.fem.Function(M)
        mD_3.interpolate(lambda x: (x[0], x[1]))
        mD_4 = dolfinx.fem.Function(M)
        mD_4.interpolate(lambda x: (x[0], x[1]))
        mD_5 = dolfinx.fem.Function(M)
        mD_5.interpolate(lambda x: (mu[0] * x[0], mu[1] * x[1]))
        mD_6 = dolfinx.fem.Function(M)
        mD_6.interpolate(lambda x: (mu[0] * x[0], mu[1] * x[1]))
        bc_1 = dolfinx.fem.dirichletbc(mD_1, dofs_bc_1)
        bc_2 = dolfinx.fem.dirichletbc(mD_2, dofs_bc_2)
        bc_3 = dolfinx.fem.dirichletbc(mD_3, dofs_bc_3)
        bc_4 = dolfinx.fem.dirichletbc(mD_4, dofs_bc_4)
        bc_5 = dolfinx.fem.dirichletbc(mD_5, dofs_bc_5)
        bc_6 = dolfinx.fem.dirichletbc(mD_6, dofs_bc_6)
        bcs_he = [bc_1, bc_2, bc_3, bc_4, bc_5, bc_6]
        # Assemble the left-hand side matrix of the harmonic extension problem
        A = dolfinx.fem.petsc.assemble_matrix(a_he_cpp, bcs=bcs_he)
        A.assemble()
        # Assemble the right-hand side vector of the harmonic extension problem # TODO Understand solving and apply_lifting
        F = dolfinx.fem.petsc.assemble_vector(f_he_cpp)
        dolfinx.fem.petsc.apply_lifting(F, [a_he_cpp], [bcs_he])
        F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(F, bcs_he)
        # Solve the harmonic extension problem
        ksp = petsc4py.PETSc.KSP()
        ksp.create(mesh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        ksp.setFromOptions()
        shape_parametrization = dolfinx.fem.Function(M)
        ksp.solve(F, shape_parametrization.vector)
        shape_parametrization.vector.ghostUpdate(
            addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
        # Initialize mesh motion object
        super().__init__(mesh, shape_parametrization)


'''
deform_mu = np.array([0.7,0.8])
harmonic_extension = HarmonicExtension(deform_mu)

with harmonic_extension:
    with dolfinx.io.XDMFFile(mpi4py.MPI.COMM_WORLD, "mesh_data/deformed_mesh_mu_" + str(deform_mu[0]) + "_" + str(deform_mu[1]) + ".xdmf", "w") as deform_mesh_file_xdmf:
        deform_mesh_file_xdmf.write_mesh(mesh)
'''


class ProblemBase(abc.ABC):
    """Define a linear problem, and solve it with KSP."""

    def __init__(self) -> None:
        # Define function space (Taylor hood P2-P1)
        P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        UP = P2*P1
        W = dolfinx.fem.FunctionSpace(mesh, UP)
        V, _ = W.sub(0).collapse()
        Q, _ = W.sub(1).collapse()
        self._W = W
        # Define trial and test functions
        (u, p) = ufl.TrialFunctions(W)
        (v, q) = ufl.TestFunctions(W)
        self._trial = (u, p)
        self._test = (v, q)
        # Define solution
        U = dolfinx.fem.Function(W)
        self._solution = U
        # Define measures for integration of forms
        dx = ufl.Measure("dx")(subdomain_data=subdomains)
        ds = ufl.Measure("ds")(subdomain_data=boundaries)
        self._dx = dx
        self._ds = ds
        # Define symbolic parameters for use in UFL forms
        mu_symb = rbnicsx.backends.SymbolicParameters(mesh, shape=(2, ))
        self._mu_symb = mu_symb
        # Define numeric parameters for use in shape parametrization
        self._mu = np.zeros(mu_symb.value.shape)

    @property
    def function_space(self) -> dolfinx.fem.FunctionSpace:
        """Return the function space of the problem."""
        return self._W

    @property
    def mu_symb(self) -> rbnicsx.backends.SymbolicParameters:
        """Return the symbolic parameters of the problem."""
        return self._mu_symb

    @property
    def trial_and_test(self) -> typing.Tuple[ufl.Argument, ufl.Argument]:  # type: ignore[no-any-unimported]
        """Return the UFL arguments used in the construction of the forms."""
        return self._trial, self._test

    @property
    def measures(self) -> typing.Tuple[ufl.Measure, ufl.Measure]:  # type: ignore[no-any-unimported]
        """Return the UFL measures used in the construction of the forms."""
        return self._dx, self._ds

    @abc.abstractproperty
    def bilinear_form(self) -> ufl.Form:  # type: ignore[no-any-unimported]
        """Return the bilinear form of the problem."""
        pass

    @abc.abstractproperty
    def linear_form(self) -> ufl.Form:  # type: ignore[no-any-unimported]
        """Return the linear form of the problem."""
        pass

    @abc.abstractproperty
    def mesh_motion(self) -> rbnicsx.backends.MeshMotion:
        """Return the mesh motion for the parameter of the latest solve."""
        pass

    @abc.abstractmethod
    def _assemble_matrix(self) -> petsc4py.PETSc.Mat:  # type: ignore[no-any-unimported]
        """Assemble the left-hand side matrix."""
        pass

    @abc.abstractmethod
    def _assemble_vector(self) -> petsc4py.PETSc.Vec:  # type: ignore[no-any-unimported]
        """Assemble the right-hand side vector."""
        pass

    def solve(self, mu: np.typing.NDArray[np.float64]) -> typing.Tuple[dolfinx.fem.Function, dolfinx.fem.Function]:
        """Assign the provided parameters value and solve the problem."""
        self._mu[:] = mu
        self._mu_symb.value[:] = mu
        return self._solve()

    def _solve(self) -> dolfinx.fem.Function:
        """Solve the linear problem with KSP."""
        A = self._assemble_matrix()
        F = self._assemble_vector()
        ksp = petsc4py.PETSc.KSP()
        ksp.create(mesh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        ksp.setFromOptions()
        solution = self._solution.copy()
        ksp.solve(F, solution.vector)
        self._solution = solution
        (solution_u, solution_p) = (self._solution.sub(0).collapse(), self._solution.sub(1).collapse())
        return (solution_u, solution_p)

# 3. FEM formulation on deformed domain


class ProblemOnDeformedDomain(ProblemBase):
    """Solve the problem on the deformed domain obtained by harmonic extension."""

    def __init__(self) -> None:
        super().__init__()
        # Split function spaces
        W = self.function_space
        V, _ = W.sub(0).collapse()
        Q, _ = W.sub(1).collapse()
        # Define trial and test functions
        (u, p), (v, q) = self.trial_and_test
        # Get measures
        dx, ds = self.measures
        # Get symbolic parameters for use in UFL forms
        mu_symb = self.mu_symb
        # Define bilinear form of the problem
        lhs = (ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.inner(p, ufl.div(v)) + ufl.inner(ufl.div(u), q)) * ufl.dx
        self._lhs = lhs
        self._lhs_cpp = dolfinx.fem.form(lhs)
        # Define linear form of the problem
        # NOTE f_term set to 0 for symmetric solution else there should be force due to gravity
        f = dolfinx.fem.Function(V)
        rhs = ufl.inner(f, v) * ufl.dx
        self._rhs = rhs
        self._rhs_cpp = dolfinx.fem.form(rhs)
        # Define BCs of the problem

        dofs_bc_1 = dolfinx.fem.locate_dofs_topological((W.sub(0), V), 1, boundaries.find(1))
        dofs_bc_2 = dolfinx.fem.locate_dofs_topological((W.sub(0), V), 1, boundaries.find(2))
        dofs_bc_3 = dolfinx.fem.locate_dofs_topological((W.sub(0), V), 1, boundaries.find(3))
        dofs_bc_4 = dolfinx.fem.locate_dofs_topological((W.sub(0), V), 1, boundaries.find(4))
        dofs_bc_5 = dolfinx.fem.locate_dofs_topological((W.sub(0), V), 1, boundaries.find(5))
        dofs_bc_6 = dolfinx.fem.locate_dofs_topological((W.sub(0), V), 1, boundaries.find(6))

        uD_1 = dolfinx.fem.Function(V)
        uD_1.interpolate(self.free_boundary)

        uD_2 = dolfinx.fem.Function(V)
        uD_2.interpolate(self.free_boundary)

        uD_3 = dolfinx.fem.Function(V)
        uD_3.interpolate(self.free_boundary)

        uD_4 = dolfinx.fem.Function(V)
        uD_4.interpolate(self.free_boundary)

        uD_5 = dolfinx.fem.Function(V)
        uD_5.interpolate(self.no_slip)

        uD_6 = dolfinx.fem.Function(V)
        uD_6.interpolate(self.no_slip)

        bc_1 = dolfinx.fem.dirichletbc(uD_1, dofs_bc_1, W.sub(0))
        bc_2 = dolfinx.fem.dirichletbc(uD_2, dofs_bc_2, W.sub(0))
        bc_3 = dolfinx.fem.dirichletbc(uD_3, dofs_bc_3, W.sub(0))
        bc_4 = dolfinx.fem.dirichletbc(uD_4, dofs_bc_4, W.sub(0))
        bc_5 = dolfinx.fem.dirichletbc(uD_5, dofs_bc_5, W.sub(0))
        bc_6 = dolfinx.fem.dirichletbc(uD_6, dofs_bc_6, W.sub(0))

        # Pressure fixation
        zero = dolfinx.fem.Function(Q)
        zero.x.set(0.0)
        dofs = dolfinx.fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x.T, [0, 0, 0]).all(axis=1))
        bc_p = dolfinx.fem.dirichletbc(zero, dofs, W.sub(1))

        bcs = [bc_1, bc_2, bc_3, bc_4, bc_5, bc_6, bc_p]
        self._bcs = bcs  # TODO Confirm BCs are evaluated on the deformed domain
        # Store mesh motion object used in the latest solve, to avoid having to solve
        # the harmonic extension once for computation and once for (optional) visualization.
        self._mesh_motion: typing.Optional[rbnicsx.backends.MeshMotion] = None

    def free_boundary(self, x):
        return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))

    def no_slip(self, x):
        return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

    @property
    def bilinear_form(self) -> ufl.Form:  # type: ignore[no-any-unimported]
        """Return the bilinear form of the problem."""
        return self._lhs

    @property
    def linear_form(self) -> ufl.Form:  # type: ignore[no-any-unimported]
        """Return the linear form of the problem."""
        return self._rhs

    @property
    def mesh_motion(self) -> rbnicsx.backends.MeshMotion:
        """Return the mesh motion object that was used in the latest solve."""
        assert self._mesh_motion is not None
        return self._mesh_motion

    def _assemble_matrix(self) -> petsc4py.PETSc.Mat:  # type: ignore[no-any-unimported]
        """Assemble the left-hand side matrix."""
        A = dolfinx.fem.petsc.assemble_matrix(self._lhs_cpp, bcs=self._bcs)
        A.assemble()
        return A

    def _assemble_vector(self) -> petsc4py.PETSc.Vec:  # type: ignore[no-any-unimported]
        """Assemble the right-hand side vector."""
        F = dolfinx.fem.petsc.assemble_vector(self._rhs_cpp)
        dolfinx.fem.petsc.apply_lifting(F, [self._lhs_cpp], bcs=[self._bcs])
        dolfinx.fem.petsc.set_bc(F, self._bcs)
        return F

    def _solve(self) -> dolfinx.fem.Function:
        """Apply shape parametrization to the mesh and solve the linear problem with KSP."""
        with HarmonicExtension(self._mu) as self._mesh_motion:
            return super()._solve()


problem = ProblemOnDeformedDomain()
mu_deformed = np.array([1.2, 1.1])
(solution_u, solution_p) = problem.solve(mu_deformed)

with problem.mesh_motion:
    with dolfinx.io.VTKFile(mpi4py.MPI.COMM_WORLD, "mesh_data/solution_velocity.pvd", "w") as ufile_pvd:
        solution_u.x.scatter_forward()
        ufile_pvd.write_mesh(mesh)
        ufile_pvd.write_function(solution_u)

with problem.mesh_motion:
    with dolfinx.io.VTKFile(mpi4py.MPI.COMM_WORLD, "mesh_data/solution_pressure.pvd", "w") as pfile_pvd:
        solution_p.x.scatter_forward()
        pfile_pvd.write_mesh(mesh)
        pfile_pvd.write_function(solution_p)

problem = ProblemOnDeformedDomain()
mu_reference = np.array([1., 1.])
(solution_u, solution_p) = problem.solve(mu_reference)

with problem.mesh_motion:
    with dolfinx.io.VTKFile(mpi4py.MPI.COMM_WORLD, "mesh_data/solution_velocity_reference.pvd", "w") as ufile_pvd:
        solution_u.x.scatter_forward()
        ufile_pvd.write_mesh(mesh)
        ufile_pvd.write_function(solution_u)

with problem.mesh_motion:
    with dolfinx.io.VTKFile(mpi4py.MPI.COMM_WORLD, "mesh_data/solution_pressure_reference.pvd", "w") as pfile_pvd:
        solution_p.x.scatter_forward()
        pfile_pvd.write_mesh(mesh)
        pfile_pvd.write_function(solution_p)


class PODANNReducedProblem(abc.ABC):
    "Projected solutions for ANN"

    def __init__(self, problem: ProblemBase) -> None:
        # Define basis functiosn storage
        W = problem.function_space
        V, _ = W.sub(0).collapse()
        Q, _ = W.sub(1).collapse()
        basis_functions_u = rbnicsx.backends.FunctionsList(V)
        self._basis_functions_u = basis_functions_u
        basis_functions_p = rbnicsx.backends.FunctionsList(Q)
        self._basis_functions_p = basis_functions_p
        u, p, v, q = ufl.TrialFunction(V), ufl.TrialFunction(Q), ufl.TestFunction(V), ufl.TestFunction(Q)
        inner_product_u = ufl.inner(u, v) * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        self._inner_product_u = inner_product_u
        inner_product_p = ufl.inner(p, q) * ufl.dx
        self._inner_product_p = inner_product_p
        self._inner_product_action_u = rbnicsx.backends.bilinear_form_action(self._inner_product_u, part="real")
        self._inner_product_action_p = rbnicsx.backends.bilinear_form_action(self._inner_product_p, part="real")
        self._mu_symb = problem.mu_symb
        self._mu = np.zeros(self._mu_symb.value.shape)

    @property
    def function_space(self) -> dolfinx.fem.FunctionSpace:
        """Return the function space of the problem."""
        return self._W

    @property
    def basis_functions_u(self) -> rbnicsx.backends.FunctionsList:
        return self._basis_functions_u

    @property
    def basis_functions_p(self) -> rbnicsx.backends.FunctionsList:
        return self._basis_functions_p

    @property
    def inner_product_form_u(self) -> ufl.Form:
        return self._inner_product_u

    @property
    def inner_product_form_p(self) -> ufl.Form:
        return self._inner_product_p

    @property
    def inner_product_action_u(self) -> typing.Callable[[dolfinx.fem.Function], typing.Callable[[dolfinx.fem.Function], petsc4py.PETSc.RealType]]:
        return self._inner_product_action_u

    @property
    def inner_product_action_p(self) -> typing.Callable[[dolfinx.fem.Function], typing.Callable[[dolfinx.fem.Function], petsc4py.PETSc.RealType]]:
        return self._inner_product_action_p

    def reconstruct_solution_u(self, reduced_solution_u: petsc4py.PETSc.Vec) -> dolfinx.fem.Function:
        return self._basis_functions_u[:reduced_solution_u.size] * reduced_solution_u

    def reconstruct_solution_p(self, reduced_solution_p: petsc4py.PETSc.Vec) -> dolfinx.fem.Function:
        return self._basis_functions_p[:reduced_solution_p.size] * reduced_solution_p

    def compute_norm_u(self, function: dolfinx.fem.Function) -> petsc4py.PETSc.RealType:
        return np.sqrt(self._inner_product_action_u(function)(function))

    def compute_norm_p(self, function: dolfinx.fem.Function) -> petsc4py.PETSc.RealType:
        return np.sqrt(self._inner_product_action_p(function)(function))

    def project_snapshot_u(self, solution: dolfinx.fem.Function, N: int) -> petsc4py.PETSc.Vec:
        return self._project_snapshot_u(solution, N)

    def _project_snapshot_u(self, solution, N) -> petsc4py.PETSc.Vec:
        projected_snapshot_u = rbnicsx.online.create_vector(N)
        A = rbnicsx.backends.project_matrix(self._inner_product_action_u, self._basis_functions_u[:N])
        F = rbnicsx.backends.project_vector(self._inner_product_action_u(solution), self._basis_functions_u[:N])
        ksp = petsc4py.PETSc.KSP()
        ksp.create(projected_snapshot_u.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        # ksp.getPC().setFactorSolverType("mumps")
        ksp.setFromOptions()
        ksp.solve(F, projected_snapshot_u)
        return projected_snapshot_u

    def project_snapshot_p(self, solution: dolfinx.fem.Function, N: int) -> petsc4py.PETSc.Vec:
        return self._project_snapshot_p(solution, N)

    def _project_snapshot_p(self, solution, N) -> petsc4py.PETSc.Vec:
        projected_snapshot_p = rbnicsx.online.create_vector(N)
        A = rbnicsx.backends.project_matrix(self._inner_product_action_p, self._basis_functions_p[:N])
        F = rbnicsx.backends.project_vector(self._inner_product_action_p(solution), self._basis_functions_p[:N])
        ksp = petsc4py.PETSc.KSP()
        ksp.create(projected_snapshot_p.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        # ksp.getPC().setFactorSolverType("mumps")
        ksp.setFromOptions()
        ksp.solve(F, projected_snapshot_p)
        return projected_snapshot_p

    def norm_error_u(self, u, v):
        return self.compute_norm_u(u-v)/self.compute_norm_u(u)

    def norm_error_p(self, p, q):
        return self.compute_norm_p(p-q)/self.compute_norm_p(p)


def generate_training_set() -> np.typing.NDArray[np.float64]:
    "Generate training set for POD"
    training_set_0 = np.linspace(0.5, 1., 6)
    training_set_1 = np.linspace(0.5, 1., 6)
    training_set = np.array(list(itertools.product(training_set_0, training_set_1)))
    return training_set


training_set = rbnicsx.io.on_rank_zero(mesh.comm, generate_training_set)

Nmax_u = 6
Nmax_p = 6

print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
print("")

print("set up snapshots matrix")
W = problem.function_space
V, _ = W.sub(0).collapse()
Q, _ = W.sub(1).collapse()
snapshots_matrix_u = rbnicsx.backends.FunctionsList(V)
snapshots_matrix_p = rbnicsx.backends.FunctionsList(Q)

print("set up reduced problem")
print(problem.__class__.__name__)
reduced_problem = PODANNReducedProblem(problem)

print("")
for (mu_index, mu) in enumerate(training_set):
    print(rbnicsx.io.TextLine(str(mu_index+1), fill="#"))

    print("Parameter number ", (mu_index+1), "of", training_set.shape[0])
    print("high fidelity solve for mu =", mu)
    (snapshot_u, snapshot_p) = problem.solve(mu)
    print(snapshot_u.x.array.shape)
    print(snapshot_p.x.array.shape)

    print("update snapshots matrix")
    snapshots_matrix_u.append(snapshot_u)
    snapshots_matrix_p.append(snapshot_p)

    print("")

print(rbnicsx.io.TextLine("Perform POD of velocity snapshots", fill="#"))
eigenvalues_u, modes_u, _ = rbnicsx.backends.proper_orthogonal_decomposition(
    snapshots_matrix_u, reduced_problem._inner_product_action_u, N=Nmax_u, tol=0)
reduced_problem._basis_functions_u.extend(modes_u)
print(rbnicsx.io.TextLine("Perform POD of pressure snapshots", fill="#"))
eigenvalues_p, modes_p, _ = rbnicsx.backends.proper_orthogonal_decomposition(
    snapshots_matrix_p, reduced_problem._inner_product_action_p, N=Nmax_p, tol=0)
reduced_problem._basis_functions_p.extend(modes_p)
print("")

print(rbnicsx.io.TextBox("POD-Galerkin offline phase ends", fill="="))

positive_eigenvalues_u = np.where(eigenvalues_u > 0., eigenvalues_u, np.nan)
singular_values_u = np.sqrt(positive_eigenvalues_u)


plt.figure(figsize=[8, 10])
xint = list()
yval = list()

for x, y in enumerate(eigenvalues_u[:Nmax_u]):
    yval.append(y)
    xint.append(x+1)

plt.plot(xint, yval, "*-", color="orange")
plt.xlabel("Eigenvalue number", fontsize=18)
plt.ylabel("Eigenvalue", fontsize=18)
plt.xticks(xint)
plt.yscale("log")
plt.title("Eigenvalue decay (Velocity)", fontsize=24)
plt.tight_layout()
# plt.show()


plt.figure(figsize=[8, 10])
xint = list()
yval = list()

for x, y in enumerate(eigenvalues_p[:Nmax_p]):
    yval.append(y)
    xint.append(x+1)

plt.plot(xint, yval, "*-", color="orange")
plt.xlabel("Eigenvalue number", fontsize=18)
plt.ylabel("Eigenvalue", fontsize=18)
plt.xticks(xint)
plt.yscale("log")
plt.title("Eigenvalue decay (Pressure)", fontsize=24)
plt.tight_layout()
# plt.show()

(solution_u, solution_p) = problem.solve(mu_deformed)
reduced_problem.project_snapshot_u(solution_u, Nmax_u)
reduced_problem.project_snapshot_p(solution_p, Nmax_p)
reduced_problem.regularisation_u = "EarlyStopping"
reduced_problem.regularisation_p = "EarlyStopping"

# ANN implementation
u_input_training_set_filepath = "ann_data/u_input_training_data.npy"
u_input_validation_set_filepath = "ann_data/u_input_validation_data.npy"
u_output_training_set_filepath = "ann_data/u_output_training_data.npy"
u_output_validation_set_filepath = "ann_data/u_output_validation_data.npy"
u_input_error_analysis_set_filepath = "ann_data/u_input_error_analysis_data.npy"

p_input_training_set_filepath = "ann_data/p_input_training_data.npy"
p_input_validation_set_filepath = "ann_data/p_input_validation_data.npy"
p_output_training_set_filepath = "ann_data/p_output_training_data.npy"
p_output_validation_set_filepath = "ann_data/p_output_validation_data.npy"
p_input_error_analysis_set_filepath = "ann_data/p_input_error_analysis_data.npy"


def generate_ann_input_set_u(filepath, samples=[4, 4]):
    """Generate an equispaced training set using numpy."""
    training_set_0 = np.linspace(0.5, 1., samples[0])  # arguments: min_parameter, max_parameter, number of smaples
    training_set_1 = np.linspace(0.5, 1., samples[1])
    training_set = np.array(list(itertools.product(training_set_0, training_set_1))).astype("f")
    np.save(filepath, training_set)


def generate_ann_output_set_u(problem, reduced_problem, N, input_file_path, output_file_path, mode=None):
    input_set = np.load(input_file_path)
    output_set = np.empty([input_set.shape[0], N])
    for i in range(input_set.shape[0]):
        if mode == None:
            print(f"Parameter number {i+1} of {input_set.shape[0]}: {input_set[i,:]}")
        else:
            print(f"{mode} parameter number {i+1} of {input_set.shape[0]}: {input_set[i,:]}")
            (u_solution, _) = problem.solve(input_set[i, :])
        output_set[i, :] = reduced_problem.project_snapshot_u(u_solution, N).array  # .astype("f")
    np.save(output_file_path, output_set)


def generate_ann_input_set_p(filepath, samples=[4, 4]):
    """Generate an equispaced training set using numpy."""
    training_set_0 = np.linspace(0.5, 1., samples[0])  # arguments: min_parameter, max_parameter, number of smaples
    training_set_1 = np.linspace(0.5, 1., samples[1])
    training_set = np.array(list(itertools.product(training_set_0, training_set_1))).astype("f")
    np.save(filepath, training_set)


def generate_ann_output_set_p(problem, reduced_problem, N, input_file_path, output_file_path, mode=None):
    input_set = np.load(input_file_path)
    output_set = np.empty([input_set.shape[0], N])
    for i in range(input_set.shape[0]):
        if mode == None:
            print(f"Parameter number {i+1} of {input_set.shape[0]}: {input_set[i,:]}")
        else:
            print(f"{mode} parameter number {i+1} of {input_set.shape[0]}: {input_set[i,:]}")
        output_set[i, :] = reduced_problem.project_snapshot_p(
            problem.solve(input_set[i, :])[1], Nmax_p).array.astype("f")
    np.save(output_file_path, output_set)


# Training dataset
generate_ann_input_set_u(u_input_training_set_filepath, samples=[5, 5])
generate_ann_output_set_u(problem, reduced_problem, Nmax_u, u_input_training_set_filepath,
                          u_output_training_set_filepath, mode="Training")
u_customDataset = CustomDataset(problem, reduced_problem, Nmax_u, u_input_training_set_filepath, u_output_training_set_filepath, input_scaling_range=[-1., 1.], output_scaling_range=[
                                -1., 1.], input_range=np.array([[0.5, 0.5], [1., 1.]]), output_range=[np.min(np.load(u_output_training_set_filepath)), np.max(np.load(u_output_training_set_filepath))])
u_train_dataloader = DataLoader(u_customDataset, batch_size=10, shuffle=True)

# Validation dataset
generate_ann_input_set_u(u_input_validation_set_filepath, samples=[3, 3])
generate_ann_output_set_u(problem, reduced_problem, Nmax_u, u_input_validation_set_filepath,
                          u_output_validation_set_filepath, mode="Validation")
u_customDataset = CustomDataset(problem, reduced_problem, Nmax_u, u_input_validation_set_filepath, u_output_validation_set_filepath, input_scaling_range=[-1., 1.], output_scaling_range=[
                                -1, 1], input_range=np.array([[0.5, 0.5], [1., 1.]]), output_range=[np.min(np.load(u_output_training_set_filepath)), np.max(np.load(u_output_training_set_filepath))])
# NOTE VVIP: output_range is from training and NOT from validation to ensure same scaling factors in training as well as validation
u_valid_dataloader = DataLoader(u_customDataset, batch_size=10, shuffle=False)

# TODO replace Nmax with with something like reduced_problem.N
u_model = HiddenLayersNet(training_set.shape[1], [25, 28], Nmax_u, Tanh())

u_training_loss = list()
u_validation_loss = list()

u_max_epochs = 10000
for epochs in range(u_max_epochs):
    print(f"Epoch: {epochs+1}/{u_max_epochs}")
    current_training_loss = train_nn(reduced_problem, u_train_dataloader, u_model,
                                     learning_rate=1e-3, loss_func="MSE", optimizer="Adam")
    u_training_loss.append(current_training_loss)
    current_validation_loss = validate_nn(reduced_problem, u_valid_dataloader, u_model, loss_func="MSE")
    u_validation_loss.append(current_validation_loss)
    # 1% safety margin against min_validation_loss before invoking eraly stopping criteria
    if epochs > 0 and current_validation_loss > 1.01 * u_min_validation_loss and reduced_problem.regularisation_u == "EarlyStopping":
        print(f"Early stopping criteria invoked at epoch: {epochs+1}")
        break
    u_min_validation_loss = min(u_validation_loss)

# Training dataset
generate_ann_input_set_p(p_input_training_set_filepath, samples=[5, 5])
generate_ann_output_set_p(problem, reduced_problem, Nmax_p, p_input_training_set_filepath,
                          p_output_training_set_filepath, mode="Training")
p_customDataset = CustomDataset(problem, reduced_problem, Nmax_p, p_input_training_set_filepath, p_output_training_set_filepath, input_scaling_range=[-1., 1.], output_scaling_range=[
                                -1., 1.], input_range=np.array([[0.5, 0.5], [1., 1.]]), output_range=[np.min(np.load(p_output_training_set_filepath)), np.max(np.load(p_output_training_set_filepath))])
p_train_dataloader = DataLoader(p_customDataset, batch_size=10, shuffle=True)

# Validation dataset
generate_ann_input_set_p(p_input_validation_set_filepath, samples=[3, 3])
generate_ann_output_set_p(problem, reduced_problem, Nmax_p, p_input_validation_set_filepath,
                          p_output_validation_set_filepath, mode="Validation")
p_customDataset = CustomDataset(problem, reduced_problem, Nmax_p, p_input_validation_set_filepath, p_output_validation_set_filepath, input_scaling_range=[-1., 1.], output_scaling_range=[
                                -1., 1.], input_range=np.array([[0.5, 0.5], [1., 1.]]), output_range=[np.min(np.load(p_output_training_set_filepath)), np.max(np.load(p_output_training_set_filepath))])
# NOTE VVIP: output_range is from training and NOT from validation to ensure same scaling factors in training as well as validation
p_valid_dataloader = DataLoader(p_customDataset, batch_size=10, shuffle=False)

# TODO replace Nmax with with something like reduced_problem.N
p_model = HiddenLayersNet(training_set.shape[1], [25, 28], Nmax_p, Tanh())

p_training_loss = list()
p_validation_loss = list()

p_max_epochs = 10000
for epochs in range(p_max_epochs):
    print(f"Epoch: {epochs+1}/{p_max_epochs}")
    current_training_loss = train_nn(reduced_problem, p_train_dataloader, p_model,
                                     learning_rate=1e-3, loss_func="MSE", optimizer="Adam")
    p_training_loss.append(current_training_loss)
    current_validation_loss = validate_nn(reduced_problem, p_valid_dataloader, p_model, loss_func="MSE")
    p_validation_loss.append(current_validation_loss)
    # 1% safety margin against min_validation_loss before invoking eraly stopping criteria
    if epochs > 0 and current_validation_loss > 1.01 * p_min_validation_loss and reduced_problem.regularisation_p == "EarlyStopping":
        print(f"Early stopping criteria invoked at epoch: {epochs+1}")
        break
    p_min_validation_loss = min(p_validation_loss)

# Error analysis dataset
print("\n")
print("Generating error analysis (only input/parameters) dataset")
print("\n")
generate_ann_input_set_u(u_input_error_analysis_set_filepath, samples=[3, 3])
u_error_analysis_set = np.load(u_input_error_analysis_set_filepath)
u_error_numpy = np.empty(u_error_analysis_set.shape[0])

for i in range(u_error_analysis_set.shape[0]):
    print(f"Error analysis parameter number {i+1} of {u_error_analysis_set.shape[0]}: {u_error_analysis_set[i,:]}")
    u_error_numpy[i] = error_analysis(reduced_problem, problem, u_error_analysis_set[i, :], u_model, Nmax_u, online_nn, device=None, norm_error=reduced_problem.norm_error_u, reconstruct_solution=reduced_problem.reconstruct_solution_u, input_scaling_range=np.array(
        [-1., 1.]), output_scaling_range=[-1., 1.], input_range=np.array([[0.5, 0.5], [1., 1.]]), output_range=[np.min(np.load(u_output_training_set_filepath)), np.max(np.load(u_output_training_set_filepath))], index=0)
    print(f"Error: {u_error_numpy[i]}")

# Error analysis dataset
print("\n")
print("Generating error analysis (only input/parameters) dataset")
print("\n")
generate_ann_input_set_p(p_input_error_analysis_set_filepath, samples=[3, 3])
p_error_analysis_set = np.load(p_input_error_analysis_set_filepath)
p_error_numpy = np.empty(p_error_analysis_set.shape[0])

for i in range(p_error_analysis_set.shape[0]):
    print(f"Error analysis parameter number {i+1} of {p_error_analysis_set.shape[0]}: {p_error_analysis_set[i,:]}")
    p_error_numpy[i] = error_analysis(reduced_problem, problem, p_error_analysis_set[i, :], p_model, Nmax_p, online_nn, device=None, norm_error=reduced_problem.norm_error_p, reconstruct_solution=reduced_problem.reconstruct_solution_p, input_scaling_range=np.array(
        [-1., 1.]), output_scaling_range=[-1., 1.], input_range=np.array([[0.5, 0.5], [1., 1.]]), output_range=[np.min(np.load(p_output_training_set_filepath)), np.max(np.load(p_output_training_set_filepath))], index=1)
    print(f"Error: {p_error_numpy[i]}")

online_mu = np.array([0.8, 0.9])
(solution_u, solution_p) = problem.solve(online_mu)
rb_solution_u = reduced_problem.reconstruct_solution_u(online_nn(reduced_problem, problem, online_mu, u_model, Nmax_u, device=None, input_scaling_range=np.array(
    [-1., 1.]), output_scaling_range=[-1., 1.], input_range=np.array([[0.5, 0.5], [1., 1.]]), output_range=[np.min(np.load(u_output_training_set_filepath)), np.max(np.load(u_output_training_set_filepath))]))
rb_solution_p = reduced_problem.reconstruct_solution_p(online_nn(reduced_problem, problem, online_mu, p_model, Nmax_p, device=None, input_scaling_range=np.array(
    [-1., 1.]), output_scaling_range=[-1., 1.], input_range=np.array([[0.5, 0.5], [1., 1.]]), output_range=[np.min(np.load(p_output_training_set_filepath)), np.max(np.load(p_output_training_set_filepath))]))

with problem.mesh_motion:
    with dolfinx.io.VTKFile(mpi4py.MPI.COMM_WORLD, "mesh_data/online_fem_solution_velocity.pvd", "w") as ufile_pvd:
        solution_u.x.scatter_forward()
        ufile_pvd.write_mesh(mesh)
        ufile_pvd.write_function(solution_u)

with problem.mesh_motion:
    with dolfinx.io.VTKFile(mpi4py.MPI.COMM_WORLD, "mesh_data/online_fem_solution_pressure.pvd", "w") as pfile_pvd:
        solution_p.x.scatter_forward()
        pfile_pvd.write_mesh(mesh)
        pfile_pvd.write_function(solution_p)

with problem.mesh_motion:
    with dolfinx.io.VTKFile(mpi4py.MPI.COMM_WORLD, "mesh_data/online_rb_solution_velocity.pvd", "w") as ufile_pvd:
        rb_solution_u.x.scatter_forward()
        ufile_pvd.write_mesh(mesh)
        ufile_pvd.write_function(rb_solution_u)

with problem.mesh_motion:
    with dolfinx.io.VTKFile(mpi4py.MPI.COMM_WORLD, "mesh_data/online_rb_solution_pressure.pvd", "w") as pfile_pvd:
        rb_solution_p.x.scatter_forward()
        pfile_pvd.write_mesh(mesh)
        pfile_pvd.write_function(rb_solution_p)

# TODO cuda vs MPI data error
# TODO for poisson problem "scaling_factors" were scalars but for Stokes
# problem they should be turned into numpy array as per scaling rule
# applied
