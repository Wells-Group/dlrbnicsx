import abc
import itertools
import typing

import dolfinx.fem
import dolfinx.io
import gmsh
import mpi4py.MPI
import numpy as np
import numpy.typing
import petsc4py.PETSc
import plotly.graph_objects as go
import ufl

import rbnicsx.backends
import rbnicsx.online
import rbnicsx.test

if __name__ == "__main__":
    from problem_reference import ProblemBase
else:
    from problems.problem_reference import ProblemBase


# 1. Geometric parametrization
class HarmonicExtension(rbnicsx.backends.MeshMotion):
    """Extend the shape parametrization from the boundary to the interior with an harmonic extension."""
    # TODO Ask Francesco: 1. Reset reference?? 2. Is mesh from global variable

    def __init__(self, mesh, subdomains, boundaries, mu: np.typing.NDArray[np.float64]) -> None:
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
        mD_1 = dolfinx.fem.Function(M)
        mD_1.interpolate(lambda x: (mu[0] * x[0], mu[1] * x[1]))
        mD_2 = dolfinx.fem.Function(M)
        mD_2.interpolate(lambda x: (mu[0] * x[0], mu[1] * x[1]))
        mD_3 = dolfinx.fem.Function(M)
        mD_3.interpolate(lambda x: (mu[0] * x[0], mu[1] * x[1]))
        mD_4 = dolfinx.fem.Function(M)
        mD_4.interpolate(lambda x: (mu[0] * x[0], mu[1] * x[1]))
        mD_5 = dolfinx.fem.Function(M)
        mD_5.interpolate(lambda x: (mu[0] * x[0], mu[1] * x[1]))
        bc_1 = dolfinx.fem.dirichletbc(mD_1, dofs_bc_1)
        bc_2 = dolfinx.fem.dirichletbc(mD_2, dofs_bc_2)
        bc_3 = dolfinx.fem.dirichletbc(mD_3, dofs_bc_3)
        bc_4 = dolfinx.fem.dirichletbc(mD_4, dofs_bc_4)
        bc_5 = dolfinx.fem.dirichletbc(mD_5, dofs_bc_5)
        bcs_he = [bc_1, bc_2, bc_3, bc_4, bc_5]
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

# 3. FEM formulation on deformed domain


class ProblemOnDeformedDomain(ProblemBase):
    """Solve the problem on the deformed domain obtained by harmonic extension."""

    def __init__(self, mesh, subdomains, boundaries) -> None:
        super().__init__(mesh, subdomains, boundaries)
        # Get trial and test functions
        u, v = self.trial_and_test
        # Get measures
        dx, ds = self.measures
        # Get symbolic parameters for use in UFL forms
        mu_symb = self.mu_symb
        # Define bilinear form of the problem
        a = (ufl.inner(ufl.grad(u), ufl.grad(v)) * dx)
        self._a = a
        self._a_cpp = dolfinx.fem.form(a)
        # Define linear form of the problem
        f_term = petsc4py.PETSc.ScalarType(-6.)  # TODO check it is taking correct mesh
        f = (ufl.inner(f_term, v) * dx)
        self._f = f
        self._f_cpp = dolfinx.fem.form(f)
        dofs_bc_1 = dolfinx.fem.locate_dofs_topological(self.function_space, 1, boundaries.find(1))
        dofs_bc_2 = dolfinx.fem.locate_dofs_topological(self.function_space, 1, boundaries.find(2))
        dofs_bc_3 = dolfinx.fem.locate_dofs_topological(self.function_space, 1, boundaries.find(3))
        dofs_bc_4 = dolfinx.fem.locate_dofs_topological(self.function_space, 1, boundaries.find(4))
        dofs_bc_5 = dolfinx.fem.locate_dofs_topological(self.function_space, 1, boundaries.find(5))
        uD_1 = dolfinx.fem.Function(self.function_space)
        uD_1.interpolate(lambda x: 1 + x[0]**2 + 2*x[1]**2)
        uD_2 = dolfinx.fem.Function(self.function_space)
        uD_2.interpolate(lambda x: 1 + x[0]**2 + 2*x[1]**2)
        uD_3 = dolfinx.fem.Function(self.function_space)
        uD_3.interpolate(lambda x: 1 + x[0]**2 + 2*x[1]**2)
        uD_4 = dolfinx.fem.Function(self.function_space)
        uD_4.interpolate(lambda x: 1 + x[0]**2 + 2*x[1]**2)
        uD_5 = dolfinx.fem.Function(self.function_space)
        uD_5.interpolate(lambda x: 1 + x[0]**2 + 2*x[1]**2)
        bc_1 = dolfinx.fem.dirichletbc(uD_1, dofs_bc_1)
        bc_2 = dolfinx.fem.dirichletbc(uD_2, dofs_bc_2)
        bc_3 = dolfinx.fem.dirichletbc(uD_3, dofs_bc_3)
        bc_4 = dolfinx.fem.dirichletbc(uD_4, dofs_bc_4)
        bc_5 = dolfinx.fem.dirichletbc(uD_5, dofs_bc_5)
        bcs = [bc_1, bc_2, bc_3, bc_4, bc_5]
        self._bcs = bcs  # TODO Confirm BCs are evaluated on the deformed domain
        # Store mesh motion object used in the latest solve, to avoid having to solve
        # the harmonic extension once for computation and once for (optional) visualization.
        self._mesh_motion: typing.Optional[rbnicsx.backends.MeshMotion] = None

    @property
    def bilinear_form(self) -> ufl.Form:  # type: ignore[no-any-unimported]
        """Return the bilinear form of the problem."""
        return self._a

    @property
    def linear_form(self) -> ufl.Form:  # type: ignore[no-any-unimported]
        """Return the linear form of the problem."""
        return self._f

    @property
    def mesh_motion(self) -> rbnicsx.backends.MeshMotion:
        """Return the mesh motion object that was used in the latest solve."""
        assert self._mesh_motion is not None
        return self._mesh_motion

    def _assemble_matrix(self) -> petsc4py.PETSc.Mat:  # type: ignore[no-any-unimported]
        """Assemble the left-hand side matrix."""
        A = dolfinx.fem.petsc.assemble_matrix(self._a_cpp, bcs=self._bcs)
        A.assemble()
        return A

    def _assemble_vector(self) -> petsc4py.PETSc.Vec:  # type: ignore[no-any-unimported]
        """Assemble the right-hand side vector."""
        F = dolfinx.fem.petsc.assemble_vector(self._f_cpp)
        F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(F, self._bcs)
        return F

    def _solve(self) -> dolfinx.fem.Function:
        """Apply shape parametrization to the mesh and solve the linear problem with KSP."""
        with HarmonicExtension(self._mesh_reference, self._subdomains_reference, self._boundaries_reference, self._mu) as self._mesh_motion:
            return super()._solve()


if __name__ == "__main__":

    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    gdim = 2
    mesh, subdomains, boundaries = dolfinx.io.gmshio.read_from_msh(
        "../mesh_data/domain_poisson.msh", mpi4py.MPI.COMM_SELF, 0, gdim=gdim)

    problem = ProblemOnDeformedDomain(mesh, subdomains, boundaries)
    mu_solve = np.array([1.1, 1.4])
    solution = problem.solve(mu_solve)
    print(solution.x.array)

    if rank == 0:
        with problem.mesh_motion:
            with dolfinx.io.XDMFFile(mpi4py.MPI.COMM_SELF, "mesh_data/solution.xdmf", "w") as solution_file_xdmf:
                # solution.x.scatter_forward()
                solution_file_xdmf.write_mesh(mesh)
                solution_file_xdmf.write_function(solution)

    mu_solve = np.array([1., 1.])
    solution = problem.solve(mu_solve)
    print(solution.x.array)

    if rank == 0:
        with problem.mesh_motion:
            with dolfinx.io.XDMFFile(mpi4py.MPI.COMM_SELF, "mesh_data/solution_reference.xdmf", "w") as solution_file_xdmf:
                # solution.x.scatter_forward()
                solution_file_xdmf.write_mesh(mesh)
                solution_file_xdmf.write_function(solution)
