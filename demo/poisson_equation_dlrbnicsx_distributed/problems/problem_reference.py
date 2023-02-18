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

# 2. FEM formulation on reference / undeformed domain
# TODO VVIP : Since FunctionSpace and dx are defined in ProblemBase and not in ProblemOnDeformedDomain, does dx and trial test function solve problem on the deformed domain?


class ProblemBase(abc.ABC):
    """Define a linear problem, and solve it with KSP."""

    def __init__(self, mesh, subdomains, boundaries) -> None:
        # Define function space
        V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
        self._V = V
        # Define trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        self._trial = u
        self._test = v
        # Define solution
        solution = dolfinx.fem.Function(V)
        self._solution = solution
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
        self._mesh_reference = mesh
        self._boundaries_reference = boundaries
        self._subdomains_reference = subdomains

    @property
    def function_space(self) -> dolfinx.fem.FunctionSpace:
        """Return the function space of the problem."""
        return self._V

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

    def solve(self, mu: np.typing.NDArray[np.float64]) -> dolfinx.fem.Function:
        """Assign the provided parameters value and solve the problem."""
        self._mu[:] = mu
        self._mu_symb.value[:] = mu
        return self._solve()

    def _solve(self) -> dolfinx.fem.Function:
        """Solve the linear problem with KSP."""
        A = self._assemble_matrix()
        F = self._assemble_vector()
        solution = (dolfinx.fem.petsc.LinearProblem(self._a, self._f, bcs=self._bcs,
                    petsc_options={"ksp_type": "preonly", "pc_type": "lu"})).solve()
        '''
        ksp = petsc4py.PETSc.KSP()
        mesh = self._mesh_reference
        ksp.create(mesh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        ksp.setFromOptions()
        solution = self._solution.copy()
        ksp.solve(F, solution.vector)
        '''
        solution.vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
        return solution


if __name__ == "__main__":

    class ProblemBaseInherited(ProblemBase):
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
            return super()._solve()

    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    gdim = 2
    mesh, subdomains, boundaries = dolfinx.io.gmshio.read_from_msh(
        "../mesh_data/domain_poisson.msh", mpi4py.MPI.COMM_SELF, 0, gdim=gdim)

    problem = ProblemBaseInherited(mesh, subdomains, boundaries)
    mu_solve = np.array([1., 1.])
    solution = problem.solve(mu_solve)
    print(solution.x.array)

    if rank == 0:
        with dolfinx.io.XDMFFile(mpi4py.MPI.COMM_SELF, "mesh_data/solution.xdmf", "w") as solution_file_xdmf:
            solution.x.scatter_forward()
            solution_file_xdmf.write_mesh(mesh)
            solution_file_xdmf.write_function(solution)
