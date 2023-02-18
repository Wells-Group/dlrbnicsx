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
        V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 2))
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
        '''ksp = petsc4py.PETSc.KSP()
        ksp.create(mesh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        #ksp.getPC().setFactorSolverType("mumps")
        ksp.setFromOptions()
        solution = self._solution.copy()
        ksp.solve(F, solution.vector)
        solution.vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)'''
        return solution
