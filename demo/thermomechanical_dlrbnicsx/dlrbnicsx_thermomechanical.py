import dolfinx
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import ufl

import rbnicsx
import rbnicsx.backends
import rbnicsx.online

from dlrbnicsx_thermomechanical_geometric_deformation import MeshDeformationWrapperClass
from dlrbnicsx_thermal import ThermalProblemOnDeformedDomain

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import sympy
from smt.sampling_methods import LHS
import itertools
import abc
import matplotlib.pyplot as plt
import os

from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory import Tanh, Sigmoid
from dlrbnicsx.dataset.custom_dataset import CustomDataset
from dlrbnicsx.interface.wrappers import DataLoader, save_model, load_model, \
    save_checkpoint, load_checkpoint, get_optimiser, get_loss_func
from dlrbnicsx.train_validate_test.train_validate_test import \
    train_nn, validate_nn, online_nn, error_analysis

class MechanicalProblemOnDeformedDomain(abc.ABC):
    def __init__(self, mesh, subdomains, boundaries, thermalproblem):
        self._mesh = mesh
        self._subdomains = subdomains
        self._boundaries = boundaries
        self._thermalproblem = thermalproblem
        dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)
        ds = ufl.Measure("ds", domain=self._mesh, subdomain_data=self._boundaries)
        self.dx = dx
        self._ds_sf = ds(11) + ds(20) + ds(21) + ds(22) + ds(23)
        self._ds_bottom = ds(1) + ds(31)
        self._ds_out = ds(30)
        self._ds_sym = ds(5) + ds(9) + ds(12)
        self._ds_top = ds(18) + ds(19) + ds(27) + ds(28) + ds(29)

        x = ufl.SpatialCoordinate(self._mesh)
        # NOTE Placeholder for ymax, Updated at each new parameter
        self._ymax = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.))
        self._VM = dolfinx.fem.VectorFunctionSpace(self._mesh, ("CG", 1))
        uM, vM = ufl.TrialFunction(self._VM), ufl.TestFunction(self._VM)
        self._trial, self._test = uM, vM
        self._inner_product = ufl.inner(uM, vM) * x[0] * ufl.dx + \
            ufl.inner(self.epsilon(uM), self.epsilon(vM)) * x[0] * ufl.dx
        self.inner_product_action = \
            rbnicsx.backends.bilinear_form_action(self._inner_product,
                                                  part="real")
        self.uT_func = dolfinx.fem.Function(self._thermalproblem._VT)
        self._rho = 77106.
        self._g = 9.8
        self._T0 = 300.
        self.mu_ref = [0.6438, 0.4313, 1., 0.5]

        self.young_modulus_6 = 1.9E11

        self.poisson_ratio_1 = 0.3
        self.poisson_ratio_2 = 0.2
        self.poisson_ratio_3 = 0.1
        self.poisson_ratio_4 = 0.1
        self.poisson_ratio_5 = 0.2
        self.poisson_ratio_6 = 0.3
        self.poisson_ratio_7 = 0.2

        self.thermal_expansion_coefficient_1 = 2.3e-6
        self.thermal_expansion_coefficient_2 = 4.6e-6
        self.thermal_expansion_coefficient_3 = 4.7e-6
        self.thermal_expansion_coefficient_4 = 4.6e-6
        self.thermal_expansion_coefficient_5 = 6.e-6
        self.thermal_expansion_coefficient_6 = 1.2e-5
        self.thermal_expansion_coefficient_7 = 4.6e-6

        dofs_bottom_1 = dolfinx.fem.locate_dofs_topological(self._VM.sub(1), self._mesh.geometry.dim-1, self._boundaries.find(1))
        dofs_bottom_31 = dolfinx.fem.locate_dofs_topological(self._VM.sub(1), self._mesh.geometry.dim-1, self._boundaries.find(31))
        dofs_sym_5 = dolfinx.fem.locate_dofs_topological(self._VM.sub(0), self._mesh.geometry.dim-1, self._boundaries.find(5))
        dofs_sym_9 = dolfinx.fem.locate_dofs_topological(self._VM.sub(0), self._mesh.geometry.dim-1, self._boundaries.find(9))
        dofs_sym_12 = dolfinx.fem.locate_dofs_topological(self._VM.sub(0), self._mesh.geometry.dim-1, self._boundaries.find(12))
        dofs_bottom_1_2 = dolfinx.fem.locate_dofs_topological(self._VM.sub(0), self._mesh.geometry.dim-1, self._boundaries.find(1))
        dofs_bottom_31_2 = dolfinx.fem.locate_dofs_topological(self._VM.sub(0), self._mesh.geometry.dim-1, self._boundaries.find(31))
        dofs_top_18 = dolfinx.fem.locate_dofs_topological(self._VM.sub(0), self._mesh.geometry.dim-1, self._boundaries.find(18))
        dofs_top_28 = dolfinx.fem.locate_dofs_topological(self._VM.sub(0), self._mesh.geometry.dim-1, self._boundaries.find(28))
        dofs_top_19 = dolfinx.fem.locate_dofs_topological(self._VM.sub(1), self._mesh.geometry.dim-1, self._boundaries.find(19))
        dofs_top_27 = dolfinx.fem.locate_dofs_topological(self._VM.sub(1), self._mesh.geometry.dim-1, self._boundaries.find(27))
        dofs_top_29 = dolfinx.fem.locate_dofs_topological(self._VM.sub(1), self._mesh.geometry.dim-1, self._boundaries.find(29))

        bc_bottom_1 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_bottom_1, self._VM.sub(1))
        bc_bottom_31 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_bottom_31, self._VM.sub(1))
        bc_sym_5 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_sym_5, self._VM.sub(0))
        bc_sym_9 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_sym_9, self._VM.sub(0))
        bc_sym_12 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_sym_12, self._VM.sub(0))
        bc_bottom_1_2 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_bottom_1_2, self._VM.sub(0))
        bc_bottom_31_2 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_bottom_31_2, self._VM.sub(0))
        bc_top_18 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_top_18, self._VM.sub(0))
        bc_top_28 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_top_28, self._VM.sub(0))
        bc_top_19 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_top_19, self._VM.sub(1))
        bc_top_27 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_top_27, self._VM.sub(1))
        bc_top_29 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_top_29, self._VM.sub(1))

        self._bcsM = [bc_bottom_1, bc_bottom_31, bc_sym_5, bc_sym_9, bc_sym_12, bc_bottom_1_2, bc_bottom_31_2,
                      bc_top_18, bc_top_28, bc_top_19, bc_top_27, bc_top_29]

    def young_modulus_1(self, sym_T):
        conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 823.)), ufl.And(ufl.ge(sym_T, 823.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
        interps = [1698.65453106434*sym_T**2 - 2185320.53818741*sym_T + 10994471124.8516, 1698.65453106434*sym_T**2 - 2185320.53818741*sym_T + 10994471124.8516, -1586.66402849173*sym_T**2 + 3222313.81084183*sym_T + 8769229590.22603, -1586.66402849173*sym_T**2 + 3222313.81084183*sym_T + 8769229590.22603]
        assert len(conditions) == len(interps)
        d_func = ufl.conditional(conditions[0], interps[0], interps[0])
        for i in range(1, len(conditions)):
            d_func = ufl.conditional(conditions[i], interps[i], d_func)
        return d_func

    def young_modulus_2(self, sym_T):
        conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 823.)), ufl.And(ufl.ge(sym_T, 823.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
        interps = [-547.091412742593*sym_T**2 - 2026218.83656489*sym_T + 16040649369.8061, -547.091412742593*sym_T**2 - 2026218.83656489*sym_T + 16040649369.8061, 8466.75900277084*sym_T**2 - 16863016.6205001*sym_T + 22145991657.8954, 8466.75900277084*sym_T**2 - 16863016.6205001*sym_T + 22145991657.8954]
        assert len(conditions) == len(interps)
        d_func = ufl.conditional(conditions[0], interps[0], interps[0])
        for i in range(1, len(conditions)):
            d_func = ufl.conditional(conditions[i], interps[i], d_func)
        return d_func

    def young_modulus_3(self, sym_T):
        conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 823.)), ufl.And(ufl.ge(sym_T, 823.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
        interps = [-105360.110803325*sym_T**2 + 123741855.955679*sym_T + 30988696357.3406, -105360.110803325*sym_T**2 + 123741855.955679*sym_T + 30988696357.3406, 61686.9806094212*sym_T**2 - 151217656.509701*sym_T + 144134535736.845, 61686.9806094212*sym_T**2 - 151217656.509701*sym_T + 144134535736.845]
        assert len(conditions) == len(interps)
        d_func = ufl.conditional(conditions[0], interps[0], interps[0])
        for i in range(1, len(conditions)):
            d_func = ufl.conditional(conditions[i], interps[i], d_func)
        return d_func

    def young_modulus_4(self, sym_T):
        conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 823.)), ufl.And(ufl.ge(sym_T, 823.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
        interps = [-781.855955678702*sym_T**2 + 927087.257617759*sym_T + 1645484985.45706, -781.855955678702*sym_T**2 + 927087.257617759*sym_T + 1645484985.45706, 656.925207756334*sym_T**2 - 1441146.53739632*sym_T + 2620013192.10535, 656.925207756334*sym_T**2 - 1441146.53739632*sym_T + 2620013192.10535]
        assert len(conditions) == len(interps)
        d_func = ufl.conditional(conditions[0], interps[0], interps[0])
        for i in range(1, len(conditions)):
            d_func = ufl.conditional(conditions[i], interps[i], d_func)
        return d_func

    def young_modulus_5(self, sym_T):
        conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 823.)), ufl.And(ufl.ge(sym_T, 823.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
        interps = [1781.75702413907*sym_T**2 + 242712.70280987*sym_T + 14275923119.3114, 1781.75702413907*sym_T**2 + 242712.70280987*sym_T + 14275923119.3114, 1781.75702413907*sym_T**2 + 242712.70280987*sym_T + 14275923119.3114, 1781.75702413907*sym_T**2 + 242712.70280987*sym_T + 14275923119.3114]
        assert len(conditions) == len(interps)
        d_func = ufl.conditional(conditions[0], interps[0], interps[0])
        for i in range(1, len(conditions)):
            d_func = ufl.conditional(conditions[i], interps[i], d_func)
        return d_func

    def young_modulus_7(self, sym_T):
        conditions = [ufl.le(sym_T, 293.), ufl.And(ufl.ge(sym_T, 293.), ufl.le(sym_T, 823.)), ufl.And(ufl.ge(sym_T, 823.), ufl.le(sym_T, 1273.)), ufl.ge(sym_T, 1273.)]
        interps = [-547.091412742593*sym_T**2 - 2026218.83656489*sym_T + 16040649369.8061, -547.091412742593*sym_T**2 - 2026218.83656489*sym_T + 16040649369.8061, 8466.75900277084*sym_T**2 - 16863016.6205001*sym_T + 22145991657.8954, 8466.75900277084*sym_T**2 - 16863016.6205001*sym_T + 22145991657.8954]
        assert len(conditions) == len(interps)
        d_func = ufl.conditional(conditions[0], interps[0], interps[0])
        for i in range(1, len(conditions)):
            d_func = ufl.conditional(conditions[i], interps[i], d_func)
        return d_func

    def epsilon(self, u):
        x = ufl.SpatialCoordinate(self._mesh)
        return ufl.as_tensor([[u[0].dx(0), 0.5*(u[0].dx(1)+u[1].dx(0)), 0.],[0.5*(u[0].dx(1)+u[1].dx(0)), u[1].dx(1), 0.],[0., 0., u[0]/x[0]]])

    @property
    def bilinear_form(self):
        uT_func = self.uT_func
        x = ufl.SpatialCoordinate(self._mesh)
        dx = self.dx
        uM, vM = self._trial, self._test
        aM = \
            ufl.inner(self.young_modulus_1(uT_func) * self.poisson_ratio_1 / ((1 - 2 * self.poisson_ratio_1) * (1 + self.poisson_ratio_1)) * (uM[0].dx(0) + uM[1].dx(1) + uM[0]/x[0]) * ufl.Identity(3) + 2 * self.young_modulus_1(uT_func) / (2 * (1 + self.poisson_ratio_1)) * self.epsilon(uM), self.epsilon(vM)) * x[0] * dx(1) + \
            ufl.inner(self.young_modulus_2(uT_func) * self.poisson_ratio_2 / ((1 - 2 * self.poisson_ratio_2) * (1 + self.poisson_ratio_2)) * (uM[0].dx(0) + uM[1].dx(1) + uM[0]/x[0]) * ufl.Identity(3) + 2 * self.young_modulus_2(uT_func) / (2 * (1 + self.poisson_ratio_2)) * self.epsilon(uM), self.epsilon(vM)) * x[0] * dx(2) + \
            ufl.inner(self.young_modulus_3(uT_func) * self.poisson_ratio_3 / ((1 - 2 * self.poisson_ratio_3) * (1 + self.poisson_ratio_3)) * (uM[0].dx(0) + uM[1].dx(1) + uM[0]/x[0]) * ufl.Identity(3) + 2 * self.young_modulus_3(uT_func) / (2 * (1 + self.poisson_ratio_3)) * self.epsilon(uM), self.epsilon(vM)) * x[0] * dx(3) + \
            ufl.inner(self.young_modulus_4(uT_func) * self.poisson_ratio_4 / ((1 - 2 * self.poisson_ratio_4) * (1 + self.poisson_ratio_4)) * (uM[0].dx(0) + uM[1].dx(1) + uM[0]/x[0]) * ufl.Identity(3) + 2 * self.young_modulus_4(uT_func) / (2 * (1 + self.poisson_ratio_4)) * self.epsilon(uM), self.epsilon(vM)) * x[0] * dx(4) + \
            ufl.inner(self.young_modulus_5(uT_func) * self.poisson_ratio_5 / ((1 - 2 * self.poisson_ratio_5) * (1 + self.poisson_ratio_5)) * (uM[0].dx(0) + uM[1].dx(1) + uM[0]/x[0]) * ufl.Identity(3) + 2 * self.young_modulus_5(uT_func) / (2 * (1 + self.poisson_ratio_5)) * self.epsilon(uM), self.epsilon(vM)) * x[0] * dx(5) + \
            ufl.inner(self.young_modulus_6 * self.poisson_ratio_6 / ((1 - 2 * self.poisson_ratio_6) * (1 + self.poisson_ratio_6)) * (uM[0].dx(0) + uM[1].dx(1) + uM[0]/x[0]) * ufl.Identity(3) + 2 * self.young_modulus_6 / (2 * (1 + self.poisson_ratio_6)) * self.epsilon(uM), self.epsilon(vM)) * x[0] * dx(6) + \
            ufl.inner(self.young_modulus_7(uT_func) * self.poisson_ratio_7 / ((1 - 2 * self.poisson_ratio_7) * (1 + self.poisson_ratio_7)) * (uM[0].dx(0) + uM[1].dx(1) + uM[0]/x[0]) * ufl.Identity(3) + 2 * self.young_modulus_7(uT_func) / (2 * (1 + self.poisson_ratio_7)) * self.epsilon(uM), self.epsilon(vM)) * x[0] * dx(7)
        return dolfinx.fem.form(aM)

    @property
    def linear_form(self):
        uT_func = self.uT_func
        dx = self.dx
        x = ufl.SpatialCoordinate(self._mesh)
        vM = self._test
        n_vec = ufl.FacetNormal(self._mesh)
        lM = \
            (uT_func - self._T0) * self.young_modulus_1(uT_func) /( 1 - 2 * self.poisson_ratio_1) * self.thermal_expansion_coefficient_1 * (vM[0].dx(0) + vM[1].dx(1) + vM[0]/x[0]) * x[0] * dx(1) + \
            (uT_func - self._T0) * self.young_modulus_2(uT_func) /( 1 - 2 * self.poisson_ratio_2) * self.thermal_expansion_coefficient_2 * (vM[0].dx(0) + vM[1].dx(1) + vM[0]/x[0]) * x[0] * dx(2) + \
            (uT_func - self._T0) * self.young_modulus_3(uT_func) /( 1 - 2 * self.poisson_ratio_3) * self.thermal_expansion_coefficient_3 * (vM[0].dx(0) + vM[1].dx(1) + vM[0]/x[0]) * x[0] * dx(3) + \
            (uT_func - self._T0) * self.young_modulus_4(uT_func) /( 1 - 2 * self.poisson_ratio_4) * self.thermal_expansion_coefficient_4 * (vM[0].dx(0) + vM[1].dx(1) + vM[0]/x[0]) * x[0] * dx(4) + \
            (uT_func - self._T0) * self.young_modulus_5(uT_func) /( 1 - 2 * self.poisson_ratio_5) * self.thermal_expansion_coefficient_5 * (vM[0].dx(0) + vM[1].dx(1) + vM[0]/x[0]) * x[0] * dx(5) + \
            (uT_func - self._T0) * self.young_modulus_6 /( 1 - 2 * self.poisson_ratio_6) * self.thermal_expansion_coefficient_6 * (vM[0].dx(0) + vM[1].dx(1) + vM[0]/x[0]) * x[0] * dx(6) + \
            (uT_func - self._T0) * self.young_modulus_7(uT_func) /( 1 - 2 * self.poisson_ratio_7) * self.thermal_expansion_coefficient_7 * (vM[0].dx(0) + vM[1].dx(1) + vM[0]/x[0]) * x[0] * dx(7) - \
            self._rho * self._g * (self._ymax - x[1]) * ufl.dot(vM, n_vec) * x[0] * self._ds_sf
        return dolfinx.fem.form(lM)

    def solve(self, mu):
        self.mu = mu
        # NOTE VVIP, make sure temperature_field is solved before geometric deformation of mechanical problem else
        # the mesh gets deformed twice for thermal problem
        # 1. One in solve of thermal problem
        # 2. Other in solve of mehcanical problem
        self.uT_func.x.array[:] = self._thermalproblem.solve(self.mu).x.array.copy()
        self.uT_func.x.scatter_forward()
        print(f"Temperature field norm: {self._thermalproblem.inner_product_action(self.uT_func)(self.uT_func)}")
        with MeshDeformationWrapperClass(self._mesh, self._boundaries,
                                         self.mu_ref, self.mu):
            
            self._ymax.value = self._mesh.comm.allreduce(np.max(self._mesh.geometry.x[:, 1]), op=MPI.MAX)
            # Bilinear side assembly
            aM_cpp = self.bilinear_form
            A = dolfinx.fem.petsc.assemble_matrix(aM_cpp, bcs=self._bcsM)
            A.assemble()

            # Linear side assembly
            lM_cpp = self.linear_form
            L = dolfinx.fem.petsc.assemble_vector(lM_cpp)
            dolfinx.fem.petsc.apply_lifting(L, [aM_cpp], [self._bcsM])
            L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            dolfinx.fem.petsc.set_bc(L, self._bcsM)

            # Solver setup
            ksp = PETSc.KSP()
            ksp.create(self._mesh.comm)
            ksp.setOperators(A)
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType("mumps")
            ksp.setFromOptions()
            uM_func = dolfinx.fem.Function(self._VM)
            ksp.solve(L, uM_func.vector)
            uM_func.x.scatter_forward()
            # print(displacement_field.x.array)
            x = ufl.SpatialCoordinate(self._mesh)
            print(f"Displacement field norm: {self._mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(uM_func, uM_func) * x[0] * ufl.dx + ufl.inner(self.epsilon(uM_func), self.epsilon(uM_func)) * x[0] * ufl.dx)))}")
        return uM_func

class MechanicalPODANNReducedProblem(abc.ABC):
    def __init__(self, mechanical_problem) -> None:
        self._basis_functions = rbnicsx.backends.FunctionsList(mechanical_problem._VM)
        uM, vM = ufl.TrialFunction(mechanical_problem._VM), ufl.TestFunction(mechanical_problem._VM)
        x = ufl.SpatialCoordinate(mechanical_problem._mesh)
        self._inner_product = ufl.inner(uM, vM) * x[0] * ufl.dx + \
            ufl.inner(mechanical_problem.epsilon(uM), mechanical_problem.epsilon(vM)) * x[0] * ufl.dx
        self._inner_product_action = \
            rbnicsx.backends.bilinear_form_action(self._inner_product,
                                                  part="real")
        self.input_scaling_range = [-1., 1.]
        self.output_scaling_range = [-1., 1.]
        self.input_range = \
            np.array([[0.55, 0.35, 0.8, 0.4], [0.75, 0.55, 1.2, 0.6]])
        self.output_range = [-6., 3.]
        self.regularisation = "EarlyStopping"

    def reconstruct_solution(self, reduced_solution):
        """Reconstructed reduced solution on the high fidelity space."""
        return self._basis_functions[:reduced_solution.size] * \
            reduced_solution

    def compute_norm(self, function):
        """Compute the norm of a function inner product
        on the reference domain."""
        return np.sqrt(self._inner_product_action(function)(function))

    def project_snapshot(self, solution, N):
        return self._project_snapshot(solution, N)

    def _project_snapshot(self, solution, N):
        projected_snapshot = rbnicsx.online.create_vector(N)
        A = rbnicsx.backends.\
            project_matrix(self._inner_product_action,
                           self._basis_functions[:N])
        F = rbnicsx.backends.\
            project_vector(self._inner_product_action(solution),
                           self._basis_functions[:N])
        ksp = PETSc.KSP()
        ksp.create(projected_snapshot.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
        ksp.solve(F, projected_snapshot)
        return projected_snapshot

    def norm_error(self, u, v):
        return self.compute_norm(u-v)/self.compute_norm(u)

# Read mesh
world_comm = MPI.COMM_WORLD
gdim = 2
gmsh_model_rank = 0
mesh, cell_tags, facet_tags = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh",
                                    world_comm, gmsh_model_rank, gdim=gdim)

# Mesh deformation parameters
mu_ref = [0.6438, 0.4313, 1., 0.5]  # reference geometry
mu = [0.8, 0.55, 0.8, 0.4]  # Parametric geometry

para_dim = 4
mechanical_ann_input_samples_num = 640
mechanical_error_analysis_samples_num = 144
num_snapshots = 700

# FEM solve
thermal_problem_parametric = \
    ThermalProblemOnDeformedDomain(mesh, cell_tags, facet_tags)

mechanical_problem_parametric = \
    MechanicalProblemOnDeformedDomain(mesh, cell_tags, facet_tags,
                                      thermal_problem_parametric)

solution_mu = mechanical_problem_parametric.solve(mu)
print(f"Solution norm at mu:{mu}: {mechanical_problem_parametric.inner_product_action(solution_mu)(solution_mu)}")

VM_plot = dolfinx.fem.VectorFunctionSpace(mesh, ("CG", mesh.geometry.cmaps[0].degree))
uM_func_plot = dolfinx.fem.Function(VM_plot)
uM_func_plot.interpolate(solution_mu)

computed_file = "solution_nonlinear_thermomechanical_mechanical/solution_computed.xdmf"
with MeshDeformationWrapperClass(mesh, facet_tags, mu_ref, mu):
    with dolfinx.io.XDMFFile(mesh.comm, computed_file, "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(uM_func_plot)

# ### Mechanical POD starts ###
def generate_training_set(num_samples, para_dim):
    training_set = np.random.uniform(size=(num_samples, para_dim))
    training_set[:, 0] = (0.75 - 0.55) * training_set[:, 0] + 0.55
    training_set[:, 1] = (0.55 - 0.35) * training_set[:, 1] + 0.35
    training_set[:, 2] = (1.20 - 0.80) * training_set[:, 2] + 0.80
    training_set[:, 3] = (0.60 - 0.40) * training_set[:, 3] + 0.40
    return training_set

mechanical_training_set = generate_training_set(num_snapshots, para_dim)

Nmax = 30

print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
print("")

print("set up snapshots matrix")
mechanical_snapshots_matrix = rbnicsx.backends.FunctionsList(mechanical_problem_parametric._VM)

print("set up reduced problem")
mechanical_reduced_problem = MechanicalPODANNReducedProblem(mechanical_problem_parametric)

print("")

for (mu_index, mu) in enumerate(mechanical_training_set):
    print(rbnicsx.io.TextLine(str(mu_index+1), fill="#"))

    print("Parameter number ", (mu_index+1), "of", mechanical_training_set.shape[0])
    print("high fidelity solve for mu =", mu)
    snapshot = mechanical_problem_parametric.solve(mu)

    print("update snapshots matrix")
    mechanical_snapshots_matrix.append(snapshot)

    print("")

print(rbnicsx.io.TextLine("perform POD", fill="#"))
mechanical_eigenvalues, mechanical_modes, _ = \
    rbnicsx.backends.\
    proper_orthogonal_decomposition(mechanical_snapshots_matrix,
                                    mechanical_reduced_problem._inner_product_action,
                                    N=Nmax, tol=1.e-10)
mechanical_reduced_problem._basis_functions.extend(mechanical_modes)
mechanical_reduced_size = len(mechanical_reduced_problem._basis_functions)
print("")

print(rbnicsx.io.TextBox("POD-Galerkin offline phase ends", fill="="))

mechanical_positive_eigenvalues = np.where(mechanical_eigenvalues > 0., mechanical_eigenvalues, np.nan)
mechanical_singular_values = np.sqrt(mechanical_positive_eigenvalues)

plt.figure(figsize=[8, 10])
xint = list()
yval = list()

for x, y in enumerate(mechanical_eigenvalues[:Nmax]):
    yval.append(y)
    xint.append(x+1)

plt.plot(xint, yval, "*-", color="orange")
plt.xlabel("Eigenvalue number", fontsize=18)
plt.ylabel("Eigenvalue", fontsize=18)
plt.xticks(xint)
plt.yscale("log")
plt.title("Eigenvalue decay (Mechanical)", fontsize=24)
plt.tight_layout()
plt.savefig("mechanical_eigenvalues.png")

print(f"Eigenvalues (Mechanical): {mechanical_positive_eigenvalues}")

# ### Mechanical POD Ends ###

def generate_ann_input_set(mechanical_ann_input_samples_num):
    xlimits = np.array([[0.55, 0.75], [0.35, 0.55],
                        [0.8, 1.2], [0.4, 0.6]])
    sampling = LHS(xlimits=xlimits)
    training_set = sampling(mechanical_ann_input_samples_num)
    return training_set


def generate_ann_output_set(problem, reduced_problem,
                            input_set, mode=None):
    output_set = np.zeros([input_set.shape[0],
                           len(reduced_problem._basis_functions)])
    for i in range(input_set.shape[0]):
        if mode is None:
            print(f"Parameter number {i+1} of {input_set.shape[0]}")
            print(f"Parameter: {input_set[i,:]}")
        else:
            print(f"{mode} parameter number {i+1} of {input_set.shape[0]}")
            print(f"Parameter: {input_set[i,:]}")
        output_set[i, :] = \
            reduced_problem.project_snapshot(problem.solve(input_set[i, :]),
                                             len(reduced_problem._basis_functions)).array.astype("f")
    return output_set

# Training dataset
mechanical_ann_input_set = generate_ann_input_set(mechanical_ann_input_samples_num)
np.random.shuffle(mechanical_ann_input_set)
mechanical_ann_output_set = \
    generate_ann_output_set(mechanical_problem_parametric,
                            mechanical_reduced_problem,
                            mechanical_ann_input_set, mode="Training")

mechanical_num_training_samples = int(0.7 * mechanical_ann_input_set.shape[0])
mechanical_num_validation_samples = \
    mechanical_ann_input_set.shape[0] - mechanical_num_training_samples

mechanical_reduced_problem.output_range[0] = np.min(mechanical_ann_output_set)
mechanical_reduced_problem.output_range[1] = np.max(mechanical_ann_output_set)
# NOTE Output_range based on the computed values instead of user guess.

mechanical_input_training_set = mechanical_ann_input_set[:mechanical_num_training_samples, :]
mechanical_output_training_set = mechanical_ann_output_set[:mechanical_num_training_samples, :]

mechanical_input_validation_set = mechanical_ann_input_set[mechanical_num_training_samples:, :]
mechanical_output_validation_set = mechanical_ann_output_set[mechanical_num_training_samples:, :]

customDataset = CustomDataset(mechanical_reduced_problem,
                              mechanical_input_training_set, mechanical_output_training_set)
mechanical_train_dataloader = DataLoader(customDataset, batch_size=40, shuffle=False)# shuffle=True)

customDataset = CustomDataset(mechanical_reduced_problem,
                              mechanical_input_validation_set, mechanical_output_validation_set)
mechanical_valid_dataloader = DataLoader(customDataset, shuffle=False)

# ANN model
mechanical_model = HiddenLayersNet(mechanical_training_set.shape[1],
                                [35, 35],
                                len(mechanical_reduced_problem._basis_functions),
                                Tanh())

mechanical_path = "mechanical_model.pth"
save_model(mechanical_model, mechanical_path)
# load_model(mechanical_model, mechanical_path)


# Training of ANN
mechanical_training_loss = list()
mechanical_validation_loss = list()

mechanical_max_epochs = 20000
mechanical_min_validation_loss = None
mechanical_start_epoch = 0
mechanical_checkpoint_path = "mechanical_checkpoint"
mechanical_checkpoint_epoch = 10

mechanical_learning_rate = 1e-6
mechanical_optimiser = get_optimiser(mechanical_model, "Adam", mechanical_learning_rate)
mechanical_loss_fn = get_loss_func("MSE", reduction="sum")

if os.path.exists(mechanical_checkpoint_path):
    mechanical_start_epoch, mechanical_min_validation_loss = \
        load_checkpoint(mechanical_checkpoint_path, mechanical_model, mechanical_optimiser)

import time
start_time = time.time()
for mechanical_epochs in range(mechanical_start_epoch, mechanical_max_epochs):
    if mechanical_epochs > 0 and mechanical_epochs % mechanical_checkpoint_epoch == 0:
        save_checkpoint(mechanical_checkpoint_path, mechanical_epochs, mechanical_model, mechanical_optimiser,
                        mechanical_min_validation_loss)
    print(f"Epoch: {mechanical_epochs+1}/{mechanical_max_epochs}")
    mechanical_current_training_loss = train_nn(mechanical_reduced_problem, mechanical_train_dataloader,
                                     mechanical_model, mechanical_loss_fn, mechanical_optimiser)
    mechanical_training_loss.append(mechanical_current_training_loss)
    mechanical_current_validation_loss = validate_nn(mechanical_reduced_problem, mechanical_valid_dataloader,
                                          mechanical_model, mechanical_loss_fn)
    mechanical_validation_loss.append(mechanical_current_validation_loss)
    if mechanical_epochs > 0 and mechanical_current_validation_loss > 1.01 * mechanical_min_validation_loss \
       and mechanical_reduced_problem.regularisation == "EarlyStopping":
        # 1% safety margin against min_validation_loss
        # before invoking early stopping criteria
        print(f"Early stopping criteria invoked at epoch: {mechanical_epochs+1}")
        break
    mechanical_min_validation_loss = min(mechanical_validation_loss)
end_time = time.time()
mechanical_elapsed_time = end_time - start_time

os.system(f"rm {mechanical_checkpoint_path}")

# Error analysis dataset
print("\n")
print("Generating error analysis (only input/parameters) dataset")
print("\n")
mechanical_error_analysis_set = generate_ann_input_set(mechanical_error_analysis_samples_num)
mechanical_error_numpy = np.zeros(mechanical_error_analysis_set.shape[0])

for i in range(mechanical_error_analysis_set.shape[0]):
    print(f"Error analysis {i+1} of {mechanical_error_analysis_set.shape[0]}")
    print(f"Parameter: : {mechanical_error_analysis_set[i,:]}")
    mechanical_error_numpy[i] = error_analysis(mechanical_reduced_problem, mechanical_problem_parametric,
                                    mechanical_error_analysis_set[i, :], mechanical_model,
                                    len(mechanical_reduced_problem._basis_functions),
                                    online_nn)
    print(f"Error: {mechanical_error_numpy[i]}")

# ### Online phase ###
online_mu = np.array([0.45, 0.56, 0.9, 0.7])

mechanical_fem_solution = mechanical_problem_parametric.solve(online_mu)
mechanical_rb_solution = \
    mechanical_reduced_problem.reconstruct_solution(
        online_nn(mechanical_reduced_problem, mechanical_problem_parametric,
                  online_mu, mechanical_model,
                  len(mechanical_reduced_problem._basis_functions)))

mechanical_fem_solution_plot = dolfinx.fem.Function(VM_plot)
mechanical_fem_solution_plot.interpolate(mechanical_fem_solution)
mechanical_rb_solution_plot = dolfinx.fem.Function(VM_plot)
mechanical_rb_solution_plot.interpolate(mechanical_rb_solution)

mechanical_fem_online_file \
    = "dlrbnicsx_solution_thermomechanical/mechanical_fem_online_mu_computed.xdmf"
with MeshDeformationWrapperClass(mesh, facet_tags, mu_ref,
                           online_mu):
    with dolfinx.io.XDMFFile(mesh.comm, mechanical_fem_online_file,
                             "w") as mechanical_solution_file:
        mechanical_solution_file.write_mesh(mesh)
        mechanical_solution_file.write_function(mechanical_fem_solution_plot)

mechanical_rb_online_file \
    = "dlrbnicsx_solution_thermomechanical/mechanical_rb_online_mu_computed.xdmf"
with MeshDeformationWrapperClass(mesh, facet_tags, mu_ref,
                           online_mu):
    with dolfinx.io.XDMFFile(mesh.comm, mechanical_rb_online_file,
                             "w") as mechanical_solution_file:
        # NOTE scatter_forward not considered for online solution
        mechanical_solution_file.write_mesh(mesh)
        mechanical_solution_file.write_function(mechanical_rb_solution_plot)

mechanical_error_function = dolfinx.fem.Function(mechanical_problem_parametric._VM)
mechanical_error_function.x.array[:] = \
    mechanical_fem_solution.x.array - mechanical_rb_solution.x.array

mechanical_error_function_plot = dolfinx.fem.Function(VM_plot)
mechanical_error_function_plot.interpolate(mechanical_error_function)

mechanical_fem_rb_error_file \
    = "dlrbnicsx_solution_thermomechanical/mechanical_fem_rb_error_computed.xdmf"
with MeshDeformationWrapperClass(mesh, facet_tags, mu_ref,
                           online_mu):
    with dolfinx.io.XDMFFile(mesh.comm, mechanical_fem_rb_error_file,
                             "w") as mechanical_solution_file:
        mechanical_solution_file.write_mesh(mesh)
        mechanical_solution_file.write_function(mechanical_error_function_plot)

with MeshDeformationWrapperClass(mesh, facet_tags, mu_ref,
                           online_mu):
    print(mechanical_reduced_problem.norm_error(mechanical_fem_solution, mechanical_rb_solution))
    print(mechanical_reduced_problem.compute_norm(mechanical_error_function))

print(f"Training time (Mechanical): {mechanical_elapsed_time}")
