import dolfinx
import ufl
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

import rbnicsx
import rbnicsx.backends
import rbnicsx.online

from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import sympy
from smt.sampling_methods import LHS
import itertools
import abc
import matplotlib.pyplot as plt
import os
import time

from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory import Tanh, Sigmoid
from dlrbnicsx.dataset.custom_dataset import CustomDataset
from dlrbnicsx.interface.wrappers import DataLoader, save_model, load_model, \
    save_checkpoint, load_checkpoint, get_optimiser, get_loss_func
from dlrbnicsx.train_validate_test.train_validate_test import \
    train_nn, validate_nn, online_nn, error_analysis
from dlrbnicsx_thermal import ProblemOnDeformedDomain


class ProblemOnDeformedDomainMechanical(abc.ABC):
    def __init__(self, mesh, cell_tags, facet_tags, thermalProblem):
        # Mesh, Subdomians and Boundaries, Mesh deformation
        self._mesh = mesh
        self.gdim = self._mesh.geometry.dim
        self._cell_tags = cell_tags
        self._facet_tags = facet_tags
        self._thermalProblem = \
            thermalProblem(self._mesh, self._cell_tags,
                           self._facet_tags, HarmonicMeshMotion)
        self.meshDeformationContext = HarmonicMeshMotion

        # Define function space, Trial and Test Function
        self._VM = dolfinx.fem.VectorFunctionSpace(mesh, ("CG", 1))
        self._uM, self._vM = ufl.TrialFunction(self._VM), ufl.TestFunction(self._VM)
        self._uM_func = dolfinx.fem.Function(self._VM)

        self._dx = ufl.Measure("dx", domain=self._mesh, subdomain_data=self._cell_tags)
        self._ds = ufl.Measure("ds", domain=self._mesh, subdomain_data=self._facet_tags)
        
        self._ds_bottom = self._ds(4) + self._ds(10)
        self._ds_outer_bottom = self._ds(5) + self._ds(11)
        self._ds_outer_top = self._ds(15) + self._ds(19)
        self._ds_1top = self._ds(8) + self._ds(14)
        self._ds_2top = self._ds(16) + self._ds(20)
        self._ds_3top = self._ds(23) + self._ds(26)
        self._ds_inner = self._ds(22) + self._ds(25)

        self._dx_sub_1 = self._dx(1) + self._dx(2)
        self._dx_sub_2 = self._dx(3) + self._dx(4)
        self._dx_sub_3 = self._dx(5) + self._dx(6)

        self._rho = 77106.
        self._g = 9.8
        self._T0 = 300.

        self._poisson_ratio_1 = 0.3
        self._poisson_ratio_2 = 0.2
        self._poisson_ratio_3 = 0.1

        self._thermal_expansion_coefficient_1 = 2.3e-6
        self._thermal_expansion_coefficient_2 = 4.6e-6
        self._thermal_expansion_coefficient_3 = 4.7e-6

        self._x = ufl.SpatialCoordinate(self._mesh)
        self._n_vec = ufl.FacetNormal(self._mesh)
        self._sym_T = sympy.Symbol("sym_T")
        self._ymax = dolfinx.fem.Constant(self._mesh, PETSc.ScalarType(0.))

        # Velocity and Pressure inner product (To be used in POD)
        self._inner_product_uM = ufl.inner(self._uM, self._vM) * self._dx + \
            ufl.inner(self.epsilon(self._uM), self.epsilon(self._vM)) * self._dx
        self._inner_product_action_uM = \
            rbnicsx.backends.bilinear_form_action(self._inner_product_uM,
                                                  part="real")

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

    def epsilon(self, u):
        return ufl.sym(ufl.grad(u))

    def sigma_mech(self, u, young_modulus, poisson_ratio, uT_func):
        lambda_ = young_modulus(uT_func) * poisson_ratio / ((1. - 2. * poisson_ratio) * (1. + poisson_ratio))
        mu = young_modulus(uT_func) / (2. * (1. + poisson_ratio))
        return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * self.epsilon(u)

    def sigma_thermal(self, young_modulus, poisson_ratio, thermal_coeff, T0, uT_func):
        lambda_ = young_modulus(uT_func) * poisson_ratio / ((1. - 2. * poisson_ratio) * (1. + poisson_ratio))
        mu = young_modulus(uT_func) / (2. * (1. + poisson_ratio))
        stress_thermal = (2 * mu + 3 * lambda_) * thermal_coeff * (uT_func - T0)
        return stress_thermal
        
    @property
    def bilinear_form(self):
        a_M = \
            ufl.inner(self.sigma_mech(self._uM, self.young_modulus_1,
                                      self._poisson_ratio_1, self._thermalProblem._uT_func),
                      self.epsilon(self._vM)) * self._dx_sub_1 + \
            ufl.inner(self.sigma_mech(self._uM, self.young_modulus_2,
                                      self._poisson_ratio_2, self._thermalProblem._uT_func),
                      self.epsilon(self._vM)) * self._dx_sub_2 + \
            ufl.inner(self.sigma_mech(self._uM, self.young_modulus_3,
                                      self._poisson_ratio_3, self._thermalProblem._uT_func),
                                      self.epsilon(self._vM)) * self._dx_sub_3
        return dolfinx.fem.form(a_M)

    @property
    def linear_form(self):
        l_M = \
            ufl.inner(self.sigma_thermal(self.young_modulus_1, self._poisson_ratio_1,
                                         self._thermal_expansion_coefficient_1, self._T0,
                                         self._thermalProblem._uT_func), ufl.div(self._vM)) * \
            self._dx_sub_1 + \
            ufl.inner(self.sigma_thermal(self.young_modulus_2, self._poisson_ratio_2,
                                         self._thermal_expansion_coefficient_2, self._T0,
                                         self._thermalProblem._uT_func), ufl.div(self._vM)) * \
            self._dx_sub_2 + \
            ufl.inner(self.sigma_thermal(self.young_modulus_3, self._poisson_ratio_3,
                                         self._thermal_expansion_coefficient_3, self._T0,
                                         self._thermalProblem._uT_func), ufl.div(self._vM)) * \
            self._dx_sub_3 - \
            self._rho * self._g * (self._ymax - self._x[1]) * ufl.dot(self._vM, self._n_vec) * \
            (self._ds_inner + self._ds_1top)
        return dolfinx.fem.form(l_M)
    
    def assemble_bcs(self):
        dofs_bottom_x_4 = \
            dolfinx.fem.locate_dofs_topological(self._VM.sub(0),
                                                self.gdim-1,
                                                self._facet_tags.find(4))
        dofs_bottom_y_4 = \
            dolfinx.fem.locate_dofs_topological(self._VM.sub(1),
                                                self.gdim-1,
                                                self._facet_tags.find(4))
        dofs_bottom_z_4 = \
            dolfinx.fem.locate_dofs_topological(self._VM.sub(2),
                                                self.gdim-1,
                                                self._facet_tags.find(4))
        dofs_bottom_x_10 = \
            dolfinx.fem.locate_dofs_topological(self._VM.sub(0),
                                                self.gdim-1,
                                                self._facet_tags.find(10))
        dofs_bottom_y_10 = \
            dolfinx.fem.locate_dofs_topological(self._VM.sub(1),
                                                self.gdim-1,
                                                self._facet_tags.find(10))
        dofs_bottom_z_10 = \
            dolfinx.fem.locate_dofs_topological(self._VM.sub(2),
                                                self.gdim-1,
                                                self._facet_tags.find(10))

        bc_bottom_x_4 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.),
                                                dofs_bottom_x_4, self._VM.sub(0))
        bc_bottom_y_4 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.),
                                                dofs_bottom_y_4, self._VM.sub(1))
        bc_bottom_z_4 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.),
                                                dofs_bottom_z_4, self._VM.sub(2))
        bc_bottom_x_10 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.),
                                                 dofs_bottom_x_10, self._VM.sub(0))
        bc_bottom_y_10 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.),
                                                 dofs_bottom_y_10, self._VM.sub(1))
        bc_bottom_z_10 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.),
                                                 dofs_bottom_z_10, self._VM.sub(2))

        dofs_top_y_16 = dolfinx.fem.locate_dofs_topological(self._VM.sub(1), self.gdim-1,
                                                            self._facet_tags.find(16))
        dofs_top_y_20 = dolfinx.fem.locate_dofs_topological(self._VM.sub(1), self.gdim-1,
                                                            self._facet_tags.find(20))
        dofs_top_y_23 = dolfinx.fem.locate_dofs_topological(self._VM.sub(1), self.gdim-1,
                                                            self._facet_tags.find(23))
        dofs_top_y_26 = dolfinx.fem.locate_dofs_topological(self._VM.sub(1), self.gdim-1,
                                                            self._facet_tags.find(26))

        bc_top_y_16 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_top_y_16,
                                              self._VM.sub(1))
        bc_top_y_20 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_top_y_20,
                                              self._VM.sub(1))
        bc_top_y_23 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_top_y_23,
                                              self._VM.sub(1))
        bc_top_y_26 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_top_y_26,
                                              self._VM.sub(1))

        bcsM = [bc_bottom_x_4, bc_bottom_y_4, bc_bottom_z_4,
                bc_bottom_x_10, bc_bottom_y_10, bc_bottom_z_10,
                bc_top_y_16, bc_top_y_20, bc_top_y_23, bc_top_y_26]
        
        return bcsM
    
    def solve(self, mu):
        # Solve the problem at given parameter mu
        self.mu = mu
        _ = self._thermalProblem.solve(mu)
        # Mesh deformation (Harmonic mesh motion)
        with HarmonicMeshMotion(self._mesh, self._facet_tags,
                                [4, 10, 5, 11, 15, 19,
                                 8, 14, 16, 20, 23, 26,
                                 22, 25, 17, 21, 6, 12,
                                 7, 13],
                                [self._thermalProblem.bc_external,
                                 self._thermalProblem.bc_external,
                                 self._thermalProblem.bc_external,
                                 self._thermalProblem.bc_external,
                                 self._thermalProblem.bc_external,
                                 self._thermalProblem.bc_external,
                                 self._thermalProblem.bc_external,
                                 self._thermalProblem.bc_external,
                                 self._thermalProblem.bc_external,
                                 self._thermalProblem.bc_external,
                                 self._thermalProblem.bc_external,
                                 self._thermalProblem.bc_external,
                                 self._thermalProblem.bc_external,
                                 self._thermalProblem.bc_external,
                                 self._thermalProblem.bc_internal,
                                 self._thermalProblem.bc_internal,
                                 self._thermalProblem.bc_external,
                                 self._thermalProblem.bc_external,
                                 self._thermalProblem.bc_external,
                                 self._thermalProblem.bc_external],
                                reset_reference=True,
                                is_deformation=True):
            bcsM = self.assemble_bcs()
            a_M_cpp = self.bilinear_form
            l_M_cpp = self.linear_form
            self._ymax.value = \
                self._mesh.comm.allreduce(np.max(self._mesh.geometry.x[:, 1]),
                                          op=MPI.MAX)
            
            # Bilinear side assembly
            A = dolfinx.fem.petsc.assemble_matrix(a_M_cpp, bcs=bcsM)
            A.assemble()

            # Linear side assembly
            L = dolfinx.fem.petsc.assemble_vector(l_M_cpp)
            dolfinx.fem.petsc.apply_lifting(L, [a_M_cpp], [bcsM])
            L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            dolfinx.fem.petsc.set_bc(L, bcsM)

            # Solver setup
            ksp = PETSc.KSP()
            ksp.create(self._mesh.comm)
            ksp.setOperators(A)
            ksp.setType("preonly")  # ("cg")
            ksp.getPC().setType("lu")  # ("gamg")
            # ksp.getPC().setFactorSolverType("mumps")
            ksp.setFromOptions()
            mechanical_start_time = time.process_time()
            ksp.solve(L, self._uM_func.vector)
            mechanical_end_time = time.process_time()
            print(f"Number of iteration: {ksp.its}")
            self._uM_func.x.scatter_forward()
            solution = dolfinx.fem.Function(self._VM)
            solution.x.array[:] = self._uM_func.x.array.copy()

            sol_norm_local = self._inner_product_action_uM(self._uM_func)(self._uM_func)
            print(f"Displacement field norm: {sol_norm_local}")
            return solution


class PODANNReducedProblem(abc.ABC):
    def __init__(self, problem) -> None:
        self._basis_functions = rbnicsx.backends.FunctionsList(problem._VM)
        self._inner_product = problem._inner_product_uM
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

# Read unit square mesh with Triangular elements
mesh_comm = MPI.COMM_WORLD
gdim = 3
gmsh_model_rank = 0
mesh, cell_tags, facet_tags = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh",
                                    mesh_comm, gmsh_model_rank, gdim=gdim)

# Parameter tuple (D_0, ??, ??, ??)
mu_ref = [0., 0.55, 0.8, 0.4]  # reference geometry
mu = [0.23, 0.55, 0.8, 0.4] # Parametrised geometry

para_dim = 4
mechanical_ann_input_samples_num = 13 # 420
mechanical_error_analysis_samples_num = 23 # 144
num_snapshots = 17 # 400
projection_error_samples_num = [7, 1, 1, 1]

problem_parametric = \
    ProblemOnDeformedDomainMechanical(mesh, cell_tags,
                                      facet_tags, ProblemOnDeformedDomain)
uM_func = problem_parametric.solve(mu)

VM_plot = dolfinx.fem.VectorFunctionSpace(mesh, ("CG", 2))
uM_func_plot = dolfinx.fem.Function(VM_plot)
uM_func_plot.interpolate(uM_func)

computed_file = "dlrbnicsx_solution_nonlinear_thermomechanical_mechanical/solution_computed.xdmf"

with dolfinx.io.XDMFFile(mesh.comm, computed_file, "w") as solution_file:
    solution_file.write_mesh(mesh)
    solution_file.write_function(uM_func_plot)

# POD Starts ###
def generate_training_set(samples=[5, 1, 1, 1]):
    training_set_0 = np.linspace(0., 0.2, samples[0])
    training_set_1 = np.linspace(0., 0., samples[1])
    training_set_2 = np.linspace(0., 0., samples[2])
    training_set_3 = np.linspace(0., 0., samples[2])
    training_set = \
        np.array(list(itertools.product(training_set_0,
                                        training_set_1,
                                        training_set_2,
                                        training_set_3)))
    return training_set


training_set = rbnicsx.io.on_rank_zero(mesh.comm, generate_training_set)

Nmax = 30

print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
print("")

print("set up snapshots matrix")
snapshots_matrix = rbnicsx.backends.FunctionsList(problem_parametric._VM)

print("set up reduced problem")
reduced_problem = PODANNReducedProblem(problem_parametric)

print("")

for (mu_index, mu) in enumerate(training_set):
    print(rbnicsx.io.TextLine(str(mu_index+1), fill="#"))

    print("Parameter number ", (mu_index+1), "of", training_set.shape[0])
    print("high fidelity solve for mu =", mu)
    snapshot = problem_parametric.solve(mu)

    print("update snapshots matrix")
    snapshots_matrix.append(snapshot)

    print("")

print(rbnicsx.io.TextLine("perform POD", fill="#"))
eigenvalues_mechanical, modes_mechanical, _ = \
    rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_matrix,
                                    problem_parametric._inner_product_action_uM,
                                    N=Nmax, tol=1.e-6)
reduced_problem._basis_functions.extend(modes_mechanical)
reduced_size = len(reduced_problem._basis_functions)
print("")

print(rbnicsx.io.TextBox("POD-Galerkin offline phase ends", fill="="))

positive_eigenvalues = np.where(eigenvalues_mechanical > 0., eigenvalues_mechanical, np.nan)
singular_values = np.sqrt(positive_eigenvalues)


plt.figure(figsize=[8, 10])
xint = list()
yval = list()

for x, y in enumerate(eigenvalues_mechanical[:reduced_size]):
    yval.append(y)
    xint.append(x+1)

plt.plot(xint, yval, "*-", color="orange")
plt.xlabel("Eigenvalue number", fontsize=18)
plt.ylabel("Eigenvalue", fontsize=18)
plt.xticks(xint)
plt.yscale("log")
plt.title("Eigenvalue decay", fontsize=24)
plt.tight_layout()
# plt.show()
plt.savefig("eigenvalue_thermomechanical.png")

# Projection errors #
projection_error_samples = \
    generate_training_set(samples=projection_error_samples_num)
projection_error_array = np.zeros(projection_error_samples.shape[0])

for i in range(projection_error_samples.shape[0]):
    fem_sol = problem_parametric.solve(projection_error_samples[i, :])
    projected_sol = reduced_problem.project_snapshot(fem_sol, reduced_size)
    rb_sol = reduced_problem.reconstruct_solution(projected_sol)
    projection_error_array[i] = \
        reduced_problem.norm_error(fem_sol, rb_sol)
    print(f"Projection error {projection_error_array[i]}, Parameter: {i}, {projection_error_samples[i, :]}")
    fem_sol_norm = reduced_problem.compute_norm(fem_sol)
    rb_sol_norm = reduced_problem.compute_norm(rb_sol)
    print(f"FEM solution norm: {fem_sol_norm}, RB solution norm: {rb_sol_norm}")

# POD Ends ###

# 5. ANN implementation

def generate_ann_input_set(num_ann_samples):
    xlimits = np.array([[0.05, 0.55], [0.35, 0.35],
                        [0.8, 0.8], [0.4, 0.4]])
    sampling = LHS(xlimits=xlimits)
    training_set = sampling(num_ann_samples)
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
mechanical_ann_input_samples = generate_ann_input_set(mechanical_ann_input_samples_num)
np.random.shuffle(mechanical_ann_input_samples)
mechanical_ann_output_set = \
    generate_ann_output_set(problem_parametric,
                            reduced_problem,
                            mechanical_ann_input_samples, mode="Training")

mechanical_num_training_samples = int(0.7 * mechanical_ann_input_samples.shape[0])
mechanical_num_validation_samples = \
mechanical_ann_input_samples.shape[0] - mechanical_num_training_samples

reduced_problem.output_range[0] = np.min(mechanical_ann_output_set)
reduced_problem.output_range[1] = np.max(mechanical_ann_output_set)
# NOTE Output_range based on the computed values instead of user guess.

mechanical_input_training_set = mechanical_ann_input_samples[:mechanical_num_training_samples, :]
mechanical_output_training_set = mechanical_ann_output_set[:mechanical_num_training_samples, :]

mechanical_input_validation_set = mechanical_ann_input_samples[mechanical_num_training_samples:, :]
mechanical_output_validation_set = mechanical_ann_output_set[mechanical_num_training_samples:, :]

customDataset = CustomDataset(reduced_problem,
                              mechanical_input_training_set,
                              mechanical_output_training_set)
mechanical_train_dataloader = DataLoader(customDataset, batch_size=40,
                                         shuffle=True)

customDataset = CustomDataset(reduced_problem,
                              mechanical_input_validation_set,
                              mechanical_output_validation_set)
mechanical_valid_dataloader = \
    DataLoader(customDataset,
               batch_size=mechanical_input_validation_set.shape[0],
               shuffle=False)

# ANN model
mechanical_model = HiddenLayersNet(training_set.shape[1],
                                   [35, 35],
                                   reduced_size,
                                   Tanh())

for params in mechanical_model.parameters():
    print(params.shape)

mechanical_path = "mechanical_model.pth"
save_model(mechanical_model, mechanical_path)
# load_model(mechanical_model, mechanical_path)


# Training of ANN
mechanical_training_loss = list()
mechanical_validation_loss = list()

mechanical_max_epochs = 20 # 20000
mechanical_min_validation_loss = None
mechanical_start_epoch = 0
mechanical_checkpoint_path = "thermal_checkpoint"
mechanical_checkpoint_epoch = 10

mechanical_learning_rate = 1.e-6
mechanical_optimiser = get_optimiser(mechanical_model, "Adam", mechanical_learning_rate)
mechanical_loss_fn = get_loss_func("MSE", reduction="sum")

if os.path.exists(mechanical_checkpoint_path):
    mechanical_start_epoch, mechanical_min_validation_loss = \
        load_checkpoint(mechanical_checkpoint_path, mechanical_model, mechanical_optimiser)

import time
start_time = time.time()
for mechanical_epochs in range(mechanical_start_epoch, mechanical_max_epochs):
    if mechanical_epochs > 0 and mechanical_epochs % mechanical_checkpoint_epoch == 0:
        save_checkpoint(mechanical_checkpoint_path, mechanical_epochs,
                        mechanical_model, mechanical_optimiser,
                        mechanical_min_validation_loss)
    print(f"Epoch: {mechanical_epochs+1}/{mechanical_max_epochs}")
    mechanical_current_training_loss = train_nn(reduced_problem, mechanical_train_dataloader,
                                    mechanical_model, mechanical_loss_fn, mechanical_optimiser)
    mechanical_training_loss.append(mechanical_current_training_loss)
    mechanical_current_validation_loss = validate_nn(reduced_problem, mechanical_valid_dataloader,
                                        mechanical_model, mechanical_loss_fn)
    mechanical_validation_loss.append(mechanical_current_validation_loss)
    if mechanical_epochs > 0 and mechanical_current_validation_loss > 1.01 * mechanical_min_validation_loss \
    and reduced_problem.regularisation == "EarlyStopping":
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
    mechanical_error_numpy[i] = error_analysis(reduced_problem, problem_parametric,
                                    mechanical_error_analysis_set[i, :], mechanical_model,
                                    len(reduced_problem._basis_functions),
                                    online_nn)
    print(f"Error: {mechanical_error_numpy[i]}")

# ### Mechanical ANN ends

# ### Online phase ###
online_mu = np.array([0.45, 0.56, 0.9, 0.7])
mechanical_fem_solution = problem_parametric.solve(online_mu)
mechanical_rb_solution = \
    reduced_problem.reconstruct_solution(
        online_nn(reduced_problem, problem_parametric,
                  online_mu, mechanical_model,
                  reduced_size))

mechanical_fem_solution_plot = dolfinx.fem.Function(VM_plot)
mechanical_fem_solution_plot.interpolate(mechanical_fem_solution)

mechanical_rb_solution_plot = dolfinx.fem.Function(VM_plot)
mechanical_rb_solution_plot.interpolate(mechanical_rb_solution)

mechanical_fem_online_file \
    = "dlrbnicsx_solution_thermomechanical/mechanical_fem_online_mu_computed.xdmf"
with HarmonicMeshMotion(problem_parametric._mesh,
                        problem_parametric._facet_tags,
                        [4, 10, 5, 11, 15, 19,
                         8, 14, 16, 20, 23, 26,
                         22, 25, 17, 21, 6, 12,
                         7, 13],
                        [problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_internal,
                         problem_parametric._thermalProblem.bc_internal,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external],
                        reset_reference=True,
                        is_deformation=True):
    with dolfinx.io.XDMFFile(mesh.comm, mechanical_fem_online_file,
                                "w") as mechanical_solution_file:
        mechanical_solution_file.write_mesh(mesh)
        mechanical_solution_file.write_function(mechanical_fem_solution_plot)

mechanical_rb_online_file \
    = "dlrbnicsx_solution_thermomechanical/mechanical_rb_online_mu_computed.xdmf"
with HarmonicMeshMotion(problem_parametric._mesh,
                        problem_parametric._facet_tags,
                        [4, 10, 5, 11, 15, 19,
                         8, 14, 16, 20, 23, 26,
                         22, 25, 17, 21, 6, 12,
                         7, 13],
                        [problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_internal,
                         problem_parametric._thermalProblem.bc_internal,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external],
                        reset_reference=True,
                        is_deformation=True):
    with dolfinx.io.XDMFFile(mesh.comm, mechanical_rb_online_file,
                             "w") as mechanical_solution_file:
        # NOTE scatter_forward not considered for online solution
        mechanical_solution_file.write_mesh(mesh)
        mechanical_solution_file.write_function(mechanical_rb_solution_plot)

mechanical_error_function = dolfinx.fem.Function(problem_parametric._VM)
mechanical_error_function.x.array[:] = \
    mechanical_fem_solution.x.array - mechanical_rb_solution.x.array

mechanical_error_function_plot = dolfinx.fem.Function(VM_plot)
mechanical_error_function_plot.interpolate(mechanical_error_function)
mechanical_fem_rb_error_file \
    = "dlbnicx_solution_thermomechanical/mechanical_fem_rb_error_computed.xdmf"
with HarmonicMeshMotion(problem_parametric._mesh,
                        problem_parametric._facet_tags,
                        [4, 10, 5, 11, 15, 19,
                         8, 14, 16, 20, 23, 26,
                         22, 25, 17, 21, 6, 12,
                         7, 13],
                        [problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_internal,
                         problem_parametric._thermalProblem.bc_internal,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external],
                        reset_reference=True,
                        is_deformation=True):
    with dolfinx.io.XDMFFile(mesh.comm, mechanical_fem_rb_error_file,
                             "w") as mechanical_solution_file:
        mechanical_solution_file.write_mesh(mesh)
        mechanical_solution_file.write_function(mechanical_error_function_plot)

with HarmonicMeshMotion(problem_parametric._mesh,
                        problem_parametric._facet_tags,
                        [4, 10, 5, 11, 15, 19,
                         8, 14, 16, 20, 23, 26,
                         22, 25, 17, 21, 6, 12,
                         7, 13],
                        [problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_internal,
                         problem_parametric._thermalProblem.bc_internal,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external,
                         problem_parametric._thermalProblem.bc_external],
                        reset_reference=True,
                        is_deformation=True):
    print(reduced_problem.norm_error(mechanical_fem_solution, mechanical_rb_solution))
    print(reduced_problem.compute_norm(mechanical_error_function))

print(reduced_problem.norm_error(mechanical_fem_solution, mechanical_rb_solution))
print(reduced_problem.compute_norm(mechanical_error_function))

print(f"Training time (Mechanical): {mechanical_elapsed_time}")
