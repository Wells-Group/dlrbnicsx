import ufl
import numpy as np

from mpi4py import MPI

import dolfinx

from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion


# Mesh deformation class (from MDFEniCSx)
class CustomMeshDeformation(HarmonicMeshMotion):
    def __init__(self, mesh, boundaries, bc_markers_list, bc_function_list,
                 mu, reset_reference=True, is_deformation=True):
        super().__init__(mesh, boundaries, bc_markers_list,
                         bc_function_list, reset_reference, is_deformation)
        self.mu = mu

    def __enter__(self):
        gdim = self._mesh.geometry.dim
        mu = self.mu
        # Compute shape parametrization
        self.shape_parametrization = self.solve()
        self._mesh.geometry.x[:, 0] += \
            (mu[2] - 1.) * (self._mesh.geometry.x[:, 0])
        self._mesh.geometry.x[:, :gdim] += \
            self.shape_parametrization.x.array.\
            reshape(self._reference_coordinates.shape[0], gdim)
        self._mesh.geometry.x[:, 0] -= min(self._mesh.geometry.x[:, 0])
        return self


# Read unit square mesh with Triangular elements
mesh_comm = MPI.COMM_WORLD
gdim = 2
gmsh_model_rank = 0
mesh, cell_tags, facet_tags = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh",
                                    mesh_comm, gmsh_model_rank, gdim=gdim)

# Mesh deformation parameters
mu = np.array([0.3, -0.413, 4.])

# Boundary conditions for custom mesh deformation (not for problem)


def u_bc_bottom(x): return (0.*x[1], mu[0]*np.sin(x[0]*np.pi))


def u_bc_right(x): return (0.*x[0], 0.*x[1])


def u_bc_top(x): return (0.*x[0], -mu[1]*np.sin(x[0]*np.pi))


def u_bc_left(x): return (0.*x[0], 0.*x[1])


# Boundary markers
boundary_markers = [1, 2, 3, 4]
# Boundary condition list for problem
bcs = list()

# Problem setup
V = dolfinx.fem.FunctionSpace(mesh, ("CG", 2))
u_D = dolfinx.fem.Function(V)

uh = dolfinx.fem.Function(V)
v = ufl.TestFunction(V)

s = - ufl.div(ufl.exp(u_D)*ufl.grad(u_D))
F = ufl.inner(ufl.exp(uh) * ufl.grad(uh), ufl.grad(v)) * ufl.dx \
    - ufl.inner(s, v) * ufl.dx

for i in boundary_markers:
    dofs = dolfinx.fem.locate_dofs_topological(V, gdim-1, facet_tags.find(i))
    bcs.append(dolfinx.fem.dirichletbc(u_D, dofs))

problem = dolfinx.fem.petsc.NonlinearProblem(F, uh, bcs=bcs)

# Deform mesh, solve the problem and post processing
with CustomMeshDeformation(mesh, facet_tags, boundary_markers,
     [u_bc_bottom, u_bc_right, u_bc_top, u_bc_left], mu,
      reset_reference=True, is_deformation=True) as mesh_class:

    solver = dolfinx.nls.petsc.NewtonSolver(mesh.comm, problem)
    solver.convergence_criterion = "incremental"

    solver.rtol = 1e-6
    solver.report = True
    ksp = solver.krylov_solver
    ksp.setFromOptions()
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

    u_D.interpolate(lambda x: x[1] * np.sin(x[0]*np.pi) * np.cos(x[1]*np.pi))

    n, converged = solver.solve(uh)
    print(f"Computed solution array: {uh.x.array}")
    assert (converged)
    print(f"Number of interations: {n:d}")

    error = np.sqrt(dolfinx.fem.assemble_scalar(
                    dolfinx.fem.form(ufl.inner(uh - u_D, uh - u_D) * ufl.dx
                                     + ufl.inner(ufl.grad(uh - u_D),
                                     ufl.grad(uh - u_D)) * ufl.dx)))
    print(f"Error: {error}")
    u_error = dolfinx.fem.Function(V)
    u_error.x.array[:] = abs(uh.x.array - u_D.x.array)

    computed_file = "solution_nonlinear_poisson/solution_computed.xdmf"
    with dolfinx.io.XDMFFile(mesh.comm, computed_file, "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(uh)
    actual_file = "solution_nonlinear_poisson/solution_actual.xdmf"
    with dolfinx.io.XDMFFile(mesh.comm, actual_file, "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(u_D)
    error_file = "solution_nonlinear_poisson/absolute_error.xdmf"
    with dolfinx.io.XDMFFile(mesh.comm, error_file, "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(u_error)
    np.save("dlrbnicsx_solution_nonlinear_poisson/dolfinx_array.npy",
            uh.x.array.copy())
