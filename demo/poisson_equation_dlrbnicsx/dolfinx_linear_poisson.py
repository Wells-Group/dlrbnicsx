import dolfinx
import ufl

from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

# Read mesh
gdim = 2
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
mesh, subdomains, boundaries = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh",
                                    mesh_comm, gmsh_model_rank, gdim=gdim)


# Mesh deformation parameters
mu = np.array([0.8, 1.1])

# Boundary conditions for harmonic mesh deformation (not for problem)


def u_bc_bottom(x): return (mu[0]*x[0], mu[1]*x[1])


def u_bc_right(x): return (mu[0]*x[0], mu[1]*x[1])


def u_bc_top(x): return (mu[0]*x[0], mu[1]*x[1])


def u_bc_curved(x): return (mu[0]*x[0], mu[1]*x[1])


def u_bc_left(x): return (mu[0]*x[0], mu[1]*x[1])


# Boundary markers
boundary_markers = [1, 2, 3, 4, 5]

# Boundary condition list for problem
bcs = list()

# Problem setup
V = dolfinx.fem.FunctionSpace(mesh, ("CG", 2))
# Function for dirichlet BC
u_D = dolfinx.fem.Function(V)
# Trial and test functions
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
# Bilinear form
a_cpp = dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
# Spatial coordinate
x = ufl.SpatialCoordinate(mesh)
# ufl form of analytical solution
u_ufl = 1 + x[0]**2 + 2*x[1]**2
# Source term
f = -ufl.div(ufl.grad(u_ufl))
# Linear form
l_cpp = dolfinx.fem.form(ufl.inner(f, v) * ufl.dx)
# Error function
error = dolfinx.fem.Function(V)

# Deform mesh
with HarmonicMeshMotion(mesh, boundaries, [1, 2, 3, 4, 5],
                        [u_bc_bottom, u_bc_right, u_bc_top,
                         u_bc_curved, u_bc_left], reset_reference=True):
    # Assemble BCs on deformed mesh
    for i in boundary_markers:
        dofs = dolfinx.fem.locate_dofs_topological(V, gdim-1,
                                                   boundaries.find(i))
        u_D.interpolate(lambda x: 1 + x[0]**2 + 2*x[1]**2)
        bc = dolfinx.fem.dirichletbc(u_D, dofs)
        bcs.append(bc)

    # Bilinear side assembly
    A = dolfinx.fem.petsc.assemble_matrix(a_cpp, bcs=bcs)
    A.assemble()

    # Linear side assembly
    L = dolfinx.fem.petsc.assemble_vector(l_cpp)
    dolfinx.fem.petsc.apply_lifting(L, [a_cpp], [bcs])
    L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(L, bcs)

    # Solver setup
    ksp = PETSc.KSP()
    ksp.create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setFromOptions()
    solution = dolfinx.fem.Function(V)
    ksp.solve(L, solution.vector)

    # Post processing
    with dolfinx.io.XDMFFile(mesh.comm,
                             "solution_poisson/solution_computed.xdmf",
                             "w") as solution_file_xdmf:
        solution.x.scatter_forward()
        np.save("dolfinx_solution_array.npy", solution.x.array)
        solution_file_xdmf.write_mesh(mesh)
        solution_file_xdmf.write_function(solution)

    with dolfinx.io.XDMFFile(mesh.comm,
                             "solution_poisson/solution_analytical.xdmf",
                             "w") as solution_file_xdmf:
        solution_file_xdmf.write_mesh(mesh)
        solution_file_xdmf.write_function(u_D)
        # u_D as exact solution

    error.x.array[:] = abs(solution.x.array - u_D.x.array)

    with dolfinx.io.XDMFFile(mesh.comm,
                             "solution_poisson/error.xdmf",
                             "w") as solution_file_xdmf:
        solution_file_xdmf.write_mesh(mesh)
        solution_file_xdmf.write_function(error)

    print(f"Computed solution array: {solution.x.array}")
    print(f"Analytical solution array: {u_D.x.array}")
    print(f"Maximum error: {max(abs(solution.x.array - u_D.x.array))}")
