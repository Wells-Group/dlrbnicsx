import dolfinx
import ufl
from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

# Read mesh
gdim = 2
comm = MPI.COMM_WORLD
gmsh_model_rank = 0
mesh, subdomains, boundaries = dolfinx.io.gmshio.read_from_msh(
    "mesh_data/domain_geometry.msh", comm, gmsh_model_rank, gdim=gdim)

P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
UP = P2 * P1
W = dolfinx.fem.FunctionSpace(mesh, UP)
V, _ = W.sub(0).collapse()
Q, _ = W.sub(1).collapse()
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)

a_cpp = dolfinx.fem.form((ufl.inner(ufl.grad(u), ufl.grad(v)) +
                          ufl.inner(p, ufl.div(v)) +
                          ufl.inner(ufl.div(u), q)) * ufl.dx)
f = dolfinx.fem.Function(V)
l_cpp = dolfinx.fem.form(ufl.inner(f, v) * ufl.dx)


def inlet(x):
    return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))


def no_slip(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))


def free_boundary_p(x):
    return np.zeros(x.shape[1],)


# Online parameter
mu = np.array([1., 1.])


def airfoil_bc(x): return (mu[0] * x[0], mu[1] * x[1])


def domain_bc(x): return (x[0], x[1])


boundary_markers = [1, 2, 3, 4, 5, 6]

with HarmonicMeshMotion(mesh, boundaries, boundary_markers,
                        [domain_bc, domain_bc, domain_bc, domain_bc,
                         airfoil_bc, airfoil_bc], reset_reference=True,
                        is_deformation=False):
    bcs = list()
    for i in boundary_markers:
        dirichletFunc_u = dolfinx.fem.Function(V)
        dirichletFunc_p = dolfinx.fem.Function(Q)
        if i == 5 or i == 6:
            dofs = \
                dolfinx.fem.locate_dofs_topological((W.sub(0), V),
                                                    gdim-1,
                                                    boundaries.find(i))
            dirichletFunc_u.interpolate(no_slip)
            bc = dolfinx.fem.dirichletbc(dirichletFunc_u, dofs, W.sub(0))
        elif i == 1 or i == 2 or i == 4:
            dofs = dolfinx.fem.locate_dofs_topological((W.sub(0), V),
                                                       gdim-1,
                                                       boundaries.find(i))
            dirichletFunc_u.interpolate(inlet)
            bc = dolfinx.fem.dirichletbc(dirichletFunc_u, dofs, W.sub(0))
        else:
            dofs = dolfinx.fem.locate_dofs_topological((W.sub(1), Q),
                                                       gdim-1,
                                                       boundaries.find(i))
            dirichletFunc_p.interpolate(free_boundary_p)
            bc = dolfinx.fem.dirichletbc(dirichletFunc_p, dofs, W.sub(1))
        bcs.append(bc)

    A = dolfinx.fem.petsc.assemble_matrix(a_cpp, bcs=bcs)
    A.assemble()

    F = dolfinx.fem.petsc.assemble_vector(l_cpp)
    dolfinx.fem.petsc.apply_lifting(F, [a_cpp], bcs=[bcs])
    F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(F, bcs)

    solution = dolfinx.fem.Function(W)
    ksp = PETSc.KSP()
    ksp.create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setFromOptions()
    ksp.solve(F, solution.vector)
    solution.x.scatter_forward()
    (solution_u, solution_p) = (solution.sub(0).collapse(),
                                solution.sub(1).collapse())

    print(max(abs(solution_u.x.array)))
    print(max(abs(solution_p.x.array)))

    with dolfinx.io.XDMFFile(mesh.comm, "dolfinx_solution/velocity.xdmf",
                             "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(solution_u)

    with dolfinx.io.XDMFFile(mesh.comm, "dolfinx_solution/pressure.xdmf",
                             "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(solution_p)
