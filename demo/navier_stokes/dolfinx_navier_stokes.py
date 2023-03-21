import dolfinx
import ufl

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

# Read mesh
gdim = 2
comm = MPI.COMM_WORLD
gmsh_model_rank = 0
mesh, subdomains, boundaries = dolfinx.io.gmshio.read_from_msh(
    "mesh_data/mesh.msh", comm, gmsh_model_rank, gdim=gdim)

V = dolfinx.fem.VectorFunctionSpace(mesh, ("CG", 2))
Q = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))

(u, v) = ufl.TrialFunction(V), ufl.TestFunction(V)
(p, q) = ufl.TrialFunction(Q), ufl.TestFunction(Q)

a = [[ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
      ufl.inner(p, ufl.div(v)) * ufl.dx],
     [ufl.inner(ufl.div(u), q) * ufl.dx,
      None]]
f_u = dolfinx.fem.Function(V)
f_q = dolfinx.fem.Function(Q)
l_u = ufl.inner(f_u, v) * ufl.dx
l_q = ufl.inner(f_q, q) * ufl.dx
l = [l_u, l_q]

a_cpp = dolfinx.fem.form(a)
l_cpp = dolfinx.fem.form(l)

def no_slip(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))


def lid(x):
    return np.stack((100.*np.ones(x.shape[1]), np.zeros(x.shape[1])))


def domain_average(msh, v):
    vol = \
        msh.comm.allreduce(dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(dolfinx.fem.Constant(msh, 1.0) * ufl.dx)),
            op =  MPI.SUM)
    return 1 / vol * msh.comm.allreduce(dolfinx.fem.assemble_scalar
                                         (dolfinx.fem.form(v * ufl.dx)),
                                         op=MPI.SUM)

dofs_1 = dolfinx.fem.locate_dofs_topological(V, gdim-1, boundaries.find(1))
dofs_2 = dolfinx.fem.locate_dofs_topological(V, gdim-1, boundaries.find(2))
dofs_3 = dolfinx.fem.locate_dofs_topological(V, gdim-1, boundaries.find(3))
dofs_4 = dolfinx.fem.locate_dofs_topological(V, gdim-1, boundaries.find(4))

lid_velocity_func = dolfinx.fem.Function(V)
no_slip_func = dolfinx.fem.Function(V)

lid_velocity_func.interpolate(lid)
no_slip_func.interpolate(no_slip)

left_bc = dolfinx.fem.dirichletbc(no_slip_func, dofs_1)
top_bc = dolfinx.fem.dirichletbc(lid_velocity_func, dofs_2)
right_bc = dolfinx.fem.dirichletbc(no_slip_func, dofs_3)
bottom_bc = dolfinx.fem.dirichletbc(no_slip_func, dofs_4)

bcs = [left_bc, top_bc, right_bc, bottom_bc]

A = dolfinx.fem.petsc.assemble_matrix_block(a_cpp, bcs=bcs)
A.assemble()

F = dolfinx.fem.petsc.assemble_vector_block(l_cpp, a_cpp, bcs=bcs)

solution = A.createVecRight()
ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
opts = PETSc.Options()
opts["mat_mumps_icntl_14"] = 80  # Increase MUMPS working memory
opts["mat_mumps_icntl_24"] = 1  # Option to support solving a singular matrix (pressure nullspace)
opts["mat_mumps_icntl_25"] = 0  # Option to support solving a singular matrix (pressure nullspace)
opts["ksp_error_if_not_converged"] = 1
ksp.setFromOptions()
ksp.solve(F, solution)

# Split the solution
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

u_h = dolfinx.fem.Function(V)
u_h.name = "velocity"
u_h.x.array[:offset] = solution.array_r[:offset]
u_h.x.scatter_forward()

print(f"Stokes solution: {u_h.x.array}")

p_h = dolfinx.fem.Function(Q)
p_h.name = "pressure"
p_h.x.array[:(len(solution.array_r) - offset)] = solution.array_r[offset:]
p_h.x.scatter_forward()
p_h.x.array[:] -= domain_average(mesh, p_h)

with dolfinx.io.XDMFFile(mesh.comm,
                         "dolfinx_solution/stokes_solution_field.xdmf",
                         "w") as solution_file:
    solution_file.write_mesh(mesh)
    solution_file.write_function(u_h)
    solution_file.write_function(p_h)

u_previous = dolfinx.fem.Function(V)
u_previous.x.array[:] = u_h.x.array.copy()

max_iter = 30

for i in range(max_iter):
    print(f"Iteration {i+1}")
    print(f"Navier stokes previous solution: {u_previous.x.array}")
    a_ns = [[ufl.inner(ufl.dot(u_previous, ufl.grad(u)), v) * ufl.dx + \
            ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
            ufl.inner(p, ufl.div(v)) * ufl.dx],
            [ufl.inner(ufl.div(u), q) * ufl.dx,
            None]]
    a_ns_cpp = dolfinx.fem.form(a_ns)
    A = dolfinx.fem.petsc.assemble_matrix_block(a_ns_cpp, bcs=bcs)
    A.assemble()
    
    F = dolfinx.fem.petsc.assemble_vector_block(l_cpp, a_cpp, bcs=bcs)
    solution = A.createVecRight()
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    opts = PETSc.Options()
    opts["mat_mumps_icntl_14"] = 80  # Increase MUMPS working memory
    opts["mat_mumps_icntl_24"] = 1  # Option to support solving a singular matrix (pressure nullspace)
    opts["mat_mumps_icntl_25"] = 0  # Option to support solving a singular matrix (pressure nullspace)
    opts["ksp_error_if_not_converged"] = 1
    ksp.setFromOptions()
    ksp.solve(F, solution)

    # Split the solution
    offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

    u_h = dolfinx.fem.Function(V)
    u_h.name = "velocity"
    u_h.x.array[:offset] = solution.array_r[:offset]
    u_h.x.scatter_forward()

    p_h = dolfinx.fem.Function(Q)
    p_h.name = "pressure"
    p_h.x.array[:(len(solution.array_r) - offset)] = solution.array_r[offset:]
    p_h.x.scatter_forward()
    p_h.x.array[:] -= domain_average(mesh, p_h)
    
    print(np.sqrt(dolfinx.fem.assemble_scalar
                  (dolfinx.fem.form(ufl.inner(u_previous - u_h,
                                              u_previous - u_h)
                  * ufl.dx))))
    
    u_previous.x.array[:] = u_h.x.array.copy()

    
    print(f"Navier stokes solution: {u_h.x.array}")

with dolfinx.io.XDMFFile(mesh.comm,
                         "dolfinx_solution/navier_stokes_solution_field.xdmf",
                         "w") as solution_file:
    solution_file.write_mesh(mesh)
    solution_file.write_function(u_h)
    solution_file.write_function(p_h)
