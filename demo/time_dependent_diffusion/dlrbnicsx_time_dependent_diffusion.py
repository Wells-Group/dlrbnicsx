from petsc4py import PETSc
from mpi4py import MPI
import numpy as np

import ufl
import dolfinx
from dolfinx.fem.petsc import \
    assemble_matrix, assemble_vector, create_vector, set_bc, apply_lifting

# Read mesh
mesh_comm = MPI.COMM_WORLD
gdim = 3
gmsh_model_rank = 0
mesh, subdomains, boundaries = \
    dolfinx.io.gmshio.read_from_msh(
        "mesh_data/mesh.msh", mesh_comm, gmsh_model_rank, gdim=gdim)

# Diffsion model
VD = dolfinx.fem.FunctionSpace(mesh, ("CG", 2))
uD, vD = ufl.TrialFunction(VD), ufl.TestFunction(VD)
uD_sol = dolfinx.fem.Function(VD)
uD_prev = dolfinx.fem.Function(VD)
# uD_prev.x.array[:] = 0.2 # Initial value
uD_prev.interpolate(lambda x: 5. * np.sin(x[0] * 2 * np.pi) * np.cos(x[1] * 2 * np.pi))
D = ufl.as_tensor([[1., 0., 0.],[0., 1.2, 0.],[0., 0., 3.]])
f_source = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.))
flux = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.))

t = 0
T = 10.
num_steps = 21
dt = T / (num_steps - 1)

aD = ufl.inner(uD, vD) * ufl.dx + dt * ufl.inner(ufl.grad(uD), ufl.grad(vD)) * ufl.dx
lD = ufl.inner(vD, uD_prev) * ufl.dx + dt * flux * vD * ufl.ds(2) + dt * ufl.inner(vD, f_source) * ufl.dx
aD_form = dolfinx.fem.form(aD)
lD_form = dolfinx.fem.form(lD)
bc = []

A = assemble_matrix(aD_form, bcs=bc)
A.assemble()
b = create_vector(lD_form)

solver = PETSc.KSP().create(mesh.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

xdmf = dolfinx.io.XDMFFile(mesh.comm, "diffusion.xdmf", "w")
xdmf.write_mesh(mesh)
xdmf.write_function(uD_prev, 0)

snapshot_matrix = np.zeros([uD_sol.x.array.shape[0], num_steps])
snapshot_matrix[:, 0] = (uD_prev.vector).copy()

for i in range(1, num_steps):
    t += dt
    print(f"Time: {t}s")
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, lD_form)
    apply_lifting(b, [aD_form], [bc])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bc)
    solver.solve(b, uD_sol.vector)
    uD_sol.x.scatter_forward()

    uD_prev.x.array[:] = uD_sol.x.array

    xdmf.write_function(uD_sol, t)
    snapshot_matrix[:, i] = (uD_sol.vector).copy() #uD_sol.x.array.copy()

xdmf.close()

adjacency_matrix = np.zeros([num_steps, num_steps])
for i in range(num_steps):
    adjacency_matrix[i, i] = 1
for i in range(1, num_steps):
    adjacency_matrix[i, i-1] = 1
    # TODO Think [i, i-1] or [i-1, i]???
