from petsc4py import PETSc
from mpi4py import MPI
import numpy as np

import ufl
import dolfinx
from dolfinx.fem.petsc import \
    assemble_matrix, assemble_vector, create_vector, set_bc, apply_lifting

import rbnicsx
import rbnicsx.online
import rbnicsx.backends

import abc

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
num_steps = 20
dt = T / num_steps

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

print("Set up snapshots matrix")
snapshots_matrix = rbnicsx.backends.FunctionsList(VD)
snapshot.x.array[:] = uD_prev.x.array.copy()
snapshots_matrix.append(snapshot)

for i in range(num_steps):
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
    snapshot = dolfinx.fem.Function(VD)
    snapshot.x.array[:] = uD_sol.x.array.copy()
    
    snapshots_matrix.append(snapshot) # TODO see if uD_sol is overwritten every time

    uD_prev.x.array[:] = uD_sol.x.array

    xdmf.write_function(uD_sol, t)
    print(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(uD_sol.dx(2), uD_sol.dx(2)) * ufl.dx)), op=MPI.SUM))

xdmf.close()

class PODANNReducedProblem(abc.ABC):
    def __init__(self, fem_space):
        self._basis_functions = rbnicsx.backends.FunctionsList(fem_space)
        u, v = ufl.TrialFunction(fem_space), ufl.TestFunction(fem_space)
        self._inner_product = ufl.inner(u, v) * ufl.dx +\
            ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        self._inner_product_action = \
            rbnicsx.backends.bilinear_form_action(self._inner_product,
                                                  part="real")

# Maximum RB size
Nmax = 30

print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
print("")

print("Set up reduced problem")
reduced_problem = PODANNReducedProblem(VD)

print("")

#for (mu_index, mu) in enumerate(training_set):
    #print(rbnicsx.io.TextLine(str(mu_index+1), fill="#"))

    #print("Parameter number ", (mu_index+1), "of", training_set.shape[0])
    #print("High fidelity solve for mu =", mu)
    #snapshot = problem_parametric.solve(mu)
    #print(f"Solution array: {snapshot.x.array}")

    #print("Update snapshots matrix")
    #snapshots_matrix.append(snapshot)

    #print("")

print(len(snapshots_matrix))
print(rbnicsx.io.TextLine("Perform POD", fill="#"))
eigenvalues, modes, _ = \
    rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_matrix,
                                    reduced_problem._inner_product_action,
                                    N=Nmax, tol=1e-10)
reduced_problem._basis_functions.extend(modes)
reduced_size = len(reduced_problem._basis_functions)
print("")

print(rbnicsx.io.TextBox("POD-Galerkin offline phase ends", fill="="))

positive_eigenvalues = np.where(eigenvalues > 0., eigenvalues, np.nan)
singular_values = np.sqrt(positive_eigenvalues)

print(f"Eigenvalues: {positive_eigenvalues}")

plt.figure(figsize=[8, 10])
xint = list()
yval = list()

for x, y in enumerate(eigenvalues[:len(reduced_problem._basis_functions)]):
    yval.append(y)
    xint.append(x+1)

plt.plot(xint, yval, "*-", color="orange")
plt.xlabel("Eigenvalue number", fontsize=18)
plt.ylabel("Eigenvalue", fontsize=18)
plt.xticks(xint)
plt.yscale("log")
plt.title("Eigenvalue decay", fontsize=24)
plt.tight_layout()
plt.savefig("eigenvalue_decay")
