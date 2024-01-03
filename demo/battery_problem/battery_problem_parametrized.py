import dolfinx
import basix
import ufl
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, \
    create_vector, apply_lifting, set_bc, NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import sympy

import matplotlib.pyplot as plt

MPI.COMM_WORLD.Barrier()
start_time = MPI.Wtime()

# Import mesh in dolfinx
gdim = 2
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh(
    "mesh_data/mesh.msh", mesh_comm, gmsh_model_rank, gdim=gdim)
n_vec = ufl.FacetNormal(mesh)

k = 2
Theta_el = basix.ufl.element("Lagrange", mesh.basix_cell(), k)
Mu_el = basix.ufl.element("Lagrange", mesh.basix_cell(), k)
U_el = basix.ufl.element("Lagrange", mesh.basix_cell(), k, shape=(mesh.geometry.dim,))
V_el = basix.ufl.mixed_element([Theta_el, Mu_el, U_el])
V = dolfinx.fem.FunctionSpace(mesh, V_el)

Theta, _ = V.sub(0).collapse()
Mu, _ = V.sub(1).collapse()
U, _ = V.sub(2).sub(1).collapse()

# x = ufl.SpatialCoordinate(mesh)

V_x = dolfinx.fem.VectorFunctionSpace(mesh, ("Lagrange", k))
x = dolfinx.fem.Function(V_x)
x.interpolate(lambda x: (x[0], x[1]))

theta_test, mu_test, u_test = ufl.TestFunctions(V)

sol_previous = dolfinx.fem.Function(V)
theta_previous, mu_previous, u_previous = ufl.split(sol_previous)
sol_current = dolfinx.fem.Function(V)
theta_current, mu_current, u_current = ufl.split(sol_current)

# sol_current.sub(2).interpolate(lambda x: (np.zeros(x[0].shape,), np.zeros(x[1].shape,)))
sol_current.sub(0).interpolate(lambda x: np.array(0.95 * np.ones(x[0].shape,)))
sol_current.x.scatter_forward()
sol_previous.sub(0).x.array[:] = sol_current.sub(0).x.array.copy()
# NOTE Why not deepcopy in Cahn Hiliard tutorial?

ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
mu_0, mu_1 = 5., 2.e-6
current_time = 0
end_time = 3600 / mu_0
num_steps = 5 # 20
dt = (end_time - current_time) / num_steps

# Constants
n_L = 49200 # Molar density of Lattice sites
f_far = 96485.3321 # Faraday constant
r_gas = 8.314 # Gas constant R
ref_T = 298. # Reference temperature
i_s = 4780 * mu_0 * mu_1 * 210 / 4 # Flux on surface # -ve implies lithiation and +ve implies delithiation

theta_current = ufl.variable(theta_current)
v_oc = - theta_current + 4.5
dv_oc_dtheta = ufl.diff(v_oc, theta_current)


def diffusivity_Li(theta_current):
    conditions = [ufl.le(theta_current, 0.23), ufl.And(ufl.ge(theta_current, 0.23), ufl.le(theta_current, 0.53)), ufl.And(ufl.ge(theta_current, 0.53), ufl.le(theta_current, 0.61)), ufl.And(ufl.ge(theta_current, 0.61), ufl.le(theta_current, 0.9)), ufl.ge(theta_current, 0.9)]
    interps = [PETSc.ScalarType(0.625e-15), 6.84665027273618e-14 * theta_current**3 - 9.34962575536366e-14 * theta_current**2 + 4.25736371145718e-14 * theta_current - 5.05401645044794e-15, -2.56806952920992e-13 * theta_current**3 + 4.23688536927247e-13 * theta_current**2 - 2.31534303960296e-13 * theta_current + 4.33717198061121e-14, 1.17694155221544e-13 * theta_current**3 - 2.61648490973595e-13 * theta_current**2 + 1.86521283059217e-13 * theta_current - 4.1632916221189e-14, PETSc.ScalarType(0.1e-15)]
    assert len(conditions) == len(interps)
    d_func = ufl.conditional(conditions[0], interps[0], interps[0])
    for i in range(1, len(conditions)):
        d_func = ufl.conditional(conditions[i], interps[i], d_func)
    return ufl.as_tensor([[d_func, 0], [0, 0]])

def epsilon(u, x):
    return ufl.as_vector([u[0].dx(0), u[1].dx(1), u[0]/x[0], (u[0].dx(1) + u[1].dx(0))])
# NOTE \epsilon_{rz} = 2 * 0.5 * (u[0].dx(1) + u[1].dx(0))

def epsilon_theta(theta_current):
    epsilon_a = 0.2572 * theta_current**5 - 0.7367 * theta_current**4 + 0.7185 * theta_current**3 - 0.2602 * theta_current**2 + 0.0446 * theta_current - 0.0025
    epsilon_b = epsilon_a
    epsilon_c = 0.2362 * theta_current**5 - 1.1269 * theta_current**4 + 2.0545 * theta_current**3 - 1.7512 * theta_current**2 + 0.6531 * theta_current - 0.0489
    return ufl.as_vector([epsilon_a, epsilon_c, epsilon_b, 0.])

computed_file = "battery_problem_parametrized/solution_computed.xdmf"

solution_file = dolfinx.io.XDMFFile(mesh.comm, computed_file, "w")
solution_file.write_mesh(mesh)
solution_file.write_function(sol_current.sub(0), current_time)
solution_file.write_function(sol_current.sub(1), current_time)
solution_file.write_function(sol_current.sub(2), current_time)

stiffness_tensor = ufl.as_tensor([[259.e9, 75.e9, 107.e9, 0.], [75.e9, 194.e9, 75.e9, 0.], [107.e9, 75.e9, 259.e9, 0.], [0., 0., 0., 59.e9]])

dofs_sym_x_1 = dolfinx.fem.locate_dofs_topological(V.sub(2).sub(0), gdim-1, facet_tags.find(5))
bc_sym_x_1 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_sym_x_1, V.sub(2).sub(0))

dofs_sym_x_2 = dolfinx.fem.locate_dofs_topological(V.sub(2).sub(0), gdim-1, facet_tags.find(6))
bc_sym_x_2 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_sym_x_2, V.sub(2).sub(0))

def center_marker(x):
    return np.logical_and(np.isclose(x[0], np.zeros_like(x[0]), atol=1.e-10), np.isclose(x[1], np.zeros_like(x[1]), atol=1.e-10))

center_disp_dofs = dolfinx.fem.locate_dofs_geometrical((V.sub(2).sub(1), U), center_marker)
print(center_disp_dofs)

bc_center = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), center_disp_dofs[0], V.sub(2).sub(1))

bc = [bc_center, bc_sym_x_1, bc_sym_x_2]

# TODO Clarify: Here theta_current is ufl.variable instead of dolfinx.fem.Function but mu_current, u_current are dolfinx.fem.Function
a0 = n_L * ufl.inner(theta_current - theta_previous, theta_test) * x[0] * dx - dt * (n_L / (r_gas * ref_T)) * theta_current * ufl.inner(diffusivity_Li(theta_current) * (f_far * dv_oc_dtheta * ufl.grad(theta_current) - ufl.grad(mu_current)), ufl.grad(theta_test)) * x[0] * dx + dt * (i_s / f_far) * theta_test * x[0] * (ds(2) + ds(3))
a1 = ufl.inner(mu_current, mu_test) * x[0] * dx + (1 / n_L) * ufl.inner(ufl.inner(stiffness_tensor * (epsilon(u_current, x) - epsilon_theta(theta_current)), ufl.diff(epsilon_theta(theta_current), theta_current)), mu_test) * x[0] * dx
a2 = ufl.inner(stiffness_tensor * epsilon(u_current, x), epsilon(u_test, x)) * x[0] * dx - ufl.inner(stiffness_tensor * epsilon_theta(theta_current), epsilon(u_test, x)) * x[0] * dx
a_tot = a0 + a1 + a2

problem = NonlinearProblem(a_tot, sol_current, bcs=bc)
solver = NewtonSolver(mesh.comm, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1.e-6
# solver.atol = 1.e-10
solver.max_it = 20
dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

if k > 1:
    V_sigma_el = basix.ufl.element("Lagrange", mesh.basix_cell(), k - 1, shape=(4,))
else:
    V_sigma_el = basix.ufl.element("DG", mesh.basix_cell(), 0, shape=(4,))

V_sigma = dolfinx.fem.FunctionSpace(mesh, V_sigma_el)
sigma_func = dolfinx.fem.Function(V_sigma)
sigma_expr = dolfinx.fem.Expression(stiffness_tensor * (epsilon(u_current, x) - epsilon_theta(theta_current)), V_sigma.element.interpolation_points())

theta_current = sol_current.sub(0)
mu_current = sol_current.sub(1)
disp_current = sol_current.sub(2)

stress_computed_file = "battery_problem_parametrized/stress_computed.xdmf"
stress_computed_file = dolfinx.io.XDMFFile(mesh.comm, stress_computed_file, "w")
stress_computed_file.write_mesh(mesh)
stress_computed_file.write_function(sigma_func.sub(0), current_time)
stress_computed_file.write_function(sigma_func.sub(1), current_time)
stress_computed_file.write_function(sigma_func.sub(2), current_time)
stress_computed_file.write_function(sigma_func.sub(3), current_time)

for i in range(num_steps):
    current_time += dt
    n, converged = solver.solve(sol_current)
    print(f"Time: {current_time}, Iteration: {n}, Converged: {converged}, Step: {i}")
    sol_current.x.scatter_forward() # NOTE Why no scatter_forward in the tutorial
    sol_previous.x.array[:] = sol_current.x.array.copy()
    # NOTE Why not deepcopy in Cahn Hiliard tutorial?
    solution_file.write_function(theta_current, current_time)
    solution_file.write_function(mu_current, current_time)
    solution_file.write_function(disp_current, current_time)

    # print(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(ufl.dot(sol_current.sub(2), n_vec), ufl.dot(sol_current.sub(2), n_vec))*(ds(5) + ds(6)))), op=MPI.SUM))

    sigma_func.interpolate(sigma_expr)
    print(f"Max stress: {np.max(sigma_func.x.array)}")
    stress_computed_file.write_function(sigma_func.sub(0), current_time)
    stress_computed_file.write_function(sigma_func.sub(1), current_time)
    stress_computed_file.write_function(sigma_func.sub(2), current_time)
    stress_computed_file.write_function(sigma_func.sub(3), current_time)

end_time = MPI.Wtime()
run_time = end_time - start_time

run_time = mesh.comm.allreduce(run_time, op=MPI.MAX)

if mesh.comm.rank == 0:
    print(f"Run time: {run_time}")
