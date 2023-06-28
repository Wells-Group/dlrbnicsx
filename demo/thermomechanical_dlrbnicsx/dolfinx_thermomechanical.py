from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import sympy
import matplotlib.pyplot as plt

import ufl
import dolfinx

from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion

# Read mesh
mesh_comm = MPI.COMM_WORLD
gdim = 2
gmsh_model_rank = 0
mesh, subdomains, boundaries = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh", mesh_comm,
                                    gmsh_model_rank, gdim=gdim)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=boundaries)
ds_sf = ds(11) + ds(20) + ds(21) + ds(22) + ds(23)
ds_bottom = ds(1) + ds(31)
ds_out = ds(30)
x = ufl.SpatialCoordinate(mesh)

V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

Q = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))
thermal_conductivity_func = dolfinx.fem.Function(Q)
thermal_conductivity_func_diff = dolfinx.fem.Function(Q)
thermal_conductivity_func_1 = dolfinx.fem.Function(Q)
thermal_conductivity_func_diff_1 = dolfinx.fem.Function(Q)
thermal_conductivity_func_2 = dolfinx.fem.Function(Q)
thermal_conductivity_func_diff_2 = dolfinx.fem.Function(Q)
thermal_conductivity_func_5 = dolfinx.fem.Function(Q)
thermal_conductivity_func_diff_5 = dolfinx.fem.Function(Q)
thermal_conductivity_func_7 = dolfinx.fem.Function(Q)
thermal_conductivity_func_diff_7 = dolfinx.fem.Function(Q)
temperature_field = dolfinx.fem.Function(V)
temperature_field.x.array[:] = 300. # Initial guess for Newton solver 300 K
# mesh_comm.Barrier()

sym_T = sympy.Symbol("sym_T")

h_cf, h_bottom, h_out = 2000., 200., 200.
T_f, T_bottom, T_out = 1773., 300., 300.

#################### Conductivity for subdomain 1 ==================

thermal_conductivity_sym_1 = sympy.interpolating_spline(2, sym_T, [293., 473., 873., 1273.], [15., 15.2, 16.2, 15.1])
thermal_conductivity_sym_1 = sympy.Piecewise(thermal_conductivity_sym_1.args[0], (thermal_conductivity_sym_1.args[1][0], True))
thermal_conductivity_sym_1_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_1)
thermal_conductivity_sym_diff_1 = sympy.diff(thermal_conductivity_sym_1, sym_T)
thermal_conductivity_sym_diff_1_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_diff_1)

def conductivity_eval_1(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return thermal_conductivity_sym_1_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

def conductivity_eval_diff_1(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return thermal_conductivity_sym_diff_1_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

omega_1_cells = subdomains.find(1)

#################### Conductivity for subdomain 2 ==================

thermal_conductivity_sym_2 = sympy.interpolating_spline(2, sym_T, [293., 473., 873., 1273.], [35.8,37.3,42.7,47.2])
thermal_conductivity_sym_2 = sympy.Piecewise(thermal_conductivity_sym_2.args[0], (thermal_conductivity_sym_2.args[1][0], True))
thermal_conductivity_sym_2_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_2)
thermal_conductivity_sym_diff_2 = sympy.diff(thermal_conductivity_sym_2, sym_T)
thermal_conductivity_sym_diff_2_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_diff_2)

def conductivity_eval_2(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return thermal_conductivity_sym_2_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

def conductivity_eval_diff_2(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return thermal_conductivity_sym_diff_2_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

omega_2_cells = subdomains.find(2)

#################### Conductivity for subdomain 5  ==================

thermal_conductivity_sym_5 = sympy.interpolating_spline(2, sym_T, [293., 473., 873., 1273.], [19.2,18.6,20.7,21.3])
thermal_conductivity_sym_5 = sympy.Piecewise(thermal_conductivity_sym_5.args[0], (thermal_conductivity_sym_5.args[1][0], True))
thermal_conductivity_sym_5_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_5)
thermal_conductivity_sym_diff_5 = sympy.diff(thermal_conductivity_sym_5, sym_T)
thermal_conductivity_sym_diff_5_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_diff_5)

def conductivity_eval_5(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return thermal_conductivity_sym_5_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

def conductivity_eval_diff_5(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return thermal_conductivity_sym_diff_5_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

omega_5_cells = subdomains.find(5)

#################### Conductivity for subdomain 7  ==================

thermal_conductivity_sym_7 = sympy.interpolating_spline(2, sym_T, [293., 473., 873., 1273.], [35.8,37.3,42.7,47.2])
thermal_conductivity_sym_7 = sympy.Piecewise(thermal_conductivity_sym_7.args[0], (thermal_conductivity_sym_7.args[1][0], True))
thermal_conductivity_sym_7_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_7)
thermal_conductivity_sym_diff_7 = sympy.diff(thermal_conductivity_sym_7, sym_T)
thermal_conductivity_sym_diff_7_lambdified = sympy.lambdify(sym_T, thermal_conductivity_sym_diff_7)

def conductivity_eval_7(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return thermal_conductivity_sym_7_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

def conductivity_eval_diff_7(x, temperature_field=temperature_field):
    tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions(tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    return thermal_conductivity_sym_diff_7_lambdified(temperature_field.eval(x.T, colliding_cells.array)[:, 0])

omega_7_cells = subdomains.find(7)

#################### Conductivity for subdomains 3, 4, 6  ==================
omega_3_cells = subdomains.find(3)
omega_4_cells = subdomains.find(4)
omega_6_cells = subdomains.find(6)
thermal_conductivity_func.x.array[omega_3_cells] = 5.5
thermal_conductivity_func.x.array[omega_4_cells] = 5.
thermal_conductivity_func.x.array[omega_6_cells] = 48.


bcs = [] # No Dirichlet BCs


a_T = ufl.inner(thermal_conductivity_func * ufl.grad(u), ufl.grad(v)) * x[0] * ufl.dx + \
    ufl.inner(u * thermal_conductivity_func_diff * ufl.grad(temperature_field), ufl.grad(v)) * x[0] * ufl.dx + \
    ufl.inner(h_cf * u, v) * x[0] * ds_sf + ufl.inner(h_bottom * u, v) * x[0] * ds_bottom + \
    ufl.inner(h_out * u, v) * x[0] * ds_out
l_T = h_cf * T_f * v * x[0] * ds_sf + h_bottom * T_bottom * v * x[0] * ds_bottom + h_out * T_out * v * x[0] * ds_out
a_T_cpp = dolfinx.fem.form(a_T)
l_T_cpp = dolfinx.fem.form(l_T)


residual_list = list()
update_tol_list = list()


max_iterations = 4
for iteration in range(max_iterations):

    print("\n =======================================================")
    print(f"\n Iteration {iteration + 1} / {max_iterations} \n")
    
    thermal_conductivity_func_1.interpolate(conductivity_eval_1)
    thermal_conductivity_func.x.array[omega_1_cells] = thermal_conductivity_func_1.x.array[omega_1_cells]

    thermal_conductivity_func_diff_1.interpolate(conductivity_eval_diff_1)
    thermal_conductivity_func_diff.x.array[omega_1_cells] = thermal_conductivity_func_diff_1.x.array[omega_1_cells]

    thermal_conductivity_func_2.interpolate(conductivity_eval_2)
    thermal_conductivity_func.x.array[omega_2_cells] = thermal_conductivity_func_2.x.array[omega_2_cells]

    thermal_conductivity_func_diff_2.interpolate(conductivity_eval_diff_2)
    thermal_conductivity_func_diff.x.array[omega_2_cells] = thermal_conductivity_func_diff_2.x.array[omega_2_cells]

    thermal_conductivity_func_5.interpolate(conductivity_eval_5)
    thermal_conductivity_func.x.array[omega_5_cells] = thermal_conductivity_func_5.x.array[omega_5_cells]

    thermal_conductivity_func_diff_5.interpolate(conductivity_eval_diff_5)
    thermal_conductivity_func_diff.x.array[omega_5_cells] = thermal_conductivity_func_diff_5.x.array[omega_5_cells]

    thermal_conductivity_func_7.interpolate(conductivity_eval_7)
    thermal_conductivity_func.x.array[omega_7_cells] = thermal_conductivity_func_7.x.array[omega_7_cells]
    
    thermal_conductivity_func_diff_7.interpolate(conductivity_eval_diff_7)
    thermal_conductivity_func_diff.x.array[omega_7_cells] = thermal_conductivity_func_diff_7.x.array[omega_7_cells]
    
    mesh_comm.Barrier()
    
    if iteration == 0:
        residual = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(thermal_conductivity_func * ufl.grad(temperature_field), ufl.grad(v)) * x[0] * ufl.dx + 
                                                                ufl.inner(h_cf * (temperature_field - T_f), v) * x[0] * ds_sf + ufl.inner( h_bottom * (temperature_field - T_bottom), v) * x[0] * ds_bottom + ufl.inner(h_out * (temperature_field - T_out), v) * x[0] * ds_out))
        residual = mesh_comm.allreduce(residual, op=MPI.SUM)
        initial_residual = residual
    else:
        if residual / initial_residual < 1.e-4:
            print(f"Residual rtol reached")
            break
    
    print(f"Relative residual: {residual/initial_residual}")
    
    # Bilinear side assembly
    A = dolfinx.fem.petsc.assemble_matrix(a_T_cpp, bcs=bcs)
    A.assemble()

    # Linear side assembly
    L = dolfinx.fem.petsc.assemble_vector(l_T_cpp)
    dolfinx.fem.petsc.apply_lifting(L, [a_T_cpp], [bcs])
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
    solution.x.scatter_forward()

    print(solution.x.array)
    
    update_function = dolfinx.fem.Function(V)
    update_function.x.array[:] = solution.x.array[:] - temperature_field.x.array[:]
    solution_update = \
        mesh_comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(update_function, update_function) * x[0] * ufl.dx + ufl.inner(ufl.grad(update_function), ufl.grad(update_function)) * x[0] * ufl.dx)), op=MPI.SUM) / \
        mesh_comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(temperature_field, temperature_field) * x[0] * ufl.dx + ufl.inner(ufl.grad(temperature_field), ufl.grad(temperature_field)) * x[0] * ufl.dx)), op=MPI.SUM)
    
    print(f"Relative update (in norm): {solution_update}")
    if solution_update < 1.e-12:
            print(f"Relative update tolerance reached")
            break
    
    temperature_field.x.array[:] = solution.x.array.copy()

    computed_file = "solution_nonlinear_thermomechanical/solution_computed.xdmf"
    with dolfinx.io.XDMFFile(mesh.comm, computed_file, "w") as solution_file:
        solution_file.write_mesh(mesh)
        solution_file.write_function(temperature_field, iteration)
    
    min_temp = mesh_comm.allreduce(np.min(temperature_field.x.array[:]), op=MPI.MIN)
    max_temp = mesh_comm.allreduce(np.max(temperature_field.x.array[:]), op=MPI.MAX)
    print(min_temp, max_temp)

    residual = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(thermal_conductivity_func * ufl.grad(temperature_field), ufl.grad(v)) * x[0] * ufl.dx + 
                                                            ufl.inner(h_cf * (temperature_field - T_f), v) * x[0] * ds_sf + ufl.inner( h_bottom * (temperature_field - T_bottom), v) * x[0] * ds_bottom + ufl.inner(h_out * (temperature_field - T_out), v) * x[0] * ds_out))
    residual = mesh_comm.allreduce(residual, op=MPI.SUM)
