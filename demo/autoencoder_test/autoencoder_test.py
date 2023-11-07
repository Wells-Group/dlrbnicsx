import dolfinx
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, \
    apply_lifting, set_bc
import ufl

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import matplotlib.pyplot as plt

from smt.sampling_methods import LHS

import rbnicsx
import rbnicsx.online
import rbnicsx.backends

import torch

mesh_comm = MPI.COMM_WORLD
gmsh_model_rank = 0
gdim = 2
mesh, subdomains, boundaries = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh",  mesh_comm,
                      gmsh_model_rank, gdim=gdim)
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=boundaries)
n_vec = ufl.FacetNormal(mesh)
para_limits = np.array([[0.2, 0.8], [0.2, 0.80]])
sampling = LHS(xlimits=para_limits)
training_set = sampling(500)


input_data = np.zeros([training_set.shape[0], mesh.geometry.x.shape[0]])

mu_0 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.6))
mu_1 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.63))

V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
u, v = ufl.TestFunction(V), ufl.TrialFunction(V)

dofs_1 = dolfinx.fem.locate_dofs_topological(V, gdim-1, boundaries.find(1))
bc_1 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_1, V)

dofs_2 = dolfinx.fem.locate_dofs_topological(V, gdim-1, boundaries.find(2))
bc_2 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_2, V)

dofs_3 = dolfinx.fem.locate_dofs_topological(V, gdim-1, boundaries.find(3))
bc_3 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_3, V)

dofs_4 = dolfinx.fem.locate_dofs_topological(V, gdim-1, boundaries.find(4))
bc_4 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.), dofs_4, V)

bcs = [bc_1, bc_2, bc_3, bc_4]

x = ufl.SpatialCoordinate(mesh)
source_term = - 10. * ((x[0] - mu_0)**2 + (x[1] - mu_1)**2)
flux_bc_1 = ufl.as_vector([ufl.sin(np.pi * x[0]), ufl.sin(np.pi * x[1])])
a_cpp = dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * dx)
l_cpp = dolfinx.fem.form(ufl.inner(source_term, v) * dx)

for para_num in range(training_set.shape[0]):
    mu_0.value = training_set[para_num, 0]
    mu_1.value = training_set[para_num, 1]
    # Bilinear side assembly
    A = assemble_matrix(a_cpp, bcs=bcs)
    A.assemble()

    # Linear side assembly
    L = assemble_vector(l_cpp)
    apply_lifting(L, [a_cpp], [bcs])
    L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(L, bcs)

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

    print(f"Update snapshots matrix: {para_num+1}/{training_set.shape[0]}, Parameter: {training_set[para_num, :]}")

    input_data[para_num, :] = solution.vector[:]

    with dolfinx.io.XDMFFile(mesh.comm,
                            "solution_poisson/solution_computed.xdmf",
                            "w") as solution_file_xdmf:
        solution_file_xdmf.write_mesh(mesh)
        solution_file_xdmf.write_function(solution)

    print(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(solution, solution) * dx)), op=MPI.SUM))

np.random.shuffle(input_data)
input_data = torch.from_numpy(input_data).to(torch.float32)
input_scaling_range = torch.from_numpy(np.array([-1., 1.]))
input_range = torch.from_numpy(np.array([torch.min(input_data), torch.max(input_data)]))
input_data_scaled = (input_scaling_range[1] - input_scaling_range[0]) * (input_data - input_range[0])/ (input_range[1] - input_range[0]) + input_scaling_range[0]
input_data_scaled_train = input_data_scaled[:int(0.7 * training_set.shape[0]), :]
input_data_scaled_valid = input_data_scaled[input_data_scaled_train.shape[0]:, :]

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_data_scaled):
        self.input_data_scaled = input_data_scaled
    
    def __len__(self):
        return self.input_data_scaled.shape[0]
    
    def __getitem__(self, idx):
        return self.input_data_scaled[idx, :]

'''
class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(381, 190),
            torch.nn.Tanh(),
            torch.nn.Linear(190, 85),
            torch.nn.Tanh(),
            torch.nn.Linear(85, 43)
            )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(43, 85),
            torch.nn.Tanh(),
            torch.nn.Linear(85, 190),
            torch.nn.Tanh(),
            torch.nn.Linear(190, 381)#,
            # torch.nn.Sigmoid() # NOTE VVIP: If input-outpt data range is [0, 1] use Sigmoid. If [-1, 1], use tanh
            )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
'''

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(381, 190),
            torch.nn.Tanh(),
            torch.nn.Linear(190, 85),
            torch.nn.Tanh(),
            torch.nn.Linear(85, 43),
            torch.nn.Tanh(),
            torch.nn.Linear(43, 20)
            )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(20, 43),
            torch.nn.Tanh(),
            torch.nn.Linear(43, 85),
            torch.nn.Tanh(),
            torch.nn.Linear(85, 190),
            torch.nn.Tanh(),
            torch.nn.Linear(190, 381)
            )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = AE()
customDataset_train = CustomDataset(input_data_scaled_train)
loader_train = torch.utils.data.DataLoader(customDataset_train, batch_size=14, shuffle=True)
customDataset_valid = CustomDataset(input_data_scaled_valid)
loader_valid = torch.utils.data.DataLoader(customDataset_valid, batch_size=input_data_scaled_valid.shape[0], shuffle=False)

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)

epochs = 1000000
losses_train = []
losses_valid = []

for epoch in range(epochs):
    loss_valid = torch.Tensor([0.])
    for batch, input_data_batch_train in enumerate(loader_train):
        
        reconstructed = model(input_data_batch_train)
        loss_train = loss_function(reconstructed, input_data_batch_train) / loss_function(reconstructed, torch.zeros_like(reconstructed))
        
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
        print(f"Epoch: {epoch}, Training loss: {loss_train.item()}")
        
        losses_train.append(loss_train.item())
    
    for batch, input_data_batch_valid in enumerate(loader_valid):

        reconstructed = model(input_data_batch_valid)
        loss_valid += loss_function(reconstructed, input_data_batch_valid)/loss_function(reconstructed, torch.zeros_like(reconstructed))

    losses_valid.append(loss_valid.item())
    print(f"Epoch: {epoch}, Validation loss: {loss_valid.item()}")
    if epoch == 0:
        min_val_loss = loss_valid.item()
    else:
        if min_val_loss >= loss_valid.item():
            min_val_loss = loss_valid.item()
        else:
            print("Early stopping criteria invoked")
            break

plt.style.use("fivethirtyeight")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.plot(losses_train, label="Training loss")
plt.plot(losses_valid, label="Validation loss")
plt.legend(loc="best")
plt.show()

error_analysis_set = sampling(90)
error_analysis_array = np.zeros(error_analysis_set.shape[0])

for para_num in range(error_analysis_set.shape[0]):
    
    mu_0.value = error_analysis_set[para_num, 0]
    mu_1.value = error_analysis_set[para_num, 1]
    # Bilinear side assembly
    A = assemble_matrix(a_cpp, bcs=bcs)
    A.assemble()

    # Linear side assembly
    L = assemble_vector(l_cpp)
    apply_lifting(L, [a_cpp], [bcs])
    L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(L, bcs)

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
    
    solution_torch = torch.from_numpy(solution.x.array).to(torch.float32)
    solution_torch_scaled = (input_scaling_range[1] - input_scaling_range[0]) * (solution_torch - input_range[0])/ (input_range[1] - input_range[0]) + input_scaling_range[0]
    reconstructed = model(solution_torch_scaled)
    reconstructed = (reconstructed - input_scaling_range[0]) * (input_range[1] - input_range[0]) / (input_scaling_range[1] - input_scaling_range[0]) + input_range[0]
    reconstructed = reconstructed.to(torch.float64).detach().numpy()
    reconstructed_func = dolfinx.fem.Function(V)
    reconstructed_func.x.array[:] = reconstructed
    
    error_analysis_array[para_num] = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(solution - reconstructed_func, solution - reconstructed_func) * dx)), op=MPI.SUM) / mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(solution, solution) * dx)), op=MPI.SUM)

print(error_analysis_array)


online_mu = np.array([0.37, 0.46])
mu_0.value = online_mu[0]
mu_1.value = online_mu[1]
# Bilinear side assembly
A = assemble_matrix(a_cpp, bcs=bcs)
A.assemble()

# Linear side assembly
L = assemble_vector(l_cpp)
apply_lifting(L, [a_cpp], [bcs])
L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(L, bcs)

# Solver setup
ksp = PETSc.KSP()
ksp.create(mesh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
ksp.setFromOptions()
solution_fem = dolfinx.fem.Function(V)
ksp.solve(L, solution_fem.vector)
solution_fem.x.scatter_forward()

solution_torch = torch.from_numpy(solution_fem.x.array).to(torch.float32)
solution_torch_scaled = (input_scaling_range[1] - input_scaling_range[0]) * (solution_torch - input_range[0])/ (input_range[1] - input_range[0]) + input_scaling_range[0]
reconstructed = model(solution_torch_scaled)
reconstructed = (reconstructed - input_scaling_range[0]) * (input_range[1] - input_range[0]) / (input_scaling_range[1] - input_scaling_range[0]) + input_range[0]
reconstructed = reconstructed.to(torch.float64).detach().numpy()
solution_reconstructed = dolfinx.fem.Function(V)
solution_reconstructed.x.array[:] = reconstructed
reconstruction_error = dolfinx.fem.Function(V)
reconstruction_error.x.array[:] = abs(solution_fem.x.array - solution_reconstructed.x.array)

with dolfinx.io.XDMFFile(mesh.comm,
                        "solution_poisson/solution_online_fem.xdmf",
                        "w") as solution_file_xdmf:
    solution_file_xdmf.write_mesh(mesh)
    solution_file_xdmf.write_function(solution_fem)

with dolfinx.io.XDMFFile(mesh.comm,
                        "solution_poisson/solution_online_reconstructed.xdmf",
                        "w") as solution_file_xdmf:
    solution_file_xdmf.write_mesh(mesh)
    solution_file_xdmf.write_function(solution_reconstructed)

with dolfinx.io.XDMFFile(mesh.comm,
                        "solution_poisson/solution_online_error.xdmf",
                        "w") as solution_file_xdmf:
    solution_file_xdmf.write_mesh(mesh)
    solution_file_xdmf.write_function(reconstruction_error)
