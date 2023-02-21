from mdfenicsx.mesh_motion_classes import MeshDeformation, HarmonicMeshMotion
import dolfinx
import ufl
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

# Read mesh
gdim = 2 # Mesh geometric dimensions
gmsh_model_rank = 0 # gmsh model rank
mesh_comm = MPI.COMM_WORLD # MPI communicator for mesh
mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh", mesh_comm, gmsh_model_rank, gdim=gdim) # Read mesh from .msh file
gdim = mesh.geometry.dim

dx = ufl.Measure("dx")(subdomain_data=cell_tags)
x = ufl.SpatialCoordinate(mesh)
V = dolfinx.fem.FunctionSpace(mesh, ("CG",1))
uh, v = dolfinx.fem.Function(V), ufl.TestFunction(V)

class BC_bottom():
    def __init__(self,mu,gdim):
        self.mu = mu
        self.gdim = gdim
    def __call__(self,x):
        values = np.zeros((self.gdim, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 0.
        values[1] = self.mu[0]*np.sin(x[0]*np.pi)
        return values

class BC_top():
    def __init__(self,mu,gdim):
        self.mu = mu
        self.gdim = gdim
    def __call__(self,x):
        values = np.zeros((self.gdim, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 0.
        values[1] = -self.mu[1]*np.sin(x[0]*np.pi)
        return values

class BC_left():
    def __init__(self,mu,gdim):
        self.mu = mu
        self.gdim = gdim
    def __call__(self,x):
        values = np.zeros((self.gdim, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = self.mu[2]*x[0]
        values[1] = 0.
        return values

class CustomMeshDeformation(HarmonicMeshMotion):
    def __init__(self, mesh, boundaries, bc_markers_list, bc_function_list, mu, reset_reference=True, is_deformation=True):
        super().__init__(mesh, boundaries, bc_markers_list, bc_function_list, reset_reference, is_deformation)
        self.mu = mu
        
    def __enter__(self):
        gdim = self._mesh.geometry.dim
        bcs = self.assemble_bcs() # Assemble BCs
        mu = self.mu
        self.shape_parametrization = self.solve() # Compute shape parametrization
        self._mesh.geometry.x[:,0] += (mu[2] - 1.) * (self._mesh.geometry.x[:,0] - 0.5)
        self._mesh.geometry.x[:,:gdim] += self.shape_parametrization.x.array.reshape(self._reference_coordinates.shape[0],gdim)
        self._mesh.geometry.x[:,0] -= min(self._mesh.geometry.x[:,0])
        return self

mu = [0.,0.,5.]#[0.349, -0.413, 4.257]
u_bc_bottom = BC_bottom(mu, gdim)
u_bc_top = BC_top(mu, gdim)
u_bc_left = BC_left(mu, gdim)

class exact_solution():
    '''
    class with __call__ method. Define boundary condition (= exact solution in this case) in __call__ method.
    '''
    def __init__(self):
        print("Initialised the class")
    def __call__(self,x):
        return x[1] * np.sin(np.pi * x[0]) * np.cos(np.pi * x[1])

u_exact = exact_solution()
u = dolfinx.fem.Function(V)
u.interpolate(u_exact)

harmonic_deformation_mesh_file_xdmf = dolfinx.io.XDMFFile(mesh.comm, "harmonic_deformation_data/2D_deformed_mesh.xdmf", "w")
print(dolfinx.fem.assemble_scalar(dolfinx.fem.form(x[1] * ufl.sin(np.pi * x[1]) * dx)))
with CustomMeshDeformation(mesh, facet_tags, [1, 5, 9, 12], [u_bc_bottom, u_bc_bottom, u_bc_top, u_bc_top], mu, reset_reference=True, is_deformation=True) as mesh_class:
    harmonic_deformation_mesh_file_xdmf.write_mesh(mesh_class._mesh)
    print(dolfinx.fem.assemble_scalar(dolfinx.fem.form(x[1] * ufl.sin(np.pi * x[1]) * dx)))
    print(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(u,u) * ufl.dx)))
print(dolfinx.fem.assemble_scalar(dolfinx.fem.form(x[1] * ufl.sin(np.pi * x[1]) * dx)))
print(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(u,u)* dx)))
