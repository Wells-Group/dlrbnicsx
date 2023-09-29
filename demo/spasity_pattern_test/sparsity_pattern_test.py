import ufl
from dolfinx.fem import VectorFunctionSpace, \
    locate_dofs_topological, create_sparsity_pattern, form
from dolfinx.graph import create_adjacencylist
from dolfinx.mesh import create_unit_square, exterior_facet_indices, CellType
from mpi4py import MPI
import numpy as np

mesh = create_unit_square(MPI.COMM_WORLD, 3, 3,
                          cell_type=CellType.quadrilateral)
gdim = mesh.geometry.dim
V = VectorFunctionSpace(mesh, ("Lagrange", 2))

mesh.topology.create_connectivity(mesh.topology.dim - 1,
                                  mesh.topology.dim)
facets = exterior_facet_indices(mesh.topology)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = form(ufl.inner(u, v) * ufl.dx)

pattern = create_sparsity_pattern(a)
pattern.finalize()
print(pattern.graph)

ad_list = create_adjacencylist(pattern.graph[0],
                               pattern.graph[1].astype(np.int32))
print(ad_list)

boundary_dofs = locate_dofs_topological(V, mesh.geometry.dim-1, facets)
# print(boundary_dofs)

'''
# NOTE s
1. create_adjacencylist has been replaced by adjacencylist in updated dolfinx
2. The graph here is NOT after bc application. If the node is in boundary,
ignore connections and only consider self connection
3. TODO check where dof_indices or node_indices is important
'''
