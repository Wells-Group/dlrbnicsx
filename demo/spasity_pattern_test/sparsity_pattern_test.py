import ufl
from dolfinx.fem import FunctionSpace, VectorFunctionSpace, \
    locate_dofs_topological, create_sparsity_pattern, form
from dolfinx.graph import create_adjacencylist
from dolfinx.mesh import create_unit_square, exterior_facet_indices, CellType
from mpi4py import MPI
import numpy as np

mesh = create_unit_square(MPI.COMM_WORLD, 3, 3,
                          cell_type=CellType.quadrilateral)
gdim = mesh.geometry.dim
V = FunctionSpace(mesh, ("Lagrange", 1))
# NOTE or FunctionSpace instead of VectorFunctionSpace

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
print(boundary_dofs)

print(V.tabulate_dof_coordinates())

'''
# NOTE s
1. Create_adjacencylist has been replaced by adjacencylist in updated dolfinx
2. The graph here is NOT after bc application. If the dof is on boundary,
ignore connections and only consider self connection
'''

adjacency_as_list = list()
adjancency_matrix = np.zeros([len(pattern.graph[1])-1, len(pattern.graph[1])-1])
for i in range(len(pattern.graph[1])-1):
    adjancency_matrix[i, pattern.graph[0][pattern.graph[1][i]:pattern.graph[1][i+1]]] = 1

print(adjancency_matrix)

for i in range(len(pattern.graph[1])-1):
    print(i, np.where(adjancency_matrix[i, :] != 0)[0])

'''
# NOTE s
1. Should boundary dofs depend only on itself and discard connection to others?
i.e.
for i in range(len(pattern.graph[1])-1):
    if len(np.where(boundary_dofs == i)[0]) > 0:
        adjancency_matrix[i, i] = 1
    else:
        adjancency_matrix[i, pattern.graph[0][pattern.graph[1][i]:pattern.graph[1][i+1]]] = 1
'''
