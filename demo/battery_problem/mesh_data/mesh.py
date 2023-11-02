import gmsh
import dolfinx
from mpi4py import MPI

comm = MPI.COMM_WORLD

# Avoid reading from logfile
gmsh.initialize(" ", False)

# Create mesh
gdim = 2

min_r, max_r = 0, 1.e-6
min_z, max_z = -1.e-6, 1.e-6

point_a = gmsh.model.occ.addPoint(min_r, min_z, 0., 5.e-8)
point_b = gmsh.model.occ.addPoint(max_r, min_z, 0., 5.e-8)
point_c = gmsh.model.occ.addPoint(max_r, 0, 0., 9.e-9)
point_d = gmsh.model.occ.addPoint(max_r, max_z, 0., 5.e-8)
point_e = gmsh.model.occ.addPoint(min_r, max_z, 0., 5.e-8)
point_f = gmsh.model.occ.addPoint(min_r, 0, 0., 9.e-9)

line_ab = gmsh.model.occ.addLine(point_a, point_b)
line_bc = gmsh.model.occ.addLine(point_b, point_c)
line_cd = gmsh.model.occ.addLine(point_c, point_d)
line_de = gmsh.model.occ.addLine(point_d, point_e)
line_ef = gmsh.model.occ.addLine(point_e, point_f)
line_fa = gmsh.model.occ.addLine(point_f, point_a)

loop_abcdefa = gmsh.model.occ.addCurveLoop([line_ab, line_bc, line_cd, line_de, line_ef, line_fa])
plane_abcdefa = gmsh.model.occ.addPlaneSurface([loop_abcdefa], 1)

gmsh.model.occ.synchronize()

# Create mesh
# 8 = Frontal-Delaunay for Quads
gmsh.option.setNumber("Mesh.Algorithm", 8)
# 2 = simple full-quad
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)
# Apply recombination algorithm
gmsh.option.setNumber("Mesh.RecombineAll", 1)
# Mesh subdivision algorithm (1: all quadrangles)
gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
# Minimum characteristic element size
# gmsh.option.setNumber("Mesh.MeshSizeMin", 1.e-9)
# Maximum characteristic element size
# gmsh.option.setNumber("Mesh.MeshSizeMax", 1.e-7)
# Mesh generation
gmsh.model.mesh.generate(gdim)
# Mesh order
gmsh.model.mesh.setOrder(1)
# Mesh optimisation or improving quality of mesh
gmsh.model.mesh.optimize("Netgen")

# Extract edges and surfaces to add physical groups
surfaces = gmsh.model.getEntities(dim=gdim)
edges = gmsh.model.getEntities(dim=gdim-1)
# Gives 'list' of boundaries in the form [(gdim-1),marker]
# with length = number of boundaries

gmsh.model.addPhysicalGroup(gdim, [surfaces[0][1]], 1)
for i in range(1, len(edges)+1):
    gmsh.model.addPhysicalGroup(gdim-1, [edges[i-1][1]],
                                edges[i-1][1])

# NOTE Remove gmsh markers and only keep physical markers
gmsh.model.occ.remove(gmsh.model.getEntities(dim=gdim-1))
gmsh.model.occ.remove(gmsh.model.getEntities(dim=gdim))

# Save and visualise the mesh
gmsh.write("mesh.msh")
gmsh.fltk.run()

# Import mesh in dolfinx
gmsh_model_rank = 0
mesh_comm = comm
mesh, subdomains, boundaries = \
    dolfinx.io.gmshio.read_from_msh("mesh.msh", mesh_comm,
                                    gmsh_model_rank, gdim=gdim)

import numpy as np
print(min(abs(mesh.geometry.x[:, 0])), min(abs(mesh.geometry.x[:, 1])))
print(np.where((mesh.geometry.x[:, 0])**2 + (mesh.geometry.x[:, 1])**2 < 1e-17))
print(mesh.geometry.x[np.where((mesh.geometry.x[:, 0])**2 + (mesh.geometry.x[:, 1])**2 < 1e-17), :])

with dolfinx.io.XDMFFile(mesh.comm, "mesh.xdmf", "w") as mesh_file_xdmf:
    mesh_file_xdmf.write_mesh(mesh)
    mesh_file_xdmf.write_meshtags(subdomains, mesh.geometry)
    mesh_file_xdmf.write_meshtags(boundaries, mesh.geometry)
