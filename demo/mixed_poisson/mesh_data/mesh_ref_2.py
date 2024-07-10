import gmsh
import sys
import math
import numpy as np

from mpi4py import MPI
import dolfinx

gmsh.initialize()
gmsh.model.add("3d_mesh")

lc = 2.e-1

gmsh.model.geo.addPoint(0., 0., 0., lc, 1)
gmsh.model.geo.addPoint(1., 0., 0., lc, 2)
gmsh.model.geo.addPoint(1., 1., 0., lc, 3)
gmsh.model.geo.addPoint(0., 1., 0., lc, 4)

gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)

gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
gmsh.model.geo.addPlaneSurface([1], 1)
# gmsh.model.geo.addPhysicalGroup(2, [1])

gmsh.model.geo.synchronize()

ov = gmsh.model.geo.copy([(2, 1)])

# vol1 = gmsh.model.geo.extrude([ov[0]], 0, 0, 1, [5, 4], [0.5, 1])
vol1 = gmsh.model.geo.extrude([(2, 1)], 0, 0, 1, [5, 4], [0.5, 0.6])

gmsh.model.geo.synchronize()

# Create mesh
gmsh.option.setNumber("Mesh.Algorithm3D", 4)
# gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
# gmsh.option.setNumber("Mesh.RecombineAll", 1)
# gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
gmsh.option.setNumber("Mesh.MeshSizeMin", lc)
gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
# Mesh generation
gmsh.model.mesh.generate(3)
# Mesh order
gmsh.model.mesh.setOrder(1)
# Mesh optimisation or improving quality of mesh
gmsh.model.mesh.optimize("Netgen")

volume = gmsh.model.getEntities(dim=3)
surfaces = gmsh.model.getEntities(dim=2)

print(volume)
print(surfaces)

gmsh.model.addPhysicalGroup(3, [volume[0][1]], 1)

for i in range(1, len(surfaces)+1):
    gmsh.model.addPhysicalGroup(2, [surfaces[i-1][1]],
                                surfaces[i-1][1])

# NOTE Remove gmsh markers and only keep physical markers
# gmsh.model.occ.remove(gmsh.model.getEntities(dim=1))
gmsh.model.occ.remove(gmsh.model.getEntities(dim=2))
gmsh.model.occ.remove(gmsh.model.getEntities(dim=3))

gmsh.write("3d_mesh.msh")

gmsh.fltk.run()

# gmsh.finalize()

# Import mesh in dolfinx
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
mesh, subdomains, boundaries = \
    dolfinx.io.gmshio.read_from_msh("3d_mesh.msh", mesh_comm,
                                    gmsh_model_rank, gdim=3)

with dolfinx.io.XDMFFile(mesh.comm, "mesh.xdmf", "w") as mesh_file_xdmf:
    mesh_file_xdmf.write_mesh(mesh)
    mesh_file_xdmf.write_meshtags(subdomains, mesh.geometry)
    mesh_file_xdmf.write_meshtags(boundaries, mesh.geometry)

print(mesh.geometry.dim)
print(mesh.geometry.x.shape)
