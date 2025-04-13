import dolfinx
import gmsh
from mpi4py import MPI
import numpy as np
import sys

gmsh.initialize('', False)
gdim = 3

# Omega_1 Standard Carbon
lc1 = 0.3
'''
gmsh.model.occ.addPoint(0., 0., 0., lc1, 1)
gmsh.model.occ.addPoint(5.9501, 0., 0., lc1, 2)
gmsh.model.occ.addPoint(5.9501, 1., 0., lc1, 3)
gmsh.model.occ.addPoint(2.1, 1., 0., lc1, 4)
gmsh.model.occ.addPoint(0., 1., 0., lc1, 5)

gmsh.model.occ.addLine(1, 2, 1)
gmsh.model.occ.addLine(2, 3, 2)
gmsh.model.occ.addLine(3, 4, 3)
gmsh.model.occ.addLine(4, 5, 4)
gmsh.model.occ.addLine(5, 1, 5)
'''

gmsh.model.occ.addPoint(0., 0., 0., lc1, 1)
gmsh.model.occ.addPoint(5.9501, 0., 0., lc1, 2)
gmsh.model.occ.addPoint(5.9501, 1., 0., lc1, 3)
gmsh.model.occ.addPoint(0., 1., 0., lc1, 4)

gmsh.model.occ.addLine(1, 2, 1)
gmsh.model.occ.addLine(2, 3, 2)
gmsh.model.occ.addLine(3, 4, 3)
gmsh.model.occ.addLine(4, 1, 4)

gmsh.model.occ.addCurveLoop([1, 2, 3, 4], 1)
gmsh.model.occ.addPlaneSurface([1], 1)

gmsh.model.occ.revolve([(2, 1)], 0, 0, 0, 0, 1, 0,
                       np.pi)

gmsh.model.occ.revolve([(2, 5)], 0, 0, 0, 0, 1, 0,
                       np.pi)

gmsh.model.occ.synchronize()

gmsh.model.occ.removeAllDuplicates() # this is the coherence in the python API

gmsh.model.occ.synchronize()

gmsh.option.setNumber("Mesh.MeshSizeMin", 0.1)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.2)
gmsh.option.setNumber("Mesh.Algorithm", 5)
gmsh.model.mesh.generate(gdim)
gmsh.model.mesh.setOrder(1)
gmsh.model.mesh.optimize("Netgen")

gmsh.model.occ.remove(gmsh.model.getEntities(dim=gdim-2))
gmsh.model.occ.remove(gmsh.model.getEntities(dim=gdim-1))
gmsh.model.occ.remove(gmsh.model.getEntities(dim=gdim))

# Extract surfaces and volumes to add physical groups
volumes = gmsh.model.getEntities(dim=gdim)
surfaces = gmsh.model.getEntities(dim=gdim-1)  # gmsh.model.getBoundary(volumes)

print(volumes)
print(surfaces)

# Subdomains markings
for i in range(1, len(volumes)+1):
    gmsh.model.addPhysicalGroup(gdim, [volumes[i-1][1]],
                                volumes[i-1][1])

# External boundaries markings
for i in range(1, len(surfaces)+1):
    gmsh.model.addPhysicalGroup(gdim-1, [surfaces[i-1][1]],
                                surfaces[i-1][1])

gmsh.write("mesh.msh")

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()

# Import mesh in dolfinx
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
                                                              
mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh(
    "mesh.msh", mesh_comm, gmsh_model_rank, gdim=gdim)

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") \
    as mesh_file_xdmf:
    mesh_file_xdmf.write_mesh(mesh)
    mesh_file_xdmf.write_meshtags(cell_tags, mesh.geometry)
    mesh_file_xdmf.write_meshtags(facet_tags, mesh.geometry)