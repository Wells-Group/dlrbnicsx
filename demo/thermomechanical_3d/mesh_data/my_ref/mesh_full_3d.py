import dolfinx
import gmsh
from mpi4py import MPI
import numpy as np
import sys

gmsh.initialize('', False)
gdim = 3

# Omega_1 Standard Carbon
lc1 = 0.2
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

gmsh.model.occ.addCurveLoop([1, 2, 3, 4, 5], 1)
gmsh.model.occ.addPlaneSurface([1], 1)

# Omega_2 Micropore Carbon
lc2 = 0.1
gmsh.model.occ.addPoint(2.1, 1.6, 0., lc2, 6)
gmsh.model.occ.addPoint(0.39, 1.6, 0., lc2, 7)
gmsh.model.occ.addPoint(0., 1.6, 0., lc2, 8)

gmsh.model.occ.addLine(4, 6, 6)
gmsh.model.occ.addLine(6, 7, 7)
gmsh.model.occ.addLine(7, 8, 8)
gmsh.model.occ.addLine(8, 5, 9)

gmsh.model.occ.addCurveLoop([-4, 6, 7, 8, 9], 2)
gmsh.model.occ.addPlaneSurface([2], 2)

# Omega_3 Corondum Brick
lc3 = 0.08
gmsh.model.occ.addPoint(0.39, 2.1, 0., lc3, 9)
gmsh.model.occ.addPoint(0., 2.1, 0., lc3, 10)

gmsh.model.occ.addLine(7, 9, 10)
gmsh.model.occ.addLine(9, 10, 11)
gmsh.model.occ.addLine(10, 8, 12)

gmsh.model.occ.addCurveLoop([-8, 10, 11, 12], 31)
gmsh.model.occ.addPlaneSurface([31], 3)

# Omega_4 Ceramic cup
lc4 = 0.08
gmsh.model.occ.addPoint(4.875, 1.6, 0., lc4, 11)
gmsh.model.occ.addPoint(4.875, 5.2, 0., lc4, 12)
gmsh.model.occ.addPoint(4.875, 6.4, 0., lc4, 13)
gmsh.model.occ.addPoint(5.5188, 6.4, 0., lc4, 14)
gmsh.model.occ.addPoint(5.5188, 7.35, 0., lc4, 15)
gmsh.model.occ.addPoint(5.5188, 7.4, 0., lc4, 16)
gmsh.model.occ.addPoint(4.875, 7.4, 0., lc4, 17)
gmsh.model.occ.addPoint(4.875, 7., 0., lc4, 18)
gmsh.model.occ.addPoint(4.475, 7., 0., lc4, 19)
gmsh.model.occ.addPoint(4.475, 2.1, 0., lc4, 20)

gmsh.model.occ.addLine(6, 11, 13)
gmsh.model.occ.addLine(11, 12, 14)
gmsh.model.occ.addLine(12, 13, 15)
gmsh.model.occ.addLine(13, 14, 16)
gmsh.model.occ.addLine(14, 15, 17)
gmsh.model.occ.addLine(15, 16, 18)
gmsh.model.occ.addLine(16, 17, 19)
gmsh.model.occ.addLine(17, 18, 20)
gmsh.model.occ.addLine(18, 19, 21)
gmsh.model.occ.addLine(19, 20, 22)
gmsh.model.occ.addLine(20, 9, 23)

gmsh.model.occ.addCurveLoop([13, 14, 15, 16, 17, 18, 19, 20,
                             21, 22, 23, -10, -7], 4)
gmsh.model.occ.addPlaneSurface([4], 4)

'''
# Omega_5 Super micropore carbon
lc5 = 0.1
gmsh.model.occ.addPoint(5.9501, 5.2, 0., lc5, 21)

gmsh.model.occ.addLine(3, 21, 24)
gmsh.model.occ.addLine(21, 12, 25)

gmsh.model.occ.addCurveLoop([-3, 24, 25, -14, -13, -6], 5)
gmsh.model.occ.addPlaneSurface([5], 5)

# Omega_7 Micropore Carbon
lc7 = 0.1
gmsh.model.occ.addPoint(5.9501, 7.35, 0., lc7, 22)

gmsh.model.occ.addLine(21, 22, 26)
gmsh.model.occ.addLine(22, 15, 27)

gmsh.model.occ.addCurveLoop([-25, 26, 27, -17, -16, -15], 7)
gmsh.model.occ.addPlaneSurface([7], 7)

# Omega_6 Stainless steel
lc6 = 0.08
gmsh.model.occ.addPoint(5.9501, 7.4, 0., lc6, 23)
gmsh.model.occ.addPoint(6.0201, 7.4, 0., lc6, 24)
gmsh.model.occ.addPoint(6.0201, 0, 0., lc6, 25)

gmsh.model.occ.addLine(22, 23, 28)
gmsh.model.occ.addLine(23, 24, 29)
gmsh.model.occ.addLine(24, 25, 30)
gmsh.model.occ.addLine(25, 2, 31)

gmsh.model.occ.addCurveLoop([2, 24, 26, 28, 29, 30, 31], 6)
gmsh.model.occ.addPlaneSurface([6], 6)
'''

gmsh.model.occ.revolve([(2, 1)], 0, 0, 0, 0, 1, 0,
                       np.pi)

gmsh.model.occ.revolve([(2, 1)], 0, 0, 0, 0, 1, 0,
                       -np.pi)

gmsh.model.occ.revolve([(2, 2)], 0, 0, 0, 0, 1, 0,
                       np.pi)

gmsh.model.occ.revolve([(2, 2)], 0, 0, 0, 0, 1, 0,
                       -np.pi)

gmsh.model.occ.revolve([(2, 3)], 0, 0, 0, 0, 1, 0,
                       np.pi)

gmsh.model.occ.revolve([(2, 3)], 0, 0, 0, 0, 1, 0,
                       -np.pi)                       

gmsh.model.occ.revolve([(2, 4)], 0, 0, 0, 0, 1, 0,
                       np.pi)

gmsh.model.occ.revolve([(2, 4)], 0, 0, 0, 0, 1, 0,
                       -np.pi)

'''
gmsh.model.occ.revolve([(2, 5)], 0, 0, 0, 0, 1, 0,
                       np.pi)

gmsh.model.occ.revolve([(2, 5)], 0, 0, 0, 0, 1, 0,
                       -np.pi)
                       
gmsh.model.occ.revolve([(2, 6)], 0, 0, 0, 0, 1, 0,
                       np.pi)

gmsh.model.occ.revolve([(2, 6)], 0, 0, 0, 0, 1, 0,
                       -np.pi)

gmsh.model.occ.revolve([(2, 7)], 0, 0, 0, 0, 1, 0,
                       np.pi)

gmsh.model.occ.revolve([(2, 7)], 0, 0, 0, 0, 1, 0,
                       -np.pi)
'''

gmsh.model.occ.synchronize()

# this is the coherence in the python API
gmsh.model.occ.removeAllDuplicates()

gmsh.model.occ.synchronize()

# gmsh.option.setNumber("Mesh.MeshSizeMin", 0.1)
# gmsh.option.setNumber("Mesh.MeshSizeMax", 0.2)
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

'''
# Subdomains markings
for i in range(1, len(volumes)+1):
    gmsh.model.addPhysicalGroup(gdim, [volumes[i-1][1]],
                                volumes[i-1][1])

# External boundaries markings
for i in range(25, len(surfaces)+1):
    gmsh.model.addPhysicalGroup(gdim-1, [surfaces[i-1][1]],
                                surfaces[i-1][1])
'''

# Volume markers subdomain 1
gmsh.model.addPhysicalGroup(gdim, [1, 2], 1)
# Boundary markers subdomain 1
gmsh.model.addPhysicalGroup(gdim-1, [7, 12], 1)
gmsh.model.addPhysicalGroup(gdim-1, [11, 6], 2)
gmsh.model.addPhysicalGroup(gdim-1, [4, 9], 3)
gmsh.model.addPhysicalGroup(gdim-1, [5, 10], 4)

# Volume markers subdomain 2
gmsh.model.addPhysicalGroup(gdim, [3, 4], 2)
# Boundary markers subdomain 2
gmsh.model.addPhysicalGroup(gdim-1, [15, 19], 5)
gmsh.model.addPhysicalGroup(gdim-1, [13, 17], 6)
gmsh.model.addPhysicalGroup(gdim-1, [14, 18], 7)

# Volume markers subdomain 3
gmsh.model.addPhysicalGroup(gdim, [5, 6], 3)
# Boundary markers subdomain 3
gmsh.model.addPhysicalGroup(gdim-1, [20, 23], 8)
gmsh.model.addPhysicalGroup(gdim-1, [21, 24], 9)

# Volume markers subdomain 4
gmsh.model.addPhysicalGroup(gdim, [7, 8], 4)
# Boundary markers subdomain 4
gmsh.model.addPhysicalGroup(gdim-1, [25, 37], 10)
gmsh.model.addPhysicalGroup(gdim-1, [26, 38], 11)
gmsh.model.addPhysicalGroup(gdim-1, [27, 39], 12)
'''
gmsh.model.addPhysicalGroup(gdim-1, [28, 40], 13)
gmsh.model.addPhysicalGroup(gdim-1, [29, 41], 14)
gmsh.model.addPhysicalGroup(gdim-1, [30, 42], 15)
gmsh.model.addPhysicalGroup(gdim-1, [31, 43], 16)
gmsh.model.addPhysicalGroup(gdim-1, [32, 44], 17)
gmsh.model.addPhysicalGroup(gdim-1, [33, 45], 18)
gmsh.model.addPhysicalGroup(gdim-1, [34, 46], 19)
gmsh.model.addPhysicalGroup(gdim-1, [35, 47], 20)
gmsh.model.addPhysicalGroup(gdim-1, [36, 48], 21)
'''

# gmsh.model.addPhysicalGroup(gdim, [9, 10], 5)
# gmsh.model.addPhysicalGroup(gdim, [11, 12], 6)
# gmsh.model.addPhysicalGroup(gdim, [13, 14], 7)

'''
gmsh.model.addPhysicalGroup(gdim-1, [1], 1)
gmsh.model.addPhysicalGroup(gdim-1, [2], 2)
gmsh.model.addPhysicalGroup(gdim-1, [3], 3)
gmsh.model.addPhysicalGroup(gdim-1, [4], 4)
gmsh.model.addPhysicalGroup(gdim-1, [5], 5)
gmsh.model.addPhysicalGroup(gdim-1, [6], 6)
gmsh.model.addPhysicalGroup(gdim-1, [7], 7)
gmsh.model.addPhysicalGroup(gdim-1, [8], 8)
gmsh.model.addPhysicalGroup(gdim-1, [9], 9)
gmsh.model.addPhysicalGroup(gdim-1, [10], 10)
gmsh.model.addPhysicalGroup(gdim-1, [11], 11)
gmsh.model.addPhysicalGroup(gdim-1, [12], 12)
gmsh.model.addPhysicalGroup(gdim-1, [13], 13)
gmsh.model.addPhysicalGroup(gdim-1, [14], 14)
gmsh.model.addPhysicalGroup(gdim-1, [15], 15)
gmsh.model.addPhysicalGroup(gdim-1, [16], 16)
'''

gmsh.write("mesh.msh")

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()

# Import mesh in dolfinx
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD

mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh(
    "mesh.msh", mesh_comm, gmsh_model_rank, gdim=gdim)

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf",
                         "w") as mesh_file_xdmf:
    mesh_file_xdmf.write_mesh(mesh)
    mesh_file_xdmf.write_meshtags(cell_tags, mesh.geometry)
    mesh_file_xdmf.write_meshtags(facet_tags, mesh.geometry)
