import dolfinx
import gmsh
from mpi4py import MPI

gmsh.initialize('', False)
gdim = 2

# Omega_1 Standard Carbon
lc1 = 0.3# 0.2
gmsh.model.geo.addPoint(0., 0., 0., lc1, 1)
gmsh.model.geo.addPoint(5.9501, 0., 0., lc1, 2)
gmsh.model.geo.addPoint(5.9501, 1., 0., lc1, 3)
gmsh.model.geo.addPoint(2.1, 1., 0., lc1, 4)
gmsh.model.geo.addPoint(0., 1., 0., lc1, 5)

gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 5, 4)
gmsh.model.geo.addLine(5, 1, 5)

gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5], 1)
gmsh.model.geo.addPlaneSurface([1], 1)

# Omega_2 Micropore Carbon
lc2 = 0.1
gmsh.model.geo.addPoint(2.1, 1.6, 0., lc2, 6)
gmsh.model.geo.addPoint(0.39, 1.6, 0., lc2, 7)
gmsh.model.geo.addPoint(0., 1.6, 0., lc2, 8)

gmsh.model.geo.addLine(4, 6, 6)
gmsh.model.geo.addLine(6, 7, 7)
gmsh.model.geo.addLine(7, 8, 8)
gmsh.model.geo.addLine(8, 5, 9)

gmsh.model.geo.addCurveLoop([-4, 6, 7, 8, 9], 2)
gmsh.model.geo.addPlaneSurface([2], 2)

# Omega_3 Corondum Brick
lc3 = 0.08
gmsh.model.geo.addPoint(0.39, 2.1, 0., lc3, 9)
gmsh.model.geo.addPoint(0., 2.1, 0., lc3, 10)

gmsh.model.geo.addLine(7, 9, 10)
gmsh.model.geo.addLine(9, 10, 11)
gmsh.model.geo.addLine(10, 8, 12)

gmsh.model.geo.addCurveLoop([-8, 10, 11, 12], 3)
gmsh.model.geo.addPlaneSurface([3], 3)

# Omega_4 Ceramic cup
lc4 = 0.08
gmsh.model.geo.addPoint(4.875, 1.6, 0., lc4, 11)
gmsh.model.geo.addPoint(4.875, 5.2, 0., lc4, 12)
gmsh.model.geo.addPoint(4.875, 6.4, 0., lc4, 13)
gmsh.model.geo.addPoint(5.5188, 6.4, 0., lc4, 14)
gmsh.model.geo.addPoint(5.5188, 7.35, 0., lc4, 15)
gmsh.model.geo.addPoint(5.5188, 7.4, 0., lc4, 16)
gmsh.model.geo.addPoint(4.875, 7.4, 0., lc4, 17)
gmsh.model.geo.addPoint(4.875, 7., 0., lc4, 18)
gmsh.model.geo.addPoint(4.475, 7., 0., lc4, 19)
gmsh.model.geo.addPoint(4.475, 2.1, 0., lc4, 20)

gmsh.model.geo.addLine(6, 11, 13)
gmsh.model.geo.addLine(11, 12, 14)
gmsh.model.geo.addLine(12, 13, 15)
gmsh.model.geo.addLine(13, 14, 16)
gmsh.model.geo.addLine(14, 15, 17)
gmsh.model.geo.addLine(15, 16, 18)
gmsh.model.geo.addLine(16, 17, 19)
gmsh.model.geo.addLine(17, 18, 20)
gmsh.model.geo.addLine(18, 19, 21)
gmsh.model.geo.addLine(19, 20, 22)
gmsh.model.geo.addLine(20, 9, 23)

gmsh.model.geo.addCurveLoop([13, 14, 15, 16, 17, 18, 19, 20,
                             21, 22, 23, -10, -7], 4)
gmsh.model.geo.addPlaneSurface([4], 4)

# Omega_5 Super micropore carbon
lc5 = 0.1
gmsh.model.geo.addPoint(5.9501, 5.2, 0., lc5, 21)

gmsh.model.geo.addLine(3, 21, 24)
gmsh.model.geo.addLine(21, 12, 25)

gmsh.model.geo.addCurveLoop([-3, 24, 25, -14, -13, -6], 5)
gmsh.model.geo.addPlaneSurface([5], 5)

# Omega_7 Micropore Carbon
lc7 = 0.1
gmsh.model.geo.addPoint(5.9501, 7.35, 0., lc7, 22)

gmsh.model.geo.addLine(21, 22, 26)
gmsh.model.geo.addLine(22, 15, 27)

gmsh.model.geo.addCurveLoop([-25, 26, 27, -17, -16, -15], 7)
gmsh.model.geo.addPlaneSurface([7], 7)

# Omega_6 Stainless steel
lc6 = 0.08
gmsh.model.geo.addPoint(5.9501, 7.4, 0., lc6, 23)
gmsh.model.geo.addPoint(6.0201, 7.4, 0., lc6, 24)
gmsh.model.geo.addPoint(6.0201, 0, 0., lc6, 25)

gmsh.model.geo.addLine(22, 23, 28)
gmsh.model.geo.addLine(23, 24, 29)
gmsh.model.geo.addLine(24, 25, 30)
gmsh.model.geo.addLine(25, 2, 31)

gmsh.model.geo.addCurveLoop([2, 24, 26, 28, 29, 30, 31], 6)
gmsh.model.geo.addPlaneSurface([6], 6)

gmsh.model.geo.synchronize()

gmsh.model.mesh.generate(gdim)
gmsh.model.mesh.setOrder(2)
gmsh.model.mesh.optimize("Netgen")

# Extract edges and surfaces to add physical groups
surfaces = gmsh.model.getEntities(dim=gdim)
edges = gmsh.model.getBoundary(surfaces)

# Subdomains markings
for i in range(1, len(surfaces)+1):
    gmsh.model.addPhysicalGroup(gdim, [surfaces[i-1][1]], surfaces[i-1][1])
# External boundaries markings
for i in range(1, len(edges)+1):
    gmsh.model.addPhysicalGroup(gdim-1, [edges[i-1][1]], edges[i-1][1])

gmsh.model.addPhysicalGroup(gdim-1, [2], 2)
gmsh.model.addPhysicalGroup(gdim-1, [3], 3)
gmsh.model.addPhysicalGroup(gdim-1, [4], 4)
gmsh.model.addPhysicalGroup(gdim-1, [6], 6)
gmsh.model.addPhysicalGroup(gdim-1, [7], 7)
gmsh.model.addPhysicalGroup(gdim-1, [8], 8)
gmsh.model.addPhysicalGroup(gdim-1, [10], 10)
gmsh.model.addPhysicalGroup(gdim-1, [13], 13)
gmsh.model.addPhysicalGroup(gdim-1, [14], 14)
gmsh.model.addPhysicalGroup(gdim-1, [15], 15)
gmsh.model.addPhysicalGroup(gdim-1, [16], 16)
gmsh.model.addPhysicalGroup(gdim-1, [17], 17)
gmsh.model.addPhysicalGroup(gdim-1, [24], 24)
gmsh.model.addPhysicalGroup(gdim-1, [25], 25)
gmsh.model.addPhysicalGroup(gdim-1, [26], 26)

gmsh.write("mesh.msh")

# gmsh.fltk.run()

gmsh.finalize()

# Import mesh in dolfinx
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh(
    "mesh.msh", mesh_comm, gmsh_model_rank, gdim=gdim)

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as mesh_file_xdmf:
    mesh_file_xdmf.write_mesh(mesh)
    mesh_file_xdmf.write_meshtags(cell_tags, mesh.geometry)
    mesh_file_xdmf.write_meshtags(facet_tags, mesh.geometry)
