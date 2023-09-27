import gmsh
from mpi4py import MPI
import dolfinx

gmsh.initialize(" ", False)
gdim = 3

center_x, center_y, center_z, radius = 0., 0., 0., 2.
circle = gmsh.model.occ.addCircle(center_x, center_y, center_z, radius)
circle_loop = gmsh.model.occ.addCurveLoop([circle])
surface = gmsh.model.occ.addPlaneSurface([circle_loop])
gmsh.model.occ.synchronize()

h = 1
extrusion = gmsh.model.occ.extrude([(2, surface)], 0, 0, h)
gmsh.model.occ.synchronize()

gmsh.option.setNumber("Mesh.MeshSizeMin", 0.05)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)
gmsh.option.setNumber("Mesh.Algorithm", 5)
gmsh.model.mesh.generate(gdim)
gmsh.model.mesh.setOrder(1)
gmsh.model.mesh.optimize("Netgen")

volumes = gmsh.model.getEntities(dim=gdim)
surfaces = gmsh.model.getEntities(dim=gdim-1)

# Surface markings
for i in range(len(volumes)):
    gmsh.model.addPhysicalGroup(gdim, [volumes[i-1][1]], volumes[i-1][1])

# Volume markings
for i in range(len(surfaces)):
    gmsh.model.addPhysicalGroup(gdim-1, [surfaces[i-1][1]], surfaces[i-1][1])

gmsh.write("mesh.msh")

gmsh.fltk.run()

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
