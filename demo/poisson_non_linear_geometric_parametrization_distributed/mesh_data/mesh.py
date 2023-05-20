import gmsh

from mpi4py import MPI

import dolfinx

gmsh.initialize('', False)

lc = 0.4#lc = 0.04
gdim = 2

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
