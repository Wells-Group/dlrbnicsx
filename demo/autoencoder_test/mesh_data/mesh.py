from mpi4py import MPI
import dolfinx
import gmsh

comm = MPI.COMM_WORLD

gmsh.initialize("", False)

gdim = 2

A = gmsh.model.occ.addPoint(0., 0., 0., 0.12)
B = gmsh.model.occ.addPoint(1., 0., 0., 0.18)
C = gmsh.model.occ.addPoint(1., 1., 0., 0.10)
D = gmsh.model.occ.addPoint(0., 1., 0., 0.15)

AB = gmsh.model.occ.addLine(A, B)
BC = gmsh.model.occ.addLine(B, C)
CD = gmsh.model.occ.addLine(C, D)
DA = gmsh.model.occ.addLine(D, A)

ABCDA = gmsh.model.occ.addCurveLoop([AB, BC, CD, DA])
rectangle = gmsh.model.occ.addPlaneSurface([ABCDA])

gmsh.model.occ.synchronize()

gmsh.option.setNumber("Mesh.Algorithm", 8)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.01)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.3)
gmsh.model.mesh.generate(gdim)
gmsh.model.mesh.setOrder(1)
gmsh.model.mesh.optimize("Netgen")

surfaces = gmsh.model.getEntities(dim=gdim)
edges = gmsh.model.getEntities(dim=gdim-1)

gmsh.model.addPhysicalGroup(gdim, [surfaces[0][1]], 1)
for i in range(1, len(edges)+1):
    gmsh.model.addPhysicalGroup(gdim-1, [edges[i-1][1]],
                                edges[i-1][1])

gmsh.model.occ.remove(gmsh.model.getEntities(dim=gdim-1))
gmsh.model.occ.remove(gmsh.model.getEntities(dim=gdim))

gmsh.write("mesh.msh")
gmsh.fltk.run()

gmsh_model_rank = 0
mesh_comm = comm
mesh, subdomains, boundaries = \
    dolfinx.io.gmshio.read_from_msh("mesh.msh", mesh_comm,
                                    gmsh_model_rank, gdim=gdim)

with dolfinx.io.XDMFFile(mesh.comm, "mesh.xdmf", "w") as mesh_file_xdmf:
    mesh_file_xdmf.write_mesh(mesh)
    mesh_file_xdmf.write_meshtags(subdomains, mesh.geometry)
    mesh_file_xdmf.write_meshtags(boundaries, mesh.geometry)
