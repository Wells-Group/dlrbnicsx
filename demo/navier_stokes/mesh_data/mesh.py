import gmsh
from mpi4py import MPI
import dolfinx

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

gmsh.initialize('', False)

gdim = 2

h1 = 0.1
h2 = 0.1
h3 = 0.1
h4 = 0.1

H = 1.0
W = 1.0

gmsh.model.occ.addPoint(0., 0., 0., h1, 1)
gmsh.model.occ.addPoint(0., W, 0., h2, 2)
gmsh.model.occ.addPoint(H, W, 0., h3, 3)
gmsh.model.occ.addPoint(H, 0., 0., h4, 4)

gmsh.model.occ.addLine(1, 2, 1)
gmsh.model.occ.addLine(2, 3, 2)
gmsh.model.occ.addLine(3, 4, 3)
gmsh.model.occ.addLine(4, 1, 4)

lineLoop = gmsh.model.occ.addCurveLoop([1, 2, 3, 4])
planeSurface = gmsh.model.occ.addPlaneSurface([lineLoop], 1)

# Synchronize
gmsh.model.occ.synchronize()

gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "CurvesList", [2])
gmsh.model.mesh.field.add("MathEval", 2)
gmsh.model.mesh.field.setString(2, "F", "F1^4 + " + str(0.01))

gmsh.model.mesh.field.setAsBackgroundMesh(2)

# Create mesh
gmsh.option.setNumber("Mesh.Algorithm", 1) # 8=Frontal-Delaunay for Quads (See section 7.4,  https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1) # 2=simple full-quad (See section 7.4,  https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options)
# gmsh.option.setNumber("Mesh.RecombineAll", 1) # Apply recombination algorithm to all surfaces, ignoring per-surface spec
gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 0) # Mesh subdivision algorithm (0: none, 1: all quadrangles, 2: all hexahedra, 3: barycentric)
# gmsh.option.setNumber("Mesh.MeshSizeMin", 0.1) # Minimum characteristic element size
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.03) # Maximum characteristic element size
gmsh.model.mesh.generate(gdim) # Mesh generation
gmsh.model.mesh.setOrder(1) # Mesh order
gmsh.model.mesh.optimize("Netgen") # Mesh optimisation or improving quality of mesh

# Extract edges and surfaces to add physical groups
surfaces = gmsh.model.getEntities(dim=gdim)
edges = gmsh.model.getBoundary(surfaces)

for i in range(1,len(surfaces)+1):
    gmsh.model.addPhysicalGroup(gdim,[surfaces[i-1][1]],surfaces[i-1][1])
for i in range(1,len(edges)+1):
    gmsh.model.addPhysicalGroup(gdim-1,[edges[i-1][1]],edges[i-1][1])

# NOTE Remove gmsh markers as dolfinx.io.gmshio extract_geometry and extract_topology_and_markers expects gmsh to provide model with only physical markers and not point/edge markers.
#gmsh.model.occ.remove(gmsh.model.getEntities(dim=gdim-1))
#gmsh.model.occ.remove(gmsh.model.getEntities(dim=gdim))

# Save and visualise the mesh
gmsh.write("mesh.msh")
gmsh.fltk.run()

gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = \
    dolfinx.io.gmshio.model_to_mesh(gmsh.model, mesh_comm,
                                    gmsh_model_rank, gdim=gdim)

with dolfinx.io.XDMFFile(comm, "mesh.xdmf", "w") as mesh_file_xdmf:
    mesh_file_xdmf.write_mesh(domain)
    mesh_file_xdmf.write_meshtags(cell_markers)
    mesh_file_xdmf.write_meshtags(facet_markers)
