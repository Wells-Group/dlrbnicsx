import gmsh
from mpi4py import MPI
import dolfinx

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Avoid reading config file such as .gmshrc as it may have old data
gmsh.initialize('', False)

# Geometric dimensions of the domain
gdim = 2

h1 = 0.2  # Very coarse
h2 = 0.1  # Coarse enough
h3 = 0.05  # Fine enough
h4 = 0.02  # Very fine

H = 10.  # Height of square
W = 10.  # Width of square
r = 5.  # Radius of circle

assert r < H
assert r < W

# Points
A = gmsh.model.occ.addPoint(0., 0., 0., h1)
B = gmsh.model.occ.addPoint(0., H, 0., h1)
C = gmsh.model.occ.addPoint(W, H, 0., h1)
D = gmsh.model.occ.addPoint(W, 0., 0., h1)

# Lines
AB = gmsh.model.occ.addLine(A, B)
BC = gmsh.model.occ.addLine(B, C)
CD = gmsh.model.occ.addLine(C, D)
DA = gmsh.model.occ.addLine(D, A)

# Surfaces and final geometry
ABCDA = gmsh.model.occ.addCurveLoop([AB, BC, CD, DA])
# Rectangle
rectangle = gmsh.model.occ.addPlaneSurface([ABCDA], 1)
# Creates Lines "equivalent" of circumference
c_r = gmsh.model.occ.addCircle(0., H, 0., r)
# Creates curve of circumference
c_r_curve = gmsh.model.occ.addCurveLoop([c_r], 2)
# Circle (plane surface type)
circle = gmsh.model.occ.addPlaneSurface([c_r_curve], 2)
# NOTE Adding disk is equivalent to creating circle from above three lines
# circle = gmsh.model.occ.addDisk(0, H, 0, r, r)
# Surface domain geometry
domain_geometry = gmsh.model.occ.cut([(gdim, rectangle)],
                                     [(gdim, circle)], 3)

# Synchronize
gmsh.model.occ.synchronize()

# Create mesh
# 8 = Frontal-Delaunay for Quads
gmsh.option.setNumber("Mesh.Algorithm", 8)
# 2 = simple full-quad
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
# Apply recombination algorithm
gmsh.option.setNumber("Mesh.RecombineAll", 1)
# Mesh subdivision algorithm (1: all quadrangles)
gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
# Minimum characteristic element size
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.1)
# Maximum characteristic element size
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.3)
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

with dolfinx.io.XDMFFile(mesh.comm, "mesh.xdmf", "w") as mesh_file_xdmf:
    mesh_file_xdmf.write_mesh(mesh)
    mesh_file_xdmf.write_meshtags(subdomains)
    mesh_file_xdmf.write_meshtags(boundaries)
