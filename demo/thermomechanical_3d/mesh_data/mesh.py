import dolfinx
import ufl
import gmsh
from mpi4py import MPI
import numpy as np
import sys

gmsh.initialize('', False)
gdim = 3

# Omega_1 Standard Carbon
lc1 = 0.6 # 0.2
gmsh.model.occ.addPoint(0., 0., 0., lc1, 1)
gmsh.model.occ.addPoint(6., 0., 0., lc1, 2)
gmsh.model.occ.addPoint(6., 2., 0., lc1, 3)
gmsh.model.occ.addPoint(4., 2., 0., lc1, 4)
gmsh.model.occ.addPoint(2., 2., 0., lc1, 5)
gmsh.model.occ.addPoint(0., 2., 0., lc1, 6)

gmsh.model.occ.addLine(1, 2, 1) # Marker 4, 10
gmsh.model.occ.addLine(2, 3, 2) # Marker 5, 11
gmsh.model.occ.addLine(3, 4, 3) # Marker 6, 12
gmsh.model.occ.addLine(4, 5, 4) # Marker 7, 13
gmsh.model.occ.addLine(5, 6, 5) # Marker 8, 14
gmsh.model.occ.addLine(6, 1, 6)

gmsh.model.occ.addCurveLoop([1, 2, 3, 4, 5, 6], 1)
gmsh.model.occ.addPlaneSurface([1], 1) # Marker 1, 2

# Omega_2 Micropore Carbon
lc2 = 0.6 # 0.2
gmsh.model.occ.addPoint(6., 6., 0., lc2, 7)
gmsh.model.occ.addPoint(4., 6., 0., lc2, 8)

gmsh.model.occ.addLine(3, 7, 7) # Marker 15, 19
gmsh.model.occ.addLine(7, 8, 8) # Marker 16, 20
gmsh.model.occ.addLine(8, 4, 9) # Marker 17, 21

gmsh.model.occ.addCurveLoop([7, 8, 9, -3], 2)
gmsh.model.occ.addPlaneSurface([2], 2) # Marker 3, 4

# Omega_3 Corondum Brick
lc3 = 0.6 # 0.2
gmsh.model.occ.addPoint(2., 6., 0., lc3, 9)

gmsh.model.occ.addLine(8, 9, 10) # Marker 23, 26
gmsh.model.occ.addLine(9, 5, 11) # Marker 22, 25

gmsh.model.occ.addCurveLoop([-9, 10, 11, -4], 3)
gmsh.model.occ.addPlaneSurface([3], 3) # Marker 5, 6

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

gmsh.model.occ.synchronize()

# this is the coherence in the python API
gmsh.model.occ.removeAllDuplicates()

gmsh.model.occ.synchronize()

# gmsh.option.setNumber("Mesh.MeshSizeMin", 0.1)
# gmsh.option.setNumber("Mesh.MeshSizeMax", 0.2)
# gmsh.option.setNumber("Mesh.Algorithm", 5)
gmsh.model.mesh.generate(gdim)
# gmsh.model.mesh.setOrder(2)
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

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf",
                         "w") as mesh_file_xdmf:
    mesh_file_xdmf.write_mesh(mesh)
    mesh_file_xdmf.write_meshtags(cell_tags, mesh.geometry)
    mesh_file_xdmf.write_meshtags(facet_tags, mesh.geometry)

V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
u = dolfinx.fem.Function(V)
u.x.array[:] = 1.

dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
dS = ufl.Measure("dS", domain=mesh, subdomain_data=facet_tags)

# print(dolfinx.mesh.interior_facet_indices(mesh.topology))

volume_1 = dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(ufl.inner(u, u) * 
    (dx(1) + dx(2)))
    )
print(f"Subdomain 1 volume: {volume_1}, True: {np.pi * 6**2 * 2}")

volume_2 = dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(ufl.inner(u, u) * 
    (dx(3) + dx(4))))
print(f"Subdomain 2 volume: {volume_2}, True: {np.pi * (6**2 - 4**2) * 4}")

volume_3 = dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(ufl.inner(u, u) * 
    (dx(5) + dx(6))))
print(f"Subdomain 3 volume: {volume_3}, True: {np.pi * (4**2 - 2**2) * 4}")

area_bottom = dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(ufl.inner(u, u) * 
    (ds(4) + ds(10)))
    )
print(f"Boundary bottom area: {area_bottom}, True: {np.pi * 6**2}")

area_outer_bottom = dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(ufl.inner(u, u) * 
    (ds(5) + ds(11)))
    )
print(f"Boundary outer bottom area: {area_outer_bottom}, True: {2 * np.pi * 6 * 2}")

area_outer_top = dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(ufl.inner(u, u) * 
    (ds(15) + ds(19)))
    )
print(f"Boundary outer top area: {area_outer_top}, True: {2 * np.pi * 6 * 4}")

area_subdomain_1top = dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(ufl.inner(u, u) * 
    (ds(8) + ds(14)))
    )
print(f"Boundary 1top area: {area_subdomain_1top}, True: {np.pi * 2**2}")

area_subdomain_2top = dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(ufl.inner(u, u) * 
    (ds(16) + ds(20)))
    )
print(f"Boundary 2top area: {area_subdomain_2top}, True: {np.pi * (6**2 - 4**2)}")

area_subdomain_3top = dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(ufl.inner(u, u) * 
    (ds(23) + ds(26)))
    )
print(f"Boundary 3top area: {area_subdomain_3top}, True: {np.pi * (4**2 - 2**2)}")

area_subdomain_inner = dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(ufl.inner(u, u) * 
    (ds(22) + ds(25)))
    )
print(f"Boundary inner area: {area_subdomain_inner}, True: {2 * np.pi * 2 * 4}")

area_subdomain_23 = dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(ufl.inner(u, u) * 
    (dS(17) + dS(21)))
    )

print(f"Boundary between subdomains 23 area: {area_subdomain_23}, True: {2 * np.pi * 4 * 4}")

area_subdomain_12 = dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(ufl.inner(u, u) * 
    (dS(6) + dS(12))
    ))

print(f"Boundary between subdomains 12 area: {area_subdomain_12}, True: {np.pi * (6**2 - 4**2)}")

area_subdomain_13 = dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(ufl.inner(u, u) * 
    (dS(7) + dS(13)))
    )

print(f"Boundary between subdomains 13 area: {area_subdomain_13}, True: {np.pi * (4**2 - 2**2)}")
