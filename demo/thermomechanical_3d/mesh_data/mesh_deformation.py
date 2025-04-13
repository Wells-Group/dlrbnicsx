from mdfenicsx.mesh_motion_classes import \
    HarmonicMeshMotion

import dolfinx
import numpy as np

from mpi4py import MPI

gdim = 3
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
mesh, cell_tags, facet_tags = \
    dolfinx.io.gmshio.read_from_msh("mesh.msh", mesh_comm,
                                    gmsh_model_rank, gdim=gdim)

# Store reference mesh
with dolfinx.io.XDMFFile(mesh_comm, "reference_mesh.xdmf",
                         "w") as reference_mesh_file:
    reference_mesh_file.write_mesh(mesh)
    reference_mesh_file.write_meshtags(cell_tags, mesh.geometry)
    reference_mesh_file.write_meshtags(facet_tags, mesh.geometry)

def bc_internal(x):
    return (0. * x[0], 0. * x[1], 0. * x[2]) # (0.1 * np.sin(x[1] * 2. * np.pi), 0. * x[1], 0. * x[2])

def bc_external(x):
    return (0. * x[0], 0. * x[1], 0. * x[2])

def bc_t1(x, mu=[2., 3., 4.]):
    r_all = np.sqrt(x[0]**2 + x[2]**2)
    indices_t1 = np.where(r_all > 4.)[0]
    r_indices_t1 = r_all[indices_t1]
    y = (0. * x[0], 0. * x[1], 0. * x[2])
    theta_cos = np.arccos(x[0][indices_t1] / r_indices_t1)
    theta_sin = np.arcsin(x[2][indices_t1] / r_indices_t1)
    y[0][indices_t1] = 4 * np.cos(theta_cos) - 2 * mu[0] * np.cos(theta_cos) + r_indices_t1 * np.cos(theta_cos) * (mu[0] / 2. - 1.)
    y[2][indices_t1] = 4 * np.sin(theta_sin) - 2 * mu[0] * np.sin(theta_sin) + r_indices_t1 * np.sin(theta_sin) * (mu[0] / 2. - 1.)
    indices_h1 = np.where(x[1] <= 2.)[0]
    y[1][indices_h1] = (mu[1] / 2. - 1.) * x[1][indices_h1]
    indices_h2 = np.where(x[1] > 2.)[0]
    y[1][indices_h2] = (mu[2] - 4.) / 4. * x[1][indices_h2] + mu[1] - mu[2] / 2.
    return y

def bc_19(x, mu=6.):
    indices_t1 = np.where(np.sqrt(x[0]**2 + x[2]**2) > 4.)[0]
    y = (0. * x[0], 0. * x[1], 0. * x[2])
    y[0][indices_t1] = (mu - 2.) * (np.sqrt(x[0][indices_t1]**2 + x[2][indices_t1]**2) / 2. - 2.) * np.cos(np.arctan(x[2][indices_t1] / x[0][indices_t1]))
    y[2][indices_t1] = (mu - 2.) * (np.sqrt(x[0][indices_t1]**2 + x[2][indices_t1]**2) / 2. - 2.) * np.sin(np.arctan(x[2][indices_t1] / x[0][indices_t1]))
    return y

def bc_15(x, mu=6.):
    indices_t1 = np.where(np.sqrt(x[0]**2 + x[2]**2) > 4.)[0]
    y = (0. * x[0], 0. * x[1], 0. * x[2])
    y[0][indices_t1] = (mu - 2.) * (np.sqrt(x[0][indices_t1]**2 + x[2][indices_t1]**2) / 2. - 2.) * np.cos(np.arctan(x[2][indices_t1] / x[0][indices_t1]))
    y[2][indices_t1] = (mu - 2.) * (np.sqrt(x[0][indices_t1]**2 + x[2][indices_t1]**2) / 2. - 2.) * np.sin(np.arctan(x[2][indices_t1] / x[0][indices_t1]))
    return y

    '''
    return ((mu - 6.) * np.cos(np.arctan(x[2] / x[0])),
            0. * x[1],
            (mu - 6.) * np.sin(np.arctan(x[2] / x[0])))
    '''

'''
# Mesh deformation (Harmonic mesh motion)
with HarmonicMeshMotion(mesh, facet_tags, [4, 10, 5, 11, 15, 19,
                                           8, 14, 16, 20, 23, 26,
                                           22, 25, 17, 21, 6, 12,
                                           7, 13],
                        [bc_external, bc_external, bc_external,
                         bc_external, bc_external, bc_external,
                         bc_external, bc_external, bc_external,
                         bc_external, bc_external, bc_external,
                         bc_external, bc_external, bc_internal,
                         bc_internal, bc_external, bc_external,
                         bc_external, bc_external],
                        reset_reference=True, is_deformation=True):
    # Store deformed mesh
    with dolfinx.io.XDMFFile(mesh.comm,
                             "deformed_harmonic.xdmf",
                             "w") as deformed_mesh_file:
        deformed_mesh_file.write_mesh(mesh)
        deformed_mesh_file.write_meshtags(cell_tags, mesh.geometry)
        deformed_mesh_file.write_meshtags(facet_tags, mesh.geometry)
'''

# Mesh deformation (Harmonic mesh motion)
with HarmonicMeshMotion(mesh, facet_tags, [4, 10, 5, 11, 15, 19,
                                           8, 14, 16, 20, 23, 26,
                                           22, 25, 17, 21, 6, 12,
                                           7, 13],
                        [bc_t1, bc_t1, bc_t1,
                         bc_t1, bc_t1, bc_t1,
                         bc_t1, bc_t1, bc_t1,
                         bc_t1, bc_t1, bc_t1,
                         bc_t1, bc_t1, bc_t1,
                         bc_t1, bc_t1, bc_t1,
                         bc_t1, bc_t1],
                        reset_reference=True, is_deformation=True):
    # Store deformed mesh
    with dolfinx.io.XDMFFile(mesh.comm,
                             "deformed_harmonic.xdmf",
                             "w") as deformed_mesh_file:
        deformed_mesh_file.write_mesh(mesh)
        deformed_mesh_file.write_meshtags(cell_tags, mesh.geometry)
        deformed_mesh_file.write_meshtags(facet_tags, mesh.geometry)