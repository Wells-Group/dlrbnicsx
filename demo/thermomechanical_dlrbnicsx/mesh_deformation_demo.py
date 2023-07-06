from mdfenicsx.mesh_motion_classes import \
    HarmonicMeshMotion, LinearElasticMeshMotion   # noqa: F401

from mpi4py import MPI
import numpy as np

import dolfinx

# Read from msh
gdim = 2
mesh_comm = MPI.COMM_WORLD
gmsh_model_rank = 0
mesh, subdomains, boundaries = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/mesh.msh", mesh_comm,
                                    gmsh_model_rank, gdim=gdim)

# Parameter tuple (D_0, D_1, t_0, t_1)
mu_ref = [0.6438, 0.4313, 1., 0.5]  # reference geometry
mu = [0.8, 0.55, 0.8, 0.4]  # Parametric geometry

# Geometric deformation boundary condition w.r.t. reference domain
# i.e. set reset_reference=True and is_deformation=True

bc_list_geometric = list()


def bc_1_geometric(x):
    indices_0 = np.where((x[0] >= 4.875) & (x[0] <= 5.5188))[0]
    indices_1 = np.where((x[0] >= 5.5188) & (x[0] <= 5.9501))[0]
    y = (0. * x[0], 0. * x[1])
    y[0][indices_0] = (x[0][indices_0] - 4.875) * (mu[0] - mu_ref[0]) / (5.5188 - 4.875)
    y[0][indices_1] = \
        (x[0][indices_1] - 5.5188) * (mu[1] - mu_ref[1]) / (5.9501 - 5.5188) + \
        (mu[0] - mu_ref[0]) * (x[0][indices_1] / x[0][indices_1])
    return y


bc_list_geometric.append(bc_1_geometric)


def bc_2_geometric(x):
    return ((mu[1] - mu_ref[1] + mu[0] - mu_ref[0]) * (x[0] / x[0]), (x[1] - 0.) * (mu[2] - mu_ref[2]) / (1. - 0.))


bc_list_geometric.append(bc_2_geometric)


def bc_3_geometric(x):
    indices_0 = np.where((x[0] >= 4.875) & (x[0] <= 5.5188))[0]
    indices_1 = np.where((x[0] >= 5.5188) & (x[0] <= 5.9501))[0]
    y = (0. * x[0], 0. * x[1])
    y[0][indices_0] = (x[0][indices_0] - 4.875) * (mu[0] - mu_ref[0]) / (5.5188 - 4.875)
    y[0][indices_1] = \
        (x[0][indices_1] - 5.5188) * (mu[1] - mu_ref[1]) / (5.9501 - 5.5188) + \
        (mu[0] - mu_ref[0]) * (x[0][indices_1] / x[0][indices_1])
    y[1][:] = (mu[2] - mu_ref[2]) * x[1]
    return y


bc_list_geometric.append(bc_3_geometric)


def bc_4_geometric(x):
    return (0. * x[0], (mu[2] - mu_ref[2]) * x[1])


bc_list_geometric.append(bc_4_geometric)


def bc_5_geometric(x):
    return (0. * x[0], (mu[2] - mu_ref[2]) * x[1])


bc_list_geometric.append(bc_5_geometric)


def bc_6_geometric(x):
    return (0. * x[0], (mu[2] - mu_ref[2]) * (x[1] / x[1]))


bc_list_geometric.append(bc_6_geometric)


def bc_7_geometric(x):
    return (0. * x[0], (mu[2] - mu_ref[2]) * (x[1] / x[1]))


bc_list_geometric.append(bc_7_geometric)


def bc_8_geometric(x):
    return (0. * x[0], (mu[2] - mu_ref[2]) * (x[1] / x[1]))


bc_list_geometric.append(bc_8_geometric)


def bc_9_geometric(x):
    return (0. * x[0], (mu[2] - mu_ref[2]) * (x[1] / x[1]))


bc_list_geometric.append(bc_9_geometric)


def bc_10_geometric(x):
    return (0. * x[0], (x[1] - 1.6) * (mu[3] - mu_ref[3]) / (2.1 - 1.6) + (mu[2] - mu_ref[2]) * (x[1] / x[1]))


bc_list_geometric.append(bc_10_geometric)


def bc_11_geometric(x):
    return (0. * x[0], (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * (x[1] / x[1]))


bc_list_geometric.append(bc_11_geometric)


def bc_12_geometric(x):
    return (0. * x[0], (x[1] - 1.6) * (mu[3] - mu_ref[3]) / (2.1 - 1.6) + (mu[2] - mu_ref[2]) * (x[1] / x[1]))


bc_list_geometric.append(bc_12_geometric)


def bc_13_geometric(x):
    return (0. * x[0], (mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_13_geometric)


def bc_14_geometric(x):
    indices_0 = np.where((x[1] >= 1.6) & (x[1] <= 2.1))[0]
    indices_1 = np.where(x[1] > 2.1)[0]
    y = (0. * x[0], 0. * x[1])
    y[1][indices_0] = (x[1][indices_0] - 1.6) * (mu[3] - mu_ref[3]) / (2.1 - 1.6) + \
        (mu[2] - mu_ref[2]) * x[1][indices_0] / x[1][indices_0]
    y[1][indices_1] = (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1][indices_1] / x[1][indices_1]
    return y


bc_list_geometric.append(bc_14_geometric)


def bc_15_geometric(x):
    return (0. * x[0], (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_15_geometric)


def bc_16_geometric(x):
    return ((x[0] - 4.875) * (mu[0] - mu_ref[0]) / (5.5188 - 4.875),
            (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_16_geometric)


def bc_17_geometric(x):
    return ((x[0] - 4.875) * (mu[0] - mu_ref[0]) / (5.5188 - 4.875),
            (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_17_geometric)


def bc_18_geometric(x):
    return ((x[0] - 4.875) * (mu[0] - mu_ref[0]) / (5.5188 - 4.875),
            (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_18_geometric)


def bc_19_geometric(x):
    return ((x[0] - 4.875) * (mu[0] - mu_ref[0]) / (5.5188 - 4.875),
            (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_19_geometric)


def bc_20_geometric(x):
    return ((x[0] - 4.875) * (mu[0] - mu_ref[0]) / (5.5188 - 4.875),
            (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_20_geometric)


def bc_21_geometric(x):
    return (0. * x[0], (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_21_geometric)


def bc_22_geometric(x):
    return (0. * x[0], (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_22_geometric)


def bc_23_geometric(x):
    return (0. * x[0], (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_23_geometric)


def bc_24_geometric(x):
    indices_0 = np.where(x[1] < 1.6)[0]
    indices_1 = np.where((x[1] >= 1.6) & (x[1] <= 2.1))[0]
    indices_2 = np.where(x[1] > 2.1)[0]
    y = (0. * x[0], 0. * x[1])
    y[1][indices_0] = (mu[2] - mu_ref[2]) * (x[1][indices_0] / x[1][indices_0])
    y[1][indices_1] = (mu[3] - mu_ref[3]) * (x[1][indices_1] - 1.6) / (2.1 - 1.6) + \
        (mu[2] - mu_ref[2]) * x[1][indices_1] / x[1][indices_1]
    y[1][indices_2] = (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1][indices_2] / x[1][indices_2]
    y[0][:] = (mu[1] - mu_ref[1] + mu[0] - mu_ref[0]) * (x[0] / x[0])
    return y


bc_list_geometric.append(bc_24_geometric)


def bc_25_geometric(x):
    indices_0 = np.where((x[0] >= 4.875) & (x[0] <= 5.5188))[0]
    indices_1 = np.where((x[0] >= 5.5188) & (x[0] <= 5.9501))[0]
    y = (0. * x[0], 0. * x[1])
    y[0][indices_0] = (x[0][indices_0] - 4.875) * (mu[0] - mu_ref[0]) / (5.5188 - 4.875)
    y[0][indices_1] = \
        (x[0][indices_1] - 5.5188) * (mu[1] - mu_ref[1]) / (5.9501 - 5.5188) + \
        (mu[0] - mu_ref[0]) * (x[1][indices_1] / x[1][indices_1])
    y[1][:] = (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1]
    return y


bc_list_geometric.append(bc_25_geometric)


def bc_26_geometric(x):
    return ((mu[1] - mu_ref[1] + mu[0] - mu_ref[0]) * x[0] / x[0],
            (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_26_geometric)


def bc_27_geometric(x):
    return ((x[0] - 5.5188) * (mu[1] - mu_ref[1]) / (5.9501 - 5.5188) + (mu[0] - mu_ref[0]) * (x[0] / x[0]),
            (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_27_geometric)


def bc_28_geometric(x):
    return ((mu[1] - mu_ref[1] + mu[0] - mu_ref[0]) * x[0] / x[0],
            (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_28_geometric)


def bc_29_geometric(x):
    return ((mu[1] - mu_ref[1] + mu[0] - mu_ref[0]) * x[0] / x[0],
            (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1] / x[1])


bc_list_geometric.append(bc_29_geometric)


def bc_30_geometric(x):
    indices_0 = np.where(x[1] < 1.)[0]
    indices_1 = np.where((x[1] >= 1.) & (x[1] <= 1.6))[0]
    indices_2 = np.where((x[1] > 1.6) & (x[1] <= 2.1))[0]
    indices_3 = np.where(x[1] > 2.1)[0]
    y = (0 * x[0], 0. * x[1])
    y[0][:] = (mu[1] - mu_ref[1] + mu[0] - mu_ref[0]) * x[0] / x[0]
    y[1][indices_0] = (x[1][indices_0] - 0.) * (mu[2] - mu_ref[2]) / (1. - 0.)
    y[1][indices_1] = (mu[2] - mu_ref[2]) * (x[1][indices_1] / x[1][indices_1])
    y[1][indices_2] = (x[1][indices_2] - 1.6) * (mu[3] - mu_ref[3]) / (2.1 - 1.6) + \
        (mu[2] - mu_ref[2]) * x[1][indices_2] / x[1][indices_2]
    y[1][indices_3] = (mu[3] - mu_ref[3] + mu[2] - mu_ref[2]) * x[1][indices_3] / x[1][indices_3]
    return y


bc_list_geometric.append(bc_30_geometric)


def bc_31_geometric(x):
    return ((mu[1] - mu_ref[1] + mu[0] - mu_ref[0]) * x[0] / x[0], 0. * x[1])


bc_list_geometric.append(bc_31_geometric)

bc_markers_list = list(np.arange(1, 32))
with HarmonicMeshMotion(mesh, boundaries, bc_markers_list,
                        bc_list_geometric, reset_reference=True,
                        is_deformation=True):
    with dolfinx.io.XDMFFile(mesh.comm, "mesh_deformation_demo/deformed_harmonic.xdmf",
                             "w") as deformed_mesh_file:
        deformed_mesh_file.write_mesh(mesh)
