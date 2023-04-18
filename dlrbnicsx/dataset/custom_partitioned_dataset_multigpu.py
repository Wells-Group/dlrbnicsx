from mpi4py import MPI
import torch  # noqa: F401
import numpy as np
import sys  # noqa: F401

comm = MPI.COMM_WORLD
num_gpus = 4
group_id = [[0, 1, 2], [4, 5], [8], [12, 13, 14, 15]]
gpu_id = [0, 1, 2, 3]

process_group_0 = comm.group.Incl(group_id[0])
process_group_1 = comm.group.Incl(group_id[1])
process_group_2 = comm.group.Incl(group_id[2])
process_group_3 = comm.group.Incl(group_id[3])

gpu_group_0 = "cuda: 0"
gpu_group_1 = "cuda: 1"
gpu_group_2 = "cuda: 2"
gpu_group_3 = "cuda: 3"

if comm.rank == 3:
    assert process_group_0 == MPI.COMM_NULL
    assert process_group_1 == MPI.COMM_NULL
    assert process_group_2 == MPI.COMM_NULL
    assert process_group_3 == MPI.COMM_NULL

process_group = comm.group.Incl(group_id[comm.rank//num_gpus])
group_comm = comm.Create_group(process_group)

shape = (2, 10)
itemsize = MPI.FLOAT.Get_size()

if comm.rank == 0:
    nbytes = np.prod(shape) * itemsize
else:
    nbytes = 0

win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)

buf, itemsize = win.Shared_query(0)
my_array = np.ndarray(buffer=buf, dtype="f", shape=shape)

print(comm.rank, my_array)

if comm.rank == 3:
    my_array[0, :3] = [10.6, 20.3, 12.4]

comm.Barrier()
print(comm.rank, my_array)
