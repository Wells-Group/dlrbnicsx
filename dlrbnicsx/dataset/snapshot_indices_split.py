import numpy as np
from mpi4py import MPI


def parameter_dataset_split(generate_data_func, comm_list, num_samples,
                            dim_sample, world_comm):

    # world_comm = MPI.COMM_WORLD

    # NOTE LIMITATION: dtype considered is MPI.DOUBLE only
    if world_comm.rank == 0:
        nbytes = num_samples * dim_sample * MPI.DOUBLE.size
    else:
        nbytes = 0

    for i in range(len(comm_list)):
        if comm_list[i] != MPI.COMM_NULL:
            local_indices = np.arange(i, num_samples, len(comm_list))
        else:
            pass

    win_world_comm = MPI.Win.Allocate_shared(
        nbytes, MPI.DOUBLE.size, comm=world_comm)
    buff0, _ = win_world_comm.Shared_query(0)
    para_samples = np.ndarray(buffer=buff0, dtype="d",
                              shape=(num_samples, dim_sample))

    if world_comm.rank == 0:
        para_samples[:, :] = generate_data_func()

    world_comm.Barrier()

    return local_indices, para_samples


def snapshot_matrix_write(local_indices, para_samples, problem,
                          comm_list, num_samples, num_dofs, world_comm):
    # NOTE:
    # world_comm = MPI.COMM_WORLD

    if world_comm.rank == 0:
        nbytes = num_samples * num_dofs * MPI.DOUBLE.size
    else:
        nbytes = 0

    win_world_comm = MPI.Win.Allocate_shared(
        nbytes, MPI.DOUBLE.size, comm=world_comm)
    buff0, _ = win_world_comm.Shared_query(0)
    fem_snapshots = np.ndarray(buffer=buff0, dtype="d",
                               shape=(num_samples, num_dofs))

    for i in range(len(comm_list)):
        print(f"World rank {world_comm.rank}, Comm number {i}")
        if comm_list[i] != MPI.COMM_NULL:
            for j in local_indices:
                solution = problem.solve(para_samples[j, :])
                rstart, rend = solution.vector.getOwnershipRange()
                fem_snapshots[j, rstart:rend] = solution.vector[rstart:rend]

    return fem_snapshots


'''
def projected_snapshot_matrix_write(local_indices, para_samples, problem,
                                    reduced_problem, comm_list, num_samples,
                                    num_dofs):
'''


if __name__ == "__main__":
    global_comm = MPI.COMM_WORLD

    group0_procs = global_comm.group.Incl(np.arange(0, global_comm.size/2, 1))
    group0_comm = global_comm.Create_group(group0_procs)

    group1_procs = global_comm.group.Incl(
        np.arange(global_comm.size/2, global_comm.size, 1))
    group1_comm = global_comm.Create_group(group1_procs)

    comm_list = [group0_comm, group1_comm]

    num_samples = 16
    dim_sample = 4

    def generate_data_func(num_samples=num_samples, dim_sample=dim_sample):
        return np.random.randn(num_samples, dim_sample)

    local_indices, para_samples = parameter_dataset_split(
        generate_data_func, comm_list, num_samples, dim_sample, global_comm)

    if global_comm.rank == 0:
        print(f"Parameter samples \n {para_samples}")

    global_comm.Barrier()  # NOTE Barrier only for good output print

    print(f"====\n Rank {global_comm.rank}" +
          f"\n Local indices: {local_indices}," +
          f"\n Parameter samples \n {para_samples[local_indices, :]}"
          f"\n====")

    global_comm.Barrier()

    if group0_comm != MPI.COMM_NULL:
        para_samples[local_indices, :] = 0

    if group1_comm != MPI.COMM_NULL:
        para_samples[local_indices, :] = 1

    global_comm.Barrier()

    print(f"====\n Rank {global_comm.rank}" +
          f"\n Local indices: {local_indices}," +
          f"\n Parameter samples \n {para_samples[:, :]}"
          f"\n====")

    for i in range(len(comm_list)):
        if comm_list[i] != MPI.COMM_NULL:
            comm_list[i].Free()