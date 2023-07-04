import os
import socket

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp  # noqa: F401
from mpi4py import MPI

from dlrbnicsx.dataset.custom_dataset import CustomDataset

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

os.environ['MASTER_ADDR'] = 'localhost'

if comm.rank == 0:
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.bind(('', 0))
	free_port = s.getsockname()[1]
	s.close()
else:
	free_port = None


free_port = comm.bcast(free_port, root=0)
comm.Barrier()

os.environ['MASTER_PORT'] = str(free_port)


'''
TODO see mp.set_start_method("spawn") to p.join() in the pytorch distributed
tutorial
(pytorch.org/tutorials/intermediate/dist_tuto.html#writing-distributed-applications-with-pytorch)

TODO mix indices before distributing data

TODO cleanup with dist.destroy_process_group()
'''

dist.init_process_group("gloo", rank=rank, world_size=size)


class CustomPartitionedDataset(CustomDataset):
    def __init__(self, problem, reduced_problem, N, input_set,
                 output_set, input_scaling_range=None,
                 output_scaling_range=None, input_range=None,
                 output_range=None, sizes=None):
        super().__init__(problem, reduced_problem, N, input_set,
                         output_set, input_scaling_range,
                         output_scaling_range, input_range, output_range)
        if sizes is None:
            world_size = dist.get_world_size()
            if isinstance(input_set, str):
                sizes = \
                    [int(np.rint(1 / world_size *
                                 len(np.load(input_set, allow_pickle=True))))
                     for _ in range(world_size-1)]
                sizes.append(len(np.load(input_set,
                                         allow_pickle=True)) - sum(sizes))
                assert sum(sizes) == \
                    len(np.load(input_set, allow_pickle=True))
            else:
                sizes = [int(np.rint(1 / world_size *
                             len(input_set)))
                         for _ in range(world_size-1)]
                sizes.append(len(input_set) - sum(sizes))
                assert sum(sizes) == len(input_set)
        # NOTE abover assertion is used to Make sure that the sum of the data
        # in distributed dataset = total number of input dataset
        self.sizes = sizes
        rank = dist.get_rank()
        global_indices = np.cumsum(self.sizes)
        if rank == 0:
            local_indices = torch.arange(0, global_indices[rank])
            print(f"Rank {rank}, Local indices {local_indices.tolist()}")
        else:
            print(global_indices[rank-1], global_indices[rank])
            local_indices = torch.arange(global_indices[rank-1],
                                         global_indices[rank])
            print(f"Rank {rank}, Local indices {local_indices.tolist()}")
        self.local_indices = local_indices

    def __len__(self):
        rank = dist.get_rank()
        return self.sizes[rank]

    def __getitem__(self, idx):
        if isinstance(self.input_set, str):
            input_data = np.load(self.input_set)[self.local_indices[idx], :]
        else:
            input_data = self.input_set[self.local_indices[idx], :]
        if isinstance(self.output_set, str):
            label = np.load(self.output_set)[self.local_indices[idx], :]
        else:
            label = self.output_set[self.local_indices[idx], :]
        return self.transform(input_data), self.target_transform(label)


if __name__ == "__main__":

    class Problem(object):
        def __init__(self):
            super().__init__()

    class ReducedProblem(object):
        def __init__(self, para_dim):
            super().__init__()
            self.input_range = np.vstack((np.zeros([1, para_dim]),
                                          np.ones([1, para_dim])))
            self.output_range = [0., 1.]
            self.input_scaling_range = [-1., 1.]
            self.output_scaling_range = [-1., 1.]

    # With numpy array as input-output
    print("\n With numpy array as input-output \n")
    input_data = np.random.uniform(0., 1., [100, 17]).astype("f")
    output_data = np.random.uniform(0., 1., [100, 7]).astype("f")
    input_data = torch.from_numpy(input_data)
    output_data = torch.from_numpy(output_data)
    dist.barrier()
    dist.all_reduce(input_data, op=dist.ReduceOp.MAX)
    dist.all_reduce(output_data, op=dist.ReduceOp.MAX)
    input_data = input_data.detach().numpy()
    output_data = output_data.detach().numpy()

    problem = Problem()
    reduced_problem = ReducedProblem(input_data.shape[1])
    custom_partitioned_dataset = \
        CustomPartitionedDataset(problem, reduced_problem, 10, input_data,
                                 output_data)
    dataloader = torch.utils.data.DataLoader(custom_partitioned_dataset,
                                             batch_size=1, shuffle=False)

    for batch, (X, y) in enumerate(dataloader):
        print(f"Rank: {rank}")
        print(custom_partitioned_dataset.transform
              (input_data[custom_partitioned_dataset.local_indices[0], :]))
        print(X)
        break

    my_first_index = custom_partitioned_dataset.local_indices[0]
    print(f"Rank {dist.get_rank()}, My first index: {my_first_index}")

    # With file_path as input-output
    print("\n With file_path as input-output \n")

    if dist.get_rank() == 0:
        np.save("input_data.npy", input_data)
        np.save("output_data.npy", output_data)

    dist.barrier()

    del input_data
    del output_data

    problem = Problem()
    reduced_problem = ReducedProblem(np.load("input_data.npy").shape[1])
    custom_partitioned_dataset = \
        CustomPartitionedDataset(problem, reduced_problem, 10,
                                 "input_data.npy", "output_data.npy")
    dataloader = torch.utils.data.DataLoader(custom_partitioned_dataset,
                                             batch_size=1, shuffle=False)
    for batch, (X, y) in enumerate(dataloader):
        print(f"Rank: {rank}")
        print(custom_partitioned_dataset.transform
              (np.load("input_data.npy")
               [my_first_index, :]))
        print(X)
        break

# TODO Use torch.tensor_split for splitting indices into different sizes.
# NOTE Do not use torch.chunk as it may leave some processes idle
