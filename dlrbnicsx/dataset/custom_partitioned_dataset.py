import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from mpi4py import MPI

from dlrbnicsx.dataset.custom_dataset import CustomDataset

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# TODO see mp.set_start_method("spawn") to p.join() in the pytorch distributed tutorial (pytorch.org/tutorials/intermediate/dist_tuto.html#writing-distributed-applications-with-pytorch), Connection closed by peer [127.0.1.1]:2975
# TODO mix indices before distributing data

dist.init_process_group("gloo", rank=rank, world_size=size)


class CustomPartitionedDataset(CustomDataset):
    def __init__(self, problem, reduced_problem, N, input_file_path, output_file_path, input_scaling_range=None, output_scaling_range=None, input_range=None, output_range=None, sizes=None):
        super().__init__(problem, reduced_problem, N, input_file_path, output_file_path,
                         input_scaling_range, output_scaling_range, input_range, output_range)
        if sizes == None:
            world_size = dist.get_world_size()
            sizes = [int(np.rint(1/world_size*len(np.load(input_file_path, allow_pickle=True))))
                     for _ in range(world_size-1)]
            sizes.append(len(np.load(input_file_path, allow_pickle=True))-sum(sizes))
        assert sum(sizes) == len(np.load(input_file_path, allow_pickle=True)
                                 ), "Make sure that the sum of the data in distributed dataset = total number of input set"
        self.sizes = sizes
        rank = dist.get_rank()
        global_indices = np.cumsum(self.sizes)
        if rank == 0:
            local_indices = torch.arange(0, global_indices[rank])
            print(f"Rank {rank}, Local indices {local_indices.tolist()}")
        else:
            local_indices = torch.arange(global_indices[rank-1], global_indices[rank])
            print(f"Rank {rank}, Local indices {local_indices.tolist()}")
        self.local_indices = local_indices

    def __len__(self):
        rank = dist.get_rank()
        return self.sizes[rank]

    def __getitem__(self, idx):
        input_data = np.load(self.input_file_path)[self.local_indices[idx], :]
        # self.reduced_problem.project_snapshot(self.problem.solve(input_data),self.N).array.astype("f")
        label = np.load(self.output_file_path)[self.local_indices[idx], :]
        return self.transform(input_data), self.target_transform(label)


if __name__ == "__main__":

    class Problem(object):
        def __init__(self):
            super().__init__()

    class ReducedProblem(object):
        def __init__(self):
            super().__init__()
            self.input_range = np.vstack((0.5*np.ones([1, 17]), np.ones([1, 17])))
            self.output_range = [0., 1.]
            self.input_scaling_range = [-1., 1.]
            self.output_scaling_range = [-1., 1.]
            self.input_file_path = "input_data.npy"
            self.output_file_path = "input_data.npy"

    problem = Problem()
    reduced_problem = ReducedProblem()
    custom_partitioned_dataset = CustomPartitionedDataset(
        problem, reduced_problem, 10, "input_data.npy", "input_data.npy")
    dataloader = torch.utils.data.DataLoader(custom_partitioned_dataset, batch_size=1, shuffle=False)
    for batch, (X, y) in enumerate(dataloader):
        print(f"Rank: {rank}")
        print(custom_partitioned_dataset.transform(np.load("input_data.npy")
              [custom_partitioned_dataset.local_indices[0], :]))
        print(X)
        exit()
