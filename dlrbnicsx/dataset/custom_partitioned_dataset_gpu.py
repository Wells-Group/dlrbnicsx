import os

import numpy as np
from mpi4py import MPI  # noqa: F401

from dlrbnicsx.dataset.custom_dataset import CustomDataset

import torch  # noqa: F401
import torch.distributed as dist
import torch.multiprocessing as mp  # noqa: F401
from torch.utils.data import DataLoader

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"


def init_gpu_process_group(comm):
    dist.init_process_group("nccl", rank=comm.rank, size=comm.size)


class CustomPartitionedDatasetGpu(CustomDataset):
    def __init__(self, problem, reduced_problem, N,
                 input_set, output_set, local_indices, cuda_rank,
                 input_scaling_range=None, output_scaling_range=None,
                 input_range=None, output_range=None):
        '''
            local_indices: np.array 1-D of indices of the dataset
            cuda_rank: (Scalar) GPU rank
            Other arguments as per CustomDataset
            (See CustomDataset in dlrbnicsx.dataset.custom_dataset)
        '''
        super().__init__(problem, reduced_problem, N, input_set, output_set,
                         input_scaling_range, output_scaling_range,
                         input_range, output_range)
        self.local_indices = local_indices
        self.cuda_rank = cuda_rank

    def __len__(self):
        input_length = self.local_indices.shape[0]
        return input_length

    def __getitem__(self, idx):
        if isinstance(self.input_set, str):
            input_data = \
                np.load(self.input_set)[self.local_indices[idx], :]
        else:
            input_data = self.input_set[self.local_indices[idx], :]
        if isinstance(self.output_set, str):
            output_data = \
                np.load(self.output_set)[self.local_indices[idx], :]
        else:
            output_data = self.output_set[self.local_indices[idx], :]
        print("transferring data to cuda")
        return self.transform(input_data).to(f"cuda:{self.cuda_rank}"), \
            self.target_transform(output_data).to(f"cuda:{self.cuda_rank}")


# TODO Verify below benchmark case
# TODO Update customDataset for transferring data to GPU once only.
# Currently the data is trabnsferred in every __getitem__ .

if __name__ == "__main__":

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    world_comm = MPI.COMM_WORLD
    gpu_group0_procs = world_comm.group.Incl([0])
    # world_comm.group.Incl([0, 1, 2, 3])
    gpu_group0_comm = world_comm.Create_group(gpu_group0_procs)

    class ReducedProblem(object):
        def __init__(self, para_dim):
            super().__init__()
            self.input_range = np.vstack((np.zeros([1, para_dim]),
                                          np.ones([1, para_dim])))
            self.input_scaling_range = [-1., 1.]
            self.output_range = [0., 1.]
            self.output_scaling_range = [-1., 1.]

    class Problem(object):
        def __init__(self):
            super().__init__()

    input_data = np.random.uniform(0., 1., [10, 5])
    output_data = np.random.uniform(0., 1., [input_data.shape[0], 7])

    problem = Problem()
    reduced_problem = ReducedProblem(input_data.shape[1])
    N = 5

    if gpu_group0_comm != MPI.COMM_NULL:
        cuda_rank = gpu_group0_comm.rank
        dist.init_process_group("nccl", rank=gpu_group0_comm.rank,
                                world_size=gpu_group0_comm.size)

        local_indices = np.arange(gpu_group0_comm.rank, input_data.shape[0],
                                  gpu_group0_comm.size)
        customDataset = \
            CustomPartitionedDatasetGpu(problem, reduced_problem, N,
                                        input_data, output_data,
                                        local_indices, cuda_rank)
        dataloader = DataLoader(customDataset, batch_size=2, shuffle=False)

        print(f"Original input data: X: \n {input_data[local_indices, :]}, \n Y: {output_data[local_indices, :]}")

        for batch, (X, Y) in enumerate(dataloader):
            print(f"Batch: {batch}, \n X: \n {X}, \n Y: \n {Y}")
