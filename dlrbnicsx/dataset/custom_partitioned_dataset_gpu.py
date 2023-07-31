import os
import socket

import numpy as np
from mpi4py import MPI  # noqa: F401

from dlrbnicsx.dataset.custom_partitioned_dataset import CustomPartitionedDataset
from dlrbnicsx.interface.wrappers import init_gpu_process_group

import torch  # noqa: F401
import torch.distributed as dist
import torch.multiprocessing as mp  # noqa: F401
from torch.utils.data import DataLoader

class CustomPartitionedDatasetGpu(CustomPartitionedDataset):
    def __init__(self, reduced_problem, input_set,
                 output_set, local_indices, cuda_rank,
                 input_scaling_range=None, output_scaling_range=None,
                 input_range=None, output_range=None, verbose=True):
        '''
            local_indices: np.array 1-D of indices of the dataset
            cuda_rank: (Scalar) GPU rank
            Other arguments as per CustomDataset
            (See CustomDataset in dlrbnicsx.dataset.custom_dataset)
        '''

        self.local_indices = local_indices
        self.cuda_rank = cuda_rank
        self.verbose = verbose

        input_set = torch.from_numpy(input_set).to(f"cuda:{cuda_rank}")
        output_set = torch.from_numpy(output_set).to(f"cuda:{cuda_rank}")


        if type(input_scaling_range) == list:
            input_scaling_range = np.array(input_scaling_range)
        if type(output_scaling_range) == list:
            output_scaling_range = np.array(output_scaling_range)
        if type(input_range) == list:
            input_range = np.array(input_range)
        if type(output_range) == list:
            output_range = np.array(output_range)


        if (np.array(input_scaling_range) == None).any():  # noqa: E711
            if type(reduced_problem.input_scaling_range) == list:
                input_scaling_range = np.array(reduced_problem.input_scaling_range)
            else:
                input_scaling_range = reduced_problem.input_scaling_range

            input_scaling_range = torch.from_numpy(input_scaling_range)#.to(torch.float32)
            input_scaling_range = input_scaling_range.to(f"cuda:{cuda_rank}")

        if (np.array(output_scaling_range) == None).any():  # noqa: E711
            if type(reduced_problem.output_scaling_range) == list:
                output_scaling_range = np.array(reduced_problem.output_scaling_range)
            else:
                output_scaling_range = reduced_problem.output_scaling_range

        output_scaling_range = torch.from_numpy(output_scaling_range)#.to(torch.float32)
        output_scaling_range = output_scaling_range.to(f"cuda:{cuda_rank}")


        if (np.array(input_range) == None).any():  # noqa: E711
            if type(reduced_problem.input_range) == list:
                input_range = np.array(reduced_problem.input_range)
            else:
                input_range = reduced_problem.input_range

        input_range = torch.from_numpy(input_range)#.to(torch.float32)
        input_range = input_range.to(f"cuda:{cuda_rank}")


        if (np.array(output_range) == None).any():  # noqa: E711
            if type(reduced_problem.output_range) == list:
                output_range = np.array(reduced_problem.output_range)
            else:
                output_range = reduced_problem.output_range

        output_range = torch.from_numpy(output_range)#.to(torch.float32)
        output_range = output_range.to(f"cuda:{cuda_rank}")

        super().__init__(reduced_problem, input_set, output_set,
                         local_indices, input_scaling_range,
                         output_scaling_range, input_range,
                         output_range, verbose=verbose)


if __name__ == "__main__":

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

    input_data = np.random.uniform(0., 1., [10, 5])
    output_data = np.random.uniform(0., 1., [input_data.shape[0], 7])

    reduced_problem = ReducedProblem(input_data.shape[1])

    if gpu_group0_comm != MPI.COMM_NULL:
        cuda_rank = [0, 1, 2, 3]
        init_gpu_process_group(gpu_group0_comm)

        local_indices = np.arange(gpu_group0_comm.rank, input_data.shape[0],
                                  gpu_group0_comm.size)
        customDataset = \
            CustomPartitionedDatasetGpu(reduced_problem,
                                        input_data, output_data,
                                        local_indices, cuda_rank[gpu_group0_comm.rank])
        dataloader = DataLoader(customDataset, batch_size=2, shuffle=False)

        print(f"Original input data: X: \n {input_data[local_indices, :]}, \n Y: {output_data[local_indices, :]}")

        for batch, (X, Y) in enumerate(dataloader):
            print(f"Batch: {batch}, \n X: \n {X}, \n Y: \n {Y}")
            exit()
