import os
import socket
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp  # noqa: F401
from torch.utils.data import DataLoader
from mpi4py import MPI

from dlrbnicsx.dataset.custom_dataset import CustomDataset
from dlrbnicsx.dataset.custom_partitioned_dataset import CustomPartitionedDataset
from dlrbnicsx.interface.wrappers import init_cpu_process_group
from dlrbnicsx.dataset.custom_partitioned_dataset_gpu import CustomPartitionedDatasetGpu
from dlrbnicsx.interface.wrappers import init_gpu_process_group

class TestDataset(unittest.TestCase):
    """Unit test for Dataset
    """
    def test_dataset_serial(self):
        """Dataset Serial case
        """
        class ReducedProblem(object):
            def __init__(self, para_dim):
                super().__init__()
                self.input_range = np.vstack((np.zeros([1, para_dim]),
                                            np.ones([1, para_dim])))
                self.input_scaling_range = [-1., 1.]
                self.output_range = [0., 1.]
                self.output_scaling_range = [-1., 1.]

        input_data = np.random.uniform(0., 1., [100, 17])
        output_data = np.random.uniform(0., 1.,
                                        [input_data.shape[0], 7])

        reduced_problem = ReducedProblem(input_data.shape[1])

        customDataset = CustomDataset(reduced_problem,
                                    input_data, output_data)

        train_dataloader = DataLoader(customDataset, batch_size=3,
                                    shuffle=True)
        test_dataloader = DataLoader(customDataset, batch_size=2)

        for X, y in train_dataloader:
            print(f"Shape of training set: {X.shape}")
            print(f"Training set requires grad: {X.requires_grad}")
            print(f"X dtype: {X.dtype}")
            # print(f"X: {X}")
            print(f"Shape of training set: {y.shape}")
            print(f"Training set requires grad: {y.requires_grad}")
            print(f"y dtype: {y.dtype}")
            # print(f"y: {y}")
            break

        for X, y in test_dataloader:
            print(f"Shape of test set: {X.shape}")
            print(f"Testing set requires grad: {X.requires_grad}")
            # print(f"X: {X}")
            print(f"Shape of test set: {y.shape}")
            print(f"Testing set requires grad: {y.requires_grad}")
            # print(f"y: {y}")
            break

    def test_dataset_distributed(self):
        """Datset Distributed case
        """
        class ReducedProblem(object):
            def __init__(self, para_dim):
                super().__init__()
                self.input_range = \
                    np.vstack((np.zeros([1, para_dim]),
                                        np.ones([1, para_dim])))
                self.output_range = [0., 1.]
                self.input_scaling_range = [-1., 1.]
                self.output_scaling_range = [-1., 1.]

        comm = MPI.COMM_WORLD
        if comm != MPI.COMM_NULL:
            init_cpu_process_group(comm)

            num_para = 100
            input_para_dim = 17
            output_para_dim = 7

            if comm.rank == 0:
                input_data = np.random.uniform(0., 1.,
                                                [num_para,
                                                input_para_dim])
                output_data = np.random.uniform(0., 1.,
                                                [num_para,
                                                output_para_dim])
            else:
                input_data = np.zeros([num_para, input_para_dim])
                output_data = np.zeros([num_para, output_para_dim])

            comm.Bcast(input_data, root=0)
            comm.Bcast(output_data, root=0)

            indices = np.arange(comm.rank, num_para, comm.size)

            reduced_problem = ReducedProblem(input_para_dim)
            custom_partitioned_dataset = \
                CustomPartitionedDataset(reduced_problem,
                                        input_data,
                                        output_data,
                                        indices, verbose=False)
            dataloader = \
                torch.utils.data.DataLoader(custom_partitioned_dataset,
                                            batch_size=2,
                                            shuffle=False)

            for batch, (X, y) in enumerate(dataloader):
                print(f"Rank: {comm.rank}")
                print(custom_partitioned_dataset.transform
                    (custom_partitioned_dataset.input_set[custom_partitioned_dataset.local_indices[0:2], :]))
                print(X)
                break

            my_first_index = custom_partitioned_dataset.local_indices[0]
            print(f"Rank {comm.rank}, My first index: {my_first_index}")

    """
    def test_dataset_multigpu(self):
        # Dataset MultiGPU
        
        world_comm = MPI.COMM_WORLD
        # gpu_group0_procs = world_comm.group.Incl([0])
        gpu_group0_procs = world_comm.group.Incl([0, 1, 2, 3])
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
        output_data = np.random.uniform(0., 1.,
                                        [input_data.shape[0], 7])

        reduced_problem = ReducedProblem(input_data.shape[1])

        if gpu_group0_comm != MPI.COMM_NULL:
            cuda_rank = [0, 1, 2, 3]
            init_gpu_process_group(gpu_group0_comm)

            local_indices = np.arange(gpu_group0_comm.rank,
                                      input_data.shape[0],
                                      gpu_group0_comm.size)
            customDataset = \
                CustomPartitionedDatasetGpu(reduced_problem,
                                            input_data,
                                            output_data,
                                            local_indices,
                                            cuda_rank[gpu_group0_comm.rank])
            dataloader = DataLoader(customDataset, batch_size=2,
                                    shuffle=False)

            print(f"Original input data: X: \n {input_data[local_indices, :]}, \n Y: {output_data[local_indices, :]}")

            for batch, (X, Y) in enumerate(dataloader):
                print(f"Batch: {batch}, \n X: \n {X}, \n Y: \n {Y}")
    """


if __name__ == "__main__":
    unittest.main()