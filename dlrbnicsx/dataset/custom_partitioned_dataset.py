import os
import socket

import numpy as np
import torch
import torch.multiprocessing as mp  # noqa: F401
from mpi4py import MPI

from dlrbnicsx.dataset.custom_dataset import CustomDataset
from dlrbnicsx.interface.wrappers import init_cpu_process_group

from mpi4py import MPI

'''
TODO see mp.set_start_method("spawn") to p.join() in the pytorch distributed
tutorial
(pytorch.org/tutorials/intermediate/dist_tuto.html#writing-distributed-applications-with-pytorch)
'''


class CustomPartitionedDataset(CustomDataset):
    def __init__(self, reduced_problem, input_set,
                 output_set, local_indices,
                 input_scaling_range=None, output_scaling_range=None,
                 input_range=None, output_range=None, verbose=True):
        super().__init__(reduced_problem, input_set,
                         output_set, input_scaling_range,
                         output_scaling_range, input_range,
                         output_range, verbose=verbose)
        self.local_indices = local_indices

    def __len__(self):
        return self.local_indices.shape[0]

    def __getitem__(self, idx):
        input_data = self.input_set[self.local_indices[idx], :]
        label = self.output_set[self.local_indices[idx], :]
        return self.transform(input_data), self.target_transform(label)


if __name__ == "__main__":

     class ReducedProblem(object):
        def __init__(self, para_dim):
            super().__init__()
            self.input_range = np.vstack((np.zeros([1, para_dim]),
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
                                        batch_size=2, shuffle=False)

        for batch, (X, y) in enumerate(dataloader):
            print(f"Rank: {comm.rank}")
            print(custom_partitioned_dataset.transform
                (custom_partitioned_dataset.input_set[custom_partitioned_dataset.local_indices[0:2], :]))
            print(X)
            break

        my_first_index = custom_partitioned_dataset.local_indices[0]
        print(f"Rank {comm.rank}, My first index: {my_first_index}")
