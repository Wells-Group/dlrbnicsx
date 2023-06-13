import os

import numpy as np
import torch  # noqa: F401
import torch.distributed as dist
import torch.multiprocessing as mp  # noqa: F401
from mpi4py import MPI  # noqa: F401

from dlrbnicsx.dataset.custom_dataset import CustomDataset

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
        super.__init__(problem, reduced_problem, N, input_set, output_set,
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
        return self.transform(input_data).to(f"cuda:{self.cuda_rank}"), \
            self.target_tranform(output_data).to(f"cuda:{self.cuda_rank}")


# TODO Set benchmark case
if __name__ == "__main__":
    pass
