import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class DataLoader(torch.utils.data.DataLoader):
    """torch.utils.data.DataLoader wrappper"""
    pass


def from_numpy(x, dtype=torch.float32):
    """
    Conversion from numpy to torch Tensor, torch.from_numpy wrapper
    Input:
    x: numpy array
    dtype: Output torch datatype, Default torch.float32
    Output:
    x: torch tensor
    """
    return torch.from_numpy(x).to(dtype)
