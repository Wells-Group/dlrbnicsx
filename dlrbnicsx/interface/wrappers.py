import torch
import torch.distributed as dist  # noqa: F401
import torch.multiprocessing as mp  # noqa: F401


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


def model_to_gpu(model, cuda_rank=0):
    model.to(f"cuda:{cuda_rank}")


def model_to_cpu(model):
    model.to("cpu")


def data_to_gpu(data, cuda_rank):
    data_on_gpu = from_numpy(data).to(f"cuda:{cuda_rank}")
    return data_on_gpu


def data_to_cpu(data):
    data_on_cpu = data.to("cpu")
    return data_on_cpu


def model_synchronise(model, verbose=False):
    for param in model.parameters():
        dist.barrier()
        if verbose is True:
            print(f"Params before synchronisation: {param.data}")
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= dist.get_world_size()
        if verbose is True:
            print(f"Params after synchronisation: {param.data}")
