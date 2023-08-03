import os
import socket

import torch
import torch.distributed as dist
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

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))

def save_checkpoint(checkpoint_path, epoch, model, optimiser):
    checkpoint = {}
    checkpoint["model_checkpoint"] = model.state_dict()
    checkpoint["optimiser_checkpoint"] = optimiser.state_dict()
    checkpoint["last_epoch"] = epoch
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(checkpoint_path):
    return torch.load(checkpoint_path)

def init_cpu_process_group(comm):
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

    dist.init_process_group("gloo", rank=comm.rank,
                            world_size=comm.size)

def init_gpu_process_group(comm):
    os.environ["MASTER_ADDR"] = "localhost"

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
    dist.init_process_group("nccl", rank=comm.rank,
                            world_size=comm.size)

def save_checkpoint(checkpoint_path, current_epoch, model, optimiser,
                    min_validation_loss):
    checkpoint = {}
    checkpoint["current_epoch"] = current_epoch
    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["optimizer_state_dict"] = optimiser.state_dict()
    checkpoint["min_validation_loss"] = min_validation_loss
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimiser):
    checkpoint = torch.load(checkpoint_path)
    current_epoch = checkpoint["current_epoch"] 
    model.load_state_dict(checkpoint["model_state_dict"])
    optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
    min_validation_loss = checkpoint["min_validation_loss"]
    return current_epoch, min_validation_loss

def get_optimiser(model, optim_name, learning_rate):
    if optim_name == "Adam":
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optim_name == "SGD":
        optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError(f"Optimiser {optim_name} not implemented")
    return optimiser

def get_loss_func(loss_name, reduction="sum"):
    if loss_name == "MSE":
        loss_func = torch.nn.MSELoss(reduction=reduction)
    else:
        raise NotImplementedError(f"Loss function {loss_name} not implemented")
    return loss_func
