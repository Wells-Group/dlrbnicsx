import torch

class DataLoader(torch.utils.data.DataLoader):
    '''
    Wrapper around DataLoader from torch.utils.data
    '''
    pass

def from_numpy(x, dtype=torch.float32):
    '''
    Conversion from numpy to torch Tensor
    Input:
    x: numpy array
    dtype: Output torch datatype, Default torch.float32
    Output:
    x: torch tensor
    '''
    return torch.from_numpy(x).to(dtype)
