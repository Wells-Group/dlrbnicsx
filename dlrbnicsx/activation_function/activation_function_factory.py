import torch
import numpy as np


class SiLU(torch.nn.Module):
    '''
    Example of  activation function WITHOUT learnable parameter and WITHOUT .backward() method
    Applies the Sigmoid Linear Unit (SiLU) activation function
    SiLU(x) = x * sigmoid(x) = x * 1 / ( 1 + exp(-x) )
    Input:
    return_numpy: bool, If return datatype is expected to be numpy array
    (during online phase or errro analysis) set the True, default:
    False, i.e. return datatype is expected to be torch.Tensor (during
    training and valiation)
    (N,) numpy array or torch tensor of dimension N
    Output: (N,) torch tensor of dimension N
    '''

    def __init__(self, return_numpy=False):
        super().__init__()
        self.return_numpy = return_numpy

    def forward(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        assert type(x) == torch.Tensor, "Only numpy array or torch tensor are supported"
        x = x.flatten()
        y = x * torch.sigmoid(x)
        if self.return_numpy == True:
            return y.detach().numpy()
        else:
            return y


class GaussianRBF(torch.nn.Module):
    '''
    Example of  activation function with learnable parameters but WITHOUT .backward() method
    Applies the Gaussian RBF activation function with LEARNABLE shape parameter gamma and centers x_c
    GaussianRBF(x) = torch.exp(-gamma * torch.abs(x-x_c)**2)
    Activation function with non-learnable parameter
    Input:
    alpha = torch.Tensor(scalar), Shape parameter (Learnable)
    return_numpy: bool, If return datatype is expected to be numpy array
    (during online phase or errro analysis) set the True, default:
    False, i.e. return datatype is expected to be torch.Tensor (during
    training and valiation)
    (N,) numpy array or torch tensor of dimension N
    Output: (N,) torch tensor of dimension N
    '''

    def __init__(self, gamma=None, centers=None, return_numpy=False):
        super().__init__()
        if gamma == None:
            self.gamma = torch.nn.Parameter(torch.Tensor([1.]))
            self.centers = centers
        else:
            self.gamma = torch.nn.Parameter(torch.Tensor([alpha]))
        self.gamma.requires_grad = True
        self.return_numpy = return_numpy

    def forward(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        assert type(x) == torch.Tensor, "Only numpy array or torch tensor are supported"
        x = x.flatten()
        if self.centers == None:
            x_c = torch.nn.Parameter(torch.ones_like(x))
        else:
            x_c = self.centers
        x_c.requires_grad = True
        y = torch.exp(-self.gamma * torch.abs(x-x_c)**2)
        if self.return_numpy == True:
            return y.detach().numpy()
        else:
            return y


class Swish(torch.nn.Module):
    '''
    Example of  activation function with learnable parameters but WITHOUT .backward() method
    Applies the Gaussian RBF activation function with LEARNABLE shape parameter gamma and centers x_c
    Swish(x) = 1 / (1 + exp(-beta * x))
    Activation function with non-learnable parameter
    Input:
    beta = torch.Tensor(scalar), Shape parameter (Learnable)
    return_numpy: bool, If return datatype is expected to be numpy array (during online phase or errro analysis) set the True, default: False, i.e. return datatype is expected to be torch.Tensor (during training and valiation)
    (N,) numpy array or torch tensor of dimension N
    Output: (N,) torch tensor of dimension N
    '''

    def __init__(self, beta=None, return_numpy=False):
        super().__init__()
        if beta == None:
            self.beta = torch.nn.Parameter(torch.Tensor([0.1]))
        else:
            self.beta = torch.nn.Parameter(torch.Tensor([beta]))
        self.beta.requires_grad = True
        self.return_numpy = return_numpy

    def forward(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        assert type(x) == torch.Tensor, "Only numpy array or torch tensor are supported"
        x = x.flatten()
        y = x / (torch.Tensor([1.]) + torch.exp(-self.beta*x))
        if self.return_numpy == True:
            return y.detach().numpy()
        else:
            return y


class Tanh(torch.nn.Module):
    def __init__(self, return_numpy=False):
        super().__init__()
        self.return_numpy = return_numpy

    def forward(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        assert type(x) == torch.Tensor, "Only numpy array or torch tensor are supported"
        y = torch.nn.Tanh()(x)
        if self.return_numpy == True:
            return y.detach().numpy()
        else:
            return y


class ReLU(torch.nn.Module):
    def __init__(self, return_numpy=False):
        super().__init__()
        self.return_numpy = return_numpy

    def forward(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        assert type(x) == torch.Tensor, "Only numpy array or torch tensor are supported"
        y = torch.nn.ReLU()(x)
        if self.return_numpy == True:
            return y.detach().numpy()
        else:
            return y


class Sigmoid(torch.nn.Module):
    def __init__(self, return_numpy=False):
        super().__init__()
        self.return_numpy = return_numpy

    def forward(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        assert type(x) == torch.Tensor, "Only numpy array or torch tensor are supported"
        y = torch.nn.Sigmoid()(x)
        if self.return_numpy == True:
            return y.detach().numpy()
        else:
            return y


class Identity(torch.nn.Module):
    def __init__(self, return_numpy=False):
        super().__init__()
        self.return_numpy = return_numpy

    def forward(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        assert type(x) == torch.Tensor, "Only numpy array or torch tensor are supported"
        y = x
        if self.return_numpy == True:
            return y.detach().numpy()
        else:
            return y

# TODO learnable and .backward() activation function


'''x = torch.Tensor([-1.2,0.,3.5])

silu_instance = SiLU(return_numpy=False)

y = silu_instance(x)

print(y)
print(x*torch.sigmoid(x))

gaussianRBF_instance = GaussianRBF(return_numpy=False)
y = gaussianRBF_instance(x)
print(y)
print(torch.exp(torch.Tensor([-1.]) * torch.abs(x-torch.ones_like(x))**2))

swish_instance = Swish(return_numpy=False)
y = swish_instance(x)
print(y)
print(x / (torch.Tensor([1.]) + torch.exp(-torch.Tensor([0.1]) * x)))

tanh_instance = Tanh(return_numpy=False)
y = tanh_instance(x)
print(y)
print(torch.nn.Tanh()(x))

sigmoid_instance = Sigmoid(return_numpy=False)
y = sigmoid_instance(x)
print(y)
print(torch.nn.Sigmoid()(x))'''
