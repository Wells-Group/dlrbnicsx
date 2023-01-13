import torch
import numpy as np
from dlrbnicsx.activation_function.activation_function_factory import Tanh, Swish, GaussianRBF

class HiddenLayersNet(torch.nn.Module):
    '''
    Inputs:
        dim_in: int, Number of input parameters
        dim_hidden_layers: List with length = number of hidden layers and the number of neurons in each hidden layer
        dim_out: int, Number of output parameters
        activation_function: torch.nn.Module, activation function
        return_numpy: bool, Default False. If True the result will be returned in numpy format, 
        include_bias: bool, Default True. If True bias will be added in the linear layers.
    '''
    def __init__(self, dim_in, dim_hidden_layers, dim_out, activation_function, return_numpy=False, include_bias=True):
        #Initialisation of class
        super().__init__()
        linear_layers = torch.nn.ModuleList()
        ann_dims = torch.nn.ModuleList()
        ann_dims = dim_hidden_layers
        ann_dims.insert(0,dim_in)
        ann_dims.insert(len(ann_dims),dim_out)
        del dim_hidden_layers
        for i in range(len(ann_dims)-1):
            linear_layers.append(torch.nn.Linear(ann_dims[i],ann_dims[i+1], bias=include_bias))
        self.linear_layers = linear_layers
        self.activation_function = activation_function
        self.return_numpy = return_numpy
    
    def forward(self, x):
        #Return the result of forward pass
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).to(torch.float32)
        result = x
        linear_layers = self.linear_layers
        for i in range(len(linear_layers)-1):
            result = self.activation_function(self.linear_layers[i](result))
        result = self.linear_layers[len(linear_layers)-1](result)
        if self.return_numpy == True:
            result = result.detach().numpy()
        return result

'''
input_parameter_set = np.ones([16,7]).astype("f")

model = HiddenLayersNet(input_parameter_set.shape[1], [4,22,8,90], 10, Tanh())
model(input_parameter_set)
print(type(model(input_parameter_set)))

model = HiddenLayersNet(input_parameter_set.shape[1], [4,22,8,90], 10, Tanh(), return_numpy=True)
print(type(model(input_parameter_set)))

for param in model.parameters():
    print(param.data.dtype)
print(input_parameter_set.dtype)
'''
# NOTE np.float64 or torch.float32 datatype error. See above: astype("f")
