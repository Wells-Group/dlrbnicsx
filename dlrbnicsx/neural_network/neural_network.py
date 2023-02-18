import torch
import numpy as np
from dlrbnicsx.activation_function.activation_function_factory import Tanh, Swish, GaussianRBF


class HiddenLayersNet(torch.nn.Module):
    '''
    Inputs:
        dim_in: int, Number of input parameters
        dim_hidden_layers: List with length = number of hidden layers
        and the number of neurons in each hidden layer dim_out: int,
        Number of output parameters
        activation_function: torch.nn.Module, activation function
        return_numpy: bool, Default False. If True the result will be returned in numpy format,
        include_bias: bool, Default True. If True bias will be added in the linear layers.
    '''

    def __init__(self, dim_in, dim_hidden_layers, dim_out, activation_function, return_numpy=False, include_bias=True):
        # Initialisation of class
        super().__init__()
        linear_layers = torch.nn.ModuleList()
        ann_dims = torch.nn.ModuleList()
        ann_dims = dim_hidden_layers
        ann_dims.insert(0, dim_in)
        ann_dims.insert(len(ann_dims), dim_out)
        del dim_hidden_layers
        for i in range(len(ann_dims)-1):
            linear_layers.append(torch.nn.Linear(ann_dims[i], ann_dims[i+1], bias=include_bias))
        self.linear_layers = linear_layers
        self.activation_function = activation_function
        self.return_numpy = return_numpy

    def forward(self, x):
        # Return the result of forward pass
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

# TODO Implement ConvNet based demos and identify common Initialisation and pattern like for HiddenLayersNet


class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(10, 32, 3, 1)  # TODO First argument (index 0) here corresponds to channel size
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        # TODO inplace argument, NOTE trainign is not set here as in the train_nn and validate_nn and online_nn functions relevant model parameters are set
        self.dropout1 = torch.nn.Dropout2d(0.25)
        # NOTE Dropout instead of Dropout2d for flattened data for compatibility in future relases
        self.dropout2 = torch.nn.Dropout(0.50)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = torch.nn.functional.relu(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = torch.nn.functional.relu(x)
        print(f"Before max pool: {x.shape}")
        x = torch.nn.functional.max_pool2d(x, 2)
        print(f"After max pool: {x.shape}")
        x = self.dropout1(x)
        print(x.shape)
        x = torch.flatten(x, 1)
        print(x.shape)
        x = self.fc1(x)
        print(x.shape)
        x = torch.nn.functional.relu(x)
        print(x.shape)
        x = self.dropout2(x)
        print(x.shape)
        x = self.fc2(x)
        print(x.shape)
        output = torch.nn.functional.log_softmax(x, dim=1)
        print(output.shape)
        return output


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

NOTE for CNN:
model = ConvNet()
model.forward(torch.randn(3,10,28,28)) # NOTE input should be in the form (batchsize, channels, height, width), second argument (index 1) here corresponds to channel size and it should match in_channel in ConvNet class initilisation
'''
# NOTE np.float64 or torch.float32 datatype error. See above: astype("f")
