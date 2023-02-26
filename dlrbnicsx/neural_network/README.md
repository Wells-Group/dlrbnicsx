# DLRBniCSx - Neural Network - Neural Network

This module implements ```HiddenLayersNet``` with ```forward``` method for initializing a fully-connected Network. The activation function can be imported from ```activation_function_factory``` module. A four hidden layers network with 4, 22, 8, 90 neurons (4 being number of neurons in first hidden layer from input side) can be initialized as:

```
input_parameter_set = np.ones([16, 7]).astype("f")
model = HiddenLayersNet(input_parameter_set.shape[1], [4, 22, 8, 90], 10, Tanh())
```
