# DLRBniCSx - Dataset - Neural Network - Neural Network

This module implements ```HiddenLayersNet``` with ```forward``` method for initializing a fully-connected Network. The activation function can be imported from ```activation_function_factory``` module. A four hidden layers network with 4, 22, 8, 90 hidden layers can be initialised as:

```
input_parameter_set = np.ones([16, 7]).astype("f")
model = HiddenLayersNet(input_parameter_set.shape[1], [4, 22, 8, 90], 10, Tanh())
```