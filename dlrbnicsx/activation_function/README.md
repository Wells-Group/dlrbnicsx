# DLRBniCSx - Activation Function - Activation Function Factory

This module implements different activation functions with learnbale parameters. Currently, following activation functions are supported:

* SiLU (Sigmoid Linear Unit)
* GaussianRBF (Gaussian Radial Basis Function)
* Swish (Swish function)
* Tanh (Hyperbolic tangent)
* ReLU (Rectified Linear Unit)
* Sigmoid (Sigmoid function)
* Identity (Identity function)

The activation function is applied to a given torch tensor or numpy array, which can be used in neural network module. Consider below function for applying SiLU activation function:
```
x = np.array([2.2, 3.1, 4.3, 5.2])
print(f"Predicted value: {SiLU()(x)}")
```
