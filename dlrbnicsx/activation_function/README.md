# DLRBniCSx - Activation Function - Activation Function Factory

This module implements different activation functions with learnable parameters ($\gamma$, $\beta$). Currently, following activation functions are supported for given torch tensor or numpy array $x$:

* SiLU (Sigmoid Linear Unit): $x * 1 / ( 1 + exp(-x))$
* GaussianRBF (Gaussian Radial Basis Function): $exp(-\gamma * |x-x_c|**2)$
* Swish (Swish function): $x / (1 + exp(-\beta * x))$
* Tanh (Hyperbolic tangent): $(exp(x) - exp(-x)) / (exp(x) + exp(-x))$
* ReLU (Rectified Linear Unit): $max(0, x)$
* Sigmoid (Sigmoid function): $1 / ( 1 + exp(-x))$
* Identity (Identity function): $x$

The activation function is applied to a given torch tensor or numpy array, which can be used in neural network module. Consider below function for applying SiLU activation function:
```
x = np.array([2.2, 3.1, 4.3, 5.2])
print(f"Predicted value: {SiLU()(x)}")
Predicted value: tensor([1.9805, 2.9664, 4.2424, 5.1715], dtype=torch.float64)
```
