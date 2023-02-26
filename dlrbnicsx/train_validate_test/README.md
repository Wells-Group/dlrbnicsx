# DLRBniCSx - Traing Validate Test - Train Validate Test

This module implements methods for training and validation of neural network as well as online phase and error analysis of the reduced basis method.

```train_nn```: It is a method for training of the neural network.

* ```train_nn(reduced_problem, dataloader, model, device=None, learning_rate=None, loss_func=None, optimizer=None)```
```learning_rate```, ```loss_func```, ```optimizer``` can be specified in ```reduced_problem```. If specified separately, the values specified in ```reduced_problem``` are ignored.

```validate_nn```: It is a method for validation of the neural network.

* ```validate_nn(reduced_problem, dataloader, model, device=None, loss_func=None)```
```loss_func``` can be specified in reduced_problem. If specified separately, the values specified in ```reduced_problem``` are ignored.

```online_nn```: It is a method for forward pass of the neural network for given ```online_mu```. It is used during online phase and error analysis.

* ```online_nn(reduced_problem, problem, online_mu, model, N, device=None, input_scaling_range=None, output_scaling_range=None, input_range=None, output_range=None)```
```input_scaling_range=None```, ```output_scaling_range=None```, ```input_range=None```, ```output_range=None``` can be specified in ```reduced_problem```. If specified separately, the values specified in ```reduced_problem``` are ignored.

```error_analysis```: It is a method for performing for measuring error between full order model solution and reduced basis solution.
* ```error_analysis(reduced_problem, problem, error_analysis_mu, model, N, online_nn, device=None, norm_error=None, reconstruct_solution=None, input_scaling_range=None,               output_scaling_range=None, input_range=None, output_range=None, index=None)```

```reduced_problem``` requires method for ```norm_error```, ```reconstruct_solution```,  ```input_scaling_range=None```, ```output_scaling_range=None```, ```input_range=None```, ```output_range=None```. If specified separately, the values specified in ```reduced_problem``` are ignored. The ```index``` argument specifies the solution on which to perform error analysis. ```problem``` require method for ```solve```. For example, in the case of Stokes's problem, if the return fields are arranged as ```(velocity, pressure)``` from ```solve``` method of the ```problem```, the ```index=0``` performs error measurement on ```velocity``` field and ```index=1``` performs error measurement on ```pressure``` field
