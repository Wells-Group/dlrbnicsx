import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import numpy as np
import rbnicsx.online

from dlrbnicsx.dataset.custom_partitioned_dataset import CustomPartitionedDataset
from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory import Tanh, Swish, GaussianRBF


def train_nn(reduced_problem, dataloader, model, device=None, learning_rate=None, loss_func=None, optimizer=None):
    '''
    Training of the Artificial Neural Network
    Inputs:
        reduced_problem: Reduced problem with attributes:
            loss_fn: loss function name (str)
            optimizer: optimizer name (str)
            lr: learning_rate (float)
        dataloder: dataloder of input dataset (dlrbnicsx.dataset.custom_dataset.DataLoader)
        model: Neural network (dlrbnicsx.neural_network.neural_network.HiddenLayersNet)
        device: cuda or cpu
    Output:
        loss: float, loss measured with loss_fn using given optimizer
    '''
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # TODO add more loss functions including PINN
    if loss_func == None:
        if reduced_problem.loss_fn == "MSE":
            loss_fn = torch.nn.MSELoss()  # TODO also add reduction argument
        else:
            NotImplementedError(f"Loss function {reduced_problem.loss_fn} is not implemented")
    else:
        print(
            f"Using training loss function = {loss_func}, ignoring loss function specified in {reduced_problem.__class__.__name__}")
        if loss_func == "MSE":
            loss_fn = torch.nn.MSELoss()  # TODO also add reduction argument
        else:
            NotImplementedError(f"Loss function {loss_fn} is not implemented")
    # TODO add more optimizers
    if learning_rate == None:
        lr = reduced_problem.learning_rate
    else:
        print(
            f"Using learning_rate = {learning_rate}, ignoring learning rate specified in {reduced_problem.__class__.__name__}")
        lr = learning_rate
    if optimizer == None:
        if reduced_problem.optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif reduced_problem.optimizer == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # TODO also add momentum argument
        else:
            NotImplementedError(f"Optimizer {reduced_problem.optimizer} is not implemented")
    else:
        print(f"Using optimizer = {optimizer}, ignoring optimizer specified in {reduced_problem.__class__.__name__}")
        if optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # TODO also add momentum argument
        else:
            NotImplementedError(f"Optimizer {optimizer} is not implemented")
    # TODO add L1/L2 and more rgularisations WITHOUT weight decay
    dataset_size = len(dataloader.dataset)
    model.train()  # NOTE
    for batch, (X, y) in enumerate(dataloader):
        # X,y = X.to(device), y.to(device) # TODO
        pred = model(X)
        loss = loss_fn(pred, y)/loss_fn(torch.zeros_like(y), y)

        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            dist.barrier()
            # print(f"param before all_reduce: {param.grad.data}")
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            # print(f"param after all_reduce: {param.grad.data}")
        optimizer.step()
        # TODO If loss reduction argument="SUM". If "MEAN", change divide the loss by dist.get_world_size()
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)

        if batch % 1 == 0:
            current = (batch+1) * len(X)
            print(f"Training loss: {loss.item(): >7f} [{current:>5d}]/[{dataset_size:>5d}]")
    return loss.item()


def validate_nn(reduced_problem, dataloader, model, device=None, loss_func=None):
    '''
    Validation of the Artificial Neural Network
    Inputs:
        reduced_problem: Reduced problem with attributes:
            loss_fn: loss function name (str)
        dataloder: dataloder of input dataset (dlrbnicsx.dataset.custom_dataset.DataLoader)
        model: Neural network (dlrbnicsx.neural_network.neural_network.HiddenLayersNet)
        device: cuda or cpu
    Output:
        loss: float, loss measured with loss_fn using given optimizer
    '''
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # TODO add more loss functions including PINN
    if loss_func == None:
        if reduced_problem.loss_fn == "MSE":
            loss_fn = torch.nn.MSELoss()  # TODO also add reduction argument
        else:
            NotImplementedError(f"Loss function {reduced_problem.loss_fn} is not implemented")
    else:
        print(
            f"Using validation loss function = {loss_func}, ignoring loss function specified in {reduced_problem.__class__.__name__}")
        if loss_func == "MSE":
            loss_fn = torch.nn.MSELoss()  # TODO also add reduction argument
        else:
            NotImplementedError(f"Loss function {loss_func} is not implemented")
    num_batches = len(dataloader)
    model.eval()  # NOTE
    valid_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            # X,y = X.to(device), y.to(device) # TODO
            pred = model(X)
            valid_loss += loss_fn(pred, y)/loss_fn(torch.zeros_like(y), y)
    # print(f"Validation loss before all_reduce: {valid_loss: >7f}")
    dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
    # print(f"Validation loss after all_reduce: {valid_loss: >7f}")
    print(f"Validation loss: {valid_loss.item(): >7f}")
    return valid_loss.item()


def online_nn(reduced_problem, problem, online_mu, model, N, device=None, input_scaling_range=None, output_scaling_range=None, input_range=None, output_range=None):
    '''
    Online phase
    Inputs:
        online_mu: np.ndarray [1,num_para] representing online parameter
        reduced_problem: reduced problem with attributes:
            input_scaling_range: (2,num_para) np.ndarray, row 0 are the
            SCALED INPUT min_values and row 1 are the SCALED INPUT
            max_values
            output_scaling_range: (2,num_para) np.ndarray, row 0 are the
            SCALED OUTPUT min_values and row 1 are the SCALED OUTPUT
            max_values
            input_range: (2,num_para) np.ndarray, row 0 are the ACTUAL
            INPUT min_values and row 1 are the ACTUAL INPUT max_values
            output_range: (2,num_para) np.ndarray, row 0 are the ACTUAL
            OUTPUT min_values and row 1 are the ACTUAL OUTPUT max_values
    Output:
        solution_reduced: rbnicsx.online.create_vector, online solution
    '''
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    if type(input_scaling_range) == list:
        input_scaling_range = np.array(input_scaling_range)
    if type(output_scaling_range) == list:
        output_scaling_range = np.array(output_scaling_range)
    if type(input_range) == list:
        input_range = np.array(input_range)
    if type(output_range) == list:
        output_range = np.array(output_range)

    if (np.array(input_scaling_range) == None).any():
        assert hasattr(reduced_problem, "input_scaling_range")
        input_scaling_range = reduced_problem.input_scaling_range
    else:
        print(
            f"Using input scaling range = {input_scaling_range}, ignoring input scaling range specified in {reduced_problem.__class__.__name__}")
    if (np.array(output_scaling_range) == None).any():
        assert hasattr(reduced_problem, "output_scaling_range")
        output_scaling_range = reduced_problem.output_scaling_range
    else:
        print(
            f"Using output scaling range = {output_scaling_range}, ignoring output scaling range specified in {reduced_problem.__class__.__name__}")
    if (np.array(input_range) == None).any():
        assert hasattr(reduced_problem, "input_range")
        input_range = reduced_problem.input_range
    else:
        print(
            f"Using input range = {input_range}, ignoring input range specified in {reduced_problem.__class__.__name__}")
    if (np.array(output_range) == None).any():
        assert hasattr(reduced_problem, "output_range")
        output_range = reduced_problem.output_range
    else:
        print(
            f"Using output range = {output_range}, ignoring output range specified in {reduced_problem.__class__.__name__}")

    online_mu_scaled = (input_scaling_range[1] - input_scaling_range[0]) * (online_mu - input_range[0, :]) / (
        input_range[1, :] - input_range[0, :]) + input_scaling_range[0]  # TODO Use transform from dataloader
    online_mu_scaled_torch = torch.from_numpy(online_mu_scaled).to(torch.float32)
    with torch.no_grad():
        X = online_mu_scaled_torch
        # X = X.to(device) # TODO
        pred_scaled = model(X)
        pred_scaled_numpy = pred_scaled.detach().numpy()
        pred = (pred_scaled_numpy - output_scaling_range[0]) * (output_range[1] - output_range[0]) / (
            output_scaling_range[1] - output_scaling_range[0]) + output_range[0]  # TODO Use reverse_target_transform from dataloader
        solution_reduced = rbnicsx.online.create_vector(N)
        solution_reduced.array = pred
    return solution_reduced


def error_analysis(reduced_problem, problem, error_analysis_mu, model, N, online_nn, device=None, norm_error=None, reconstruct_solution=None, input_scaling_range=None, output_scaling_range=None, input_range=None, output_range=None, index=None):
    '''
    Inputs:
        error_analysis_mu: np.ndarray of size [1,num_para] representing
        parameter set at which error analysis needs to be evaluated
        problem: full order model with method:
            norm_error(fem_solution,ann_reconstructed_solution) and
            methods required for online_nn (Used ONLY if norm_error is
            not specified) solve: method to compute full order model
            solution
        reduced_problem: reduced problem with attributes:
            reconstruct_solution: Reconstruct FEM solution from reduced
            basis solution
            input_scaling_range: (2,num_para) np.ndarray, row 0 are the
            SCALED INPUT min_values and row 1 are the SCALED INPUT
            max_values
            output_scaling_range: (2,num_para) np.ndarray, row 0 are the
            SCALED OUTPUT min_values and row 1 are the SCALED OUTPUT
            max_values
            input_range: (2,num_para) np.ndarray, row 0 are the ACTUAL
            INPUT min_values and row 1 are the ACTUAL INPUT max_values
            output_range: (2,num_para) np.ndarray, row 0 are the ACTUAL
            OUTPUT min_values and row 1 are the ACTUAL OUTPUT max_values
    Outputs:
        error: float, Error computed with norm_error between FEM and RB solution
    '''
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    ann_prediction = online_nn(reduced_problem, problem, error_analysis_mu, model, N, device,
                               input_scaling_range, output_scaling_range, input_range, output_range)
    if reconstruct_solution == None:
        ann_reconstructed_solution = reduced_problem.reconstruct_solution(ann_prediction)
    else:
        print(f"Using {reconstruct_solution.__name__}, ignoring RB to FEM solution construction specified in {reduced_problem.__class__.__name__}")
        ann_reconstructed_solution = reconstruct_solution(ann_prediction)
    fem_solution = problem.solve(error_analysis_mu)
    if type(fem_solution) == tuple:
        assert index != None
        fem_solution = fem_solution[index]
    if norm_error == None:
        error = reduced_problem.norm_error(fem_solution, ann_reconstructed_solution)
    else:
        print(f"Using {norm_error.__name__}, ignoring error norm specified in {reduced_problem.__class__.__name__}")
        error = norm_error(fem_solution, ann_reconstructed_solution)
    return error


if __name__ == "__main__":

    class Problem(object):
        def __init__(self):
            super().__init__()

    class ReducedProblem(object):
        def __init__(self):
            super().__init__()
            self.input_range = np.vstack((0.5*np.ones([1, 2]), np.ones([1, 2])))
            self.output_range = [0., 1.]
            self.input_scaling_range = [-1., 1.]
            self.output_scaling_range = [-1., 1.]
            self.learning_rate = 1e-4
            self.optimizer = "Adam"
            self.loss_fn = "MSE"

    problem = Problem()
    reduced_problem = ReducedProblem()
    # NOTE Updating output_range based on the computed values instead of user guess.
    reduced_problem.output_range[0], reduced_problem.output_range[1] = np.min(
        np.load("ann_data/output_training_data.npy")), np.max(np.load("ann_data/output_training_data.npy"))
    custom_partitioned_dataset = CustomPartitionedDataset(
        problem, reduced_problem, 10, "ann_data/input_training_data.npy", "ann_data/output_training_data.npy")
    train_dataloader = torch.utils.data.DataLoader(custom_partitioned_dataset, batch_size=100, shuffle=True)
    custom_partitioned_dataset = CustomPartitionedDataset(
        problem, reduced_problem, 10, "ann_data/input_validation_data.npy", "ann_data/output_validation_data.npy")
    valid_dataloader = torch.utils.data.DataLoader(custom_partitioned_dataset, batch_size=100, shuffle=False)
    dim_in, dim_out = np.load(
        "ann_data/input_training_data.npy").shape[1], np.load("ann_data/output_training_data.npy").shape[1]
    model = HiddenLayersNet(dim_in, [4], dim_out, Tanh())
    for param in model.parameters():
        # print(f"Params before all_reduce: {param.data}")
        # NOTE This ensures that models in all processes start with same weights and biases
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        # print(f"Params after all_reduce: {param.data}")
    max_epochs = 5  # 20000
    for epoch in range(max_epochs):
        print(f"Rank {dist.get_rank()} Epoch {epoch+1} of Maximum epochs {max_epochs}")
        train_loss = train_nn(reduced_problem, train_dataloader, model)
        valid_loss = validate_nn(reduced_problem, valid_dataloader, model)
    online_nn(reduced_problem, problem, np.array([0.2, 0.8]), model, dim_out)
    # error = error_analysis(reduced_problem, problem, error_analysis_mu, model, dim_out, online_nn) # NOTE reduced_problem requires reconstruct_solution, norm_error method. problem requires solve method
