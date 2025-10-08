import torch
import torch.distributed as dist
import torch.multiprocessing as mp  # noqa: F401

import numpy as np
import rbnicsx.online

from dlrbnicsx.dataset.custom_partitioned_dataset import CustomPartitionedDataset
from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory import Tanh

def train_nn(reduced_problem, dataloader, model, loss_fn,
             optimizer, report=True, verbose=False):
    '''
    Training of the Artificial Neural Network
    Inputs:
        reduced_problem: Reduced problem with attributes:
            loss_fn: loss function name (str)
            optimizer: optimizer name (str)
            lr: learning_rate (float)
        dataloder: dataloder of input dataset
            (dlrbnicsx.dataset.custom_dataset.DataLoader)
        model: Neural network
            (dlrbnicsx.neural_network.neural_network.HiddenLayersNet)
        device: cuda or cpu
    Output:
        loss: float, loss measured with loss_fn using given optimizer
    '''
    # TODO add more loss functions including PINN
    # TODO add more optimizers
    # TODO add L1/L2 and more rgularisations WITHOUT weight decay
    dataset_size = len(dataloader.dataset)
    current_size = 0
    model.train()  # NOTE
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        for param in model.parameters():
            dist.barrier()
            if verbose is True:
                print(f"param before all_reduce: {param.grad.data}")

            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

            if verbose is True:
                print(f"param after all_reduce: {param.grad.data}")

        optimizer.step()
        dist.barrier()
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        optimizer.zero_grad()

        current_size += X.shape[0]
        if report is True and batch % 1 == 0:
            print(f"Loss: {loss.item()} {current_size}/{dataset_size}")
    return loss.item()


def validate_nn(reduced_problem, dataloader, model, loss_fn,
                report=True, verbose=False):
    '''
    Validation of the Artificial Neural Network
    Inputs:
        reduced_problem: Reduced problem with attributes:
            loss_fn: loss function name (str)
        dataloader: dataloader of input dataset
            (dlrbnicsx.dataset.custom_dataset.DataLoader)
        model: Neural network
            (dlrbnicsx.neural_network.neural_network.HiddenLayersNet)
        device: cuda or cpu
    Output:
        loss: float, loss measured with loss_fn using given optimizer
    '''
    # TODO add more loss functions including PINN
    model.eval()  # NOTE
    valid_loss = torch.tensor([0.])
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            valid_loss += loss_fn(pred, y)
    dist.barrier()

    if verbose is True:
        print(f"Validation loss before all_reduce: {valid_loss: >7f}")

    dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)

    if verbose is True:
        print(f"Validation loss after all_reduce: {valid_loss: >7f}")

    if report is True:
        print(f"Validation loss: {valid_loss.item(): >7f}")
    return valid_loss.item()


def online_nn(reduced_problem, problem, online_mu, model, rb_size,
              input_scaling_range=None, output_scaling_range=None,
              input_range=None, output_range=None, verbose=False):
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
    model.eval()

    if type(input_scaling_range) == list:
        input_scaling_range = np.array(input_scaling_range)
    if type(output_scaling_range) == list:
        output_scaling_range = np.array(output_scaling_range)
    if type(input_range) == list:
        input_range = np.array(input_range)
    if type(output_range) == list:
        output_range = np.array(output_range)

    if (np.array(input_scaling_range) == None).any():  # noqa: E711
        assert hasattr(reduced_problem, "input_scaling_range")
        input_scaling_range = reduced_problem.input_scaling_range
    else:
        if verbose is True:
            print(f"Using input scaling range = {input_scaling_range}, " +
                "ignoring input scaling range specified in "
                f"{reduced_problem.__class__.__name__}")
    if (np.array(output_scaling_range) == None).any():  # noqa: E711
        assert hasattr(reduced_problem, "output_scaling_range")
        output_scaling_range = reduced_problem.output_scaling_range
    else:
        if verbose is True:
            print(f"Using output scaling range = {output_scaling_range}, " +
                "ignoring output scaling range specified in " +
                f"{reduced_problem.__class__.__name__}")
    if (np.array(input_range) == None).any():  # noqa: E711
        assert hasattr(reduced_problem, "input_range")
        input_range = reduced_problem.input_range
    else:
        if verbose is True:
            print(f"Using input range = {input_range}, " +
                "ignoring input range specified in " +
                f"{reduced_problem.__class__.__name__}")
    if (np.array(output_range) == None).any():  # noqa: E711
        assert hasattr(reduced_problem, "output_range")
        output_range = reduced_problem.output_range
    else:
        if verbose is True:
            print(f"Using output range = {output_range}, " +
                "ignoring output range specified in " +
                f"{reduced_problem.__class__.__name__}")

    online_mu_scaled = (input_scaling_range[1] - input_scaling_range[0]) * \
                       (online_mu - input_range[0, :]) / \
                       (input_range[1, :] - input_range[0, :]) + \
        input_scaling_range[0]  # TODO Use transform from dataloader
    online_mu_scaled_torch = \
        torch.from_numpy(online_mu_scaled).to(torch.float32)
    with torch.no_grad():
        X = online_mu_scaled_torch
        pred_scaled = model(X)
        pred_scaled_numpy = pred_scaled.detach().numpy()
        pred = (pred_scaled_numpy - output_scaling_range[0]) * \
               (output_range[1] - output_range[0]) / \
               (output_scaling_range[1] - output_scaling_range[0]) + \
            output_range[0]
    # TODO Use reverse_target_transform from dataloader
        solution_reduced = rbnicsx.online.create_vector(rb_size)
        solution_reduced.array = pred
    return solution_reduced


def error_analysis(reduced_problem, problem, error_analysis_mu, model,
                   rb_size, online_nn, norm_error=None,
                   reconstruct_solution=None, input_scaling_range=None,
                   output_scaling_range=None, input_range=None,
                   output_range=None, index=None, verbose=False):
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
        error: float, Error computed with norm_error between FEM
            and RB solution
    '''
    model.eval()
    ann_prediction = online_nn(reduced_problem, problem, error_analysis_mu,
                               model, rb_size, input_scaling_range,
                               output_scaling_range, input_range,
                               output_range)
    if reconstruct_solution is None:
        ann_reconstructed_solution = \
            reduced_problem.reconstruct_solution(ann_prediction)
    else:
        if verbose is True:
            print(f"Using {reconstruct_solution.__name__}, " +
                "ignoring RB to FEM solution construction specified in " +
                f"{reduced_problem.__class__.__name__}")
        ann_reconstructed_solution = reconstruct_solution(ann_prediction)
    fem_solution = problem.solve(error_analysis_mu)
    if type(fem_solution) == tuple:
        assert index is not None
        fem_solution = fem_solution[index]
    if norm_error is None:
        error = reduced_problem.norm_error(fem_solution,
                                           ann_reconstructed_solution)
    else:
        if verbose is True:
            print(f"Using {norm_error.__name__}, " +
                "ignoring error norm specified in " +
                f"{reduced_problem.__class__.__name__}")
        error = norm_error(fem_solution, ann_reconstructed_solution)
    return error


if __name__ == "__main__":

    '''
    NOTE
    Run below code with 2 processes

    ```mpiexec -n 2 python3 train_validate_test_distributed.py```

    and verify from printed terminal output whether the params
    after all_reduce are same in all processes.

    Higher number of processes can also be used instead of only 2.
    '''

    class Problem(object):
        def __init__(self):
            super().__init__()

    class ReducedProblem(object):
        def __init__(self):
            super().__init__()
            self.input_range = np.vstack((0.5*np.ones([1, 4]),
                                          np.ones([1, 4])))
            self.output_range = [0., 1.]
            self.input_scaling_range = [-1., 1.]
            self.output_scaling_range = [-1., 1.]
            self.learning_rate = 1e-4
            self.optimizer = "Adam"
            self.loss_fn = "MSE"

    problem = Problem()
    reduced_problem = ReducedProblem()

    input_training_data = np.random.default_rng().uniform(0., 1.,
                                                          (30, 4)).astype("f")
    output_training_data = \
        np.random.default_rng().uniform(0., 1.,
                                        (input_training_data.shape[0],
                                         6)).astype("f")
    input_training_data = \
        torch.from_numpy(input_training_data).to(torch.float32)
    output_training_data = \
        torch.from_numpy(output_training_data).to(torch.float32)
    dist.barrier()
    dist.all_reduce(input_training_data, op=dist.ReduceOp.MAX)
    dist.all_reduce(output_training_data, op=dist.ReduceOp.MAX)
    input_training_data = input_training_data.detach().numpy()
    output_training_data = output_training_data.detach().numpy()

    # NOTE Updating output_range based on the computed values
    reduced_problem.output_range[0] = np.min(output_training_data)
    reduced_problem.output_range[1] = np.max(output_training_data)

    custom_partitioned_dataset = \
        CustomPartitionedDataset(problem, reduced_problem, 10,
                                 input_training_data, output_training_data)

    train_dataloader = \
        torch.utils.data.DataLoader(custom_partitioned_dataset,
                                    batch_size=100, shuffle=True)

    input_validation_data = \
        np.random.default_rng().uniform(0., 1.,
                                        (3, input_training_data.shape[1])
                                        ).astype("f")
    output_validation_data = \
        np.random.default_rng().uniform(0., 1.,
                                        (input_validation_data.shape[0],
                                         output_training_data.shape[1])
                                        ).astype("f")
    input_validation_data = \
        torch.from_numpy(input_validation_data).to(torch.float32)
    output_validation_data = \
        torch.from_numpy(output_validation_data).to(torch.float32)
    dist.barrier()
    dist.all_reduce(input_validation_data, op=dist.ReduceOp.MAX)
    dist.all_reduce(output_validation_data, op=dist.ReduceOp.MAX)
    input_validation_data = input_validation_data.detach().numpy()
    output_validation_data = output_validation_data.detach().numpy()

    custom_partitioned_dataset = \
        CustomPartitionedDataset(problem, reduced_problem, 10,
                                 input_validation_data,
                                 output_validation_data)
    valid_dataloader = \
        torch.utils.data.DataLoader(custom_partitioned_dataset,
                                    shuffle=False)

    dim_in = input_training_data.shape[1]
    dim_out = output_training_data.shape[1]

    model = HiddenLayersNet(dim_in, [4], dim_out, Tanh())

    for param in model.parameters():
        print(f"Rank: {dist.get_rank()}, " +
              f"Params before all_reduce: {param.data}")
        '''
        NOTE This ensures that models in all processes start with
        same weights and biases
        '''
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        print(f"Rank: {dist.get_rank()}, " +
              f"Params after all_reduce: {param.data}")

    max_epochs = 5  # 20000

    for epoch in range(max_epochs):
        print(f"Rank {dist.get_rank()} Epoch {epoch+1} of Maximum " +
              f"epochs {max_epochs}")
        train_loss = train_nn(reduced_problem, train_dataloader, model)
        valid_loss = validate_nn(reduced_problem, valid_dataloader, model)

    online_mu = \
        np.random.default_rng().uniform(0., 1., input_training_data.shape[1])
    _ = online_nn(reduced_problem, problem, online_mu, model, dim_out)

    '''
    error_analysis_mu = \
        np.random.default_rng().uniform(0., 1.,
                                        (30, input_training_data.shape[1]))
    for i in range(error_analysis_mu.shape[0]):
        error = error_analysis(reduced_problem, problem,
                               error_analysis_mu[i,:], model,
                               dim_out, online_nn)
    '''
    # TODO Dummy problem for error analysis
