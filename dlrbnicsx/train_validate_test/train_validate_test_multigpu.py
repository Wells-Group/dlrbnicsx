import torch
import torch.distributed as dist
import torch.multiprocessing as mp  # noqa: F401

import numpy as np
from mpi4py import MPI

import rbnicsx.online

from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory \
    import Tanh


def train_nn(reduced_problem, dataloader, model, device=None,
             learning_rate=None, loss_func=None, optimizer=None):
    # TODO add more loss functions including PINN
    if loss_func is None:
        if reduced_problem.loss_fn == "MSE":
            loss_fn = torch.nn.MSELoss(reduction="sum")
        else:
            NotImplementedError(f"Loss function {reduced_problem.loss_fn}" +
                                "is not implemented")
    else:
        print(f"Using training loss function = {loss_func}," +
              "ignoring loss function specified in " +
              f"{reduced_problem.__class__.__name__}")
        if loss_func == "MSE":
            loss_fn = torch.nn.MSELoss(reduction="sum")
        else:
            NotImplementedError(f"Loss function {loss_fn} " +
                                "is not implemented")
    # TODO add more optimizers
    if learning_rate is None:
        lr = reduced_problem.learning_rate
    else:
        print(f"Using learning_rate = {learning_rate}," +
              "ignoring learning rate specified in " +
              f"{reduced_problem.__class__.__name__}")
        lr = learning_rate
    if optimizer is None:
        if reduced_problem.optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif reduced_problem.optimizer == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            # TODO also add momentum argument
        else:
            NotImplementedError(f"Optimizer {reduced_problem.optimizer} " +
                                "is not implemented")
    else:
        print(f"Using optimizer = {optimizer}, " +
              "ignoring optimizer specified in " +
              f"{reduced_problem.__class__.__name__}")
        if optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            # TODO also add momentum argument
        else:
            NotImplementedError(f"Optimizer {optimizer} " +
                                "is not implemented")

    # TODO add L1/L2 and more rgularisations WITHOUT weight decay
    dataset_size = len(dataloader.dataset)
    current_size = 0
    model.train()  # NOTE
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            dist.barrier()
            print(f"param before all_reduce: {param.grad.data}")
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            print(f"param after all_reduce: {param.grad.data}")
        optimizer.step()
        dist.barrier()
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)

        current_size += X.shape[0]
        if batch % 1 == 0:
            print(f"Loss: {loss.item()} {current_size}/{dataset_size}")
    return loss.item()


def validate_nn(reduced_problem, dataloader, model, loss_func=None):
    # TODO add more loss functions including PINN
    if loss_func is None:
        if reduced_problem.loss_fn == "MSE":
            loss_fn = torch.nn.MSELoss(reduction="sum")
        else:
            NotImplementedError(f"Loss function {reduced_problem.loss_fn} " +
                                "is not implemented")
    else:
        print(f"Using validation loss function = {loss_func}," +
              "ignoring loss function specified in " +
              f"{reduced_problem.__class__.__name__}")
        if loss_func == "MSE":
            loss_fn = torch.nn.MSELoss(reduction="sum")
        else:
            NotImplementedError(f"Loss function {loss_func} " +
                                "is not implemented")
    # dataset_size = len(dataloader.dataset)
    model.eval()  # NOTE
    valid_loss = torch.tensor([0.])
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            valid_loss += loss_fn(pred, y)
    dist.barrier()
    print(f"Validation loss before all_reduce: {valid_loss: >7f}")
    dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
    print(f"Validation loss after all_reduce: {valid_loss: >7f}")
    print(f"Validation loss: {valid_loss.item(): >7f}")
    return valid_loss.item()


def online_nn(reduced_problem, problem, online_mu, model, N, cuda_rank,
              input_scaling_range=None, output_scaling_range=None,
              input_range=None, output_range=None, error_mode=False):
    '''
    NOTE Addition of error_mode (default False): so that error_anaysis_set
    transferred to cpu and solution_reduced to gpu only once
    also NOTE solution_reduced is array of reduced dofs as compared
    to reduced solution if error_mode=False
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
        print(f"Using input scaling range = {input_scaling_range}, " +
              "ignoring input scaling range specified in "
              f"{reduced_problem.__class__.__name__}")
    if (np.array(output_scaling_range) == None).any():  # noqa: E711
        assert hasattr(reduced_problem, "output_scaling_range")
        output_scaling_range = reduced_problem.output_scaling_range
    else:
        print(f"Using output scaling range = {output_scaling_range}, " +
              "ignoring output scaling range specified in " +
              f"{reduced_problem.__class__.__name__}")
    if (np.array(input_range) == None).any():  # noqa: E711
        assert hasattr(reduced_problem, "input_range")
        input_range = reduced_problem.input_range
    else:
        print(f"Using input range = {input_range}, " +
              "ignoring input range specified in " +
              f"{reduced_problem.__class__.__name__}")
    if (np.array(output_range) == None).any():  # noqa: E711
        assert hasattr(reduced_problem, "output_range")
        output_range = reduced_problem.output_range
    else:
        print(f"Using output range = {output_range}, " +
              "ignoring output range specified in " +
              f"{reduced_problem.__class__.__name__}")

    online_mu_scaled = (input_scaling_range[1] - input_scaling_range[0]) * \
        (online_mu - input_range[0, :]) / (input_range[1, :] -
                                           input_range[0, :]) + \
        input_scaling_range[0]
    online_mu_scaled_torch = \
        torch.from_numpy(online_mu_scaled).to(torch.float32)
    online_mu_scaled_torch = online_mu_scaled_torch.to(f"cuda:{cuda_rank}")

    with torch.no_grad():
        pred_scaled = model(online_mu_scaled_torch)
        pred_scaled_numpy = pred_scaled.cpu().detach().numpy()
        pred = (pred_scaled_numpy - output_scaling_range[0]) / \
            (output_scaling_range[1] - output_scaling_range[0]) * \
            (output_range[1] - output_range[0]) + \
            output_range[0]
        if error_mode is True:
            solution_reduced = pred
        else:
            solution_reduced = rbnicsx.online.create_vector(N)
            solution_reduced.array = pred
    return solution_reduced


# TODO error_analysis test and debugging
# NOTE Here error_anaysis_set is passed instead of only error_anaysis_mu
def error_analysis(reduced_problem, problem, error_analysis_set, model, N,
                   online_nn, cuda_rank_list, cpu_comm_list, gpu_comm, world_comm,
                   norm_error=None, reconstruct_solution=None,
                   input_scaling_range=None, output_scaling_range=None,
                   input_range=None, output_range=None, index=None):
    model.eval()
    itemsize = MPI.DOUBLE.Get_size()
    if world_comm.rank == 0:
        nbytes_rb_solution = error_analysis_set.shape[0] * itemsize * N
    else:
        nbytes_rb_solution = 0

    win = MPI.Win.Allocate_shared(nbytes_rb_solution, itemsize,
                                  comm=world_comm)
    buf, itemsize = win.Shared_query(0)
    rb_solution = np.ndarray(buffer=buf, dtype="d",
                             shape=(error_analysis_set.shape[0], N))

    # NOTE : gpu_comm is communicator not list of communicator
    if gpu_comm != MPI.COMM_NULL:
        gpu_indices = np.arange(gpu_comm.rank, error_analysis_set.shape[0],
                                gpu_comm.size)
        rb_solution[gpu_indices, :] = \
            online_nn(reduced_problem, problem,
                      error_analysis_set[gpu_indices, :], model, N,
                      cuda_rank_list[gpu_comm.rank], input_scaling_range,
                      output_scaling_range, input_range,
                      output_range, error_mode=True)

    world_comm.Barrier()

    if world_comm.rank == 0:
        nbytes_error_array = error_analysis_set.shape[0] * itemsize
    else:
        nbytes_error_array = 0

    win_error = MPI.Win.Allocate_shared(nbytes_error_array,
                                        itemsize, comm=world_comm)
    buf_error, itemsize = win_error.Shared_query(0)
    error_array = np.ndarray(buffer=buf_error, dtype="d",
                             shape=(error_analysis_set.shape[0], 1))

    for i in range(len(cpu_comm_list)):
        if cpu_comm_list[i] != MPI.COMM_NULL:
            cpu_indices = np.arange(i, error_analysis_set.shape[0],
                                    len(cpu_comm_list))
            for k in cpu_indices:
                fem_solution = problem.solve(error_analysis_set[k, :])
                if type(fem_solution) == tuple:
                    assert index is not None
                    fem_solution = fem_solution[index]
                ann_prediction = rbnicsx.online.create_vector(N)
                ann_prediction.array = rb_solution[k, :]
                if reconstruct_solution is None:
                    ann_reconstructed_solution = \
                        reduced_problem.reconstruct_solution(ann_prediction)
                else:
                    print(f"Using {reconstruct_solution.__name__}, " +
                          "ignoring RB to FEM solution construction specified in " +
                          f"{reduced_problem.__class__.__name__}")
                    ann_reconstructed_solution = \
                        reconstruct_solution(ann_prediction)
                if norm_error is None:
                    error_array[k] = \
                        reduced_problem.norm_error(fem_solution,
                                                   ann_reconstructed_solution)
                else:
                    print(f"Using {norm_error.__name__}, " +
                          "ignoring error norm specified in " +
                          f"{reduced_problem.__class__.__name__}")
                    error_array[k] = \
                        norm_error(fem_solution, ann_reconstructed_solution)

    world_comm.Barrier()
    return error_array


if __name__ == "__main__":

    world_comm = MPI.COMM_WORLD
    gpu_group0_procs = world_comm.group.Incl([0, 1, 2])
    gpu_group0_comm = world_comm.Create_group(gpu_group0_procs)

    num_input_samples = 8
    para_dim = 2
    len_rb_space = 5
    itemsize = MPI.DOUBLE.Get_size()

    if world_comm.rank == 0:
        nbytes_para = itemsize * para_dim * num_input_samples
    else:
        nbytes_para = 0

    win0 = MPI.Win.Allocate_shared(nbytes_para, itemsize,
                                   comm=MPI.COMM_WORLD)
    buf0, itemsize = win0.Shared_query(0)
    error_anaysis_mu = np.ndarray(buffer=buf0, dtype="d",
                                  shape=(num_input_samples, para_dim))

    if world_comm.rank == 0:
        error_anaysis_mu[:, :] = np.random.rand(num_input_samples, para_dim)

    world_comm.Barrier()

    cuda_rank = [0, 0, 0, 0]

    if gpu_group0_comm != MPI.COMM_NULL:
        indices = np.arange(gpu_group0_comm.rank, num_input_samples,
                            gpu_group0_comm.size)
        model = \
            HiddenLayersNet(para_dim, [30, 30], len_rb_space, Tanh()
                            ).to(f"cuda:{cuda_rank[gpu_group0_comm.rank]}")
        '''
        try:
            model.load_state_dict(torch.load("model.pth"))
        except:
            torch.save(model.state_dict(), "model.pth")
        '''
        model.load_state_dict(torch.load("model.pth"))

        input_range = np.vstack([np.array([0, 0]), np.array([1., 1.])])
        output_range = [0., -1.]
        input_scaling_range = \
            np.vstack([np.array([-1., -1.]), np.array([1., 1.])])
        output_scaling_range = [-1., 1.]

        if gpu_group0_comm.rank == 0:
            rb_solution_array = \
                online_nn(num_input_samples, num_input_samples,
                          error_anaysis_mu, model, len_rb_space,
                          cuda_rank[gpu_group0_comm.rank],
                          input_scaling_range=input_scaling_range,
                          output_scaling_range=output_scaling_range,
                          input_range=input_range, output_range=output_range,
                          error_mode=True)
            print(f"Prediction (error_mode = True): {rb_solution_array}")

        gpu_group0_comm.Barrier()

        for i in indices:
            rb_solution = \
                online_nn(num_input_samples, num_input_samples,
                          error_anaysis_mu[i, :], model, len_rb_space,
                          cuda_rank[gpu_group0_comm.rank],
                          input_scaling_range=input_scaling_range,
                          output_scaling_range=output_scaling_range,
                          input_range=input_range, output_range=output_range)
            print(f"Prediction online (error_mode = False): {rb_solution.array}")
