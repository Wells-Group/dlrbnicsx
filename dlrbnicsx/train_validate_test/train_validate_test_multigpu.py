import torch
import torch.distributed as dist
import torch.multiprocessing as mp  # noqa: F401
from torch.utils.data import DataLoader

import numpy as np
from mpi4py import MPI
import os

import rbnicsx.online

from dlrbnicsx.dataset.custom_partitioned_dataset_gpu import \
    CustomPartitionedDatasetGpu
from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory \
    import Tanh
from dlrbnicsx.interface.wrappers \
    import model_to_gpu, data_to_gpu, model_synchronise  # noqa: F401


def train_nn(reduced_problem, dataloader, model, loss_fn,
             optimizer, verbose=False, report=True):
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


def validate_nn(reduced_problem, dataloader, model, cuda_rank,
                loss_fn, verbose=False, report=True):
    # TODO add more loss functions including PINN
    # dataset_size = len(dataloader.dataset)
    model.eval()  # NOTE
    valid_loss = torch.tensor([0.]).to(f"cuda:{cuda_rank}")
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            valid_loss += loss_fn(pred, y)
    dist.barrier()
    if verbose is True:
        print(f"Validation loss before all_reduce: {valid_loss.item(): >7f}")

    dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)

    if verbose is True:
        print(f"Validation loss after all_reduce: {valid_loss.item(): >7f}")

    if report is True:
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
                   online_nn, cuda_num, cpu_comm_list, gpu_comm,
                   world_comm, cpu_indices, norm_error=None,
                   reconstruct_solution=None, input_scaling_range=None,
                   output_scaling_range=None, input_range=None,
                   output_range=None, index=None):

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
        model.eval()
        gpu_indices = np.arange(gpu_comm.rank, error_analysis_set.shape[0],
                                gpu_comm.size)
        rb_solution[gpu_indices, :] = \
            online_nn(reduced_problem, problem,
                      error_analysis_set[gpu_indices, :], model, N,
                      cuda_num, input_scaling_range,
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
            # cpu_indices = np.arange(i, error_analysis_set.shape[0],
            #                         len(cpu_comm_list))
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

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    # train_nn and validate_nn test
    world_comm = MPI.COMM_WORLD
    gpu_group0_procs = world_comm.group.Incl([0, 1, 2, 3])
    gpu_group0_comm = world_comm.Create_group(gpu_group0_procs)

    num_training_samples = 12
    input_features = 5
    output_features = 4

    class ReducedProblem(object):
        def __init__(self, para_dim):
            super().__init__()
            self.input_range = np.vstack((np.zeros([1, para_dim]),
                                          np.ones([1, para_dim])))
            self.input_scaling_range = [-1., 1.]
            self.output_range = [0., 1.]
            self.output_scaling_range = [-1., 1.]
            self.learning_rate = 1.e-4
            self.optimizer = "Adam"
            self.loss_fn = "MSE"

    class Problem(object):
        def __init__(self):
            super().__init__()

    problem = Problem()
    reduced_problem = ReducedProblem(input_features)
    cuda_rank = [0, 1, 2, 3]

    if gpu_group0_comm != MPI.COMM_NULL:
        dist.init_process_group("nccl", rank=gpu_group0_comm.rank,
                                world_size=gpu_group0_comm.size)

        X = np.random.randn(num_training_samples, input_features)
        X_recv = np.zeros_like(X)
        gpu_group0_comm.Allreduce(X, X_recv, op=MPI.SUM)
        weight = np.random.randn(input_features, output_features)
        weight_recv = np.zeros_like(weight)
        gpu_group0_comm.Allreduce(weight, weight_recv, op=MPI.SUM)
        X = X_recv
        weight = weight_recv
        np.save("input_set.npy", X)
        np.save("output_set.npy", weight)

        '''
        X = np.load("input_set.npy")
        weight = np.load("output_set.npy")
        '''

        Y = np.matmul(X, weight)

        indices = np.arange(gpu_group0_comm.rank, X.shape[0],
                            gpu_group0_comm.size)

        print(f"Gpu comm rank: {gpu_group0_comm.rank}, world comm rank: {world_comm.rank}, indices: {indices}")

        # NOTE Same customDataset and dataloader only for testing NOT in demos
        customDataset = \
            CustomPartitionedDatasetGpu(problem, reduced_problem,
                                        output_features, X, Y,
                                        indices, cuda_rank[gpu_group0_comm.rank])

        # NOTE shuffle=False in training only for testing NOT in demos
        train_dataloader = DataLoader(customDataset, batch_size=1000, shuffle=False)
        valid_dataloader = DataLoader(customDataset, shuffle=False)

        model = HiddenLayersNet(input_features, [], output_features, Tanh())
        model_to_gpu(model, cuda_rank=cuda_rank[gpu_group0_comm.rank])
        torch.save(model.state_dict(), "model_state_dict.pth")
        # model.load_state_dict(torch.load("model_state_dict.pth"))

        model_synchronise(model, verbose=True)

        max_iter = 10
        for current_iter in range(max_iter):
            print(f"Iteration: {current_iter+1}")
            train_loss = \
                train_nn(reduced_problem, train_dataloader, model)
            valid_loss = \
                validate_nn(reduced_problem, valid_dataloader, model,
                            cuda_rank[gpu_group0_comm.rank])

    world_comm.Barrier()

    try:
        for param in model.parameters():
            print(param)
    except:
        print(f"print failed on rank: {world_comm.rank}")

    # Error analysis test
    num_error_input_samples = 8
    para_dim = input_features
    len_rb_space = output_features
    itemsize = MPI.DOUBLE.Get_size()

    if world_comm.rank == 0:
        nbytes_para = itemsize * para_dim * num_error_input_samples
    else:
        nbytes_para = 0

    win0 = MPI.Win.Allocate_shared(nbytes_para, itemsize,
                                   comm=MPI.COMM_WORLD)
    buf0, itemsize = win0.Shared_query(0)
    error_anaysis_mu = np.ndarray(buffer=buf0, dtype="d",
                                  shape=(num_error_input_samples, para_dim))

    if world_comm.rank == 0:
        error_anaysis_mu[:, :] = \
            np.random.rand(num_error_input_samples, para_dim)

    world_comm.Barrier()

    if gpu_group0_comm != MPI.COMM_NULL:
        indices = np.arange(gpu_group0_comm.rank, num_error_input_samples,
                            gpu_group0_comm.size)

        if gpu_group0_comm.rank == 0:
            rb_solution_array = \
                online_nn(reduced_problem, problem,
                          error_anaysis_mu, model, len_rb_space,
                          cuda_rank[gpu_group0_comm.rank],
                          input_scaling_range=reduced_problem.input_scaling_range,
                          output_scaling_range=reduced_problem.output_scaling_range,
                          input_range=reduced_problem.input_range,
                          output_range=reduced_problem.output_range,
                          error_mode=True)
            print(f"Prediction (error_mode = True): {rb_solution_array}")

        gpu_group0_comm.Barrier()

        for i in indices:
            rb_solution = \
                online_nn(reduced_problem, problem,
                          error_anaysis_mu[i, :], model, len_rb_space,
                          cuda_rank[gpu_group0_comm.rank],
                          input_scaling_range=reduced_problem.input_scaling_range,
                          output_scaling_range=reduced_problem.output_scaling_range,
                          input_range=reduced_problem.input_range,
                          output_range=reduced_problem.output_range)
            print(f"Prediction online (error_mode = False): {rb_solution.array}")
