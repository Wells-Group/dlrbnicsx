import unittest

from mpi4py import MPI

import torch
import torch.distributed as dist
import torch.multiprocessing as mp  # noqa: F401

import numpy as np
import rbnicsx.online

from dlrbnicsx.dataset.custom_partitioned_dataset import CustomDataset
from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory import Tanh
from dlrbnicsx.dataset.custom_partitioned_dataset import CustomPartitionedDataset
from dlrbnicsx.train_validate_test.train_validate_test import train_nn, validate_nn, error_analysis, online_nn
from dlrbnicsx.train_validate_test.train_validate_test_distributed import train_nn as train_nn_distributed
from dlrbnicsx.train_validate_test.train_validate_test_distributed import validate_nn as validate_nn_distributed
from dlrbnicsx.train_validate_test.train_validate_test_distributed import error_analysis as error_analysis_distributed
from dlrbnicsx.train_validate_test.train_validate_test_distributed import online_nn as online_nn_distributed
from dlrbnicsx.interface.wrappers import init_cpu_process_group

class TestTrainValidateTest(unittest.TestCase):
    """Testing routine for Train Validate Test routines
    """
    def test_train_validate_test(self):
        """Serial case
        """
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
                # self.optimizer = torch.optim.SGD()
                # self.loss_fn = torch.nn.MSELoss()

        dim_in = input_training_data.shape[1]
        dim_out = output_training_data.shape[1]

        model = HiddenLayersNet(dim_in, [4], dim_out, Tanh())

        problem = Problem()
        reduced_problem = ReducedProblem()

        input_training_data = \
            np.random.default_rng().uniform(0., 1.,
                                            (10, 4)).astype("f")
        output_training_data = \
            np.random.default_rng().uniform(0., 1.,
                                            (input_training_data.shape[0],
                                            6)).astype("f")

        # NOTE Updating output_range based on the computed values
        reduced_problem.output_range[0] = \
            np.min(output_training_data)
        reduced_problem.output_range[1] = \
            np.max(output_training_data)

        customDataset = \
            CustomDataset(reduced_problem,
                          input_training_data,
                          output_training_data)

        train_dataloader = \
            torch.utils.data.DataLoader(customDataset,
                                        batch_size=5,
                                        shuffle=True)

        input_validation_data = \
            np.random.default_rng().uniform(0., 1.,
                                            (3, input_training_data.shape[1])
                                            ).astype("f")
        output_validation_data = \
            np.random.default_rng().uniform(0., 1.,
                                            (input_validation_data.shape[0],
                                            output_training_data.shape[1])
                                            ).astype("f")

        customDataset = CustomDataset(reduced_problem,
                                      input_validation_data,
                                      output_validation_data)
        valid_dataloader = \
            torch.utils.data.DataLoader(customDataset,
                                        batch_size=100,
                                        shuffle=False)

        max_epochs = 5  # 20000

        loss_func = torch.nn.MSELoss()
        optimiser = torch.optim.SGD(model.parameters(),
                                    lr=reduced_problem.learning_rate)

        for epoch in range(max_epochs):
            print(f"Epoch {epoch+1} of Maximum epochs {max_epochs}")
            train_loss = train_nn(reduced_problem,
                                  train_dataloader,
                                  model, loss_func, 
                                  optimiser)
            valid_loss = validate_nn(reduced_problem,
                                     valid_dataloader,
                                     model, loss_func)

        online_mu = \
            np.random.default_rng().uniform(0., 1.,
                                            input_training_data.shape[1])
        _ = online_nn(reduced_problem, problem, online_mu,
                      model, dim_out).array
        
    def test_train_validate_test_distributed(self):
        """Distributed case
        """
        '''
        NOTE
        Run below code with 2 processes

        ```mpiexec -n 2 python3 train_validate_test_distributed.py```

        and verify from printed terminal output whether the params
        after all_reduce are same in all processes.

        Higher number of processes could also be used instead of only 2.
        '''

        comm = MPI.COMM_WORLD
        init_cpu_process_group(comm)
        
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

        indices_train = \
            np.arange(comm.rank, input_training_data.shape[0],
                      comm.size)

        custom_partitioned_dataset = \
            CustomPartitionedDataset(reduced_problem,
                                     input_training_data,
                                     output_training_data,
                                     indices_train)

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
        dist.all_reduce(input_validation_data,
                        op=dist.ReduceOp.MAX)
        dist.all_reduce(output_validation_data,
                        op=dist.ReduceOp.MAX)
        input_validation_data = \
            input_validation_data.detach().numpy()
        output_validation_data = \
            output_validation_data.detach().numpy()

        indices_val = \
            np.arange(comm.rank, input_validation_data.shape[0],
                      comm.size)

        custom_partitioned_dataset = \
            CustomPartitionedDataset(reduced_problem,
                                     input_validation_data,
                                     output_validation_data,
                                     indices_val)
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
        loss_func = torch.nn.MSELoss()
        optimiser = torch.optim.SGD(model.parameters(),
                                    lr=reduced_problem.learning_rate)


        for epoch in range(max_epochs):
            print(f"Rank {dist.get_rank()} Epoch {epoch+1} of Maximum " +
                f"epochs {max_epochs}")
            train_loss = train_nn_distributed(reduced_problem,
                                              train_dataloader,
                                              model, loss_func,
                                              optimiser)
            valid_loss = validate_nn_distributed(reduced_problem,
                                                 valid_dataloader,
                                                 model, loss_func)

        online_mu = \
            np.random.default_rng().uniform(0., 1., input_training_data.shape[1])
        _ = online_nn_distributed(reduced_problem, problem, online_mu, model, dim_out)

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



if __name__ == "__main__":
    unittest.main()