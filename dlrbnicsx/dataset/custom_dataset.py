import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np


class CustomDataset(Dataset):
    def __init__(self, reduced_problem, input_set,
                 output_set, input_scaling_range=None,
                 output_scaling_range=None, input_range=None,
                 output_range=None, verbose=True):
        '''
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

        input_set: numpy array of ACTUAL INPUT

        output_set: numpy array of ACTUAL OUTPUT
        '''
        self.reduced_problem = reduced_problem
        self.verbose = verbose
        if type(input_set) == np.ndarray:
            if self.verbose is True:
                print("Converting input set from numpy array to torch tensor")
            self.input_set = torch.from_numpy(input_set)#.to(torch.float32)
        else:
            self.input_set = input_set
        if type(output_set) == np.ndarray:
            if self.verbose is True:
                print("Converting output set from numpy array to torch tensor")
            self.output_set = torch.from_numpy(output_set)#.to(torch.float32)
        else:
            self.output_set = output_set

        if isinstance(input_scaling_range, list):
            if self.verbose is True:
                print(f"Using input scaling range {input_scaling_range}")
            self.input_scaling_range = \
                torch.from_numpy(np.array(input_scaling_range))
        elif isinstance(input_scaling_range, np.ndarray):
            if self.verbose is True:
                print(f"Using input scaling range {input_scaling_range}")
            self.input_scaling_range = \
                torch.from_numpy(input_scaling_range)
        elif isinstance(input_scaling_range, torch.Tensor):
            if self.verbose is True:
                print(f"Using input scaling range {input_scaling_range}")
            self.input_scaling_range = \
                input_scaling_range
        else:
            if self.verbose is True:
                print(f"Using input scaling range {reduced_problem.input_scaling_range}")
            if isinstance(reduced_problem.input_scaling_range, list):
                self.input_scaling_range = \
                    torch.from_numpy(np.array(reduced_problem.input_scaling_range))
            elif isinstance(reduced_problem.input_scaling_range, np.ndarray):
                self.input_scaling_range = torch.from_numpy(reduced_problem.input_scaling_range)
            elif isinstance(reduced_problem.input_scaling_range, torch.tensor):
                self.input_scaling_range = reduced_problem.input_scaling_range

        if isinstance(output_scaling_range, list):
            if self.verbose is True:
                print(f"Using output scaling range {output_scaling_range}")
            self.output_scaling_range = \
                torch.from_numpy(np.array(output_scaling_range))
        elif isinstance(output_scaling_range, np.ndarray):
            if self.verbose is True:
                print(f"Using output scaling range {output_scaling_range}")
            self.output_scaling_range = \
                torch.from_numpy(output_scaling_range)
        elif isinstance(output_scaling_range, torch.Tensor):
            if self.verbose is True:
                print(f"Using output scaling range {output_scaling_range}")
            self.output_scaling_range = \
                output_scaling_range
        else:
            if self.verbose is True:
                print(f"Using output scaling range {reduced_problem.output_scaling_range}")
            if isinstance(reduced_problem.output_scaling_range, list):
                self.output_scaling_range = \
                    torch.from_numpy(np.array(reduced_problem.output_scaling_range))
            elif isinstance(reduced_problem.output_scaling_range, np.ndarray):
                self.output_scaling_range = torch.from_numpy(reduced_problem.output_scaling_range)
            elif isinstance(reduced_problem.output_scaling_range, torch.tensor):
                self.output_scaling_range = reduced_problem.output_scaling_range

        if isinstance(input_range, list):
            if self.verbose is True:
                print(f"Using input range {input_range}")
            self.input_range = \
                torch.from_numpy(np.array(input_range))
        elif isinstance(input_range, np.ndarray):
            if self.verbose is True:
                print(f"Using input range {input_range}")
            self.input_range = \
                torch.from_numpy(input_range)
        elif isinstance(input_range, torch.Tensor):
            if self.verbose is True:
                print(f"Using input range {input_range}")
            self.input_range = input_range
        else:
            if self.verbose is True:
                print(f"Using input range {reduced_problem.input_range}")
            if isinstance(reduced_problem.input_range, list):
                self.input_range = \
                    torch.from_numpy(np.array(reduced_problem.input_range))
            elif isinstance(reduced_problem.input_range, np.ndarray):
                self.input_range = torch.from_numpy(reduced_problem.input_range)
            elif isinstance(reduced_problem.input_range, torch.tensor):
                self.input_range = reduced_problem.input_range

        if isinstance(output_range, list):
            if self.verbose is True:
                print(f"Output range {output_range}")
            self.output_range = \
                torch.from_numpy(np.array(output_range))
        elif isinstance(output_range, np.ndarray):
            if self.verbose is True:
                print(f"Output range {output_range}")
            self.output_range = \
                torch.from_numpy(output_range)
        elif isinstance(output_range, torch.Tensor):
            if self.verbose is True:
                print(f"Output range {output_range}")
            self.output_range = output_range
        else:
            if self.verbose is True:
                print(f"Output range {reduced_problem.output_range}")
            if isinstance(reduced_problem.output_range, list):
                self.output_range = \
                    torch.from_numpy(np.array(reduced_problem.output_range))
            elif isinstance(reduced_problem.output_range, np.ndarray):
                self.output_range = torch.from_numpy(reduced_problem.output_range)
            elif isinstance(reduced_problem.output_range, torch.tensor):
                self.output_range = reduced_problem.output_range


    def __len__(self):
        input_length = self.input_set.shape[0]
        return input_length

    def __getitem__(self, idx):
        input_data = self.input_set[idx, :]
        label = self.output_set[idx, :]
        return self.transform(input_data), self.target_transform(label)

    def transform(self, input_data):
        input_data_scaled = (self.input_scaling_range[1] -
                             self.input_scaling_range[0]) * \
            (input_data - self.input_range[0, :]) / \
            (self.input_range[1, :] - self.input_range[0, :]) + \
            self.input_scaling_range[0]
        if input_data_scaled.dtype != torch.float32:
            input_data_scaled = input_data_scaled.to(torch.float32)
        return input_data_scaled

    def target_transform(self, label):
        output_data_scaled = (self.output_scaling_range[1] - self.output_scaling_range[0]) * (
            label - self.output_range[0]) / (self.output_range[1] - self.output_range[0]) + self.output_scaling_range[0]
        if output_data_scaled.dtype != torch.float32:
            output_data_scaled = output_data_scaled.to(torch.float32)
        return output_data_scaled

    def reverse_transform(self, input_data_scaled):
        input_data = (input_data_scaled - self.input_scaling_range[0]) * \
            (self.input_range[1, :] - self.input_range[0, :]) / \
            (self.input_scaling_range[1] - self.input_scaling_range[0]) + \
            self.input_range[0, :]
        if input_data.dtype != torch.float32:
            input_data = input_data.to(torch.float32)
        return input_data

    def reverse_target_transform(self, output_data_scaled):
        output_data = \
            (output_data_scaled - self.output_scaling_range[0]) * \
            (self.output_range[1] - self.output_range[0]) / \
            (self.output_scaling_range[1] - self.output_scaling_range[0]) \
            + self.output_range[0]
        if output_data.dtype != torch.float32:
            output_data = output_data.to(torch.float32)
        return output_data


if __name__ == "__main__":

    class ReducedProblem(object):
        def __init__(self, para_dim):
            super().__init__()
            self.input_range = np.vstack((np.zeros([1, para_dim]),
                                          np.ones([1, para_dim])))
            self.input_scaling_range = [-1., 1.]
            self.output_range = [0., 1.]
            self.output_scaling_range = [-1., 1.]

    input_data = np.random.uniform(0., 1., [100, 17])
    output_data = np.random.uniform(0., 1.,
                                    [input_data.shape[0], 7])

    reduced_problem = ReducedProblem(input_data.shape[1])

    customDataset = CustomDataset(reduced_problem,
                                  input_data, output_data)

    train_dataloader = DataLoader(customDataset, batch_size=3,
                                  shuffle=True)
    test_dataloader = DataLoader(customDataset, batch_size=2)

    for X, y in train_dataloader:
        print(f"Shape of training set: {X.shape}")
        print(f"Training set requires grad: {X.requires_grad}")
        print(f"X dtype: {X.dtype}")
        # print(f"X: {X}")
        print(f"Shape of training set: {y.shape}")
        print(f"Training set requires grad: {y.requires_grad}")
        print(f"y dtype: {y.dtype}")
        # print(f"y: {y}")
        break

    for X, y in test_dataloader:
        print(f"Shape of test set: {X.shape}")
        print(f"Testing set requires grad: {X.requires_grad}")
        # print(f"X: {X}")
        print(f"Shape of test set: {y.shape}")
        print(f"Testing set requires grad: {y.requires_grad}")
        # print(f"y: {y}")
        break
