import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np


class CustomDataset(Dataset):
    def __init__(self, reduced_problem, input_set,
                 output_set, input_scaling_range=None,
                 output_scaling_range=None, input_range=None,
                 output_range=None, verbose=False):
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
        self.input_set = torch.from_numpy(input_set)#.to(torch.float32)
        self.output_set = torch.from_numpy(output_set)#.to(torch.float32)

        if type(input_scaling_range) == list:
            input_scaling_range = np.array(input_scaling_range)
        if type(output_scaling_range) == list:
            output_scaling_range = np.array(output_scaling_range)
        if type(input_range) == list:
            input_range = np.array(input_range)
        if type(output_range) == list:
            output_range = np.array(output_range)

        if (np.array(input_scaling_range) == None).any():  # noqa: E711
            assert hasattr(self.reduced_problem, "input_scaling_range")
            if type(self.reduced_problem.input_scaling_range) == list:
                self.input_scaling_range = \
                    torch.from_numpy(np.array(self.reduced_problem.input_scaling_range))#.to(torch.float32)
            else:
                self.input_scaling_range = \
                    torch.from_numpy(self.reduced_problem.input_scaling_range)#.to(torch.float32)
        else:
            print(f"Using input scaling range = {input_scaling_range}," +
                  "ignoring input scaling range specified in " +
                  f"{reduced_problem.__class__.__name__}")
            self.input_scaling_range = \
                torch.from_numpy(input_scaling_range).to(torch.float32)

        if (np.array(output_scaling_range) == None).any():  # noqa: E711
            assert hasattr(reduced_problem, "output_scaling_range")
            if type(self.reduced_problem.output_scaling_range) == list:
                self.output_scaling_range = \
                    torch.from_numpy(np.array(self.reduced_problem.output_scaling_range))#.to(torch.float32)
            else:
                self.output_scaling_range = \
                    torch.from_numpy(self.reduced_problem.output_scaling_range)#.to(torch.float32)
        else:
            print(f"Using output scaling range = {output_scaling_range}," +
                  "ignoring output scaling range specified in " +
                  f"{reduced_problem.__class__.__name__}")
            self.output_scaling_range = \
                torch.from_numpy(output_scaling_range)#.to(torch.float32)

        if (np.array(input_range) == None).any():  # noqa: E711
            assert hasattr(self.reduced_problem, "input_range")
            if type(self.reduced_problem.input_range) == list:
                self.input_range = \
                    torch.from_numpy(np.array(self.reduced_problem.input_range))#.to(torch.float32)
            else:
                self.input_range = \
                    torch.from_numpy(self.reduced_problem.input_range)#.to(torch.float32)
        else:
            print(f"Using input range = {input_range}," +
                  "ignoring input range specified in " +
                  f"{reduced_problem.__class__.__name__}")
            self.input_range = \
                torch.from_numpy(input_range).to(torch.float32)

        if (np.array(output_range) == None).any():  # noqa: E711
            assert hasattr(self.reduced_problem, "output_range")
            if type(self.reduced_problem.output_range) == list:
                self.output_range = \
                    torch.from_numpy(np.array(self.reduced_problem.output_range))#.to(torch.float32)
            else:
                self.output_range = \
                    torch.from_numpy(self.reduced_problem.output_range)#.to(torch.float32)
        else:
            print(f"Using output range = {output_range}, " +
                  "ignoring output range specified in " +
                  f"{reduced_problem.__class__.__name__}")
            self.output_range = \
                torch.from_numpy(output_range)#.to(torch.float32)

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
    output_data = np.random.uniform(0., 1., [input_data.shape[0], 7])

    reduced_problem = ReducedProblem(input_data.shape[1])

    # With numpy array as input-output
    print("\n With numpy array as input-output \n")
    customDataset = CustomDataset(reduced_problem,
                                  input_data, output_data)

    train_dataloader = DataLoader(customDataset, batch_size=3, shuffle=True)
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
