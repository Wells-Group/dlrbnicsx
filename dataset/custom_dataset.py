import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader # wrapper for iterables over dataset

import numpy as np

class CustomDataset(Dataset):
    def __init__(self, problem, reduced_problem, N):
        '''
        input_scaling_range: (2,num_para) np.ndarray, row 0 are the min_values and row 1 are the max_values
        output_scaling_range: (2,1) np.ndarray, row 0 are the min_values and row 1 are the max_values
        N: int, size of reduced basis
        '''
        assert hasattr(problem,"input_range")
        assert hasattr(problem,"input_scaling_range")
        assert hasattr(problem,"input_file_path")
        assert hasattr(problem,"output_range")
        assert hasattr(problem,"output_scaling_range")
        self.problem = problem
        self.reduced_problem = reduced_problem
        self.N = N
        
    def __len__(self):
        return np.load(self.problem.input_file_path).shape[0]
    
    def __getitem__(self,idx):
        input_data = np.load(self.problem.input_file_path)[idx,:]
        label = self.reduced_problem.project_snapshot(self.problem.solve(input_data),self.N).array
        return self.transform(input_data), self.target_transform(label)
            
    def transform(self, input_data):
        input_data_scaled = (self.problem.input_scaling_range[1] - self.problem.input_scaling_range[0]) * (input_data - self.problem.input_range[0,:]) / (self.problem.input_range[1,:] - self.problem.input_range[0,:]) + self.problem.input_scaling_range[0]
        return torch.from_numpy(input_data_scaled)
    
    def target_transform(self,label):
        output_data_scaled = (self.problem.output_scaling_range[1] - self.problem.output_scaling_range[0]) * (label - self.problem.output_range[0]) / (self.problem.output_range[1] - self.problem.output_range[0]) + self.problem.output_scaling_range[0]
        return torch.from_numpy(output_data_scaled)
    
    def reverse_transform(self, input_data_scaled): # TODO verify formula
        input_data_scaled = input_data_scaled.detach().numpy()
        input_data = (input_data_scaled - self.problem.input_scaling_range[0]) * (self.problem.input_range[1,:] - self.problem.input_range[0,:]) / (self.problem.input_scaling_range[1] - self.problem.input_scaling_range[0]) + self.problem.input_range[0,:]
        return input_data
    
    def reverse_target_transform(self,output_data_scaled):# TODO verify formula
        output_data_scaled = output_data_scaled.detach().numpy()
        output_data = (output_data_scaled - self.problem.output_scaling_range[0]) * (self.problem.output_range[1] - self.problem.output_range[0]) / (self.problem.output_scaling_range[1] - self.problem.output_scaling_range[0]) + self.problem.output_range[0]
        return output_data

'''
class Problem(object):
    def __init__(self):
        super().__init__()
        para_dim = 17
        self.input_range = np.vstack((np.zeros([1,para_dim]),np.ones([1,para_dim])))
        print(f"input_range shape: {self.input_range.shape}")
        self.input_scaling_range = [0.2,0.7]
        self.output_range = np.vstack((np.zeros([1,2*para_dim])-1.,np.ones([1,2*para_dim])))
        self.output_scaling_range = [0.3,0.8]
        self.input_file_path = "input_data.npy"
    def solve(self,input_para):
        return np.hstack((10. * input_para - 1.,10. * input_para + 1.))

input_data = np.random.rand(41,17)
np.save("input_data.npy",input_data)

problem = Problem()
customDataset = CustomDataset(problem)

train_dataloader = DataLoader(customDataset, batch_size=9, shuffle=True)
test_dataloader = DataLoader(customDataset, batch_size=7)

for X,y in train_dataloader:
    print(f"Shape of training set: {X.shape}")
    print(f"Training set requires grad: {X.requires_grad}")
    #print(f"X: {X}")
    print(f"Shape of training set: {y.shape}")
    print(f"Training set requires grad: {y.requires_grad}")
    #print(f"y: {y}")

for X,y in test_dataloader:
    print(f"Shape of test set: {X.shape}")
    print(f"Testing set requires grad: {X.requires_grad}")
    #print(f"X: {X}")
    print(f"Shape of test set: {y.shape}")
    print(f"Testing set requires grad: {y.requires_grad}")
    #print(f"y: {y}")
'''
