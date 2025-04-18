# DLRBniCSx - Dataset - Custom Dataset - Custom Dataset (Distributed)

This module implements class ```CustomDataset```, which stores the samples and their corresponding labels. ```CustomDataset``` contains methods:

* ```__len__```: Returns size of the dataset
* ```__getitem__```: Gives $i^{th}$ SCALED sample from the dataset
* ```transform```: Scaling of given INPUT data within given range
* ```reverse_transform```: Reverse scaling of SCALED INPUT data to the original data range
* ```transform```: Scaling of given OUTPUT data within given range
* ```reverse_transform```: Reverse scaling of SCALED OUTPUT data to the original data range

```DataLoader``` can be used to wrap an iterable around the ```CustomDataset``` to enable easy access to the samples during training and validation. Typical application of the ```CustomDataset``` is as follows:

```
customDataset = CustomDataset(problem, reduced_problem, N, input_data, output_data)
train_dataloader = DataLoader(customDataset, batch_size=training_batch_size, shuffle=True)
test_dataloader = DataLoader(customDataset)
```

The ```CustomPartitionedDataset``` is used to distribute the dataset across different processes for distributed training. ```CustomPartitionedDataset``` is inherited from ```CustomDataset``` class, with overwritten methods:

* ```__len__```: Returns size of the dataset within given process
* ```__getitem__```: Returns $i^{th}$ SCALED sample from the dataset within given process

```DataLoader``` can be used to wrap an iterable around the ```CustomPartitionedDataset``` to enable easy access to the samples during training and validation. Typical application of the ```CustomPartitionedDataset``` is as follows:

```
custom_partitioned_dataset = CustomPartitionedDataset(problem, reduced_problem, N, input_data, output_data)
train_dataloader = DataLoader(custom_partitioned_dataset, batch_size=training_batch_size, shuffle=True)
test_dataloader = DataLoader(custom_partitioned_dataset)
```
