## POD-ANN for the Stokes equation with geometric parametrization (Distributed) ##

### 1. Problem statement

We solve the problem reported in **POD-ANN for the Stokes equation with geometric parametrization**. However, we now compute one snapshot on one process for the POD and use data parallel training of the neural network using **MPI**.

### 2. Implementation

Below are the notable differences:

- We now use the distributed modules

```
from dlrbnicsx.dataset.custom_partitioned_dataset \
    import CustomPartitionedDataset

from dlrbnicsx.train_validate_test.train_validate_test_distributed \
    import train_nn, validate_nn, online_nn, error_analysis
```

- MPI communicator for mesh:

We create separate mesh on each process by ```mesh_comm = MPI.COMM_SELF``` to ```dolfinx.io.gmshio.read_from_msh```.

```
# Read mesh
mesh_comm = MPI.COMM_SELF  # NOTE
gdim = 2
gmsh_model_rank = 0
mesh, cell_tags, facet_tags = \
    dolfinx.io.gmshio.read_from_msh("mesh_data/domain_geometry.msh",
                                    mesh_comm, gmsh_model_rank, gdim=gdim)
```

- POD training set:

We now generate samples for POD on process with rank 0 and Bcast the sample to other processes.

```
world_comm = MPI.COMM_WORLD

.
.
.

# Generate samples on rank 0 and Bcast to other processes
if rank == 0:
    training_set = generate_training_set()
else:
    training_set = np.zeros_like(generate_training_set())

world_comm.Bcast(training_set, root=0)
```

Next, each process computes part of POD snapshot matrix according to ```training_set_indices```. The entries in the snapshot are then collected together using MPI.
```
training_set_indices = np.arange(rank, training_set.shape[0], size)

.
.
.


for mu_index in training_set_indices:
    print(rbnicsx.io.TextLine(str(mu_index+1) + f"/{training_set.shape[0]}",
                              fill="#"))
    print(f"High fidelity solve for mu = {training_set[mu_index,:]}")
    solution_vel_mu, solution_pre_mu = \
        problem_parametric.solve(training_set[mu_index, :])
    training_set_solutions_u[mu_index, :] = solution_vel_mu.x.array
    training_set_solutions_p[mu_index, :] = solution_pre_mu.x.array

.
.
.

world_comm.Allreduce(training_set_solutions_u, training_set_solutions_recv_u,
                     op=MPI.SUM)
world_comm.Allreduce(training_set_solutions_p, training_set_solutions_recv_p,
                     op=MPI.SUM)
```

**NOTE**: Eigenvectors and Eigenvalues for POD are computed on each of the processes.

- Neural network training set:

The samples for ANN input training set are generated on process with rank 0 and Bcast to other processes.

```
# Generate ANN input TRAINING samples on the rank 0 and Bcast to other processes
if rank == 0:
    ann_input_set = generate_ann_input_set(samples=[8, 8])
    np.random.shuffle(ann_input_set)
else:
    ann_input_set = np.zeros_like(generate_ann_input_set(samples=[8, 8]))

world_comm.Bcast(ann_input_set, root=0)
```

Similar to POD, each process computes part of the output training set according to ```indices```.

```
def generate_ann_output_set(problem, reduced_problem, input_set,
                            indices, mode=None):
    # Compute output set for ANN based on input set
    output_set_u = np.zeros([input_set.shape[0],
                             len(reduced_problem._basis_functions_u)])
    output_set_p = np.zeros([input_set.shape[0],
                             len(reduced_problem._basis_functions_p)])
    rb_size_u = len(reduced_problem._basis_functions_u)
    rb_size_p = len(reduced_problem._basis_functions_p)
    for i in indices:
        if mode is None:
            print(f"Parameter number {i+1} of ")
            print(f"{input_set.shape[0]}: {input_set[i, :]}")
        else:
            print(f"{mode} parameter number {i+1} of ")
            print(f"{input_set.shape[0]}: {input_set[i, :]}")
        print(input_set[i, :])
        (solution_u, solution_p) = problem.solve(input_set[i, :])
        print(solution_u.x.array)
        print(solution_p.x.array)
        output_set_u[i, :] = \
            reduced_problem.project_snapshot_u(solution_u, rb_size_u).array
        output_set_p[i, :] = \
            reduced_problem.project_snapshot_p(solution_p, rb_size_p).array
    return output_set_u, output_set_p

.
.
.

indices = np.arange(rank, ann_input_set.shape[0], size)
```

The output training set is collected using MPI.

```
# Start zero matrix for MPI all reduce and collect output set using MPI.SUM
ann_output_set_recv_u = np.zeros_like(ann_output_set_u)
ann_output_set_recv_p = np.zeros_like(ann_output_set_p)
world_comm.Barrier()
world_comm.Allreduce(ann_output_set_u, ann_output_set_recv_u, op=MPI.SUM)
world_comm.Allreduce(ann_output_set_p, ann_output_set_recv_p, op=MPI.SUM)
```

- Dataset distribution across different processes:

We now use ```CustomPartitionedDataset``` instead of ```CustomDataset```. This will distribute the input data across different processes. Neural network will perform data parallel training of neural network using chunk of available data with gradient synchronisation at the end of each iteration.

- Neural network model initialisation:

Since, the neural network is initialised on each process using random neural network parameters (weights and biases), it is important to synchronise these random parameters before start of the training for example, by averaging these parameters.

```
for param in model_u.parameters():
    print(f"Rank {rank} \n Params before all_reduce: {param.data}")
    # NOTE This ensures that models in all processes start with same weights and biases
    dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
    param.data /= dist.get_world_size()
    print(f"Rank {rank} \n Params after all_reduce: {param.data}")
```

- Error analysis:

The samples for error analysis are generated on process with rank 0 and Bcast to other processes. Each process computes error for part of the error analysis set as specified in ```indices```.

```
if rank == 0:
    error_analysis_set_u = generate_ann_input_set(samples=[3, 3])
else:
    error_analysis_set_u = np.zeros_like(generate_ann_input_set(samples=[3, 3]))

world_comm.Bcast(error_analysis_set_u, root=0)

.
.
.

indices = np.arange(rank, error_analysis_set_u.shape[0], size)
```

- Online phase: 

It is performed on only one process (for example, rank 0).

```
# Online phase

if rank == 0:

.
.
.

```
