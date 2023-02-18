import torch
import numpy as np
import rbnicsx


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
        optimizer.step()

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
            valid_loss += loss_fn(pred, y).item()/loss_fn(torch.zeros_like(y), y).item()
    print(f"Validation loss: {valid_loss: >7f}")
    return valid_loss


def online_nn(reduced_problem, problem, online_mu, model, N, device=None, input_scaling_range=None, output_scaling_range=None, input_range=None, output_range=None):
    '''
    Online phase
    Inputs:
        online_mu: np.ndarray [1,num_para] representing online parameter
        reduced_problem: reduced problem with attributes:
            input_scaling_range: (2,num_para) np.ndarray, row 0 are the SCALED INPUT min_values and row 1 are the SCALED INPUT max_values
            output_scaling_range: (2,num_para) np.ndarray, row 0 are the SCALED OUTPUT min_values and row 1 are the SCALED OUTPUT max_values
            input_range: (2,num_para) np.ndarray, row 0 are the ACTUAL INPUT min_values and row 1 are the ACTUAL INPUT max_values
            output_range: (2,num_para) np.ndarray, row 0 are the ACTUAL OUTPUT min_values and row 1 are the ACTUAL OUTPUT max_values
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
        error_analysis_mu: np.ndarray of size [1,num_para] representing parameter set at which error analysis needs to be evaluated
        problem: full order model with method:
            norm_error(fem_solution,ann_reconstructed_solution) and methods required for online_nn (Used ONLY if norm_error is not specified)
            solve: method to compute full order model solution
        reduced_problem: reduced problem with attributes:
            reconstruct_solution: Reconstruct FEM solution from reduced basis solution
            input_scaling_range: (2,num_para) np.ndarray, row 0 are the SCALED INPUT min_values and row 1 are the SCALED INPUT max_values
            output_scaling_range: (2,num_para) np.ndarray, row 0 are the SCALED OUTPUT min_values and row 1 are the SCALED OUTPUT max_values
            input_range: (2,num_para) np.ndarray, row 0 are the ACTUAL INPUT min_values and row 1 are the ACTUAL INPUT max_values
            output_range: (2,num_para) np.ndarray, row 0 are the ACTUAL OUTPUT min_values and row 1 are the ACTUAL OUTPUT max_values
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
