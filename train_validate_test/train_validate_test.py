import torch
import numpy as np
import rbnicsx

def train_nn(reduced_problem, problem, dataloader, model, loss_fn, optimizer, device=None):
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # TODO add more loss functions including PINN
    if reduced_problem.loss_fn == "MSE":
        loss_fn = torch.nn.MSELoss() #  TODO also add reduction argument
    # TODO add more optimizers
    if reduced_problem.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    elif reduced_problem.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr) #  TODO also add momentum argument
    else:
        NotImplementedError(f"Optimizer {reduced_problem.optimizer} is not implemented")
    # TODO add L1/L2 and more rgularisations WITHOUT weight decay
    lr = reduced_problem.learning_rate
    size = len(dataloader.dataset)
    model.train() # NOTE
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y) # TODO relative loss??
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss: >7f} [{current:>5d}]/[{size:>5d}]")
    return loss

def validate_nn(reduced_problem, problem, dataloader, model, loss_fn, device=None):
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # TODO add more loss functions including PINN
    if reduced_problem.loss_fn == "MSE":
        loss_fn = torch.nn.MSELoss() #  TODO also add reduction argument
    num_batches = len(dataloader)
    model.eval() # NOTE
    valid_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            valid_loss += loss_fn(pred, y).item() # TODO relative loss??
    print(f"Validation loss: {valid_loss: >7f}")
    return valid_loss

def online_nn(reduced_problem, problem, online_mu, model, N, device=None):
    '''
    online_mu: np.ndarray [1,num_para] representing online parameter
    '''
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    online_mu_scaled = (problem.input_scaling_range[1] - problem.input_scaling_range[0]) * (online_mu - problem.input_range[0,:]) / (problem.input_range[1,:] - problem.input_range[0,:]) + problem.input_scaling_range[0]
    online_mu_scaled_torch = torch.from_numpy(online_mu_scaled)
    with torch.no_grad():
        X = online_mu_scaled_torch.to(device)
        pred_scaled = model(X)
        pred_scaled_numpy = pred_scaled.detach().numpy()
        pred = (pred_scaled_numpy - problem.output_scaling_range[0]) * (problem.output_range[1] - problem.output_range[0]) / (problem.output_scaling_range[1] - problem.output_scaling_range[0]) + problem.output_range[0]
        solution_reduced = rbnicsx.online.create_vector(N)
        solution_reduced.array = pred
    return solution_reduced

def error_analysis(reduced_problem, problem, error_analysis_mu, model, N, online_nn, device=None):
    '''
    error_analysis_mu: np.ndarray of size [1,num_para] representing parameter set at which error analysis needs to be evaluated
    problem: full order model with method norm_error(fem_solution,ann_reconstructed_solution) and methods required for online_nn
    '''
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    ann_prediction = online_nn(reduced_problem, problem, error_analysis_mu, model, N, device=device)
    ann_reconstructed_solution = reduced_problem.reconstruct_solution(ann_prediction)
    fem_solution = problem.solve(error_analysis_mu)
    error = problem.norm_error(fem_solution,ann_reconstructed_solution)
    return error
